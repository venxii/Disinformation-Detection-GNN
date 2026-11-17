"""Paired comparison of an experiment against the multi-seed baseline (same seeds, same folds).

    .venv/bin/python -m experiments.generalization.compare --exp artifacts/generalization/masked \
        --task 4class --models dgtr,gat_full

For each model, checkpoint and seed present in both roots:
  * pooled LOEO macro-F1 (all 9 held-out events concatenated), exp - base, per seed;
  * per-fold macro-F1, exp - base, paired by (seed, held-out event);
  * Wilcoxon signed-rank over the 9 folds of seed-averaged Δ (the event is the unit of generalisation);
  * pooled per-class F1 (mean over seeds).
Writes <exp>/compare_<task>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from lifecycle.data import TASKS

from .multiseed_summary import softmax

CPS = {"10%": 0, "100%": -1}


def load(root: Path, task: str, model: str):
    """{seed: {event: (gold, pred[n, K])}} for complete seeds only."""
    out = {}
    for sd in sorted(root.glob("seed*")):
        vdir = sd / task / "verifiers"
        files = sorted(vdir.glob("*.json"))
        if len(files) != 9:
            continue
        folds = {}
        for j in files:
            meta = json.loads(j.read_text()); f = np.load(j.with_suffix(".npz"))
            if f"{model}_test" not in f:
                break
            folds[j.stem] = (f["test_labels"], softmax(f[f"{model}_test"] / meta["temperatures"][model]).argmax(-1))
        else:
            out[sd.name] = folds
    return out


def mf1(g, p, C):
    return float(f1_score(g, p, labels=range(C), average="macro", zero_division=0))


def compare(base, exp, C, names):
    seeds = sorted(set(base) & set(exp))
    events = sorted(base[seeds[0]])
    res = {"seeds": seeds}
    for cp, k in CPS.items():
        pooled = {s: {} for s in seeds}
        fold_d = np.zeros((len(seeds), len(events)))
        per_class = {"base": [], "exp": []}
        for i, s in enumerate(seeds):
            for tag, src in (("base", base), ("exp", exp)):
                g = np.concatenate([src[s][e][0] for e in events]); p = np.concatenate([src[s][e][1][:, k] for e in events])
                pooled[s][tag] = mf1(g, p, C)
                per_class[tag].append(f1_score(g, p, labels=range(C), average=None, zero_division=0))
            for j, e in enumerate(events):
                fold_d[i, j] = mf1(exp[s][e][0], exp[s][e][1][:, k], C) - mf1(base[s][e][0], base[s][e][1][:, k], C)
        d = np.array([pooled[s]["exp"] - pooled[s]["base"] for s in seeds])
        fd = fold_d.mean(0)
        try:
            p_w = float(wilcoxon(fd).pvalue)
        except ValueError:
            p_w = 1.0
        res[cp] = {
            "base_pooled": [pooled[s]["base"] for s in seeds], "exp_pooled": [pooled[s]["exp"] for s in seeds],
            "delta_pooled_mean": float(d.mean()), "delta_pooled_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "seeds_improved": int((d > 0).sum()),
            "fold_delta_mean": dict(zip(events, fd.round(4).tolist())), "fold_delta_std": dict(zip(events, fold_d.std(0, ddof=1).round(4).tolist())) if len(seeds) > 1 else {},
            "folds_improved": int((fd > 0).sum()), "wilcoxon_p_folds": p_w,
            "per_class_f1_base": dict(zip(names, np.mean(per_class["base"], 0).round(4).tolist())),
            "per_class_f1_exp": dict(zip(names, np.mean(per_class["exp"], 0).round(4).tolist())),
        }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/generalization/baseline")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--task", default="4class")
    parser.add_argument("--models", default="dgtr")
    parser.add_argument("--base-model", default=None, help="compare every exp model against this base model instead of itself")
    args = parser.parse_args()
    C = len(set(TASKS[args.task].values()))
    names = [n for n, _ in sorted({v: k for k, v in TASKS[args.task].items()}.items())] if args.task != "binary" else ["non-rumour", "rumour"]
    out = {}
    for m in args.models.split(","):
        bm = args.base_model or m
        base, exp = load(Path(args.base), args.task, bm), load(Path(args.exp), args.task, m)
        if not set(base) & set(exp):
            print(f"{m}: no shared complete seeds"); continue
        r = out[m] = compare(base, exp, C, names)
        print(f"\n### {args.task}: {m} (exp) vs {bm} (baseline), seeds {', '.join(r['seeds'])}")
        print("| checkpoint | base pooled mF1 | exp pooled mF1 | Δ (mean ± std over seeds) | seeds Δ>0 | folds Δ>0 | Wilcoxon p (9 folds) |")
        print("|---|---|---|---|---|---|---|")
        for cp in CPS:
            x = r[cp]; f = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
            print(f"| {cp} | {f(x['base_pooled'])} | {f(x['exp_pooled'])} | {x['delta_pooled_mean']:+.3f} ± {x['delta_pooled_std']:.3f} | {x['seeds_improved']}/{len(r['seeds'])} | {x['folds_improved']}/9 | {x['wilcoxon_p_folds']:.3f} |")
        x = r["100%"]
        print("\nper-fold Δ macro-F1 @100% (mean ± std over seeds): " + ", ".join(f"{e.replace('-all-rnr-threads', '')} {v:+.3f}±{x['fold_delta_std'].get(e, 0):.3f}" for e, v in x["fold_delta_mean"].items()))
        print("per-class F1 @100% base → exp: " + ", ".join(f"{c} {x['per_class_f1_base'][c]:.3f}→{x['per_class_f1_exp'][c]:.3f}" for c in names))
    Path(args.exp).mkdir(parents=True, exist_ok=True)
    (Path(args.exp) / f"compare_{args.task}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
