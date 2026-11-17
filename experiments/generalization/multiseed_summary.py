"""Mean ± std over seeds of LOEO verifier results; per-fold scores kept for paired comparisons.

    .venv/bin/python -m experiments.generalization.multiseed_summary --root artifacts/generalization/baseline
Reads <root>/seed*/<task>/verifiers/<event>.{npz,json} written by experiments/run_verifiers.py.
Writes <root>/summary.json and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lifecycle.data import TASKS

MODELS = ("dgtr", "gat_full", "prefix_lr", "source_lr")
CHECKPOINTS = {"10%": 0, "100%": -1}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def scores(gold, pred, C):
    return {"macro_f1": float(f1_score(gold, pred, labels=range(C), average="macro", zero_division=0)), "acc": float((gold == pred).mean())}


def seed_results(seed_dir: Path, task: str):
    C = len(set(TASKS[task].values()))
    vdir = seed_dir / task / "verifiers"
    events = sorted(p.stem for p in vdir.glob("*.json"))
    out = {m: {"per_fold": {}, "pooled": {}} for m in MODELS}
    gold_all = []
    preds = {m: {cp: [] for cp in CHECKPOINTS} for m in MODELS}
    for e in events:
        meta = json.loads((vdir / f"{e}.json").read_text())
        f = np.load(vdir / f"{e}.npz")
        gold = f["test_labels"]; gold_all.append(gold)
        for m in MODELS:
            p = softmax(f[f"{m}_test"] / meta["temperatures"][m]).argmax(-1)
            out[m]["per_fold"][e] = {cp: scores(gold, p[:, k], C) for cp, k in CHECKPOINTS.items()}
            for cp, k in CHECKPOINTS.items():
                preds[m][cp].append(p[:, k])
    gold_all = np.concatenate(gold_all)
    for m in MODELS:
        for cp in CHECKPOINTS:
            out[m]["pooled"][cp] = scores(gold_all, np.concatenate(preds[m][cp]), C)
            out[m]["pooled"][cp]["fold_mean_macro_f1"] = float(np.mean([v[cp]["macro_f1"] for v in out[m]["per_fold"].values()]))
    return out, events


def ms(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": [float(v) for v in values]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/generalization/baseline")
    parser.add_argument("--tasks", default="4class,binary")
    args = parser.parse_args()
    root = Path(args.root)
    seeds = sorted(root.glob("seed*"), key=lambda p: int(p.name[4:]))
    summary = {}
    for task in args.tasks.split(","):
        per_seed = {}
        for s in seeds:
            if (s / task / "verifiers").exists() and len(list((s / task / "verifiers").glob("*.json"))) == 9:
                per_seed[s.name], events = seed_results(s, task)
        if not per_seed:
            continue
        task_sum = {"seeds": sorted(per_seed), "models": {}}
        print(f"\n### {task} — {len(per_seed)} seeds ({', '.join(sorted(per_seed))}); mean ± std over seeds\n")
        print("| model | pooled macro-F1 @10% | pooled macro-F1 @100% | fold-mean macro-F1 @100% | pooled acc @100% |")
        print("|---|---|---|---|---|")
        for m in MODELS:
            agg = {}
            for cp in CHECKPOINTS:
                agg[cp] = {k: ms([per_seed[s][m]["pooled"][cp][k] for s in per_seed]) for k in ("macro_f1", "acc", "fold_mean_macro_f1")}
            folds = {e: {cp: ms([per_seed[s][m]["per_fold"][e][cp]["macro_f1"] for s in per_seed]) for cp in CHECKPOINTS} for e in events}
            task_sum["models"][m] = {"pooled": agg, "per_fold_macro_f1": folds}
            f = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"
            print(f"| {m} | {f(agg['10%']['macro_f1'])} | {f(agg['100%']['macro_f1'])} | {f(agg['100%']['fold_mean_macro_f1'])} | {f(agg['100%']['acc'])} |")
        print(f"\nPer-fold macro-F1 @100% (mean ± std over seeds):\n")
        print("| held-out event | " + " | ".join(MODELS) + " |")
        print("|---|" + "---|" * len(MODELS))
        for e in events:
            print(f"| {e.replace('-all-rnr-threads', '')} | " + " | ".join(f"{task_sum['models'][m]['per_fold_macro_f1'][e]['100%']['mean']:.3f} ± {task_sum['models'][m]['per_fold_macro_f1'][e]['100%']['std']:.3f}" for m in MODELS) + " |")
        summary[task] = task_sum
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {root / 'summary.json'}")


if __name__ == "__main__":
    main()
