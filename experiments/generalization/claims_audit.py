"""Claims audit: every quantitative claim of docs/lifecycle_report.md and docs/professor_summary.md,
recomputed for the 5-seed baseline (clean nested protocol) and for the committed single run.

    .venv/bin/python -m experiments.generalization.claims_audit [--tasks 4class,veracity]

Runs
  committed  artifacts/lifecycle                         (the run the docs were written from; seed 7)
  seed<S>    artifacts/generalization/rq/baseline/seed<S> (S = 7, 13, 21, 42, 99; gat_full seed fix applied)

For every metric the script reports
  * committed value, rerun-seed-7 value, and noise = |committed − rerun seed 7| (same seed, same code,
    same data: pure run-to-run noise; LR baselines and the stance model are deterministic, noise 0);
  * 5-seed mean ± std and per-seed values;
  * for differences, a 95% CI from a crossed bootstrap that resamples seeds and held-out events
    independently (2,000 reps); every statistic is computed from per-(seed, event) sufficient statistics,
    so pairing (same thread, same seed) is kept.
A difference is **established** iff its CI excludes 0 and |5-seed mean| > noise.
Writes artifacts/generalization/claims_audit_<task>.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.rq_analysis import WEIGHTS, utilities
from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads
from lifecycle.trajectory_metrics import ece, thread_summary

SEEDS = (7, 13, 21, 42, 99)
COMMITTED = Path("artifacts/lifecycle")
RQ = Path("artifacts/generalization/rq/baseline")
REPS = 2000
SCEN = ("strong_single", "distinct_volume", "weak_duplicates", "neutral", "supporting")


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- sufficient statistics per (run, event)

def macro_f1_from_cm(cm):
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def f1_counts(c):
    tp, fp, fn = c
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def cm(gold, pred, C):
    m = np.zeros((C, C)); np.add.at(m, (gold, pred), 1)
    return m.ravel()


def run_cells(root: Path, task: str, threads, events):
    """{stat_name: array[n_events, k]} for one run."""
    C = len(TASKS[task])
    labels = np.array([TASKS[task][t.label] for t in threads])
    vdir = root / task / "verifiers"
    cells = defaultdict(lambda: [None] * len(events))
    scalars = {"temperature_dgtr": [], "gate_current": 0, "min_units_1": 0, "tau_prov_min": 0, "ece_probs": [], "ece_gold": []}
    rec = np.load(root / task / "commit_policies" / "test_records.npz")
    cp_summary = json.loads((root / task / "commit_policies" / "summary.json").read_text())
    clean = root / task / "variants" / "clean"
    inj = defaultdict(list)
    with gzip.open(clean / "injection_records.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line); inj[(r["event"], r["system"])].append(r)
    offset = 0
    for j, (held_out, validation, train_events) in enumerate(loeo_folds(threads)):
        assert held_out == events[j]
        f = np.load(vdir / f"{held_out}.npz"); meta = json.loads((vdir / f"{held_out}.json").read_text())
        gold = f["test_labels"]; n = len(gold)
        T = meta["temperatures"]
        p = {m: softmax(f[f"{m}_test"] / T[m]) for m in ("dgtr", "gat_full", "source_lr", "prefix_lr")}
        scalars["temperature_dgtr"].append(T["dgtr"])
        scalars["ece_probs"].append(p["dgtr"][:, -1]); scalars["ece_gold"].append(gold)
        maj = Counter(labels[i] for i, t in enumerate(threads) if t.event in train_events or t.event == validation).most_common(1)[0][0]
        for m in p:
            cells[f"cm_{m}_100"][j] = cm(gold, p[m][:, -1].argmax(1), C)
        cells["cm_dgtr_10"][j] = cm(gold, p["dgtr"][:, 0].argmax(1), C)
        cells["cm_majority"][j] = cm(gold, np.full_like(gold, maj), C)
        d100 = (p["dgtr"][:, -1].argmax(1) == gold).astype(float); d10 = (p["dgtr"][:, 0].argmax(1) == gold).astype(float)
        cells["acc_dgtr_100"][j] = [d100.sum(), n]; cells["acc_dgtr_10"][j] = [d10.sum(), n]
        cells["acc_majority"][j] = [(gold == maj).sum(), n]
        cells["acc_gat_100"][j] = [(p["gat_full"][:, -1].argmax(1) == gold).sum(), n]
        cells["rq1_always10_minus_100"][j] = [(d10 - d100).sum(), n]
        cells["dgtr_beats_majority_fold"][j] = [float(d100.mean() > (gold == maj).mean()), 1]
        # RQ1: frozen commit policies on held-out threads (records are in loeo order)
        for pol in ("calibrated", "raw_confidence", "cost_sensitive", "fixed_checkpoint"):
            r = rec[f"w0__{pol}"][offset:offset + n]
            assert (r[:, 2] == gold).all()
            forced = np.where(r[:, 0] >= 0, r[:, 1], p["dgtr"][:, -1].argmax(1))
            cells[f"rq1_{pol}_minus_final"][j] = [((forced == gold).astype(float) - d100).sum(), n]
            t = np.where(r[:, 0] >= 0, np.asarray(CHECKPOINTS)[np.maximum(r[:, 0], 0)], 1.0)
            cells[f"rq1_{pol}_share_by_20"][j] = [(t <= 0.2).sum(), n]
        offset += n
        # RQ2 + natural trajectories
        trajs = {}
        for sys in ("b_early_no_revision", "d_full", "c_revision_no_conflict_gate"):
            with gzip.open(clean / "trajectories" / f"{held_out}_{sys}.jsonl.gz", "rt") as fh:
                trajs[sys] = {r["root_id"]: r for r in map(json.loads, fh)}
        keys = sorted(trajs["b_early_no_revision"])
        lam = 1.0 / (C - 1)
        fb = np.array([thread_summary(trajs["b_early_no_revision"][k]["gold"], trajs["b_early_no_revision"][k]["log"])["final_forced"] == trajs["b_early_no_revision"][k]["gold"] for k in keys], float)
        fd = np.array([thread_summary(trajs["d_full"][k]["gold"], trajs["d_full"][k]["log"])["final_forced"] == trajs["d_full"][k]["gold"] for k in keys], float)
        cells["rq2_final_acc_d_minus_b"][j] = [(fd - fb).sum(), len(keys)]
        cells["rq2_harmed"][j] = [((fb == 1) & (fd == 0)).sum(), len(keys)]
        cells["rq2_helped"][j] = [((fb == 0) & (fd == 1)).sum(), len(keys)]
        for u in WEIGHTS:
            ub = np.array([utilities(trajs["b_early_no_revision"][k], lam)[u] for k in keys]); ud = np.array([utilities(trajs["d_full"][k], lam)[u] for k in keys])
            cells[f"rq2_utility_{u}_d_minus_b"][j] = [(ud - ub).sum(), len(keys)]
        ok = bad = 0
        touched = 0
        for k in keys:
            ev = [e for row in trajs["d_full"][k]["log"] for e in row["events"] if e["type"] == "reopen"]
            touched += bool(ev)
            for e in ev:
                ok += e["label"] == trajs["d_full"][k]["gold"]; bad += e["label"] != trajs["d_full"][k]["gold"]
        cells["rq2_threads_reopened"][j] = [touched, len(keys)]
        cells["rq2_reopen_precision"][j] = [bad, ok + bad]
        # RQ3: injection
        dk = defaultdict(list); ck = defaultdict(list)
        for r in inj[(held_out, "d_full")]:
            dk[r["root_id"]].append(r)
        for r in inj[(held_out, "c_revision_no_conflict_gate")]:
            ck[r["root_id"]].append(r)
        common = set(dk) & set(ck)
        for name, src in (("d", dk), ("c", ck)):
            rs = [r for k in common for r in src[k]]
            cells[f"rq3_counts_{name}"][j] = [sum(r["reopened"] and r["should_reopen"] for r in rs), sum(r["reopened"] and not r["should_reopen"] for r in rs),
                                               sum((not r["reopened"]) and r["should_reopen"] for r in rs)]
        for sys in ("d_full", "c_revision_no_conflict_gate", "abl_no_dedup", "abl_volume_path_only", "abl_confirmed_locked", "abl_no_hysteresis", "abl_strength_path_only"):
            rs = inj[(held_out, sys)]
            cells[f"inj_counts_{sys}"][j] = [sum(r["reopened"] and r["should_reopen"] for r in rs), sum(r["reopened"] and not r["should_reopen"] for r in rs),
                                              sum((not r["reopened"]) and r["should_reopen"] for r in rs)]
            for sc in SCEN:
                x = [r["reopened"] for r in rs if r["scenario"] == sc]
                cells[f"inj_rate_{sys}_{sc}"][j] = [sum(x), len(x)]
        fold = json.loads((clean / "folds" / f"{held_out}.json").read_text())
        cur = fold.get("gate_design") == "current"
        scalars["gate_current"] += cur
        scalars["min_units_1"] += fold["commit_config"]["min_units_prov"] == 1
        scalars["tau_prov_min"] += abs(fold["commit_config"]["tau_prov"] - 0.4) < 1e-9
        cells["rq3_d_tp_current_gate_folds"][j] = [cells["rq3_counts_d"][j][0] if cur else 0, cells["rq3_counts_d"][j][0]]
    out = {k: np.array(v, dtype=float) for k, v in cells.items()}
    s = json.loads((clean / "summary.json").read_text())
    scalars["natural"] = {sys: {k: v for k, v in s["systems"][sys].items() if not isinstance(v, (list, dict))} for sys in s["systems"]}
    scalars["natural_paired"] = s["paired_bootstrap"]
    scalars["ece_dgtr_100"] = ece(np.concatenate(scalars.pop("ece_probs")), np.concatenate(scalars.pop("ece_gold")))
    val = cp_summary["validation_crossfit"]; sanity = cp_summary["sanity_unselected"]
    scalars["rq1_validation"] = {u: {**{pol: val[u][pol][f"U_{u}"] for pol in val[u]}, "always_10": sanity["always_at_10pct"][f"U_{u}"],
                                     "always_100": sanity["always_at_100pct"][f"U_{u}"]} for u in ("w0", "w05", "hold")}
    scalars["median_minutes_calibrated"] = cp_summary["heldout_frozen"]["w0"]["calibrated"]["median_minutes_to_commit"]
    return out, scalars


# ---------------------------------------------------------------- statistics

STATS = {
    # name: (cells used, function of summed cells, is_difference)
    "dgtr_macro_f1_100": (("cm_dgtr_100",), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)), False),
    "dgtr_macro_f1_10": (("cm_dgtr_10",), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)), False),
    "gat_macro_f1_100": (("cm_gat_full_100",), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)), False),
    "majority_macro_f1": (("cm_majority",), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)), False),
    "dgtr_minus_gat_macro_f1_100": (("cm_dgtr_100", "cm_gat_full_100"), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)) - macro_f1_from_cm(c[1].reshape(C, C)), True),
    "dgtr_minus_source_lr_macro_f1_100": (("cm_dgtr_100", "cm_source_lr_100"), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)) - macro_f1_from_cm(c[1].reshape(C, C)), True),
    "dgtr_minus_prefix_lr_macro_f1_100": (("cm_dgtr_100", "cm_prefix_lr_100"), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)) - macro_f1_from_cm(c[1].reshape(C, C)), True),
    "dgtr_minus_majority_macro_f1": (("cm_dgtr_100", "cm_majority"), lambda c, C: macro_f1_from_cm(c[0].reshape(C, C)) - macro_f1_from_cm(c[1].reshape(C, C)), True),
    "dgtr_acc_100": (("acc_dgtr_100",), lambda c, C: c[0][0] / c[0][1], False),
    "dgtr_acc_10": (("acc_dgtr_10",), lambda c, C: c[0][0] / c[0][1], False),
    "gat_acc_100": (("acc_gat_100",), lambda c, C: c[0][0] / c[0][1], False),
    "majority_acc": (("acc_majority",), lambda c, C: c[0][0] / c[0][1], False),
    "dgtr_minus_majority_acc": (("acc_dgtr_100", "acc_majority"), lambda c, C: c[0][0] / c[0][1] - c[1][0] / c[1][1], True),
    "folds_dgtr_beats_majority": (("dgtr_beats_majority_fold",), lambda c, C: c[0][0], False),
    "rq1_always10_minus_always100_acc": (("rq1_always10_minus_100",), lambda c, C: c[0][0] / c[0][1], True),
    "rq1_calibrated_minus_final_acc": (("rq1_calibrated_minus_final",), lambda c, C: c[0][0] / c[0][1], True),
    "rq1_raw_confidence_minus_final_acc": (("rq1_raw_confidence_minus_final",), lambda c, C: c[0][0] / c[0][1], True),
    "rq1_cost_sensitive_minus_final_acc": (("rq1_cost_sensitive_minus_final",), lambda c, C: c[0][0] / c[0][1], True),
    "rq1_calibrated_share_by_20pct": (("rq1_calibrated_share_by_20",), lambda c, C: c[0][0] / c[0][1], False),
    "rq2_final_acc_d_minus_b": (("rq2_final_acc_d_minus_b",), lambda c, C: c[0][0] / c[0][1], True),
    "rq2_utility_w0_d_minus_b": (("rq2_utility_w0_d_minus_b",), lambda c, C: c[0][0] / c[0][1], True),
    "rq2_utility_w05_d_minus_b": (("rq2_utility_w05_d_minus_b",), lambda c, C: c[0][0] / c[0][1], True),
    "rq2_utility_hold_d_minus_b": (("rq2_utility_hold_d_minus_b",), lambda c, C: c[0][0] / c[0][1], True),
    "rq2_threads_reopened": (("rq2_threads_reopened",), lambda c, C: c[0][0] / c[0][1], False),
    "rq2_harmed_threads": (("rq2_harmed",), lambda c, C: c[0][0], False),
    "rq2_helped_threads": (("rq2_helped",), lambda c, C: c[0][0], False),
    "rq2_reopen_precision": (("rq2_reopen_precision",), lambda c, C: c[0][0] / max(c[0][1], 1), False),
    "rq3_d_f1": (("rq3_counts_d",), lambda c, C: f1_counts(c[0]), False),
    "rq3_c_f1": (("rq3_counts_c",), lambda c, C: f1_counts(c[0]), False),
    "rq3_f1_d_minus_c": (("rq3_counts_d", "rq3_counts_c"), lambda c, C: f1_counts(c[0]) - f1_counts(c[1]), True),
    "rq3_d_precision": (("rq3_counts_d",), lambda c, C: c[0][0] / max(c[0][0] + c[0][1], 1), False),
    "rq3_d_recall": (("rq3_counts_d",), lambda c, C: c[0][0] / max(c[0][0] + c[0][2], 1), False),
    "rq3_share_of_d_correct_reopens_from_current_gate_folds": (("rq3_d_tp_current_gate_folds",), lambda c, C: c[0][0] / max(c[0][1], 1), False),
    "inj_d_strong_single_rate": (("inj_rate_d_full_strong_single",), lambda c, C: c[0][0] / max(c[0][1], 1), False),
    "inj_d_weak_duplicates_rate": (("inj_rate_d_full_weak_duplicates",), lambda c, C: c[0][0] / max(c[0][1], 1), False),
    "abl_no_dedup_minus_d_weak_dup_rate": (("inj_rate_abl_no_dedup_weak_duplicates", "inj_rate_d_full_weak_duplicates"), lambda c, C: c[0][0] / max(c[0][1], 1) - c[1][0] / max(c[1][1], 1), True),
    "abl_volume_only_recall": (("inj_counts_abl_volume_path_only",), lambda c, C: c[0][0] / max(c[0][0] + c[0][2], 1), False),
    "abl_confirmed_locked_minus_d_inj_recall": (("inj_counts_abl_confirmed_locked", "inj_counts_d_full"), lambda c, C: c[0][0] / max(c[0][0] + c[0][2], 1) - c[1][0] / max(c[1][0] + c[1][2], 1), True),
    "abl_no_hysteresis_minus_d_inj_f1": (("inj_counts_abl_no_hysteresis", "inj_counts_d_full"), lambda c, C: f1_counts(c[0]) - f1_counts(c[1]), True),
    "abl_strength_only_minus_d_inj_f1": (("inj_counts_abl_strength_path_only", "inj_counts_d_full"), lambda c, C: f1_counts(c[0]) - f1_counts(c[1]), True),
}


def stat(name, cell_sums, C):
    keys, fn, _ = STATS[name]
    return fn([cell_sums[k] for k in keys], C)


def crossed_ci(name, runs, C, rng):
    """Resample seeds and events independently; statistic on summed cells."""
    keys = STATS[name][0]
    cube = {k: np.stack([runs[s][0][k] for s in runs]) for k in keys}  # [S, E, d]
    S, E = cube[keys[0]].shape[:2]
    vals = []
    for _ in range(REPS):
        ws = np.bincount(rng.integers(0, S, S), minlength=S).astype(float); we = np.bincount(rng.integers(0, E, E), minlength=E).astype(float)
        sums = {k: np.einsum("s,e,sed->d", ws, we, cube[k]) for k in keys}
        vals.append(stat(name, sums, C))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="4class")
    args = parser.parse_args()
    all_threads, _ = load_threads()
    for task in args.tasks.split(","):
        C = len(TASKS[task])
        threads = task_threads(all_threads, task)
        events = [h for h, _, _ in loeo_folds(threads)]
        runs = {f"seed{s}": run_cells(RQ / f"seed{s}", task, threads, events) for s in SEEDS}
        committed = run_cells(COMMITTED, task, threads, events)
        rng = np.random.default_rng(7)
        res = {}
        for name, (_, _, is_diff) in STATS.items():
            per_seed = {s: stat(name, {k: v.sum(0) for k, v in runs[s][0].items()}, C) for s in runs}
            com = stat(name, {k: v.sum(0) for k, v in committed[0].items()}, C)
            x = np.array(list(per_seed.values()))
            noise = abs(com - per_seed["seed7"])
            row = {"committed": com, "rerun_seed7": per_seed["seed7"], "noise": noise, "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                   "per_seed": per_seed, "is_difference": is_diff}
            if is_diff:
                ci = crossed_ci(name, runs, C, rng)
                row["ci95_seeds_x_events"] = ci
                row["established"] = bool((ci[0] > 0 or ci[1] < 0) and abs(row["mean"]) > noise)
            res[name] = row
        # scalar / point metrics with no CI
        sc = {"committed": {k: v for k, v in committed[1].items()}, **{s: runs[s][1] for s in runs}}
        res["_scalars"] = sc
        Path(f"artifacts/generalization/claims_audit_{task}.json").write_text(json.dumps(res, indent=2, default=float))
        print(f"\n### {task}\n| metric | committed | rerun s7 | noise | 5-seed mean ± std | per seed | CI (seeds×events) | established |\n|---|---|---|---|---|---|---|---|")
        for name, r in res.items():
            if name.startswith("_"):
                continue
            ci = r.get("ci95_seeds_x_events")
            print(f"| {name} | {r['committed']:.4f} | {r['rerun_seed7']:.4f} | {r['noise']:.4f} | {r['mean']:.4f} ± {r['std']:.4f} | "
                  f"{', '.join(f'{v:.3f}' for v in r['per_seed'].values())} | {'' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {r.get('established', '')} |")


if __name__ == "__main__":
    main()
