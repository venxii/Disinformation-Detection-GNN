"""Reframed results: RQ1 (commit timing), RQ2 (revision does no harm), RQ3 (revision recovers).

Reads outputs of run_commit_policies, run_lifecycle and run_injection; writes
artifacts/lifecycle/rq_results.json and figures under artifacts/lifecycle/<task>/figures/.

    .venv/bin/python -m experiments.rq_analysis
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads
from lifecycle.trajectory_metrics import thread_summary

ROOT = Path("artifacts/lifecycle")
K = len(CHECKPOINTS)
WEIGHTS = {"w0": lambda k: 1.0, "w05": lambda k: 1.0 - 0.5 * CHECKPOINTS[k], "hold": lambda k: (K - k) / K}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REPS = 2000


def boot(values_fn, n, reps=REPS, seed=7):
    """Percentile CI of a statistic computed on resampled thread indices."""
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n, n)) for _ in range(reps)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def ci_mean(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "ci95": boot(lambda i: x[i].mean(), len(x))}


# ---------------------------------------------------------------- RQ1

def rq1(task):
    C = len(TASKS[task])
    s = json.loads((ROOT / task / "commit_policies" / "summary.json").read_text())
    rec = np.load(ROOT / task / "commit_policies" / "test_records.npz")
    final = []
    for e, _, _ in loeo_folds(task_threads(THREADS, task)):
        f = np.load(ROOT / task / "verifiers" / f"{e}_calibrated_test.npz")
        final.append(np.stack([f["dgtr"][:, -1].argmax(1), f["dgtr"][:, 0].argmax(1), f["gold"]], 1))
    final = np.concatenate(final)
    lam = 1.0 / (C - 1)
    confirm = {}
    for u in WEIGHTS:
        confirm[u] = {}
        for p in ("fixed_checkpoint", "raw_confidence", "calibrated", "cost_sensitive"):
            r = rec[f"{u}__{p}"]
            assert (r[:, 2] == final[:, 2]).all()
            commit, label, gold = r[:, 0], r[:, 1], r[:, 2]
            forced = np.where(commit >= 0, label, final[:, 0])            # abstentions fall back to the final prediction
            d_acc = (forced == gold).astype(float) - (final[:, 0] == gold)
            t = np.where(commit >= 0, np.asarray(CHECKPOINTS)[np.maximum(commit, 0)], 1.0)
            confirm[u][p] = {"acc_minus_final_only": ci_mean(d_acc), "mean_commit_fraction": float(t.mean()),
                             "share_committed_by_20pct": float((t <= 0.2).mean())}
    early = (final[:, 1] == final[:, 2]).astype(float) - (final[:, 0] == final[:, 2])
    return {"lambda": lam, "validation": s["validation_crossfit"], "heldout_frozen": s["heldout_frozen"],
            "heldout_accuracy_cost_vs_final_only": confirm,
            "heldout_always10_minus_always100_acc": ci_mean(early),
            "sanity_unselected_validation": s["sanity_unselected"], "unlocked_flips_validation": s["unlocked_flips_example"],
            "validation_acc_by_checkpoint": s["verifier_val_crossfit_acc_by_checkpoint"], "curves": s["curves_validation"]}


def plot_rq1(task, r):
    figs = ROOT / task / "figures"; figs.mkdir(exist_ok=True)
    cur = r["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = {"fixed_checkpoint": "fixed checkpoint", "raw_confidence": "raw-confidence threshold",
             "calibrated": "calibration-aware threshold", "cost_sensitive": "cost-sensitive stopping"}
    for j, (p, label) in enumerate(names.items()):
        pts = [c for c in cur[p] if c["selective_acc"] is not None and c["coverage"] >= 0.02]  # <2% coverage = a handful of threads
        if p == "cost_sensitive":
            pts = [c for c in pts if abs(c["param"]["lambda"] - r["lambda"]) < 1e-9]
        x = [c["mean_commit_checkpoint"] for c in pts]; y = [c["selective_acc"] for c in pts]
        order = np.argsort(x)
        axes[0].plot(np.array(x)[order], np.array(y)[order], color=SERIES[j], linewidth=2, marker="o", markersize=4, label=label)
        cov = [c["coverage"] for c in pts]; risk = [1 - c["selective_acc"] for c in pts]
        order = np.argsort(cov)
        axes[1].plot(np.array(cov)[order], np.array(risk)[order], color=SERIES[j], linewidth=2, marker="o", markersize=4, label=label)
    ref = [c for c in cur["final_confidence_reference"] if c["selective_acc"] is not None]
    order = np.argsort([c["coverage"] for c in ref])
    axes[1].plot(np.array([c["coverage"] for c in ref])[order], 1 - np.array([c["selective_acc"] for c in ref])[order],
                 color=MUTED, linewidth=1.5, linestyle="--", label="final-snapshot confidence (reference)")
    axes[0].set_title("Earliness vs accuracy of commitments (coverage ≥ 2%)", fontsize=10, color=INK, loc="left")
    axes[0].set_xlabel("mean commit point (fraction of thread observed)", fontsize=9, color=MUTED)
    axes[0].set_ylabel("accuracy of committed decisions", fontsize=9, color=MUTED)
    axes[1].set_title("Risk vs coverage", fontsize=10, color=INK, loc="left")
    axes[1].set_xlabel("coverage (share of threads committed)", fontsize=9, color=MUTED)
    axes[1].set_ylabel("selective risk (1 − accuracy)", fontsize=9, color=MUTED)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5); ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{task}: validation events, 2-way cross-fitted", fontsize=9, color=MUTED, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(figs / "rq1_earliness_accuracy_risk_coverage.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- RQ2

def load_logs(dest, name):
    rows = {}
    for path in sorted((dest / "trajectories").glob(f"*-threads_{name}.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line); rows[r["root_id"]] = r
    return rows


def final_commit(r):
    """(label held at the end or None, checkpoint at which that commitment started)."""
    s = thread_summary(r["gold"], r["log"])
    if s["final_held"] is None:
        return None, None
    last = s["episodes"][-1]
    return s["final_held"], last["start"]


def utilities(r, lam):
    label, k = final_commit(r)
    if label is None:
        return {u: 0.0 for u in WEIGHTS}
    reward = 1.0 if label == r["gold"] else -lam
    return {u: reward * w(k) for u, w in WEIGHTS.items()}


def rq2(task, dest):
    C = len(TASKS[task]); lam = 1.0 / (C - 1)
    b, d = load_logs(dest, "b_early_no_revision"), load_logs(dest, "d_full")
    keys = sorted(b)
    ub = {u: np.array([utilities(b[k], lam)[u] for k in keys]) for u in WEIGHTS}
    ud = {u: np.array([utilities(d[k], lam)[u] for k in keys]) for u in WEIGHTS}
    fb = np.array([thread_summary(b[k]["gold"], b[k]["log"])["final_forced"] == b[k]["gold"] for k in keys], dtype=float)
    fd = np.array([thread_summary(d[k]["gold"], d[k]["log"])["final_forced"] == d[k]["gold"] for k in keys], dtype=float)
    harm = (fb == 1) & (fd == 0); help_ = (fb == 0) & (fd == 1)
    touched = np.array([any(e["type"] == "reopen" for row in d[k]["log"] for e in row["events"]) for k in keys])
    # false reopens: reopens fired against a correct commitment, per correct commitment episode
    reopen_ok, reopen_bad = [], []
    for k in keys:
        for row in d[k]["log"]:
            for e in row["events"]:
                if e["type"] == "reopen":
                    (reopen_ok if e["label"] == d[k]["gold"] else reopen_bad).append(k)
    n = len(keys)
    return {
        "n_threads": n,
        "utility_d_minus_b": {u: ci_mean(ud[u] - ub[u]) for u in WEIGHTS},
        "final_acc_b": float(fb.mean()), "final_acc_d": float(fd.mean()),
        "final_acc_d_minus_b": ci_mean(fd - fb),
        "threads_reopened": float(touched.mean()),
        "harmed_threads": int(harm.sum()), "helped_threads": int(help_.sum()),
        "net_help_rate": ci_mean(help_.astype(float) - harm.astype(float)),
        "harm_rate_among_reopened": float(harm[touched].mean()) if touched.any() else None,
        "help_rate_among_reopened": float(help_[touched].mean()) if touched.any() else None,
        "reopens_against_correct_commitment": len(reopen_ok), "reopens_against_wrong_commitment": len(reopen_bad),
        "base_rate_wrong_commitment": float(np.mean([final_commit(b[k])[0] != b[k]["gold"] for k in keys if final_commit(b[k])[0] is not None])),
    }


# ---------------------------------------------------------------- RQ3

def prf(rs):
    tp = sum(r["reopened"] and r["should_reopen"] for r in rs); fp = sum(r["reopened"] and not r["should_reopen"] for r in rs)
    fn = sum(not r["reopened"] and r["should_reopen"] for r in rs)
    return (tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan, 2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def rq3(task, dest):
    with gzip.open(dest / "injection_records.jsonl.gz", "rt") as f:
        recs = [json.loads(l) for l in f]
    systems = ("c_revision_no_conflict_gate", "d_full", "abl_no_dedup", "abl_volume_path_only", "abl_confirmed_locked")
    by = {s: defaultdict(list) for s in systems}
    for r in recs:
        if r["system"] in by:
            by[r["system"]][(r["event"], r["root_id"])].append(r)
    out = {}
    for s in systems:
        groups = list(by[s].values()); n = len(groups)
        flat = [r for g in groups for r in g]
        p, rc, f1 = prf(flat)
        idx_stat = lambda i, j: prf([r for t in i for r in groups[t]])[j]
        scen = {}
        for sc in ("strong_single", "distinct_volume", "weak_duplicates", "neutral", "supporting"):
            x = np.array([r["reopened"] for r in flat if r["scenario"] == sc], dtype=float)
            scen[sc] = ci_mean(x) if len(x) else None
        recovered = [r for r in flat if r["should_reopen"] and r["reopened"]]
        out[s] = {"eligible_threads": n, "precision": p, "recall": rc, "f1": f1,
                  "f1_ci95": boot(lambda i: idx_stat(i, 2), n), "precision_ci95": boot(lambda i: idx_stat(i, 0), n),
                  "recall_ci95": boot(lambda i: idx_stat(i, 1), n), "reopen_rate_by_scenario": scen,
                  "label_changed_after_justified_reopen": float(np.mean([r["final_label"] != r["committed"] for r in recovered])) if recovered else None,
                  "abstained_after_justified_reopen": float(np.mean([r["final_label"] is None for r in recovered])) if recovered else None}
    # paired bootstrap of F1(d) - F1(c) over threads eligible for both
    common = sorted(set(by["d_full"]) & set(by["c_revision_no_conflict_gate"]))
    gd = [by["d_full"][k] for k in common]; gc = [by["c_revision_no_conflict_gate"][k] for k in common]
    diff = lambda i: prf([r for t in i for r in gd[t]])[2] - prf([r for t in i for r in gc[t]])[2]
    out["f1_d_minus_c"] = {"mean": diff(np.arange(len(common))), "ci95": boot(diff, len(common)), "threads": len(common)}
    return out


def main():
    import argparse
    global THREADS, ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--tasks", default="4class,veracity")
    parser.add_argument("--output", default="rq_results.json", help="file name under --root")
    args = parser.parse_args()
    ROOT = Path(args.root)
    THREADS, _ = load_threads()
    results = {}
    for task in args.tasks.split(","):
        r1 = rq1(task); plot_rq1(task, r1); r1.pop("curves")
        results[task] = {"RQ1": r1}
        for name, d in (("", ROOT / task), ("_legacy_gate", ROOT / task / "variants" / "legacy_gate")):
            if (d / "injection_records.jsonl.gz").exists():  # optional in roots that only ran the clean protocol
                results[task]["RQ2" + name], results[task]["RQ3" + name] = rq2(task, d), rq3(task, d)
        clean = ROOT / task / "variants" / "clean"
        if (clean / "injection_records.jsonl.gz").exists():
            results[task]["RQ2_clean"] = rq2(task, clean)
            results[task]["RQ3_clean"] = rq3(task, clean)
            folds = {p.stem: json.loads(p.read_text()) for p in sorted((clean / "folds").glob("*.json"))}
            results[task]["clean_selection"] = {e: {"gate_design": r["gate_design"], "stance_config": r["stance_config"],
                                                    "stance_heldout_macro_f1": r["stance_heldout"].get("macro_f1"),
                                                    "gate": {k: r["full_config"]["conflict"][k] for k in ("theta_net", "theta_strong", "v_min")},
                                                    "commit": {k: r["commit_config"][k] for k in ("tau_prov", "tau_conf", "min_units_prov")}}
                                                for e, r in folds.items()}
        print(f"done {task}", flush=True)
    (ROOT / args.output).write_text(json.dumps(results, indent=2, default=float))
    print(f"saved {ROOT / args.output}")


if __name__ == "__main__":
    main()
