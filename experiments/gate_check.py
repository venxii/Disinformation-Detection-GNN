"""Sensitivity check for the post-smoke-test gate change (see docs/lifecycle_report.md §6).

Compares the current gate with the pre-change ("legacy") gate on every headline number,
and re-scores the current gate with the germanwings fold (whose test metrics were seen
before the change) excluded.

    .venv/bin/python -m experiments.gate_check > artifacts/lifecycle/gate_check.md
"""
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.run_lifecycle import paired_bootstrap
from lifecycle.trajectory_metrics import lifecycle_metrics, thread_summary, thread_utility

SMOKE_FOLD = "germanwings-crash-all-rnr-threads"


def load(dest, name, exclude=()):
    rows = []
    for path in sorted((dest / "trajectories").glob(f"*-threads_{name}.jsonl.gz")):
        if any(path.name.startswith(e + "_") for e in exclude):
            continue
        with gzip.open(path, "rt") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def injection(dest, exclude=()):
    with gzip.open(dest / "injection_records.jsonl.gz", "rt") as f:
        recs = [json.loads(l) for l in f if json.loads(l)["event"] not in exclude]
    out = {}
    for sys_ in ("c_revision_no_conflict_gate", "d_full"):
        rs = [r for r in recs if r["system"] == sys_]
        tp = sum(r["reopened"] and r["should_reopen"] for r in rs); fp = sum(r["reopened"] and not r["should_reopen"] for r in rs)
        fn = sum(not r["reopened"] and r["should_reopen"] for r in rs)
        strong = [r["reopened"] for r in rs if r["scenario"] == "strong_single"]
        out[sys_] = (tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1), float(np.mean(strong)))
    return out


def row(label, dest, C, exclude=()):
    lam = 1 / (C - 1)
    b, d = load(dest, "b_early_no_revision", exclude), load(dest, "d_full", exclude)
    ub = [thread_utility(r["gold"], r["log"], lam) for r in b]; ud = [thread_utility(r["gold"], r["log"], lam) for r in d]
    fb = [float(thread_summary(r["gold"], r["log"])["final_forced"] == r["gold"]) for r in b]
    fd = [float(thread_summary(r["gold"], r["log"])["final_forced"] == r["gold"]) for r in d]
    m = lifecycle_metrics(d, C)
    u, a = paired_bootstrap(ud, ub), paired_bootstrap(fd, fb)
    inj = injection(dest, exclude)
    print(f"| {label} | {len(d)} | {u['mean']:+.4f} [{u['ci95'][0]:+.4f}, {u['ci95'][1]:+.4f}] | {a['mean']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | "
          f"{m['reopens_per_100_threads']:.1f} | {m['reopen_precision']:.2f} | {m['reopen_recall']:.2f} | {m['false_reopen_rate']:.2f} | "
          f"{inj['d_full'][0]:.2f} / {inj['d_full'][1]:.2f} / {inj['d_full'][2]:.2f} | {inj['d_full'][3]:.2f} | {inj['c_revision_no_conflict_gate'][2]:.2f} |")


def main():
    print("| task · gate · folds | test threads | utility d−b [95% CI] | final acc d−b [95% CI] | reopens/100 | reopen P | reopen R | false-reopen | injection d P / R / F1 | d strong_single rate | injection c F1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for task, C in (("4class", 4), ("veracity", 3)):
        out = Path("artifacts/lifecycle") / task
        row(f"{task} · current · all 9", out, C)
        row(f"{task} · legacy · all 9", out / "variants" / "legacy_gate", C)
        row(f"{task} · current · excl. germanwings", out, C, (SMOKE_FOLD,))
        row(f"{task} · legacy · excl. germanwings", out / "variants" / "legacy_gate", C, (SMOKE_FOLD,))


if __name__ == "__main__":
    main()
