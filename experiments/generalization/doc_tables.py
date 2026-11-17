"""5-seed tables for docs/lifecycle_report.md (verifiers, natural lifecycle, injection), clean protocol.

    .venv/bin/python -m experiments.generalization.doc_tables --task 4class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lifecycle.data import TASKS
from lifecycle.trajectory_metrics import ece

from .compare import load
from .multiseed_summary import softmax

SEEDS = (7, 13, 21, 42, 99)
BASE = Path("artifacts/generalization/baseline")
RQ = Path("artifacts/generalization/rq/baseline")
ms = lambda x: f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


def verifier_table(task):
    C = len(set(TASKS[task].values()))
    print(f"\n#### Verifiers, {task} (pooled LOEO; macro-F1 / accuracy / ECE; mean ± std over 5 seeds)\n")
    print("| model | 10% | 50% | 100% |\n|---|---|---|---|")
    for m in ("dgtr", "gat_full", "source_lr", "prefix_lr"):
        cells = []
        for k in (0, 3, 5):
            f1s, accs, eces = [], [], []
            for s in SEEDS:
                vdir = BASE / f"seed{s}" / task / "verifiers"
                g, p = [], []
                for j in sorted(vdir.glob("*-threads.json")):
                    meta = json.loads(j.read_text()); f = np.load(j.with_suffix(".npz"))
                    g.append(f["test_labels"]); p.append(softmax(f[f"{m}_test"][:, k] / meta["temperatures"][m]))
                g, p = np.concatenate(g), np.concatenate(p)
                f1s.append(f1_score(g, p.argmax(1), labels=range(C), average="macro", zero_division=0)); accs.append((p.argmax(1) == g).mean()); eces.append(ece(p, g))
            sd = lambda x: f"{np.mean(x):.3f}" + (f"±{np.std(x, ddof=1):.3f}" if np.std(x) > 0 else "")
            cells.append(f"{sd(f1s)} / {sd(accs)} / {sd(eces)}")
        print(f"| {m} | " + " | ".join(cells) + " |")


def natural_table(task):
    suffix = "" if task == "4class" else f"_{task}"
    print(f"\n#### Lifecycle on natural trajectories, {task}, clean protocol (mean ± std over 5 seeds)\n")
    keys = ("utility", "final_acc_forced", "final_macro_f1_forced", "final_coverage", "earliness_mean_checkpoint", "reopens_per_100_threads",
            "reopen_precision", "reopen_recall", "false_reopen_rate", "justified_reopens_corrected", "mean_flips")
    print("| system | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    S = [json.loads((RQ / f"seed{s}" / task / "variants" / "clean" / "summary.json").read_text())["systems"] for s in SEEDS]
    for sys in ("a_final_only", "b_early_no_revision", "c_revision_no_conflict_gate", "d_full", "abl_confirmed_locked", "abl_no_support_offset", "abl_reverify_verifier_only"):
        vals = []
        for k in keys:
            x = [s[sys][k] for s in S if s[sys][k] is not None]
            vals.append(ms(x) if x else "–")
        print(f"| {sys} | " + " | ".join(vals) + " |")
    print("\nPaired per-seed differences (natural):")
    for a, b, k in (("abl_no_support_offset", "d_full", "reopen_recall"), ("abl_no_support_offset", "d_full", "false_reopen_rate"),
                    ("abl_confirmed_locked", "d_full", "reopen_precision"), ("abl_confirmed_locked", "d_full", "reopen_recall"),
                    ("d_full", "abl_reverify_verifier_only", "justified_reopens_corrected"), ("d_full", "c_revision_no_conflict_gate", "reopen_recall")):
        x = [s[a][k] - s[b][k] for s in S]
        print(f"  {a} − {b} [{k}]: {ms(x)}  per seed {np.round(x, 3).tolist()}")
    rqs = [json.loads((RQ / f"seed{s}" / f"rq_results{suffix}.json").read_text())[task]["RQ2_clean"] for s in SEEDS]
    lift = [r["reopens_against_wrong_commitment"] / (r["reopens_against_wrong_commitment"] + r["reopens_against_correct_commitment"]) - r["base_rate_wrong_commitment"] for r in rqs]
    print(f"  (d) reopen precision − base rate of wrong commitments: {ms(lift)} per seed {np.round(lift, 3).tolist()}")


def injection_table(task):
    suffix = "" if task == "4class" else f"_{task}"
    print(f"\n#### Injection, {task}, clean protocol: reopen rate per scenario and P/R/F1 (mean ± std over 5 seeds)\n")
    sc = ("strong_single", "distinct_volume", "weak_duplicates", "neutral", "supporting")
    print("| system | " + " | ".join(sc) + " | precision | recall | F1 |\n|---|" + "---|" * (len(sc) + 3))
    R = [json.loads((RQ / f"seed{s}" / f"rq_results{suffix}.json").read_text())[task]["RQ3_clean"] for s in SEEDS]
    for sys in ("c_revision_no_conflict_gate", "d_full", "abl_no_dedup", "abl_volume_path_only", "abl_confirmed_locked"):
        row = [ms([r[sys]["reopen_rate_by_scenario"][x]["mean"] for r in R]) for x in sc]
        row += [ms([r[sys][k] for r in R if r[sys][k] == r[sys][k]]) for k in ("precision", "recall", "f1")]
        print(f"| {sys} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="4class"); a = ap.parse_args()
    verifier_table(a.task)
    if (RQ / "seed99" / a.task / "variants" / "clean" / "summary.json").exists():
        natural_table(a.task); injection_table(a.task)
