"""Step 5: RQ1–RQ3 headline numbers per configuration, mean ± std over seeds (4-class, clean protocol).

    .venv/bin/python -m experiments.generalization.rq_summary --configs baseline,masked

Reads artifacts/generalization/rq/<config>/seed*/rq_results.json plus the calibrated DGTR test
probabilities (<seed>/4class/verifiers/*_calibrated_test.npz) for accuracy by checkpoint.
"Sig" columns count seeds whose own 95% thread-bootstrap CI excludes 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("artifacts/generalization/rq")


def by_checkpoint(seed_dir):
    files = sorted((seed_dir / "4class" / "verifiers").glob("*_calibrated_test.npz"))
    p = np.concatenate([np.load(f)["dgtr"] for f in files]); g = np.concatenate([np.load(f)["gold"] for f in files])
    pred = p.argmax(-1)
    return [float((pred[:, k] == g).mean()) for k in range(p.shape[1])], [float(f1_score(g, pred[:, k], labels=range(4), average="macro")) for k in range(p.shape[1])]


def seed_metrics(seed_dir):
    r = json.loads((seed_dir / "rq_results.json").read_text())["4class"]
    acc, mf1 = by_checkpoint(seed_dir)
    rq1, rq2, rq3 = r["RQ1"], r["RQ2_clean"], r["RQ3_clean"]
    cal = rq1["heldout_accuracy_cost_vs_final_only"]["w0"]["calibrated"]
    a10 = rq1["heldout_always10_minus_always100_acc"]
    sig = lambda ci: int(ci[0] > 0 or ci[1] < 0)
    return {
        "acc_by_ckpt": acc, "mf1_by_ckpt": mf1, "acc_100_minus_10": acc[-1] - acc[0], "mf1_100_minus_10": mf1[-1] - mf1[0],
        "RQ1_always10_minus_always100_acc": a10["mean"], "RQ1_always10_sig": sig(a10["ci95"]),
        "RQ1_calibrated_minus_final_acc": cal["acc_minus_final_only"]["mean"], "RQ1_calibrated_sig": sig(cal["acc_minus_final_only"]["ci95"]),
        "RQ1_calibrated_share_by_20pct": cal["share_committed_by_20pct"],
        "RQ2_final_acc_d_minus_b": rq2["final_acc_d_minus_b"]["mean"], "RQ2_final_acc_sig": sig(rq2["final_acc_d_minus_b"]["ci95"]),
        "RQ2_utility_w05_d_minus_b": rq2["utility_d_minus_b"]["w05"]["mean"], "RQ2_w05_sig": sig(rq2["utility_d_minus_b"]["w05"]["ci95"]),
        "RQ2_threads_reopened": rq2["threads_reopened"],
        "RQ2_reopen_precision": rq2["reopens_against_wrong_commitment"] / max(1, rq2["reopens_against_wrong_commitment"] + rq2["reopens_against_correct_commitment"]),
        "RQ2_base_rate_wrong": rq2["base_rate_wrong_commitment"],
        "RQ3_d_f1": rq3["d_full"]["f1"], "RQ3_d_precision": rq3["d_full"]["precision"], "RQ3_d_recall": rq3["d_full"]["recall"],
        "RQ3_c_f1": rq3["c_revision_no_conflict_gate"]["f1"],
        "RQ3_f1_d_minus_c": rq3["f1_d_minus_c"]["mean"], "RQ3_d_minus_c_sig": sig(rq3["f1_d_minus_c"]["ci95"]),
        "RQ3_strong_single_reopen_rate": rq3["d_full"]["reopen_rate_by_scenario"]["strong_single"]["mean"],
        "gate_current_folds": sum(v["gate_design"] == "current" for v in r["clean_selection"].values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="baseline")
    args = parser.parse_args()
    table = {}
    for c in args.configs.split(","):
        seeds = sorted(d for d in (ROOT / c).glob("seed*") if (d / "rq_results.json").exists())
        if seeds:
            table[c] = {d.name: seed_metrics(d) for d in seeds}
    keys = [k for k in next(iter(next(iter(table.values())).values())) if not k.endswith("_by_ckpt")]
    print("| metric | " + " | ".join(f"{c} (n={len(v)})" for c, v in table.items()) + " |")
    print("|---|" + "---|" * len(table))
    for k in ["acc_by_ckpt", "mf1_by_ckpt"]:
        print(f"| {k} (10/20/30/50/75/100%) | " + " | ".join(" / ".join(f"{x:.3f}" for x in np.mean([s[k] for s in v.values()], 0)) for v in table.values()) + " |")
    for k in keys:
        cells = []
        for v in table.values():
            x = np.array([s[k] for s in v.values()], dtype=float)
            cells.append(f"{int(x.sum())}/{len(x)}" if k.endswith("_sig") else f"{x.mean():+.4f} ± {x.std(ddof=1) if len(x) > 1 else 0:.4f}")
        print(f"| {k} | " + " | ".join(cells) + " |")
    (ROOT / "rq_summary.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
