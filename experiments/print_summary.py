"""Print markdown tables from artifacts/lifecycle/<task>/summary.json (and injection_summary.json if present)."""
import json
import sys
from pathlib import Path


def fmt(v, p=3):
    return "–" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def main(task):
    out = Path("artifacts/lifecycle") / task
    s = json.loads((out / "summary.json").read_text())
    print(f"## {task}  (C={s['num_classes']}, pooled LOEO test threads={s['systems']['d_full']['n_threads']})\n")
    print("### Verifiers (calibrated, pooled over 9 held-out events): macro-F1 / accuracy / ECE per checkpoint\n")
    print("| model | " + " | ".join(f"{int(c*100)}%" for c in s["checkpoints"]) + " |")
    print("|---|" + "---|" * len(s["checkpoints"]))
    for m, rows in s["verifiers"].items():
        print(f"| {m} | " + " | ".join(f"{r['macro_f1']:.3f} / {r['acc']:.3f} / {r['ece']:.3f}" for r in rows) + " |")
    print("\n### Lifecycle systems (pooled test)\n")
    cols = ["utility", "final_acc_forced", "final_macro_f1_forced", "final_coverage", "final_selective_acc", "first_commit_acc", "earliness_mean_checkpoint",
            "median_minutes_to_first_commit", "reopens_per_100_threads", "reopen_precision", "reopen_recall", "false_reopen_rate",
            "justified_reopens_corrected", "mean_checkpoints_to_correct", "mean_flips", "final_ece"]
    print("| system | " + " | ".join(cols) + " | U(λ=1) |")
    print("|---|" + "---|" * (len(cols) + 1))
    for name, m in s["systems"].items():
        print(f"| {name} | " + " | ".join(fmt(m[c]) for c in cols) + f" | {fmt(s['systems_utility_lambda1'][name])} |")
    print("\n### Accuracy of committed decisions / coverage by checkpoint\n")
    print("| system | " + " | ".join(f"{int(c*100)}%" for c in s["checkpoints"]) + " |")
    print("|---|" + "---|" * len(s["checkpoints"]))
    for name in ("a_final_only", "b_early_no_revision", "c_revision_no_conflict_gate", "d_full"):
        pc = s["systems"][name]["per_checkpoint"]
        print(f"| {name} | " + " | ".join(f"{p['decision_acc']:.3f} / {p['coverage']:.2f}" for p in pc) + " |")
    print("\n### Paired bootstrap (thread resampling, 95% CI)\n")
    for k, v in s["paired_bootstrap"].items():
        print(f"- {k}: " + "; ".join(f"{m} Δ={x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for m, x in v.items()))
    print("\n### Tuned on validation event\n")
    for e, t in s["tuned"].items():
        print(f"- {e.replace('-all-rnr-threads','')}: {t['commit']} {t['gate']} stance-F1={fmt(s['stance_heldout_macro_f1'][e])}")
    inj = out / "injection_summary.json"
    if inj.exists():
        j = json.loads(inj.read_text())
        print("\n### Injection stress test: reopen rate per scenario (should reopen: strong_single, distinct_volume)\n")
        scen = list(next(iter(j.values()))["reopen_rate"])
        print("| system | eligible | " + " | ".join(scen) + " | precision | recall | F1 |")
        print("|---|---|" + "---|" * (len(scen) + 3))
        for name, v in j.items():
            print(f"| {name} | {v['eligible_threads']} | " + " | ".join(fmt(v['reopen_rate'][x], 2) for x in scen) + f" | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "veracity")
