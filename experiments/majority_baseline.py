"""Per-fold held-out accuracy of the verifier vs the majority-class baseline (majority taken from training events).

    .venv/bin/python -m experiments.majority_baseline > artifacts/lifecycle/majority_baseline.md
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from lifecycle.data import TASKS, load_threads, loeo_folds, task_threads


def main():
    threads_all, _ = load_threads()
    out = {}
    for task in ("4class", "veracity"):
        labels = TASKS[task]; C = len(labels)
        threads = task_threads(threads_all, task)
        rows, pooled = [], {"gold": [], "dgtr": [], "gat": [], "maj": []}
        print(f"\n### {task}\n")
        print("| held-out event | n | train-majority class | majority acc | DGTR acc @10% | DGTR acc @100% | GAT acc @100% | DGTR beats majority? | DGTR mF1 @100% | majority mF1 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for held_out, validation, train_events in loeo_folds(threads):
            f = np.load(Path("artifacts/lifecycle") / task / "verifiers" / f"{held_out}_calibrated_test.npz")
            gold = f["gold"]
            train_counts = Counter(labels[t.label] for t in threads if t.event in train_events or t.event == validation)
            maj = train_counts.most_common(1)[0][0]
            d10, d100, g100 = (f["dgtr"][:, 0].argmax(1) == gold).mean(), (f["dgtr"][:, -1].argmax(1) == gold).mean(), (f["gat_full"][:, -1].argmax(1) == gold).mean()
            m = (gold == maj).mean()
            mf = f1_score(gold, f["dgtr"][:, -1].argmax(1), labels=range(C), average="macro", zero_division=0)
            mm = f1_score(gold, np.full_like(gold, maj), labels=range(C), average="macro", zero_division=0)
            name = list(labels)[maj]
            rows.append({"event": held_out, "n": int(len(gold)), "majority_class": name, "majority_acc": float(m), "dgtr_acc_10": float(d10),
                         "dgtr_acc_100": float(d100), "gat_acc_100": float(g100), "dgtr_beats_majority": bool(d100 > m), "dgtr_mf1": float(mf), "majority_mf1": float(mm)})
            print(f"| {held_out.replace('-all-rnr-threads', '')} | {len(gold)} | {name} | {m:.3f} | {d10:.3f} | {d100:.3f} | {g100:.3f} | {'yes' if d100 > m else '**no**'} | {mf:.3f} | {mm:.3f} |")
            pooled["gold"].append(gold); pooled["dgtr"].append(f["dgtr"][:, -1].argmax(1)); pooled["gat"].append(f["gat_full"][:, -1].argmax(1)); pooled["maj"].append(np.full_like(gold, maj))
        g = np.concatenate(pooled["gold"])
        res = {k: float((np.concatenate(v) == g).mean()) for k, v in pooled.items() if k != "gold"}
        mf1 = {k: float(f1_score(g, np.concatenate(v), labels=range(C), average="macro", zero_division=0)) for k, v in pooled.items() if k != "gold"}
        folds_beaten = sum(r["dgtr_beats_majority"] for r in rows)
        print(f"\nPooled accuracy: DGTR {res['dgtr']:.3f}, GAT {res['gat']:.3f}, train-majority {res['maj']:.3f}. "
              f"Pooled macro-F1: DGTR {mf1['dgtr']:.3f}, GAT {mf1['gat']:.3f}, majority {mf1['maj']:.3f}. DGTR beats majority accuracy in {folds_beaten}/9 folds.")
        out[task] = {"folds": rows, "pooled_acc": res, "pooled_macro_f1": mf1, "folds_beating_majority": folds_beaten}
    Path("artifacts/lifecycle/majority_baseline.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
