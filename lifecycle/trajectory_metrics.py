"""Phase 6: metrics computed on decision trajectories, not only on final outputs.

Input: ``results`` = list of {"gold": int, "log": [row per checkpoint]} produced by
lifecycle.state_machine.run_thread.

Definitions (natural PHEME data, no per-post contradiction labels exist):
  held_k            committed label at checkpoint k (PROVISIONAL or CONFIRMED), else None
  commitment episode a maximal run of checkpoints holding one label, ended early by a reopen
  justified reopen  a reopen whose invalidated label != gold
  reopen precision  justified reopens / all reopens
  reopen recall     wrong episodes ended by a reopen / all wrong episodes
  false-reopen rate correct episodes ended by a reopen / all correct episodes
  time-to-correct   checkpoints from a justified reopen until held == gold (0 = re-verified correctly at once)
  flips             changes between distinct committed labels (WAIT gaps ignored)
  utility (CU)      mean over threads x checkpoints of +1 (held correct), -lambda (held wrong), 0 (WAIT);
                    lambda = 1/(C-1) makes uniform guessing worth exactly 0 (chance-corrected)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .data import CHECKPOINTS

COMMITTED = ("PROVISIONAL", "CONFIRMED")


def held(row):
    return row["label"] if row["state"] in COMMITTED else None


def ece(probs, labels, bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = edges == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - (pred[mask] == labels[mask]).mean())
    return float(total)


def thread_summary(gold, log):
    K = len(log)
    holds = [held(r) for r in log]
    reopens = [(k, e) for k, r in enumerate(log) for e in r["events"] if e["type"] == "reopen"]
    episodes, current, start = [], None, None
    for k, r in enumerate(log):
        reopened_here = any(e["type"] == "reopen" for e in r["events"])
        if current is not None and (reopened_here or holds[k] != current):
            episodes.append({"label": current, "start": start, "end": k - 1, "reopened": reopened_here})
            current = None
        if holds[k] is not None and current is None:
            current, start = holds[k], k
    if current is not None:
        episodes.append({"label": current, "start": start, "end": K - 1, "reopened": False})
    ttc = []
    for k, e in reopens:
        if e["label"] != gold:
            later = [j - k for j in range(k, K) if holds[j] == gold]
            ttc.append(later[0] if later else None)
    sequence = [h for h in holds if h is not None]
    flips = sum(a != b for a, b in zip(sequence, sequence[1:]))
    first = next((k for k, h in enumerate(holds) if h is not None), None)
    final_signal = np.asarray(log[-1]["signal_probs"])
    return {
        "holds": holds, "reopens": reopens, "episodes": episodes, "ttc": ttc, "flips": flips, "first": first,
        "first_elapsed": log[first]["elapsed_min"] if first is not None else None,
        "final_held": holds[-1], "final_forced": holds[-1] if holds[-1] is not None else int(final_signal.argmax()),
        "final_signal": final_signal,
    }


def thread_utility(gold, log, lam):
    return float(np.mean([(1.0 if held(r) == gold else -lam) if held(r) is not None else 0.0 for r in log]))


def lifecycle_metrics(results, num_classes, checkpoints=CHECKPOINTS, lam=None):
    lam = 1.0 / (num_classes - 1) if lam is None else lam
    golds = np.array([r["gold"] for r in results])
    S = [thread_summary(r["gold"], r["log"]) for r in results]
    K = len(checkpoints)
    out = {"n_threads": len(results)}
    per_k = []
    for k in range(K):
        h = np.array([s["holds"][k] if s["holds"][k] is not None else -1 for s in S])
        committed = h >= 0
        correct = committed & (h == golds)
        per_k.append({"checkpoint": checkpoints[k], "coverage": float(committed.mean()),
                      "decision_acc": float(correct.sum() / max(committed.sum(), 1)),
                      "wrong_held_rate": float((committed & ~correct).mean())})
    out["per_checkpoint"] = per_k
    utility = np.array([[(1.0 if s["holds"][k] == g else -lam) if s["holds"][k] is not None else 0.0 for k in range(K)] for s, g in zip(S, golds)])
    out["utility"] = float(utility.mean())
    out["utility_lambda"] = lam
    firsts = [s["first"] for s in S if s["first"] is not None]
    out["ever_committed"] = len(firsts) / max(len(S), 1)
    out["earliness_mean_checkpoint"] = float(np.mean([checkpoints[f] for f in firsts])) if firsts else None
    out["median_minutes_to_first_commit"] = float(np.median([s["first_elapsed"] for s in S if s["first"] is not None])) if firsts else None
    first_correct = [s["holds"][s["first"]] == g for s, g in zip(S, golds) if s["first"] is not None]
    out["first_commit_acc"] = float(np.mean(first_correct)) if first_correct else None
    forced = np.array([s["final_forced"] for s in S])
    final_held = np.array([s["final_held"] if s["final_held"] is not None else -1 for s in S])
    out["final_acc_forced"] = float((forced == golds).mean())
    out["final_macro_f1_forced"] = float(f1_score(golds, forced, labels=range(num_classes), average="macro", zero_division=0))
    out["final_coverage"] = float((final_held >= 0).mean())
    out["final_selective_acc"] = float((final_held[final_held >= 0] == golds[final_held >= 0]).mean()) if (final_held >= 0).any() else None
    out["final_ece"] = ece(np.stack([s["final_signal"] for s in S]), golds)
    reopens = [(e, g) for s, g in zip(S, golds) for _, e in s["reopens"]]
    justified = sum(e["label"] != g for e, g in reopens)
    episodes = [(ep, g) for s, g in zip(S, golds) for ep in s["episodes"]]
    wrong = [ep for ep, g in episodes if ep["label"] != g]
    right = [ep for ep, g in episodes if ep["label"] == g]
    out["reopens"] = len(reopens)
    out["reopens_per_100_threads"] = 100.0 * len(reopens) / max(len(S), 1)
    out["reopen_precision"] = justified / len(reopens) if reopens else None
    out["reopen_recall"] = sum(ep["reopened"] for ep in wrong) / len(wrong) if wrong else None
    out["false_reopen_rate"] = sum(ep["reopened"] for ep in right) / len(right) if right else None
    out["wrong_episodes"], out["correct_episodes"] = len(wrong), len(right)
    ttc = [t for s in S for t in s["ttc"]]
    out["justified_reopens_corrected"] = sum(t is not None for t in ttc) / len(ttc) if ttc else None
    out["mean_checkpoints_to_correct"] = float(np.mean([t for t in ttc if t is not None])) if any(t is not None for t in ttc) else None
    out["mean_flips"] = float(np.mean([s["flips"] for s in S]))
    out["threads_with_flip"] = float(np.mean([s["flips"] > 0 for s in S]))
    return out


def verifier_metrics(probs, golds, num_classes, checkpoints=CHECKPOINTS):
    """probs: [N, K, C] calibrated. Accuracy, macro-F1 and ECE per checkpoint."""
    rows = []
    for k, cp in enumerate(checkpoints):
        pred = probs[:, k].argmax(1)
        rows.append({"checkpoint": cp, "acc": float((pred == golds).mean()),
                     "macro_f1": float(f1_score(golds, pred, labels=range(num_classes), average="macro", zero_division=0)),
                     "ece": ece(probs[:, k], golds)})
    return rows
