"""RQ1: when to commit. Compares four commit policies on the validation events only.

Every LOEO fold has its own verifier and its own validation event. Each validation
set is split 50/50 by a hash of the thread id; temperature, policy hyperparameters
and the stopping rule's estimators are fitted on one half and scored on the other
(2-way cross-fitting), so the comparison is out-of-sample even within validation.
Results are pooled over the 9 folds. A frozen held-out confirmation (hyperparameters
from the full validation set, applied once to the test events) is reported
separately and was not used for any choice.

Policies (all commit once; revision is a separate layer, RQ2/RQ3):
  fixed_checkpoint  commit argmax at checkpoint k*                         (no abstention)
  raw_confidence    first k with uncalibrated max-softmax >= tau          (abstain if never)
  calibrated        first k with calibrated max >= tau, same argmax for 2 checkpoints,
                    evidence units >= u (the lifecycle's commit rule)      (abstain if never)
  cost_sensitive    one-step-lookahead Bayes stopping rule: commit at k iff
                    E[U | commit now] >= max(0, E[U | commit at k+1]); estimators
                    P(correct_k | p_k) and P(correct_{k+1} | p_k) are logistic fits
                    on the other half; no tuned hyperparameter (abstain if E[U] <= 0 at the end)

Utilities (per thread; r = +1 correct, -lambda wrong, 0 abstain; lambda = 1/(C-1)):
  w0    r                       no reward for earliness at all
  w05   r * (1 - 0.5 * t)       t = fraction of thread observed at commit
  hold  r * (K - k) / K         time the decision is held (the Phase 6 utility)

    .venv/bin/python -m experiments.run_commit_policies --task 4class
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from decision.calibration import TemperatureScaler
from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads
from lifecycle.evidence import EvidenceStore
from lifecycle.trajectory_metrics import ece

K = len(CHECKPOINTS)
TAUS = np.round(np.arange(0.30, 0.96, 0.05), 2)
UTILITIES = {
    "w0": lambda k: np.ones_like(np.asarray(k, dtype=float)),
    "w05": lambda k: 1.0 - 0.5 * np.asarray(CHECKPOINTS)[np.asarray(k)],
    "hold": lambda k: (K - np.asarray(k, dtype=float)) / K,
}
POLICIES = ("fixed_checkpoint", "raw_confidence", "calibrated", "cost_sensitive")


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logits, gold):
    flat = torch.from_numpy(logits.reshape(-1, logits.shape[-1])).float()
    return float(TemperatureScaler().fit(flat, torch.from_numpy(np.repeat(gold, K)).long()).temperature.detach())


def evidence_units(thread, embeddings):
    store, units = EvidenceStore(), []
    seen = 0
    for m in thread.checkpoint_lengths(CHECKPOINTS):
        for i in range(seen, m):
            store.add(i, thread.texts[i], embeddings[thread.emb_rows[i]], np.zeros(4), is_source=bool(thread.is_source[i]))
        seen = max(seen, m)
        units.append(store.evidence_units)
    return units


# ---- policies: each returns (commit_k [-1 = abstain], label) per thread -------------------------

def fixed_checkpoint(cal, raw, units, k):
    return np.full(len(cal), k), cal[:, k].argmax(1)


def first_true(mask):
    any_ = mask.any(1)
    return np.where(any_, mask.argmax(1), -1)


def raw_confidence(cal, raw, units, tau):
    k = first_true(raw.max(2) >= tau)
    return k, raw.argmax(2)[np.arange(len(raw)), np.maximum(k, 0)]


def calibrated(cal, raw, units, tau, u):
    pred = cal.argmax(2)
    stable = np.concatenate([np.zeros((len(cal), 1), bool), pred[:, 1:] == pred[:, :-1]], axis=1)
    k = first_true((cal.max(2) >= tau) & stable & (units >= u))
    return k, pred[np.arange(len(cal)), np.maximum(k, 0)]


class StoppingEstimators:
    """P(argmax_k correct | p_k) and P(argmax_{k+1} correct | p_k), logistic in logit(max p_k)."""

    def __init__(self, cal, gold):
        self.now, self.next = [], []
        for k in range(K):
            x = self._x(cal[:, k])
            self.now.append(self._fit(x, cal[:, k].argmax(1) == gold))
            self.next.append(self._fit(x, cal[:, k + 1].argmax(1) == gold) if k + 1 < K else None)

    @staticmethod
    def _x(p):
        m = np.clip(p.max(1), 1e-4, 1 - 1e-4)
        return np.log(m / (1 - m))[:, None]

    @staticmethod
    def _fit(x, y):
        if y.all() or not y.any():
            return float(y.mean())
        return LogisticRegression(C=1.0).fit(x, y.astype(int))

    def prob(self, model, p):
        return np.full(len(p), model) if isinstance(model, float) else model.predict_proba(self._x(p))[:, 1]


def cost_sensitive(cal, raw, units, est: StoppingEstimators, weight, lam):
    n = len(cal)
    commit = np.full(n, -1)
    for k in range(K):
        open_ = commit < 0
        a_now = est.prob(est.now[k], cal[:, k])
        v_now = (a_now * (1 + lam) - lam) * weight(k)
        if k + 1 < K:
            a_next = est.prob(est.next[k], cal[:, k])
            v_wait = np.maximum(0.0, (a_next * (1 + lam) - lam) * weight(k + 1))
            take = open_ & (v_now > 0) & (v_now >= v_wait)
        else:
            take = open_ & (v_now > 0)
        commit[take] = k
    return commit, cal.argmax(2)[np.arange(n), np.maximum(commit, 0)]


# ---- scoring ----------------------------------------------------------------------------------

def per_thread_utility(commit, label, gold, lam, weight):
    r = np.where(commit < 0, 0.0, np.where(label == gold, 1.0, -lam))
    return r * np.where(commit < 0, 0.0, weight(np.maximum(commit, 0)))


def unlocked_flips(cal, raw, policy, params):
    """Label changes a user would see if the threshold rule were re-applied every checkpoint without locking."""
    if policy == "raw_confidence":
        show = raw.max(2) >= params["tau"]; pred = raw.argmax(2)
    elif policy == "calibrated":
        show = cal.max(2) >= params["tau"]; pred = cal.argmax(2)
    else:
        return 0.0
    flips = []
    for s, p in zip(show, pred):
        seq = p[s]
        flips.append(int((seq[1:] != seq[:-1]).sum()) if len(seq) > 1 else 0)
    return float(np.mean(flips))


def describe(commit, label, gold, minutes, lam):
    committed = commit >= 0
    out = {"coverage": float(committed.mean()), "abstention": float(1 - committed.mean()),
           "selective_acc": float((label[committed] == gold[committed]).mean()) if committed.any() else None,
           "mean_commit_checkpoint": float(np.asarray(CHECKPOINTS)[commit[committed]].mean()) if committed.any() else None,
           "median_minutes_to_commit": float(np.median(minutes[committed, commit[committed]])) if committed.any() else None,
           "locked_flips": 0.0}
    for name, weight in UTILITIES.items():
        out[f"U_{name}"] = float(per_thread_utility(commit, label, gold, lam, weight).mean())
    return out


def grid(policy):
    if policy == "fixed_checkpoint":
        return [{"k": k} for k in range(K)]
    if policy == "raw_confidence":
        return [{"tau": float(t)} for t in TAUS]
    if policy == "calibrated":
        return [{"tau": float(t), "u": u} for t in TAUS for u in (3, 2, 1)]
    return [{}]


def apply(policy, params, cal, raw, units, est, weight, lam):
    if policy == "fixed_checkpoint":
        return fixed_checkpoint(cal, raw, units, **params)
    if policy == "raw_confidence":
        return raw_confidence(cal, raw, units, **params)
    if policy == "calibrated":
        return calibrated(cal, raw, units, **params)
    return cost_sensitive(cal, raw, units, est, weight, lam)


def select(policy, cal, raw, units, gold, est, weight, lam):
    best, best_u = None, -np.inf
    for params in grid(policy):
        c, l = apply(policy, params, cal, raw, units, est, weight, lam)
        u = per_thread_utility(c, l, gold, lam, weight).mean()
        if u > best_u + 1e-12:
            best, best_u = params, u
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="4class")
    parser.add_argument("--root", default="artifacts/lifecycle")
    args = parser.parse_args()
    out = Path(args.root) / args.task
    dest = out / "commit_policies"
    dest.mkdir(parents=True, exist_ok=True)
    all_threads, embeddings = load_threads()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    threads = task_threads(all_threads, args.task)
    C = len(TASKS[args.task])
    lam = 1.0 / (C - 1)
    pooled_val = {u: {p: [] for p in POLICIES} for u in UTILITIES}       # per-thread records, cross-fitted
    pooled_test = {u: {p: [] for p in POLICIES} for u in UTILITIES}
    curves_val = {"cal": [], "raw": [], "units": [], "gold": [], "minutes": []}
    chosen = {}
    for held_out, validation, _ in loeo_folds(threads):
        fold = np.load(out / "verifiers" / f"{held_out}.npz")
        meta = json.loads((out / "verifiers" / f"{held_out}.json").read_text())
        split = {}
        for part in ("val", "test"):
            idx = fold[f"{part}_idx"]
            split[part] = {"raw_logits": fold[f"dgtr_{part}"], "gold": fold[f"{part}_labels"],
                           "units": np.array([evidence_units(threads[i], embeddings) for i in idx]),
                           "minutes": np.array([[threads[i].rel_time[m - 1] / 60.0 for m in threads[i].checkpoint_lengths(CHECKPOINTS)] for i in idx]),
                           "half": np.array([zlib.crc32(threads[i].root_id.encode()) % 2 for i in idx])}
        v = split["val"]
        cal_cross = np.zeros_like(v["raw_logits"])
        records = {u: {p: [None] * len(v["gold"]) for p in POLICIES} for u in UTILITIES}
        for h in (0, 1):
            fit, ev = v["half"] != h, v["half"] == h
            T = fit_temperature(v["raw_logits"][fit], v["gold"][fit])
            cal_fit, cal_ev = softmax(v["raw_logits"][fit] / T), softmax(v["raw_logits"][ev] / T)
            raw_fit, raw_ev = softmax(v["raw_logits"][fit]), softmax(v["raw_logits"][ev])
            cal_cross[ev] = v["raw_logits"][ev] / T
            est_fit = StoppingEstimators(cal_fit, v["gold"][fit])
            for uname, weight in UTILITIES.items():
                for p in POLICIES:
                    params = select(p, cal_fit, raw_fit, v["units"][fit], v["gold"][fit], est_fit, weight, lam)
                    c, l = apply(p, params, cal_ev, raw_ev, v["units"][ev], est_fit, weight, lam)
                    for j, (ci, li) in zip(np.nonzero(ev)[0], zip(c, l)):
                        records[uname][p][j] = (int(ci), int(li), int(v["gold"][j]), v["minutes"][j], params)
        for uname in UTILITIES:
            for p in POLICIES:
                pooled_val[uname][p].extend(records[uname][p])
        curves_val["cal"].append(softmax(cal_cross)); curves_val["raw"].append(softmax(v["raw_logits"]))
        curves_val["units"].append(v["units"]); curves_val["gold"].append(v["gold"]); curves_val["minutes"].append(v["minutes"])
        # frozen held-out confirmation: everything fitted on the full validation event
        T = meta["temperatures"]["dgtr"]
        cal_v, raw_v = softmax(v["raw_logits"] / T), softmax(v["raw_logits"])
        t = split["test"]
        cal_t, raw_t = softmax(t["raw_logits"] / T), softmax(t["raw_logits"])
        est = StoppingEstimators(cal_v, v["gold"])
        chosen[held_out] = {}
        for uname, weight in UTILITIES.items():
            chosen[held_out][uname] = {}
            for p in POLICIES:
                params = select(p, cal_v, raw_v, v["units"], v["gold"], est, weight, lam)
                chosen[held_out][uname][p] = params
                c, l = apply(p, params, cal_t, raw_t, t["units"], est, weight, lam)
                pooled_test[uname][p].extend((int(ci), int(li), int(g), mm, params) for ci, li, g, mm in zip(c, l, t["gold"], t["minutes"]))
        print(f"[{args.task}] {held_out}: val={len(v['gold'])} test={len(t['gold'])}", flush=True)

    def table(pooled):
        result = {}
        for uname in UTILITIES:
            result[uname] = {}
            for p in POLICIES:
                recs = pooled[uname][p]
                c = np.array([r[0] for r in recs]); l = np.array([r[1] for r in recs]); g = np.array([r[2] for r in recs])
                minutes = np.stack([r[3] for r in recs])
                result[uname][p] = describe(c, l, g, minutes, lam)
        return result

    cal = np.concatenate(curves_val["cal"]); raw = np.concatenate(curves_val["raw"]); units = np.concatenate(curves_val["units"])
    gold = np.concatenate(curves_val["gold"]); minutes = np.concatenate(curves_val["minutes"])
    # sanity rows: unselected fixed policies under every utility, and unlocked flip counts
    sanity = {}
    for k in (0, K - 1):
        c, l = fixed_checkpoint(cal, raw, units, k)
        sanity[f"always_at_{int(CHECKPOINTS[k] * 100)}pct"] = describe(c, l, gold, minutes, lam)
    flips = {p: unlocked_flips(cal, raw, p, {"tau": tau}) for p, tau in (("raw_confidence", 0.6), ("calibrated", 0.4))}
    # curves on cross-fitted calibrated validation probabilities
    curves = {"fixed_checkpoint": [], "raw_confidence": [], "calibrated": [], "cost_sensitive": [], "final_confidence_reference": []}
    for k in range(K):
        c, l = fixed_checkpoint(cal, raw, units, k)
        curves["fixed_checkpoint"].append({"param": CHECKPOINTS[k], **describe(c, l, gold, minutes, lam)})
    for tau in np.round(np.arange(0.25, 0.991, 0.025), 3):
        c, l = raw_confidence(cal, raw, units, tau)
        curves["raw_confidence"].append({"param": float(tau), **describe(c, l, gold, minutes, lam)})
        c, l = calibrated(cal, raw, units, tau, 1)
        curves["calibrated"].append({"param": float(tau), **describe(c, l, gold, minutes, lam)})
        keep = cal[:, -1].max(1) >= tau
        c = np.where(keep, K - 1, -1); l = cal[:, -1].argmax(1)
        curves["final_confidence_reference"].append({"param": float(tau), **describe(c, l, gold, minutes, lam)})
    est_all = StoppingEstimators(cal, gold)   # descriptive sweep only; in-sample, not used for any number in the tables
    for w in np.round(np.arange(0.0, 1.0001, 0.1), 2):
        for lam_s in (lam, 2 * lam, 4 * lam):
            c, l = cost_sensitive(cal, raw, units, est_all, lambda k, w=w: 1.0 - w * np.asarray(CHECKPOINTS)[np.asarray(k)], lam_s)
            curves["cost_sensitive"].append({"param": {"w": float(w), "lambda": float(lam_s)}, **describe(c, l, gold, minutes, lam)})
    summary = {"task": args.task, "lambda": lam, "n_val_threads": int(len(gold)),
               "validation_crossfit": table(pooled_val), "heldout_frozen": table(pooled_test),
               "sanity_unselected": sanity, "unlocked_flips_example": flips, "curves_validation": curves,
               "verifier_val_crossfit_ece_by_checkpoint": [ece(cal[:, k], gold) for k in range(K)],
               "verifier_val_crossfit_acc_by_checkpoint": [float((cal[:, k].argmax(1) == gold).mean()) for k in range(K)],
               "chosen_params_full_validation": chosen}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(dest / "test_records.npz", **{f"{u}__{p}": np.array([[r[0], r[1], r[2]] for r in pooled_test[u][p]]) for u in UTILITIES for p in POLICIES})
    print(f"saved={dest / 'summary.json'}")


if __name__ == "__main__":
    main()
