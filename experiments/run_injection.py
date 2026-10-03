"""Phase 6 stress test: controlled late-evidence injection with known reopen ground truth.

PHEME has no per-post "this should reopen the decision" labels, so the lifecycle is
also tested on synthetic-but-real evidence: for each LOEO test thread whose system
holds a TRUE or FALSE commitment at the 75% checkpoint, real replies carrying
*human* RumourEval stance labels (taken from other threads; same event preferred)
are appended at the 100% checkpoint, after the thread's real posts.

  contradicting stance: deny for a TRUE commitment, support for a FALSE commitment
  scenario            injected posts                                         should reopen?
  strong_single       1 contradicting post, top-quartile stance confidence    yes
  distinct_volume     5 distinct contradicting posts (any confidence)         yes
  weak_duplicates     1 bottom-half-confidence contradicting post x 8 copies  no
  neutral             5 distinct "comment" posts                              no
  supporting          5 distinct posts supporting the committed label         no

The verifier is re-run on the injected 100% snapshot, and the stance model scores the
injected replies against the target thread's source, so every component sees them.
Thresholds are the validation-tuned ones from experiments/run_lifecycle.py.

Command:
    .venv/bin/python -m experiments.run_injection --task 4class
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from lifecycle.data import CHECKPOINTS, STANCES, TASKS, Thread, load_threads, loeo_folds, task_threads
from lifecycle.evidence import ConflictConfig, EvidenceModel
from lifecycle.stance import StanceModel, _source_row, load_stance_labels
from lifecycle.state_machine import LifecycleConfig, run_thread
from lifecycle.verifier import ThreadCache, TemporalVerifier, VerifierConfig, predict_logits
from experiments.run_lifecycle import softmax, systems

FALSE, TRUE = 0, 1
SCENARIOS = {"strong_single": True, "distinct_volume": True, "weak_duplicates": False, "neutral": False, "supporting": False}
TESTED = ("b_early_no_revision", "c_revision_no_conflict_gate", "d_full", "abl_no_dedup", "abl_strength_path_only",
          "abl_volume_path_only", "abl_confirmed_locked", "abl_no_hysteresis")


def config_from_dict(d):
    return LifecycleConfig(**{**d, "conflict": ConflictConfig(**d["conflict"])})


def build_pools(all_threads, gold_stance, stance_cache):
    """Human-labelled replies with the fold stance model's confidence in the human label (in original context)."""
    pools = {s: [] for s in range(len(STANCES))}
    for t in all_threads:
        probs = stance_cache[t.root_id]
        for i, pid in enumerate(t.post_ids):
            if not t.is_source[i] and pid in gold_stance:
                s = gold_stance[pid]
                pools[s].append({"event": t.event, "root": t.root_id, "text": t.texts[i], "row": int(t.emb_rows[i]), "conf": float(probs[i, s])})
    return pools


def pick(pool, target, n, rng, *, distinct=True):
    """Prefer posts from other threads of the target's event; fall back to all other threads."""
    same = [p for p in pool if p["event"] == target.event and p["root"] != target.root_id]
    candidates = same if len(same) >= n else [p for p in pool if p["root"] != target.root_id]
    order = rng.permutation(len(candidates))[:n]
    return [candidates[i] for i in order]


def scenario_posts(name, label, target, pools, rng):
    contradict = STANCES.index("deny") if label == TRUE else STANCES.index("support")
    agree = STANCES.index("support") if label == TRUE else STANCES.index("deny")
    pool = pools[contradict]
    confs = np.array([p["conf"] for p in pool])
    if name == "strong_single":
        return pick([p for p in pool if p["conf"] >= np.percentile(confs, 75)], target, 1, rng)
    if name == "weak_duplicates":
        return pick([p for p in pool if p["conf"] <= np.percentile(confs, 50)], target, 1, rng) * 8
    if name == "distinct_volume":
        return pick(pool, target, 5, rng)
    if name == "neutral":
        return pick(pools[STANCES.index("comment")], target, 5, rng)
    return pick(pools[agree], target, 5, rng)


def injected_thread(t: Thread, posts) -> Thread:
    k = len(posts)
    src = int(np.argmax(t.is_source)) if t.is_source.any() else 0
    last = float(t.rel_time[-1])
    return Thread(t.root_id, t.event, t.label, t.post_ids + [f"inj{j}" for j in range(k)], t.texts + [p["text"] for p in posts],
                  np.concatenate([t.rel_time, last + 60.0 * np.arange(1, k + 1, dtype=np.float32)]),
                  np.concatenate([t.parent, np.full(k, src)]), np.concatenate([t.is_source, np.zeros(k, dtype=bool)]),
                  np.concatenate([t.emb_rows, np.array([p["row"] for p in posts], dtype=np.int64)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="4class")
    parser.add_argument("--root", default="artifacts/lifecycle")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = Path(args.root) / args.task
    all_threads, embeddings = load_threads()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    emb_t = torch.from_numpy(embeddings)
    threads = task_threads(all_threads, args.task)
    by_root = {t.root_id: i for i, t in enumerate(threads)}
    labels = np.array([TASKS[args.task][t.label] for t in threads])
    C = len(TASKS[args.task])
    gold_stance = load_stance_labels()
    K = len(CHECKPOINTS)
    records = []
    for held_out, validation, train_events in loeo_folds(threads):
        report = json.loads((out / "folds" / f"{held_out}.json").read_text())
        meta = json.loads((out / "verifiers" / f"{held_out}.json").read_text())
        stance_cache = pickle.loads((Path(args.root) / "stance" / f"{held_out}.pkl").read_bytes())["by_root"]
        stance_model = StanceModel().fit(all_threads, embeddings, gold_stance, exclude_events=[held_out])
        train_idx = [i for i, t in enumerate(threads) if t.event in train_events]
        evidence_model = EvidenceModel.fit([stance_cache[threads[i].root_id][~threads[i].is_source] for i in train_idx], labels[train_idx], C)
        verifier = TemporalVerifier(VerifierConfig(**meta["dgtr_config"]))
        verifier.load_state_dict(torch.load(out / "verifiers" / f"{held_out}_dgtr.pt"))
        temperature = meta["temperatures"]["dgtr"]
        fold = np.load(out / "verifiers" / f"{held_out}.npz")
        test_probs = {threads[i].root_id: softmax(fold["dgtr_test"][j] / temperature) for j, i in enumerate(fold["test_idx"])}
        # Pools may include other threads of the held-out event: human stance labels are used only to
        # construct the stress test, never to train anything that scores it.
        pools = build_pools(all_threads, gold_stance, stance_cache)
        configs = systems(config_from_dict(report["commit_config"]), config_from_dict(report["full_config"]))
        # 1. eligibility per system from its own natural trajectory
        jobs = {}
        for name in TESTED:
            with gzip.open(out / "trajectories" / f"{held_out}_{name}.jsonl.gz", "rt") as f:
                for line in f:
                    r = json.loads(line)
                    row = r["log"][K - 2]
                    if row["state"] in ("PROVISIONAL", "CONFIRMED") and row["label"] in (FALSE, TRUE):
                        for scen in SCENARIOS:
                            jobs.setdefault((r["root_id"], row["label"], scen), []).append((name, row["state"]))
        if not jobs:
            continue
        # 2. build injected threads once per (thread, committed label, scenario)
        keys = sorted(jobs)
        variants, items = [], {}
        for key in keys:
            root, label, scen = key
            t = threads[by_root[root]]
            rng = np.random.default_rng(zlib.crc32(f"{args.seed}|{root}|{label}|{scen}".encode()))
            posts = scenario_posts(scen, label, t, pools, rng)
            vecs = embeddings[[p["row"] for p in posts]]
            st = stance_model.predict_proba([p["text"] for p in posts], vecs, embeddings[_source_row(t)])
            items[key] = [{"text": p["text"], "vector": v, "stance": s} for p, v, s in zip(posts, vecs, st)]
            variants.append(injected_thread(t, posts))
        cache = ThreadCache(variants)
        full_logits = []
        for start in range(0, len(variants), 256):
            idx = np.arange(start, min(start + 256, len(variants)))
            full_logits.append(predict_logits(verifier, cache, idx, emb_t, (1.0,)))
        full_probs = softmax(np.concatenate(full_logits)[:, 0] / temperature)
        # 3. replay each eligible system with the injection at the last checkpoint
        for key, p_last in zip(keys, full_probs):
            root, label, scen = key
            t = threads[by_root[root]]
            probs = test_probs[root].copy()
            probs[K - 1] = p_last
            for name, state_before in jobs[key]:
                log = run_thread(configs[name], evidence_model, t, embeddings, stance_cache[root], probs,
                                 t.checkpoint_lengths(CHECKPOINTS), override_posts={K - 1: items[key]})
                last = log[-1]
                records.append({"event": held_out, "root_id": root, "system": name, "scenario": scen, "committed": label,
                                "state_before": state_before, "gold": int(labels[by_root[root]]),
                                "should_reopen": SCENARIOS[scen], "reopened": any(e["type"] == "reopen" for e in last["events"]),
                                "final_label": last["label"], "final_state": last["state"],
                                "verifier_flipped": int(np.argmax(p_last)) != label, "conflict": last["conflict"]})
        print(f"[{args.task}] {held_out}: {len(keys)} injected variants", flush=True)
    with gzip.open(out / "injection_records.jsonl.gz", "wt") as f:
        for r in records:
            f.write(json.dumps(r, default=float) + "\n")
    summary = {}
    for name in TESTED:
        rs = [r for r in records if r["system"] == name]
        by_s = {s: [r for r in rs if r["scenario"] == s] for s in SCENARIOS}
        tp = sum(r["reopened"] and r["should_reopen"] for r in rs)
        fp = sum(r["reopened"] and not r["should_reopen"] for r in rs)
        fn = sum((not r["reopened"]) and r["should_reopen"] for r in rs)
        summary[name] = {"eligible_threads": len(by_s["neutral"]),
                         "reopen_rate": {s: float(np.mean([r["reopened"] for r in v])) if v else None for s, v in by_s.items()},
                         "reopen_rate_from_confirmed": {s: float(np.mean([r["reopened"] for r in v if r["state_before"] == "CONFIRMED"])) if any(r["state_before"] == "CONFIRMED" for r in v) else None for s, v in by_s.items()},
                         "verifier_flip_rate": {s: float(np.mean([r["verifier_flipped"] for r in v])) if v else None for s, v in by_s.items()},
                         "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                         "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    (out / "injection_summary.json").write_text(json.dumps(summary, indent=2))
    for name, s in summary.items():
        print(name, json.dumps(s["reopen_rate"]), "P/R/F1", s["precision"], s["recall"], round(s["f1"], 3))


if __name__ == "__main__":
    main()
