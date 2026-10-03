"""Phases 5-6: run the decision lifecycle, baselines and ablations on LOEO test events.

Requires the Phase 3 outputs from experiments/run_verifiers.py.

Per held-out event:
  1. stance model trained on RumourEval labels from the other events (cached per event);
  2. evidence model P(stance | label) fitted on training-event threads only;
  3. thresholds tuned on the validation event only, maximising chance-corrected utility:
       stage A  commit policy (tau_prov, tau_conf, min_units_prov) with gate="none";
                shared by baselines (b), (c) and the full system so they differ only in revision
       stage B  conflict gate + re-verifier (theta_net, theta_strong, v_min, gamma) with gate="conflict"
  4. every system replayed on the test event.

Commands (folds are independent and can run in parallel with --only-event):
    .venv/bin/python -m experiments.run_lifecycle --task 4class [--only-event EVENT]
    .venv/bin/python -m experiments.run_lifecycle --task 4class --summarize
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads
from lifecycle.evidence import EvidenceModel
from lifecycle.stance import StanceModel, load_stance_labels
from lifecycle.state_machine import LifecycleConfig, run_thread
from lifecycle.trajectory_metrics import lifecycle_metrics, thread_summary, thread_utility, verifier_metrics

VERIFIER = "dgtr"


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fold_stance(all_threads, embeddings, held_out, cache_dir: Path):
    path = cache_dir / f"{held_out}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    labels = load_stance_labels()
    model = StanceModel().fit(all_threads, embeddings, labels, exclude_events=[held_out])
    result = {"by_root": {t.root_id: model.predict_thread(t, embeddings) for t in all_threads},
              "heldout_eval": model.evaluate(all_threads, embeddings, labels, [held_out]), "n_train": model.n_train}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def replay(config, model, threads, idx, embeddings, stance, probs, golds):
    return [{"gold": int(g), "root_id": threads[i].root_id, "event": threads[i].event,
             "log": run_thread(config, model, threads[i], embeddings, stance[threads[i].root_id], probs[j], threads[i].checkpoint_lengths(CHECKPOINTS))}
            for j, (i, g) in enumerate(zip(idx, golds))]


def tune(base, grid, evaluate):
    """Grid is ordered most-conservative first; a later config must strictly improve utility."""
    best_cfg, best_u, table = None, -np.inf, []
    for values in grid:
        cfg = base.with_(**values)
        u = evaluate(cfg)
        table.append({**values, "utility": u})
        if u > best_u + 1e-9:
            best_cfg, best_u = cfg, u
    return best_cfg, best_u, table


def systems(commit, full):
    return {
        "a_final_only": commit.with_(final_only=True),
        "b_early_no_revision": commit.with_(gate="none"),
        "c_revision_no_conflict_gate": commit.with_(gate="prediction_change", reverify="verifier"),
        "d_full": full,
        "abl_no_dedup": full.with_(dedup=False),
        "abl_strength_path_only": full.with_(use_volume=False),
        "abl_volume_path_only": full.with_(use_strength=False),
        "abl_confirmed_locked": full.with_(reopen_confirmed=False),
        "abl_no_hysteresis": full.with_(kappa_confirmed=1.0),
        "abl_no_support_offset": full.with_(beta_support=0.0),
        "abl_reverify_verifier_only": full.with_(reverify="verifier"),
        "abl_no_min_evidence": full.with_(min_units_prov=1),
        "abl_conflict_or_change_gate": full.with_(gate="conflict+change"),
    }


def oracle_stance(threads, stance, gold_stance):
    """Replace predicted stance with one-hot human labels wherever RumourEval annotates the reply."""
    out = {}
    for t in threads:
        rows = stance[t.root_id].copy()
        for i, pid in enumerate(t.post_ids):
            if not t.is_source[i] and pid in gold_stance:
                rows[i] = np.eye(4, dtype=np.float32)[gold_stance[pid]]
        out[t.root_id] = rows
    return out


def run_fold(args, held_out, validation, train_events, threads, labels, embeddings, all_threads, C, root, out):
    start = time.time()
    fold = np.load(out / "verifiers" / f"{held_out}.npz")
    meta = json.loads((out / "verifiers" / f"{held_out}.json").read_text())
    assert meta["validation"] == validation
    val_idx, test_idx = fold["val_idx"], fold["test_idx"]
    probs = {m: {s: softmax(fold[f"{m}_{s}"] / meta["temperatures"][m]) for s in ("val", "test")} for m in meta["temperatures"]}
    stance_info = fold_stance(all_threads, embeddings, held_out, root / "stance")
    stance = stance_info["by_root"]
    train_idx = [i for i, t in enumerate(threads) if t.event in train_events]
    evidence_model = EvidenceModel.fit([stance[threads[i].root_id][~threads[i].is_source] for i in train_idx], labels[train_idx], C)
    val_gold, test_gold = labels[val_idx], labels[test_idx]
    p_val, p_test = probs[VERIFIER]["val"], probs[VERIFIER]["test"]

    def evaluate(cfg):
        return lifecycle_metrics(replay(cfg, evidence_model, threads, val_idx, embeddings, stance, p_val, val_gold), C)["utility"]

    commit, u_a, table_a = tune(LifecycleConfig(gate="none"), STAGE_A_GRID, evaluate)
    full, u_b, table_b = tune(commit.with_(gate="conflict"), STAGE_B_GRID, evaluate)
    report = {"held_out": held_out, "validation": validation, "stance_heldout": stance_info["heldout_eval"],
              "stance_train_n": stance_info["n_train"], "llr_table": evidence_model.llr.round(3).tolist(),
              "stance_given_label": evidence_model.p.round(4).tolist(),
              "val_utility": {"stage_a": u_a, "stage_b": u_b}, "commit_config": asdict(commit), "full_config": asdict(full),
              "tuning_tables": {"stage_a": table_a, "stage_b": table_b}, "systems": {}}
    logs_dir = out / "trajectories"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs = {name: (cfg, evidence_model, stance, test_idx) for name, cfg in systems(commit, full).items()}
    # Oracle-stance condition on test threads with >= 1 human-annotated reply (evidence model refit the same way).
    gold_stance = load_stance_labels()
    annotated = np.array([i for i in test_idx if any(p in gold_stance for j, p in enumerate(threads[i].post_ids) if not threads[i].is_source[j])], dtype=int)
    if len(annotated):
        o_stance = oracle_stance(threads, stance, gold_stance)
        o_model = EvidenceModel.fit([o_stance[threads[i].root_id][~threads[i].is_source] for i in train_idx], labels[train_idx], C)
        runs["subset_b_early_no_revision"] = (commit.with_(gate="none"), evidence_model, stance, annotated)
        runs["subset_d_full"] = (full, evidence_model, stance, annotated)
        runs["subset_d_full_oracle_stance"] = (full, o_model, o_stance, annotated)
        report["oracle_llr_table"] = o_model.llr.round(3).tolist()
    position = {i: j for j, i in enumerate(test_idx)}
    for name, (cfg, model, st, idx) in runs.items():
        rows = np.array([position[i] for i in idx], dtype=int)
        results = replay(cfg, model, threads, idx, embeddings, st, p_test[rows], test_gold[rows])
        report["systems"][name] = lifecycle_metrics(results, C)
        with gzip.open(logs_dir / f"{held_out}_{name}.jsonl.gz", "wt") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    np.savez_compressed(out / "verifiers" / f"{held_out}_calibrated_test.npz", gold=test_gold, **{m: probs[m]["test"] for m in probs})
    (out / "folds").mkdir(exist_ok=True)
    (out / "folds" / f"{held_out}.json").write_text(json.dumps(report, indent=2, default=float))
    d, b = report["systems"]["d_full"], report["systems"]["b_early_no_revision"]
    print(f"[{args.task}] {held_out}: val U a/b={u_a:.3f}/{u_b:.3f} commit tau={commit.tau_prov}/{commit.tau_conf} units={commit.min_units_prov} "
          f"gate net={full.conflict.theta_net} strong={full.conflict.theta_strong} v={full.conflict.v_min} gamma={full.gamma} | "
          f"test U b={b['utility']:.3f} d={d['utility']:.3f} reopens d={d['reopens']} ({time.time() - start:.0f}s)", flush=True)


def paired_bootstrap(a, b, reps=2000, seed=7):
    """95% CI of mean(a - b) resampling threads (pooled over folds)."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(reps)]
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def summarize(out: Path, C: int, task: str):
    reports = {p.stem: json.loads(p.read_text()) for p in sorted((out / "folds").glob("*.json"))}
    pooled = {}
    for event in reports:
        for name in reports[event]["systems"]:
            with gzip.open(out / "trajectories" / f"{event}_{name}.jsonl.gz", "rt") as f:
                pooled.setdefault(name, []).extend(json.loads(line) for line in f)
    calibrated = [np.load(out / "verifiers" / f"{e}_calibrated_test.npz") for e in reports]
    gold = np.concatenate([c["gold"] for c in calibrated])
    lam = 1.0 / (C - 1)
    per_thread = {name: np.array([thread_utility(r["gold"], r["log"], lam) for r in rs]) for name, rs in pooled.items()}
    final_correct = {name: np.array([thread_summary(r["gold"], r["log"])["final_forced"] == r["gold"] for r in rs], dtype=float) for name, rs in pooled.items()}
    comparisons = {}
    for other in ("a_final_only", "b_early_no_revision", "c_revision_no_conflict_gate"):
        comparisons[f"d_full_vs_{other}"] = {"utility": paired_bootstrap(per_thread["d_full"], per_thread[other]),
                                             "final_acc": paired_bootstrap(final_correct["d_full"], final_correct[other])}
    comparisons["oracle_vs_predicted_stance_subset"] = {"utility": paired_bootstrap(per_thread["subset_d_full_oracle_stance"], per_thread["subset_d_full"])}
    summary = {"task": task, "num_classes": C, "checkpoints": CHECKPOINTS,
               "verifiers": {m: verifier_metrics(np.concatenate([c[m] for c in calibrated]), gold, C) for m in calibrated[0].files if m != "gold"},
               "systems": {name: lifecycle_metrics(results, C) for name, results in pooled.items()},
               "systems_utility_lambda1": {name: lifecycle_metrics(results, C, lam=1.0)["utility"] for name, results in pooled.items()},
               "paired_bootstrap": comparisons,
               "per_fold_utility": {e: {s: r["systems"][s]["utility"] for s in r["systems"]} for e, r in reports.items()},
               "per_fold_threads": {e: r["systems"]["d_full"]["n_threads"] for e, r in reports.items()},
               "tuned": {e: {"commit": {k: r["commit_config"][k] for k in ("tau_prov", "tau_conf", "min_units_prov")},
                             "gate": {k: r["full_config"]["conflict"][k] for k in ("theta_net", "theta_strong", "v_min")} | {"gamma": r["full_config"]["gamma"]}}
                         for e, r in reports.items()},
               "stance_heldout_macro_f1": {e: r["stance_heldout"].get("macro_f1") for e, r in reports.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"saved={out / 'summary.json'}")


STAGE_A_GRID = [dict(tau_prov=tp, tau_conf=min(tp + d, 0.95), min_units_prov=u)
                for tp, d, u in itertools.product([0.8, 0.7, 0.6, 0.5, 0.4], [0.2, 0.1], [3, 2, 1])]
STAGE_B_GRID = [dict(theta_net=tn, theta_strong=ts, v_min=v, gamma=g)
                for tn, ts, v, g in itertools.product([2.0, 1.0, 0.5, 0.25], [0.95, 0.85, 0.7], [5, 3, 2], [0.3, 1.0, 3.0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="4class")
    parser.add_argument("--root", default="artifacts/lifecycle")
    parser.add_argument("--only-event", default=None)
    parser.add_argument("--summarize", action="store_true", help="pool saved fold outputs into summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    out = root / args.task
    C = len(TASKS[args.task])
    if args.summarize:
        return summarize(out, C, args.task)
    all_threads, embeddings = load_threads()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    threads = task_threads(all_threads, args.task)
    labels = np.array([TASKS[args.task][t.label] for t in threads])
    for held_out, validation, train_events in loeo_folds(threads):
        if args.only_event and held_out != args.only_event:
            continue
        run_fold(args, held_out, validation, train_events, threads, labels, embeddings, all_threads, C, root, out)


if __name__ == "__main__":
    main()
