"""Phase 3: train the temporal verifier and baselines under leave-one-event-out.

Writes, per task and held-out event, raw logits for validation and test threads at
every checkpoint plus a validation-fitted temperature per model:

    artifacts/lifecycle/<task>/verifiers/<event>.npz
    artifacts/lifecycle/<task>/verifiers/<event>.json   (split, temperatures, history)

Command (both tasks, seed 7):
    .venv/bin/python -m experiments.run_verifiers --task 4class
    .venv/bin/python -m experiments.run_verifiers --task veracity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from decision.calibration import TemperatureScaler
from lifecycle import baselines
from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads
from lifecycle.verifier import ThreadCache, VerifierConfig, config_dict, fit_verifier, predict_logits

MODELS = ("dgtr", "source_lr", "prefix_lr", "gat_full")


def fit_temperature(logits, labels):
    flat = torch.from_numpy(logits.reshape(-1, logits.shape[-1])).float()
    y = torch.from_numpy(np.repeat(labels, logits.shape[1])).long()
    return float(TemperatureScaler().fit(flat, y).temperature.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="4class")
    parser.add_argument("--output", default="artifacts/lifecycle")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--only-event", default=None)
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.output) / args.task / "verifiers"
    out.mkdir(parents=True, exist_ok=True)
    all_threads, emb_np = load_threads()
    threads = task_threads(all_threads, args.task)
    emb_np = np.ascontiguousarray(emb_np, dtype=np.float32)
    emb_t = torch.from_numpy(emb_np)
    label_map = TASKS[args.task]
    labels = np.array([label_map[t.label] for t in threads])
    cache = ThreadCache(threads)
    by_event = {}
    for i, t in enumerate(threads):
        by_event.setdefault(t.event, []).append(i)
    models = args.models.split(",")
    for held_out, validation, train_events in loeo_folds(threads):
        if args.only_event and held_out != args.only_event:
            continue
        start = time.time()
        train_idx = np.array(sorted(i for e in train_events for i in by_event[e]))
        val_idx, test_idx = np.array(by_event[validation]), np.array(by_event[held_out])
        print(f"[{args.task}] held_out={held_out} validation={validation} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        arrays = {"val_idx": val_idx, "test_idx": test_idx, "val_labels": labels[val_idx], "test_labels": labels[test_idx]}
        meta = {"held_out": held_out, "validation": validation, "train_events": train_events, "seed": args.seed, "checkpoints": CHECKPOINTS, "temperatures": {}, "task": args.task}
        eval_idx = {"val": val_idx, "test": test_idx}
        for name in models:
            if name == "dgtr":
                config = VerifierConfig(num_classes=len(label_map))
                model, history = fit_verifier(cache, train_idx, val_idx, labels, emb_t, config, checkpoints=CHECKPOINTS, epochs=args.epochs, seed=args.seed)
                logits = {k: predict_logits(model, cache, v, emb_t, CHECKPOINTS) for k, v in eval_idx.items()}
                meta["dgtr_history"], meta["dgtr_config"] = history, config_dict(config)
                torch.save(model.state_dict(), out / f"{held_out}_dgtr.pt")
            else:
                fn = getattr(baselines, name)
                logits = fn(threads, train_idx, eval_idx, labels, emb_np, CHECKPOINTS, len(label_map))
            temperature = fit_temperature(logits["val"], labels[val_idx])
            meta["temperatures"][name] = temperature
            arrays[f"{name}_val"], arrays[f"{name}_test"] = logits["val"], logits["test"]
            acc = (logits["test"].argmax(-1) == labels[test_idx][:, None]).mean(0)
            print(f"  {name:10s} T={temperature:.2f} test acc by checkpoint={np.round(acc, 3).tolist()}", flush=True)
        np.savez_compressed(out / f"{held_out}.npz", **arrays)
        (out / f"{held_out}.json").write_text(json.dumps(meta, indent=2))
        print(f"  fold done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
