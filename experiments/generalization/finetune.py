"""Step 3: fine-tune BERTweet end-to-end as a causal prefix classifier, under the same LOEO folds.

Input for thread t at checkpoint k: (source tweet, replies observed in the prefix, oldest first),
encoded as a BERTweet sentence pair truncated on the replies side only. Nothing after the prefix
is read, so the leakage guarantee is the same as lifecycle.data's (rows < n_k).

Training: each epoch draws one random checkpoint prefix per training thread. Loss = cross-entropy
with class-balanced weights (Cui et al. 2019, effective number, beta=0.999), renormalised to mean 1.
Selection: the epoch with best validation-event macro-F1 averaged over checkpoints (the DGTR rule);
the held-out event is only scored with the selected weights. Optional gradient-reversal event head
(step 4): an event classifier on the [CLS] vector whose gradient into the encoder is multiplied by -λ.

Output matches experiments/run_verifiers.py (model name ``ft_bertweet``), so compare.py and the
lifecycle runners read it unchanged:
    <output>/seed<S>/<task>/verifiers/<event>.{npz,json}

    .venv/bin/python -m experiments.generalization.finetune --task 4class --seed 7 --text masked
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lifecycle.data import CHECKPOINTS, TASKS, load_threads, loeo_folds, task_threads

from .masking import FULL, REVISION, load_bertweet, normalize_for_bertweet, post_texts

NAME = "ft_bertweet"


def class_balanced_weights(y, C, beta=0.999):
    n = np.bincount(y, minlength=C).astype(np.float64)
    w = (1 - beta) / (1 - beta ** np.maximum(n, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class PrefixClassifier(nn.Module):
    def __init__(self, encoder, C, n_events=0):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(768, C)
        self.event_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, n_events)) if n_events else None

    def forward(self, enc, lam=0.0):
        h = self.encoder(**enc).last_hidden_state[:, 0]
        logits = self.head(self.drop(h))
        if self.event_head is None:
            return logits, None
        return logits, self.event_head(GradReverse.apply(h, lam))


def thread_texts(text_mode):
    """post_id -> BERTweet-normalised text (original or event-masked)."""
    if text_mode == "masked":
        df = pd.read_csv("data/nlp/embeddings/masked/masked_texts.csv", dtype=str, keep_default_na=False)
        return dict(zip(df.post_id, df.text_masked))
    ids, raw = post_texts()
    return {i: normalize_for_bertweet(t) for i, t in zip(ids, raw)}


def prefix_pair(t, m, text):
    src = int(np.argmax(t.is_source[:m])) if t.is_source[:m].any() else 0
    replies = [text[t.post_ids[j]] for j in range(m) if j != src]
    return text[t.post_ids[src]], " </s> ".join(replies)


def encode(tok, pairs, device, max_length):
    a, b = zip(*pairs)
    b = [x if x else None for x in b]
    if all(x is None for x in b):
        enc = tok(list(a), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    else:  # tokenizer needs a string for every pair; an empty second segment is just "</s></s>"
        enc = tok(list(a), [x or "" for x in b], padding=True, truncation="only_second", max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict(model, tok, threads, idx, text, device, max_length, batch=64):
    """Logits [len(idx), K, C]; identical prefix inputs are encoded once."""
    model.eval()
    keys, uniq = [], {}
    for i in idx:
        for m in threads[i].checkpoint_lengths(CHECKPOINTS):
            p = prefix_pair(threads[i], m, text)
            keys.append(uniq.setdefault(p, len(uniq)))
    pairs = list(uniq)
    order = sorted(range(len(pairs)), key=lambda j: len(pairs[j][0]) + len(pairs[j][1]))
    out = np.zeros((len(pairs), model.head.out_features), dtype=np.float32)
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            logits, _ = model(encode(tok, [pairs[j] for j in rows], device, max_length))
        out[rows] = logits.float().cpu().numpy()
    return out[np.array(keys)].reshape(len(idx), len(CHECKPOINTS), -1)


def fit_fold(threads, labels, train_idx, val_idx, test_idx, text, C, args, device, events_of=None):
    from sklearn.metrics import f1_score
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tok, encoder = load_bertweet(device)
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    train_events = sorted({events_of[i] for i in train_idx}) if args.adv_lambda > 0 else []
    ev_id = {e: j for j, e in enumerate(train_events)}
    model = PrefixClassifier(encoder, C, n_events=len(train_events)).to(device)
    steps = args.epochs * int(np.ceil(len(train_idx) / args.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))  # linear warm-up over 10% of steps, then linear decay to 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    w = class_balanced_weights(labels[train_idx], C).to(device)
    best, best_state, history, step = -1.0, None, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        order = rng.permutation(train_idx)
        for s in range(0, len(order), args.batch):
            chunk = order[s:s + args.batch]
            pairs = []
            for i in chunk:
                lens = threads[i].checkpoint_lengths(CHECKPOINTS)
                pairs.append(prefix_pair(threads[i], lens[rng.integers(len(lens))], text))
            y = torch.tensor(labels[chunk], device=device)
            # DANN schedule: λ ramps 0 → adv_lambda over training
            lam = args.adv_lambda * (2 / (1 + np.exp(-10 * step / steps)) - 1)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
                logits, ev_logits = model(encode(tok, pairs, device, args.max_length), lam)
                loss = nn.functional.cross_entropy(logits.float(), y, weight=w)
                if ev_logits is not None:
                    ey = torch.tensor([ev_id[events_of[i]] for i in chunk], device=device)
                    loss = loss + nn.functional.cross_entropy(ev_logits.float(), ey)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            total += loss.item() * len(chunk)
        val = predict(model, tok, threads, val_idx, text, device, args.max_length)
        score = float(np.mean([f1_score(labels[val_idx], val[:, k].argmax(1), average="macro") for k in range(len(CHECKPOINTS))]))
        history.append({"epoch": epoch, "train_loss": total, "val_macro_f1_mean": score, "seconds": round(time.time() - t0, 1)})
        print(f"    epoch {epoch} loss={total:.1f} val_mF1={score:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_val = score, val
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = predict(model, tok, threads, test_idx, text, device, args.max_length)
    return {"val": best_val, "test": test}, history


def main():
    from experiments.run_verifiers import fit_temperature
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="4class")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--text", choices=("original", "masked"), default="original")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--adv-lambda", type=float, default=0.0, help=">0 adds the gradient-reversal event head")
    parser.add_argument("--only-event", default=None)
    parser.add_argument("--max-train", type=int, default=0, help="debug: subsample training threads")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tag = "ft" + ("_masked" if args.text == "masked" else "") + (f"_adv{args.adv_lambda:g}" if args.adv_lambda else "")
    out = Path(args.output or f"artifacts/generalization/{tag}") / f"seed{args.seed}" / args.task / "verifiers"
    out.mkdir(parents=True, exist_ok=True)
    all_threads, _ = load_threads(embedding_dir=FULL)
    threads = task_threads(all_threads, args.task)
    label_map = TASKS[args.task]; C = len(set(label_map.values()))
    labels = np.array([label_map[t.label] for t in threads])
    events_of = [t.event for t in threads]
    text = thread_texts(args.text)
    by_event = {}
    for i, t in enumerate(threads):
        by_event.setdefault(t.event, []).append(i)
    for held_out, validation, train_events in loeo_folds(threads):
        if (args.only_event and held_out != args.only_event) or (out / f"{held_out}.json").exists():
            continue
        start = time.time()
        train_idx = np.array(sorted(i for e in train_events for i in by_event[e]))
        if args.max_train:
            train_idx = np.sort(np.random.default_rng(0).choice(train_idx, min(args.max_train, len(train_idx)), replace=False))
        val_idx, test_idx = np.array(by_event[validation]), np.array(by_event[held_out])
        print(f"[{tag} {args.task} seed{args.seed}] held_out={held_out} validation={validation} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        logits, history = fit_fold(threads, labels, train_idx, val_idx, test_idx, text, C, args, device, events_of)
        T = fit_temperature(logits["val"], labels[val_idx])
        acc = (logits["test"].argmax(-1) == labels[test_idx][:, None]).mean(0)
        print(f"  {NAME} T={T:.2f} test acc by checkpoint={np.round(acc, 3).tolist()}  fold {time.time() - start:.0f}s", flush=True)
        np.savez_compressed(out / f"{held_out}.npz", val_idx=val_idx, test_idx=test_idx, val_labels=labels[val_idx], test_labels=labels[test_idx],
                            **{f"{NAME}_val": logits["val"], f"{NAME}_test": logits["test"]})
        (out / f"{held_out}.json").write_text(json.dumps({
            "held_out": held_out, "validation": validation, "train_events": train_events, "seed": args.seed, "checkpoints": CHECKPOINTS,
            "temperatures": {NAME: T}, "task": args.task, "text": args.text, "history": {NAME: history},
            "config": {k: v for k, v in vars(args).items() if k not in ("output", "only_event")}, "model_revision": REVISION}, indent=2))


if __name__ == "__main__":
    main()
