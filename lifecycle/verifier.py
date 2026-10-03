"""Phase 3: DGTR-style temporal verifier (after Wei et al., 2023, "DGTR: Dynamic
Graph Transformer for Rumor Detection").

What is kept from DGTR: a shared *structural* encoder applied to a sequence of
nested propagation sub-graphs, followed by a *temporal* transformer over the
resulting snapshot embeddings. What differs (deliberately):

* The sub-graph sequence for an observed prefix is defined by fixed absolute
  deadlines after the source post (5m, 15m, 1h, 4h, 24h, everything observed),
  never by checkpoint index or fraction. A fraction-of-final-size index would
  leak the eventual thread size (which correlates with the label) into the input.
* The temporal transformer is causal and the classifier reads the last token.
* The structural encoder is a 2-layer GAT over frozen BERTweet node vectors plus
  prefix-local features, not DGTR's full graph transformer.

Leakage contract: ``predict(thread, m)`` reads only rows ``< m`` of the thread
(tested in tests/test_verifier.py by perturbing rows >= m).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

DEADLINES = (300.0, 900.0, 3600.0, 4 * 3600.0, 24 * 3600.0, math.inf)
NODE_FEATS = 4   # log-minutes since source, depth, is_source, log out-degree within prefix
GRAPH_FEATS = 3  # log size, log elapsed minutes, log max depth


@dataclass
class VerifierConfig:
    num_classes: int = 4
    embed_dim: int = 768
    hidden: int = 128
    heads: int = 4
    dropout: float = 0.3
    temporal_layers: int = 1


def _causal_depth(parent: np.ndarray) -> np.ndarray:
    """Depth using only parents that arrive earlier; otherwise the node is treated as a root.

    This makes depth prefix-invariant, so a node's features never change when
    later posts arrive (except out-degree, which is recomputed per prefix)."""
    depth = np.zeros(len(parent), dtype=np.float32)
    for i, p in enumerate(parent):
        if 0 <= p < i:
            depth[i] = depth[p] + 1
    return depth


class ThreadCache:
    """Per-thread arrays the verifier needs; built once per run."""

    def __init__(self, threads):
        self.threads = threads
        self.depth = [_causal_depth(t.parent) for t in threads]
        self.valid_parent = [np.where((t.parent >= 0) & (t.parent < np.arange(t.n)), t.parent, -1) for t in threads]

    def window_lengths(self, ti: int, m: int) -> list[int]:
        rel = self.threads[ti].rel_time[:m]
        return [max(1, min(m, int(np.searchsorted(rel, d, side="right")))) for d in DEADLINES]


def build_batch(cache: ThreadCache, samples, embeddings: torch.Tensor):
    """samples: list of (thread_index, prefix_length). Returns tensors for the model.

    Each sample becomes len(DEADLINES) tokens; identical sub-graphs are encoded once."""
    unique: dict[tuple[int, int], int] = {}
    token_index = []
    for ti, m in samples:
        row = []
        for length in cache.window_lengths(ti, m):
            key = (ti, length)
            if key not in unique:
                unique[key] = len(unique)
            row.append(unique[key])
        token_index.append(row)
    emb_rows, node_feats, edges, batch, graph_feats, source_pos = [], [], [], [], [], []
    offset = 0
    for g, (ti, length) in enumerate(unique):
        t = cache.threads[ti]
        parent = cache.valid_parent[ti][:length]
        children = parent >= 0
        outdeg = np.bincount(parent[children], minlength=length).astype(np.float32)
        depth = cache.depth[ti][:length]
        rel_min = t.rel_time[:length] / 60.0
        emb_rows.append(t.emb_rows[:length])
        node_feats.append(np.stack([np.log1p(rel_min), np.log1p(depth), t.is_source[:length].astype(np.float32), np.log1p(outdeg)], axis=1))
        child_idx = np.nonzero(children)[0]
        if len(child_idx):
            src, dst = parent[child_idx] + offset, child_idx + offset
            edges.append(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]))
        batch.append(np.full(length, g))
        graph_feats.append([math.log1p(length), math.log1p(float(rel_min.max())), math.log1p(float(depth.max()))])
        src_local = int(np.argmax(t.is_source[:length])) if t.is_source[:length].any() else 0
        source_pos.append(offset + src_local)
        offset += length
    rows = torch.from_numpy(np.concatenate(emb_rows))
    return {
        "x": embeddings[rows],
        "node_feats": torch.from_numpy(np.concatenate(node_feats)).float(),
        "edge_index": torch.from_numpy(np.concatenate(edges, axis=1)).long() if edges else torch.empty((2, 0), dtype=torch.long),
        "batch": torch.from_numpy(np.concatenate(batch)).long(),
        "graph_feats": torch.tensor(graph_feats, dtype=torch.float32),
        "source_pos": torch.tensor(source_pos, dtype=torch.long),
        "token_index": torch.tensor(token_index, dtype=torch.long),
    }


class TemporalVerifier(nn.Module):
    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = c = config
        self.inp = nn.Sequential(nn.Dropout(c.dropout), nn.Linear(c.embed_dim + NODE_FEATS, c.hidden), nn.LayerNorm(c.hidden), nn.GELU())
        self.gat = nn.ModuleList([GATConv(c.hidden, c.hidden, heads=c.heads, concat=False, add_self_loops=True) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(c.hidden) for _ in range(2)])
        self.pool = nn.Sequential(nn.Linear(3 * c.hidden + GRAPH_FEATS, c.hidden), nn.LayerNorm(c.hidden), nn.GELU())
        self.window_embedding = nn.Parameter(torch.zeros(len(DEADLINES), c.hidden))
        layer = nn.TransformerEncoderLayer(c.hidden, nhead=c.heads, dim_feedforward=2 * c.hidden, dropout=c.dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=c.temporal_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(c.hidden), nn.Dropout(c.dropout), nn.Linear(c.hidden, c.num_classes))
        mask = torch.triu(torch.full((len(DEADLINES), len(DEADLINES)), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, b):
        h = self.inp(torch.cat([b["x"], b["node_feats"]], dim=1))
        for conv, norm in zip(self.gat, self.norms):
            h = norm(h + torch.nn.functional.gelu(conv(h, b["edge_index"])))
        pooled = torch.cat([h[b["source_pos"]], global_mean_pool(h, b["batch"]), global_max_pool(h, b["batch"]), b["graph_feats"]], dim=1)
        z = self.pool(pooled)[b["token_index"]] + self.window_embedding  # [B, T, H]
        z = self.temporal(z, mask=self.causal_mask)
        return self.head(z[:, -1])


def class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0) / num_classes
    return torch.tensor(weights, dtype=torch.float32)


def fit_verifier(cache, train_idx, val_idx, labels, embeddings, config: VerifierConfig, *, checkpoints, epochs=12, batch_threads=64, lr=1e-3, seed=7, log=print):
    """Train on every checkpoint prefix of training threads; select epoch by validation macro-F1 averaged over checkpoints."""
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalVerifier(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    weights = class_weights(labels[train_idx], config.num_classes)
    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for start in range(0, len(order), batch_threads):
            chunk = order[start:start + batch_threads]
            samples = [(ti, m) for ti in chunk for m in sorted(set(cache.threads[ti].checkpoint_lengths(checkpoints)))]
            b = build_batch(cache, samples, embeddings)
            y = torch.tensor([labels[ti] for ti, _ in samples])
            loss = nn.functional.cross_entropy(model(b), y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(samples)
        logits = predict_logits(model, cache, val_idx, embeddings, checkpoints)
        f1s = [f1_score(labels[val_idx], logits[:, k].argmax(1), average="macro") for k in range(len(checkpoints))]
        score = float(np.mean(f1s))
        history.append({"epoch": epoch, "train_loss": total, "val_macro_f1_mean": score})
        log(f"    epoch {epoch:2d} loss={total:.1f} val_mF1(mean over checkpoints)={score:.3f}")
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_logits(model, cache, idx, embeddings, checkpoints, batch_threads=128):
    """Return logits [len(idx), K, C] computed independently for every checkpoint prefix."""
    model.eval()
    out = np.zeros((len(idx), len(checkpoints), model.config.num_classes), dtype=np.float32)
    for start in range(0, len(idx), batch_threads):
        chunk = idx[start:start + batch_threads]
        samples = [(ti, m) for ti in chunk for m in cache.threads[ti].checkpoint_lengths(checkpoints)]
        logits = model(build_batch(cache, samples, embeddings)).numpy()
        out[start:start + len(chunk)] = logits.reshape(len(chunk), len(checkpoints), -1)
    return out


def config_dict(config: VerifierConfig):
    return asdict(config)
