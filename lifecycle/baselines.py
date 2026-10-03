"""Phase 3 comparison verifiers, all evaluated on the same causal prefixes.

* ``source_lr``  – logistic regression on the source tweet's BERTweet vector only
                   (constant over checkpoints; the strongest "no propagation" control).
* ``prefix_lr``  – logistic regression on [mean BERTweet over the observed prefix,
                   source vector, log size, log elapsed]; trained on all training prefixes.
* ``gat_full``   – the project's existing models.gnn.GraphClassifier (GAT, BERTweet +
                   5 structural features) trained on full training graphs with class
                   weights and train-only feature normalisation, then run on prefixes.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.gnn import GraphClassifier, ModelConfig, set_seed
from .verifier import class_weights


def _source_vec(thread, embeddings):
    i = int(np.argmax(thread.is_source)) if thread.is_source.any() else 0
    return embeddings[thread.emb_rows[i]]


def _prefix_vec(thread, m, embeddings):
    mean = embeddings[thread.emb_rows[:m]].mean(0)
    extra = np.array([np.log1p(m), np.log1p(thread.rel_time[m - 1] / 60.0)], dtype=np.float32)
    return np.concatenate([mean, _source_vec(thread, embeddings), extra])


def fit_lr(X, y, C=0.05, seed=7):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed).fit(scaler.transform(X), y)
    return scaler, clf


def source_lr(threads, train_idx, eval_idx, labels, embeddings, checkpoints, num_classes):
    X = np.stack([_source_vec(threads[i], embeddings) for i in train_idx])
    scaler, clf = fit_lr(X, labels[train_idx])
    out = {}
    for name, idx in eval_idx.items():
        z = clf.decision_function(scaler.transform(np.stack([_source_vec(threads[i], embeddings) for i in idx])))
        out[name] = np.repeat(_pad(z, clf.classes_, num_classes)[:, None, :], len(checkpoints), axis=1)
    return out


def prefix_lr(threads, train_idx, eval_idx, labels, embeddings, checkpoints, num_classes):
    X, y = [], []
    for i in train_idx:
        for m in sorted(set(threads[i].checkpoint_lengths(checkpoints))):
            X.append(_prefix_vec(threads[i], m, embeddings)); y.append(labels[i])
    scaler, clf = fit_lr(np.stack(X), np.array(y))
    out = {}
    for name, idx in eval_idx.items():
        rows = [_prefix_vec(threads[i], m, embeddings) for i in idx for m in threads[i].checkpoint_lengths(checkpoints)]
        z = _pad(clf.decision_function(scaler.transform(np.stack(rows))), clf.classes_, num_classes)
        out[name] = z.reshape(len(idx), len(checkpoints), num_classes)
    return out


def _pad(z, classes, num_classes):
    full = np.full((z.shape[0], num_classes), -1e4, dtype=np.float32)
    full[:, classes] = z
    return full


def _structural(thread, m):
    """Same five features as models.data._features, computed on the prefix only."""
    parent = thread.parent[:m]
    valid = (parent >= 0) & (parent < m)
    depth = np.zeros(m, dtype=np.float32)
    for i in range(m):
        p, d, seen = parent[i], 0, set()
        while 0 <= p < m and p not in seen:
            seen.add(p); d += 1; p = parent[p]
        depth[i] = d
    outdeg = np.bincount(parent[valid], minlength=m).astype(np.float32)
    return np.stack([thread.is_source[:m].astype(np.float32), thread.rel_time[:m], depth, valid.astype(np.float32), outdeg], axis=1)


def _graph(thread, m, embeddings, label):
    parent = thread.parent[:m]
    child = np.nonzero((parent >= 0) & (parent < m))[0]
    edge_index = torch.tensor(np.stack([parent[child], child]), dtype=torch.long) if len(child) else torch.empty((2, 0), dtype=torch.long)
    x = np.concatenate([embeddings[thread.emb_rows[:m]], _structural(thread, m)], axis=1)
    return Data(x=torch.from_numpy(x).float(), edge_index=edge_index, y=torch.tensor([label]))


def gat_full(threads, train_idx, eval_idx, labels, embeddings, checkpoints, num_classes, epochs=10, seed=7):
    set_seed(seed)
    train = [_graph(threads[i], threads[i].n, embeddings, labels[i]) for i in train_idx]
    feats = torch.cat([d.x for d in train])
    mean, std = feats.mean(0), feats.std(0).clamp_min(1e-6)
    for d in train:
        d.x = (d.x - mean) / std
    model = GraphClassifier(ModelConfig(model_type="gat", input_dim=feats.size(1), hidden_dim=64, num_classes=num_classes, dropout=0.3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = class_weights(labels[train_idx], num_classes)
    loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for batch in loader:
            loss = torch.nn.functional.cross_entropy(model(batch)["logits"], batch.y, weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for name, idx in eval_idx.items():
            graphs = []
            for i in idx:
                for m in threads[i].checkpoint_lengths(checkpoints):
                    g = _graph(threads[i], m, embeddings, labels[i]); g.x = (g.x - mean) / std; graphs.append(g)
            logits = torch.cat([model(b)["logits"] for b in DataLoader(graphs, batch_size=256)]).numpy()
            out[name] = logits.reshape(len(idx), len(checkpoints), num_classes)
    return out
