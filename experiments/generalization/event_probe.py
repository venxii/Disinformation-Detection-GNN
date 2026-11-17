"""How much event identity survives in the input representation (masking diagnostic).

Linear probe (standardised logistic regression, 5-fold stratified CV over threads) predicting
the thread's event from (a) the source-tweet embedding and (b) the mean embedding of the full
thread, for each embedding directory. Lower = fewer event cues. Chance macro-F1 ≈ 1/9.

    .venv/bin/python -m experiments.generalization.event_probe
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lifecycle.data import load_threads

DIRS = {"original": "data/nlp/embeddings/full", "masked": "data/nlp/embeddings/masked"}


def main():
    out = {}
    for name, d in DIRS.items():
        threads, emb = load_threads(embedding_dir=d)
        y = np.array([t.event for t in threads])
        src = np.stack([emb[t.emb_rows[int(np.argmax(t.is_source)) if t.is_source.any() else 0]] for t in threads])
        mean = np.stack([emb[t.emb_rows].mean(0) for t in threads])
        out[name] = {}
        for view, X in (("source", src), ("thread_mean", mean)):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=2)
            out[name][view] = {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}
            print(f"{name:9s} {view:12s} event acc={out[name][view]['acc']:.3f} macro-F1={out[name][view]['macro_f1']:.3f}", flush=True)
    Path("artifacts/generalization/event_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
