"""Phase 4 (a): reply stance model (support / deny / query / comment).

Supervision: RumourEval-2019 subtask-A labels (Gorrell et al., 2019), which annotate
replies inside PHEME threads (5,026 PHEME posts overlap; 4,709 are replies). The
classifier is logistic regression on word (1-2) and char (2-5) TF-IDF of the reply
text plus down-weighted frozen BERTweet vectors [reply, source, reply*source]. On
held-out events this scored 0.37-0.45 macro-F1 vs 0.37-0.40 for embeddings alone;
"deny" F1 stays at ~0.12-0.21, which is the main limitation of the evidence side.
Per LOEO fold it is trained only on annotations from non-test events, so no stance
labels from the held-out event are used.

Data: data/external/rumoureval2019 (figshare 8845580, md5 85b37dc70106c2b9f584e63164fcbb92).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from .data import STANCES
from .evidence import normalise_text

RUMOUREVAL = Path("data/external/rumoureval2019/rumoureval-2019-training-data")


def load_stance_labels(root: Path = RUMOUREVAL) -> dict[str, int]:
    labels = {}
    for key in ("train-key.json", "dev-key.json"):
        labels.update(json.loads((root / key).read_text())["subtaskaenglish"])
    return {pid: STANCES.index(s) for pid, s in labels.items()}


def _source_row(thread):
    return thread.emb_rows[int(np.argmax(thread.is_source))] if thread.is_source.any() else thread.emb_rows[0]


def pair_features(reply_vecs: np.ndarray, source_vec: np.ndarray) -> np.ndarray:
    source = np.broadcast_to(source_vec, reply_vecs.shape)
    return np.concatenate([reply_vecs, source, reply_vecs * source], axis=1)


def _text(text: str) -> str:
    return normalise_text(re.sub(r"https?://\S+", " url ", text))


class StanceModel:
    def __init__(self, C: float = 1.0, embedding_weight: float = 0.05, seed: int = 7):
        self.C, self.embedding_weight, self.seed = C, embedding_weight, seed

    def _features(self, texts, pair_vecs):
        texts = [_text(t) for t in texts]
        dense = sparse.csr_matrix(self.scaler.transform(pair_vecs) * self.embedding_weight)
        return sparse.hstack([self.words.transform(texts), self.chars.transform(texts), dense]).tocsr()

    def fit(self, threads, embeddings, stance_labels, exclude_events=()):
        texts, X, y = [], [], []
        for t in threads:
            if t.event in exclude_events:
                continue
            src = embeddings[_source_row(t)]
            for i, pid in enumerate(t.post_ids):
                if not t.is_source[i] and pid in stance_labels:
                    texts.append(t.texts[i]); X.append(pair_features(embeddings[t.emb_rows[i]][None], src)[0]); y.append(stance_labels[pid])
        X, y = np.stack(X), np.array(y)
        clean = [_text(t) for t in texts]
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(clean)
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=30000).fit(clean)
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced", random_state=self.seed).fit(self._features(texts, X), y)
        self.n_train = len(y)
        return self

    def predict_proba(self, texts, reply_vecs: np.ndarray, source_vec: np.ndarray) -> np.ndarray:
        if len(reply_vecs) == 0:
            return np.zeros((0, len(STANCES)), dtype=np.float32)
        return self.clf.predict_proba(self._features(texts, pair_features(reply_vecs, source_vec))).astype(np.float32)

    def predict_thread(self, thread, embeddings) -> np.ndarray:
        """Stance probabilities for every post in chronological order (source rows are zeros)."""
        out = np.zeros((thread.n, len(STANCES)), dtype=np.float32)
        replies = np.nonzero(~thread.is_source)[0]
        out[replies] = self.predict_proba([thread.texts[i] for i in replies], embeddings[thread.emb_rows[replies]], embeddings[_source_row(thread)])
        return out

    def evaluate(self, threads, embeddings, stance_labels, events):
        y, p = [], []
        for t in threads:
            if t.event not in events:
                continue
            probs = self.predict_thread(t, embeddings)
            for i, pid in enumerate(t.post_ids):
                if not t.is_source[i] and pid in stance_labels:
                    y.append(stance_labels[pid]); p.append(int(probs[i].argmax()))
        if not y:
            return {"n": 0}
        report = classification_report(y, p, labels=range(len(STANCES)), target_names=STANCES, output_dict=True, zero_division=0)
        return {"n": len(y), "macro_f1": f1_score(y, p, labels=range(len(STANCES)), average="macro", zero_division=0),
                "per_class_f1": {s: report[s]["f1-score"] for s in STANCES}}
