"""Memory-mapped loader for Person 2's validated BERTweet artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class EmbeddingStore:
    def __init__(self, embeddings: np.ndarray, index: dict[str, int], dimension: int):
        self.embeddings = embeddings
        self.index = index
        self.dimension = dimension

    def lookup(self, post_ids):
        post_ids = [str(post_id) for post_id in post_ids]
        missing = [post_id for post_id in post_ids if post_id not in self.index]
        if missing:
            raise KeyError(f"Missing embeddings for {len(missing)} post IDs; first missing ID: {missing[0]}")
        rows = np.asarray([self.index[post_id] for post_id in post_ids], dtype=np.int64)
        return np.asarray(self.embeddings[rows], dtype=np.float32)


def load_embedding_store(directory: str | Path, *, mmap=True) -> EmbeddingStore:
    directory = Path(directory)
    matrix_path = directory / "embeddings.npy"
    ids_path = directory / "ids.csv"
    if not matrix_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Expected {matrix_path} and {ids_path}; Person 2's large artifacts are not present")
    embeddings = np.load(matrix_path, mmap_mode="r" if mmap else None)
    with ids_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    post_ids = [row["post_id"] for row in rows]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(post_ids):
        raise ValueError("embedding matrix and ids.csv row counts do not match")
    if len(set(post_ids)) != len(post_ids):
        raise ValueError("ids.csv contains duplicate post IDs")
    return EmbeddingStore(embeddings, {post_id: index for index, post_id in enumerate(post_ids)}, embeddings.shape[1])
