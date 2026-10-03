"""
Look up BERTweet embeddings for graph nodes by post_id.

Person 1's graph JSON files (data/graphs/graph_XXXXXX.json) store nodes as
a list; each node has a string "post_id". The embedding matrix
(data/nlp/embeddings/full/) has its own row order, given by ids.csv.
These two orders are unrelated, so rows are always looked up by post_id:

    row i of the returned matrix  <->  node i of the graph's "nodes" list

Usage:
    from nlp.embedding_node_helper import (
        load_embedding_store, get_graph_node_embeddings
    )
    store = load_embedding_store()                    # memory-mapped
    node_ids, x = get_graph_node_embeddings(
        "data/graphs/graph_000001.json", store
    )
    x.shape   # (num_nodes, 768); x[i] belongs to node_ids[i]
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nlp.bertweet_embeddings import load_full_embeddings
from nlp.config import FULL_EMBEDDINGS_DIR


# ============================================================
# ERRORS
# ============================================================

class EmbeddingLookupError(Exception):
    """Base class for all errors raised by this module."""


class InvalidPostIdError(EmbeddingLookupError, ValueError):
    """A post_id is not a non-empty string (e.g. int, float, None, '')."""


class DuplicatePostIdError(EmbeddingLookupError, ValueError):
    """The same post_id appears more than once where it must be unique."""


class MissingPostIdError(EmbeddingLookupError, KeyError):
    """A requested post_id has no embedding."""


class MappingMismatchError(EmbeddingLookupError, ValueError):
    """The embedding matrix and the ID mapping do not line up."""


# ============================================================
# EMBEDDING STORE
# ============================================================

@dataclass(frozen=True)
class EmbeddingStore:
    """
    embeddings : (num_posts, dim) array, possibly memory-mapped (read-only)
    index      : post_id (str) -> row in `embeddings`
    metadata   : contents of metadata.json (empty for synthetic stores)
    """

    embeddings: np.ndarray
    index: dict
    metadata: dict = field(default_factory=dict)

    @property
    def dim(self):
        return self.embeddings.shape[1]

    def __len__(self):
        return self.embeddings.shape[0]

    def __contains__(self, post_id):
        return post_id in self.index


def _check_post_id(post_id, position, source):

    if not isinstance(post_id, str):
        raise InvalidPostIdError(
            f"{source}[{position}] is {type(post_id).__name__} {post_id!r}; "
            "post_ids must be strings. Read IDs as text (e.g. "
            "pd.read_csv(..., dtype=str)) - converting through float "
            "corrupts 18-digit Twitter IDs."
        )

    if post_id == "" or post_id != post_id.strip():
        raise InvalidPostIdError(
            f"{source}[{position}] is {post_id!r}; post_ids must be "
            "non-empty and have no surrounding whitespace."
        )


def _duplicates(ids):

    seen = set()
    duplicates = []

    for post_id in ids:
        if post_id in seen:
            duplicates.append(post_id)
        seen.add(post_id)

    return duplicates


def build_embedding_store(embeddings, post_ids, metadata=None):
    """
    Build a store from an embedding matrix and the post_id of each row.

    Raises MappingMismatchError if the matrix is not 2-D or its row count
    differs from len(post_ids); InvalidPostIdError / DuplicatePostIdError
    if the IDs are not unique non-empty strings.
    """

    post_ids = list(post_ids)

    if embeddings.ndim != 2:
        raise MappingMismatchError(
            f"embeddings must be 2-D, got shape {embeddings.shape}"
        )

    if embeddings.shape[0] != len(post_ids):
        raise MappingMismatchError(
            f"embedding matrix has {embeddings.shape[0]} rows but the ID "
            f"mapping has {len(post_ids)} post_ids; they must be equal"
        )

    for position, post_id in enumerate(post_ids):
        _check_post_id(post_id, position, "ID mapping")

    duplicates = _duplicates(post_ids)

    if duplicates:
        raise DuplicatePostIdError(
            f"ID mapping contains {len(duplicates)} duplicate post_ids, "
            f"e.g. {duplicates[:5]}; each post must map to exactly one row"
        )

    index = {post_id: row for row, post_id in enumerate(post_ids)}

    return EmbeddingStore(embeddings, index, dict(metadata or {}))


def load_embedding_store(output_dir=FULL_EMBEDDINGS_DIR, mmap=True):
    """
    Load the full BERTweet embeddings (only if marked complete) as a store.

    mmap=True (default) keeps the 321 MB matrix on disk; only the rows
    that are requested are read into memory.
    """

    embeddings, ids, metadata = load_full_embeddings(output_dir, mmap=mmap)

    return build_embedding_store(embeddings, ids["post_id"].tolist(), metadata)


# ============================================================
# LOOKUP
# ============================================================

def get_node_embeddings(node_post_ids, store):
    """
    Return a float32 array of shape (len(node_post_ids), store.dim) where
    row i is the embedding of node_post_ids[i]. Input order is preserved
    exactly; an empty sequence gives shape (0, dim).

    Raises InvalidPostIdError, DuplicatePostIdError or MissingPostIdError.
    Nothing is ever skipped or replaced by another post's vector.
    """

    if isinstance(node_post_ids, (str, bytes)):
        raise InvalidPostIdError(
            "node_post_ids must be a list of post_id strings, not a single "
            "string; use [post_id] for one node"
        )

    node_post_ids = list(node_post_ids)

    for position, post_id in enumerate(node_post_ids):
        _check_post_id(post_id, position, "node_post_ids")

    duplicates = _duplicates(node_post_ids)

    if duplicates:
        raise DuplicatePostIdError(
            f"node_post_ids contains {len(duplicates)} duplicate post_ids, "
            f"e.g. {duplicates[:5]}; each graph node must be a distinct post"
        )

    missing = [post_id for post_id in node_post_ids if post_id not in store.index]

    if missing:
        raise MissingPostIdError(
            f"{len(missing)} of {len(node_post_ids)} post_ids have no "
            f"embedding, e.g. {missing[:5]}"
        )

    rows = np.fromiter(
        (store.index[post_id] for post_id in node_post_ids),
        dtype=np.int64,
        count=len(node_post_ids),
    )

    # Copy only the requested rows into a plain in-memory array
    result = np.empty((len(rows), store.dim), dtype=np.float32)

    if len(rows):
        result[:] = store.embeddings[rows]

    return result


# ============================================================
# GRAPH JSON HELPERS
# ============================================================

def load_graph(path):
    """Read one of Person 1's graph JSON files (read-only)."""

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def graph_node_post_ids(graph):
    """post_id of every node, in the order of the graph's "nodes" list."""

    nodes = graph.get("nodes")

    if not isinstance(nodes, list):
        raise InvalidPostIdError("graph has no 'nodes' list")

    post_ids = []

    for position, node in enumerate(nodes):

        if not isinstance(node, dict) or "post_id" not in node:
            raise InvalidPostIdError(f"graph node {position} has no 'post_id'")

        post_ids.append(node["post_id"])

    return post_ids


def get_graph_node_embeddings(graph, store):
    """
    graph: a graph dict or a path to a graph JSON file.

    Returns (node_post_ids, features) where features[i] is the embedding
    of node_post_ids[i], i.e. of graph["nodes"][i].
    """

    if isinstance(graph, (str, Path)):
        graph = load_graph(graph)

    node_post_ids = graph_node_post_ids(graph)

    return node_post_ids, get_node_embeddings(node_post_ids, store)
