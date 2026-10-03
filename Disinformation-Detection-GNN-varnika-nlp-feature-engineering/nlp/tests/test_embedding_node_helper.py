"""
Tests for nlp/embedding_node_helper.py using small synthetic matrices.
The real 321 MB embedding file is never loaded here.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nlp.bertweet_embeddings import save_embeddings
from nlp.embedding_node_helper import (
    DuplicatePostIdError,
    EmbeddingLookupError,
    InvalidPostIdError,
    MappingMismatchError,
    MissingPostIdError,
    build_embedding_store,
    get_graph_node_embeddings,
    get_node_embeddings,
    graph_node_post_ids,
    load_embedding_store,
)


DIM = 768

# Long, numeric-looking Twitter IDs (would lose digits if parsed as float)
POST_IDS = [
    "552783238415265792",
    "552787794503143424",
    "552804911797530625",
    "498235547685756928",
    "498243332204949504",
]


def synthetic_matrix(n, dim=DIM):
    """Row r is filled with the value r, so every row is recognisable."""
    return np.repeat(np.arange(n, dtype=np.float32)[:, None], dim, axis=1)


@pytest.fixture
def store():
    return build_embedding_store(synthetic_matrix(len(POST_IDS)), POST_IDS)


def row_values(features):
    """Which stored row each returned vector came from."""
    return features[:, 0].astype(int).tolist()


# ============================================================
# LOOKUP AND ORDER
# ============================================================

def test_lookup_by_post_id(store):
    features = get_node_embeddings(["552804911797530625"], store)
    assert row_values(features) == [2]


def test_order_follows_graph_not_embedding_file(store):
    graph_order = [POST_IDS[3], POST_IDS[0], POST_IDS[4], POST_IDS[1]]
    features = get_node_embeddings(graph_order, store)
    assert row_values(features) == [3, 0, 4, 1]


def test_multiple_nodes_all_rows_correct(store):
    features = get_node_embeddings(POST_IDS[::-1], store)
    assert row_values(features) == [4, 3, 2, 1, 0]
    for i in range(len(POST_IDS)):
        assert np.all(features[i] == features[i, 0])


def test_single_node_graph(store):
    features = get_node_embeddings([POST_IDS[1]], store)
    assert features.shape == (1, DIM)
    assert row_values(features) == [1]


def test_empty_graph_returns_empty_matrix(store):
    features = get_node_embeddings([], store)
    assert features.shape == (0, DIM)
    assert features.dtype == np.float32


def test_shape_and_dtype(store):
    features = get_node_embeddings(POST_IDS[:3], store)
    assert features.shape == (3, DIM)
    assert features.dtype == np.float32
    assert type(features) is np.ndarray


def test_same_input_same_output(store):
    order = [POST_IDS[2], POST_IDS[0]]
    assert np.array_equal(
        get_node_embeddings(order, store), get_node_embeddings(order, store)
    )


def test_accepts_tuples_and_generators(store):
    from_tuple = get_node_embeddings(tuple(POST_IDS[:2]), store)
    from_gen = get_node_embeddings((p for p in POST_IDS[:2]), store)
    assert np.array_equal(from_tuple, from_gen)


def test_returned_matrix_is_a_copy(store):
    features = get_node_embeddings([POST_IDS[0]], store)
    features[:] = 999
    assert store.embeddings[0, 0] == 0


# ============================================================
# STRING IDS
# ============================================================

def test_long_numeric_ids_stay_strings(store):
    assert all(isinstance(k, str) for k in store.index)
    assert "552783238415265792" in store


def test_integer_id_is_rejected(store):
    with pytest.raises(InvalidPostIdError, match="must be strings"):
        get_node_embeddings([552783238415265792], store)


def test_float_id_is_rejected(store):
    with pytest.raises(InvalidPostIdError, match="float"):
        get_node_embeddings([5.527832384152658e17], store)


@pytest.mark.parametrize("bad", ["", " 552783238415265792", None])
def test_empty_padded_or_none_id_is_rejected(store, bad):
    with pytest.raises(InvalidPostIdError):
        get_node_embeddings([POST_IDS[0], bad], store)


def test_single_string_instead_of_list_is_rejected(store):
    with pytest.raises(InvalidPostIdError, match="not a single string"):
        get_node_embeddings(POST_IDS[0], store)


def test_float_rounded_id_is_reported_missing(store):
    rounded = "552783238415265800"
    with pytest.raises(MissingPostIdError, match=rounded):
        get_node_embeddings([rounded], store)


# ============================================================
# ERRORS
# ============================================================

def test_missing_post_id(store):
    with pytest.raises(MissingPostIdError, match="1 of 2"):
        get_node_embeddings([POST_IDS[0], "999999999999999999"], store)


def test_missing_error_is_a_lookup_error_and_key_error(store):
    with pytest.raises(KeyError):
        get_node_embeddings(["123"], store)
    with pytest.raises(EmbeddingLookupError):
        get_node_embeddings(["123"], store)


def test_duplicate_ids_in_mapping():
    with pytest.raises(DuplicatePostIdError, match="ID mapping"):
        build_embedding_store(synthetic_matrix(3), [POST_IDS[0], POST_IDS[1], POST_IDS[0]])


def test_duplicate_requested_node_ids(store):
    with pytest.raises(DuplicatePostIdError, match="node_post_ids"):
        get_node_embeddings([POST_IDS[0], POST_IDS[1], POST_IDS[0]], store)


def test_row_count_mismatch():
    with pytest.raises(MappingMismatchError, match="4 rows"):
        build_embedding_store(synthetic_matrix(4), POST_IDS)


def test_non_2d_matrix_is_rejected():
    with pytest.raises(MappingMismatchError, match="2-D"):
        build_embedding_store(np.zeros(5, np.float32), POST_IDS)


def test_invalid_id_in_mapping():
    with pytest.raises(InvalidPostIdError, match="ID mapping"):
        build_embedding_store(synthetic_matrix(2), [POST_IDS[0], 498235547685756928])


# ============================================================
# MEMORY-MAPPED MATRICES
# ============================================================

def test_memory_mapped_matrix(tmp_path):
    path = tmp_path / "embeddings.npy"
    np.save(path, synthetic_matrix(len(POST_IDS)))
    mapped = np.load(path, mmap_mode="r")

    store = build_embedding_store(mapped, POST_IDS)
    features = get_node_embeddings([POST_IDS[4], POST_IDS[2]], store)

    assert isinstance(store.embeddings, np.memmap)
    assert type(features) is np.ndarray
    assert row_values(features) == [4, 2]


def test_load_embedding_store_from_completed_output(tmp_path):
    subset = pd.DataFrame({"post_id": POST_IDS, "root_id": POST_IDS})
    metadata = {"status": "complete", "num_posts": len(POST_IDS), "embedding_dim": DIM}
    save_embeddings(tmp_path, synthetic_matrix(len(POST_IDS)), subset, metadata,
                    index_column="embedding_index")

    store = load_embedding_store(tmp_path, mmap=True)

    assert isinstance(store.embeddings, np.memmap)
    assert len(store) == len(POST_IDS)
    assert store.metadata["status"] == "complete"
    assert row_values(get_node_embeddings([POST_IDS[3]], store)) == [3]


def test_load_embedding_store_refuses_incomplete_output(tmp_path):
    subset = pd.DataFrame({"post_id": POST_IDS, "root_id": POST_IDS})
    metadata = {"status": "failed_validation", "num_posts": 5, "embedding_dim": DIM}
    save_embeddings(tmp_path, synthetic_matrix(5), subset, metadata,
                    index_column="embedding_index")
    with pytest.raises(RuntimeError):
        load_embedding_store(tmp_path)


# ============================================================
# GRAPH JSON HELPERS
# ============================================================

def make_graph(node_ids):
    """Same structure as Person 1's graph JSON files."""
    return {
        "root_id": node_ids[0] if node_ids else "",
        "label": "false",
        "nodes": [
            {"post_id": p, "text": "t", "timestamp": "", "parent_id": None,
             "post_type": "source" if i == 0 else "reaction"}
            for i, p in enumerate(node_ids)
        ],
        "edges": [{"source": node_ids[0], "target": p} for p in node_ids[1:]],
    }


def test_graph_node_post_ids_keeps_node_order():
    graph = make_graph([POST_IDS[4], POST_IDS[1], POST_IDS[3]])
    assert graph_node_post_ids(graph) == [POST_IDS[4], POST_IDS[1], POST_IDS[3]]


def test_get_graph_node_embeddings_from_dict(store):
    graph = make_graph([POST_IDS[4], POST_IDS[1], POST_IDS[3]])
    node_ids, features = get_graph_node_embeddings(graph, store)
    assert node_ids == [POST_IDS[4], POST_IDS[1], POST_IDS[3]]
    assert row_values(features) == [4, 1, 3]


def test_get_graph_node_embeddings_from_file(tmp_path, store):
    path = tmp_path / "graph_000001.json"
    path.write_text(json.dumps(make_graph([POST_IDS[2]])))
    node_ids, features = get_graph_node_embeddings(path, store)
    assert node_ids == [POST_IDS[2]]
    assert features.shape == (1, DIM)


def test_graph_without_nodes_list_is_rejected(store):
    with pytest.raises(InvalidPostIdError, match="nodes"):
        get_graph_node_embeddings({"edges": []}, store)


def test_graph_node_without_post_id_is_rejected(store):
    graph = make_graph([POST_IDS[0]])
    del graph["nodes"][0]["post_id"]
    with pytest.raises(InvalidPostIdError, match="node 0"):
        get_graph_node_embeddings(graph, store)


def test_graph_with_empty_node_list(store):
    node_ids, features = get_graph_node_embeddings({"nodes": []}, store)
    assert node_ids == []
    assert features.shape == (0, DIM)
