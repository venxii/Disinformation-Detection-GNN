"""
Tests for checkpointed extraction in nlp/bertweet_embeddings.py.

A deterministic fake embed_fn replaces BERTweet, so these tests are fast
and never touch the real model or the full dataset.
"""

import json

import numpy as np
import pytest

from nlp import bertweet_embeddings as be
from nlp.bertweet_embeddings import (
    CheckpointMismatchError,
    assemble_chunks,
    chunk_bounds,
    chunk_file,
    completed_output_exists,
    extract_with_checkpoints,
    extraction_config,
    load_full_embeddings,
    load_valid_chunk,
    save_embeddings,
)
from nlp.config import EMBEDDING_DIM


# ============================================================
# HELPERS
# ============================================================

def fake_vector(text):
    """Same text -> same vector, different texts -> different vectors."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**32)
    return np.random.default_rng(seed).normal(size=EMBEDDING_DIM).astype(np.float32)


class CountingEmbedder:
    """Fake embed_fn that records calls and can fail on a chosen call."""

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.rows = 0
        self.fail_on_call = fail_on_call

    def __call__(self, texts):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise KeyboardInterrupt("simulated interruption")
        self.rows += len(texts)
        return np.vstack([fake_vector(t) for t in texts])


@pytest.fixture
def data():
    n = 23
    post_ids = [f"5500000000000000{i:02d}" for i in range(n)]
    texts = [f"tweet number {i}" for i in range(n)]
    return post_ids, texts


def config_for(post_ids, **changes):
    config = extraction_config(post_ids, "sha-of-input", 4, 5, "cpu")
    config.update(changes)
    return config


def expected_matrix(texts):
    return np.vstack([fake_vector(t) for t in texts])


# ============================================================
# BASIC BEHAVIOUR
# ============================================================

def test_chunk_bounds_cover_everything_once():
    bounds = chunk_bounds(23, 5)
    assert bounds == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)]
    covered = [i for start, end in bounds for i in range(start, end)]
    assert covered == list(range(23))


def test_full_run_keeps_input_order(tmp_path, data):
    post_ids, texts = data
    embedder = CountingEmbedder()

    embeddings, stats = extract_with_checkpoints(
        post_ids, texts, embedder, tmp_path, config_for(post_ids),
        chunk_size=5, log=lambda _: None,
    )

    assert np.array_equal(embeddings, expected_matrix(texts))
    assert embeddings.dtype == np.float32
    assert stats["chunks_total"] == 5
    assert stats["chunks_computed"] == 5
    assert stats["chunks_reused"] == 0
    assert stats["assembled_ids_match_input"] is True
    assert stats["chunk_indices"] is True
    assert embedder.rows == len(texts)


def test_progress_file_records_last_completed_position(tmp_path, data):
    post_ids, texts = data
    extract_with_checkpoints(
        post_ids, texts, CountingEmbedder(), tmp_path, config_for(post_ids),
        chunk_size=5, log=lambda _: None,
    )
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed_chunks"] == 5
    assert progress["last_completed_row"] == 22
    assert progress["last_completed_post_id"] == post_ids[-1]


# ============================================================
# INTERRUPT AND RESUME
# ============================================================

def test_interrupted_run_resumes_without_recomputing(tmp_path, data):
    post_ids, texts = data
    config = config_for(post_ids)

    with pytest.raises(KeyboardInterrupt):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(fail_on_call=3), tmp_path,
            config, chunk_size=5, log=lambda _: None,
        )

    assert chunk_file(tmp_path, 0).exists()
    assert chunk_file(tmp_path, 1).exists()
    assert not chunk_file(tmp_path, 2).exists()

    resumed = CountingEmbedder()
    embeddings, stats = extract_with_checkpoints(
        post_ids, texts, resumed, tmp_path, config,
        chunk_size=5, log=lambda _: None,
    )

    assert stats["chunks_reused"] == 2
    assert stats["chunks_computed"] == 3
    assert resumed.rows == 23 - 10
    assert np.array_equal(embeddings, expected_matrix(texts))


def test_interrupted_run_is_not_a_completed_output(tmp_path, data):
    post_ids, texts = data
    with pytest.raises(KeyboardInterrupt):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(fail_on_call=2), tmp_path / "ckpt",
            config_for(post_ids), chunk_size=5, log=lambda _: None,
        )
    assert completed_output_exists(tmp_path / "out") is False


def test_leftover_temp_file_is_removed_and_not_counted(tmp_path, data):
    post_ids, texts = data
    config = config_for(post_ids)

    with pytest.raises(KeyboardInterrupt):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(fail_on_call=2), tmp_path,
            config, chunk_size=5, log=lambda _: None,
        )

    partial = tmp_path / "chunk_00001.npz.tmp"
    partial.write_bytes(b"half written")

    messages = []
    _, stats = extract_with_checkpoints(
        post_ids, texts, CountingEmbedder(), tmp_path, config,
        chunk_size=5, log=messages.append,
    )

    assert not partial.exists()
    assert any("temporary" in m for m in messages)
    assert stats["chunks_reused"] == 1


def test_failed_write_never_creates_a_chunk_file(tmp_path, monkeypatch, data):
    post_ids, texts = data

    def broken_savez(file, **arrays):
        file.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(be.np, "savez", broken_savez)

    with pytest.raises(OSError):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(), tmp_path, config_for(post_ids),
            chunk_size=5, log=lambda _: None,
        )

    assert not chunk_file(tmp_path, 0).exists()


# ============================================================
# CORRUPTED OR WRONG CHUNKS
# ============================================================

def run_once(tmp_path, post_ids, texts, config):
    return extract_with_checkpoints(
        post_ids, texts, CountingEmbedder(), tmp_path, config,
        chunk_size=5, log=lambda _: None,
    )


def test_corrupted_chunk_is_reported_and_recomputed(tmp_path, data):
    post_ids, texts = data
    config = config_for(post_ids)
    run_once(tmp_path, post_ids, texts, config)

    chunk_file(tmp_path, 2).write_bytes(b"garbage")

    messages = []
    embedder = CountingEmbedder()
    embeddings, stats = extract_with_checkpoints(
        post_ids, texts, embedder, tmp_path, config,
        chunk_size=5, log=messages.append,
    )

    assert stats["chunks_recomputed_invalid"] == 1
    assert stats["chunks_reused"] == 4
    assert embedder.rows == 5
    assert any("Chunk 2 is invalid" in m for m in messages)
    assert np.array_equal(embeddings, expected_matrix(texts))


def test_chunk_with_wrong_ids_is_rejected(tmp_path, data):
    post_ids, _ = data
    path = tmp_path / "chunk.npz"
    np.savez(path, embeddings=np.zeros((5, EMBEDDING_DIM), np.float32),
             post_ids=np.array(post_ids[5:10]))
    _, problem = load_valid_chunk(path, post_ids[0:5])
    assert "post_ids" in problem


@pytest.mark.parametrize(
    "embeddings, reason",
    [
        (np.zeros((4, EMBEDDING_DIM), np.float32), "shape"),
        (np.zeros((5, EMBEDDING_DIM), np.float64), "dtype"),
        (np.full((5, EMBEDDING_DIM), np.nan, np.float32), "NaN"),
    ],
)
def test_invalid_chunk_contents_are_rejected(tmp_path, data, embeddings, reason):
    post_ids, _ = data
    path = tmp_path / "chunk.npz"
    np.savez(path, embeddings=embeddings, post_ids=np.array(post_ids[:5]))
    _, problem = load_valid_chunk(path, post_ids[:5])
    assert reason in problem


def test_assemble_detects_missing_chunk(tmp_path, data):
    post_ids, texts = data
    run_once(tmp_path, post_ids, texts, config_for(post_ids))
    chunk_file(tmp_path, 3).unlink()
    with pytest.raises(RuntimeError, match="chunk files"):
        assemble_chunks(tmp_path, post_ids, 5)


def test_bad_embed_output_is_never_saved(tmp_path, data):
    post_ids, texts = data

    def wrong_shape(chunk_texts):
        return np.zeros((len(chunk_texts), 10), np.float32)

    with pytest.raises(ValueError, match="shape"):
        extract_with_checkpoints(
            post_ids, texts, wrong_shape, tmp_path, config_for(post_ids),
            chunk_size=5, log=lambda _: None,
        )
    assert not chunk_file(tmp_path, 0).exists()


# ============================================================
# SETTINGS MISMATCH
# ============================================================

@pytest.mark.parametrize(
    "changes",
    [
        {"model_revision": "different"},
        {"preprocessing_version": "9.9"},
        {"input_sha256": "other-input"},
        {"batch_size": 64},
        {"device": "mps"},
        {"embedding_dim": 1024},
        {"pooling": "cls token"},
    ],
)
def test_settings_mismatch_stops_resume(tmp_path, data, changes):
    post_ids, texts = data

    with pytest.raises(KeyboardInterrupt):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(fail_on_call=2), tmp_path,
            config_for(post_ids), chunk_size=5, log=lambda _: None,
        )

    key = next(iter(changes))
    with pytest.raises(CheckpointMismatchError, match=key):
        extract_with_checkpoints(
            post_ids, texts, CountingEmbedder(), tmp_path,
            config_for(post_ids, **changes), chunk_size=5, log=lambda _: None,
        )


def test_different_post_ids_stop_resume(tmp_path, data):
    post_ids, texts = data
    run_once(tmp_path, post_ids, texts, config_for(post_ids))

    reordered = post_ids[::-1]
    with pytest.raises(CheckpointMismatchError, match="post_ids_sha256"):
        extract_with_checkpoints(
            reordered, texts, CountingEmbedder(), tmp_path,
            config_for(reordered), chunk_size=5, log=lambda _: None,
        )


# ============================================================
# COMPLETED OUTPUT
# ============================================================

def make_output(directory, post_ids, status):
    import pandas as pd
    subset = pd.DataFrame({"post_id": post_ids, "root_id": post_ids})
    metadata = {"status": status, "num_posts": len(post_ids),
                "embedding_dim": EMBEDDING_DIM}
    save_embeddings(directory, np.zeros((len(post_ids), EMBEDDING_DIM), np.float32),
                    subset, metadata, index_column="embedding_index")


def test_completed_output_detection(tmp_path, data):
    post_ids, _ = data
    assert completed_output_exists(tmp_path) is False
    make_output(tmp_path, post_ids, "complete")
    assert completed_output_exists(tmp_path) is True


def test_load_full_refuses_incomplete_output(tmp_path, data):
    post_ids, _ = data
    make_output(tmp_path, post_ids, "failed_validation")
    with pytest.raises(RuntimeError, match="not 'complete'"):
        load_full_embeddings(tmp_path)


def test_load_full_refuses_missing_metadata(tmp_path, data):
    post_ids, _ = data
    make_output(tmp_path, post_ids, "complete")
    (tmp_path / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_full_embeddings(tmp_path)


def test_load_full_returns_aligned_data(tmp_path, data):
    post_ids, _ = data
    make_output(tmp_path, post_ids, "complete")
    embeddings, ids, metadata = load_full_embeddings(tmp_path, mmap=True)
    assert embeddings.shape == (len(post_ids), EMBEDDING_DIM)
    assert ids["post_id"].tolist() == post_ids
    assert list(ids.columns) == ["embedding_index", "post_id", "root_id"]
    assert metadata["status"] == "complete"


def test_cli_refuses_to_overwrite_completed_output(tmp_path, monkeypatch, data):
    post_ids, _ = data
    make_output(tmp_path, post_ids, "complete")
    monkeypatch.setattr(be, "FULL_EMBEDDINGS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        be.main(["full"])
