"""
Tests for nlp/bertweet_embeddings.py.

The real BERTweet model is never downloaded here: a tiny fake tokenizer
and fake model with the same call interface are used instead.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from nlp.bertweet_embeddings import (
    embed_texts,
    freeze_model,
    load_embeddings,
    mean_pool,
    parameter_fingerprint,
    prepare_texts,
    save_embeddings,
    select_device,
    select_pilot_posts,
)
from nlp.config import EMBEDDING_DIM


# ============================================================
# FAKE TOKENIZER AND MODEL
# ============================================================

BOS, PAD, EOS = 0, 1, 2


class FakeTokenizer:
    """Splits on spaces; each word gets a stable ID. Pads on the right."""

    unk_token_id = 3

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        words = text.split()
        return [BOS] + [self.vocab.setdefault(w, 4 + len(self.vocab)) for w in words] + [EOS]

    def __call__(self, texts, padding=False, truncation=False,
                 max_length=None, return_tensors=None):

        single = isinstance(texts, str)
        batch = [self._ids(t) for t in ([texts] if single else texts)]

        if truncation and max_length:
            batch = [ids[:max_length] for ids in batch]

        if return_tensors != "pt":
            return {"input_ids": batch[0] if single else batch}

        width = max(len(ids) for ids in batch)
        input_ids = [ids + [PAD] * (width - len(ids)) for ids in batch]
        mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in batch]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(mask),
        }


class FakeModel(torch.nn.Module):
    """
    Each token's vector depends only on its ID, so a correct pooling
    gives the same result with or without padding. Padding positions
    get a huge value so any leak into the mean would be obvious.
    """

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = torch.nn.Parameter(
            torch.randn(1000, EMBEDDING_DIM, generator=generator)
        )
        with torch.no_grad():
            self.table[PAD] = 1e6

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.table[input_ids])


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def model():
    return freeze_model(FakeModel())


# ============================================================
# MEAN POOLING
# ============================================================

def test_mean_pool_matches_manual_average():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = torch.tensor([[1, 1, 1]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[3.0, 4.0]]))


def test_mean_pool_ignores_padding_positions():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [999.0, -999.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_mean_pool_handles_rows_of_different_lengths():
    hidden = torch.tensor([
        [[2.0], [4.0], [6.0]],
        [[10.0], [500.0], [500.0]],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[4.0], [10.0]]))


def test_mean_pool_all_padding_row_gives_zeros_not_nan():
    hidden = torch.ones(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.long)
    pooled = mean_pool(hidden, mask)
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled, torch.zeros(1, 4))


# ============================================================
# EMBEDDING
# ============================================================

def test_embed_texts_shape_and_dtype(tokenizer, model):
    out = embed_texts(["a b", "c"], tokenizer, model, "cpu")
    assert out.shape == (2, EMBEDDING_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_embed_texts_padding_does_not_change_embeddings(tokenizer, model):
    texts = ["short", "a much longer text with many words in it", "mid size text"]
    batched = embed_texts(texts, tokenizer, model, "cpu", batch_size=3)
    alone = np.vstack([embed_texts([t], tokenizer, model, "cpu") for t in texts])
    assert np.allclose(batched, alone, atol=1e-5)


def test_embed_texts_keeps_input_order(tokenizer, model):
    texts = ["one two three four", "x", "five six", "y z"]
    out = embed_texts(texts, tokenizer, model, "cpu", batch_size=2)
    for row, text in enumerate(texts):
        alone = embed_texts([text], tokenizer, model, "cpu")[0]
        assert np.allclose(out[row], alone, atol=1e-5)


def test_embed_texts_without_length_sorting_gives_same_result(tokenizer, model):
    texts = ["one two three four", "x", "five six", "y z", ""]
    sorted_run = embed_texts(texts, tokenizer, model, "cpu", batch_size=2)
    in_order = embed_texts(texts, tokenizer, model, "cpu", batch_size=2,
                           sort_by_length=False)
    assert np.allclose(sorted_run, in_order, atol=1e-5)


def test_embed_texts_empty_and_whitespace_text(tokenizer, model):
    out = embed_texts(["", "   ", None, "word"], tokenizer, model, "cpu")
    assert out.shape == (4, EMBEDDING_DIM)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], out[1])
    assert np.allclose(out[0], out[2])


def test_embed_texts_truncates_long_text(tokenizer, model):
    long_text = " ".join(f"w{i}" for i in range(300))
    out = embed_texts([long_text], tokenizer, model, "cpu", max_length=128)
    assert out.shape == (1, EMBEDDING_DIM)
    assert np.isfinite(out).all()


def test_embed_texts_does_not_update_weights(tokenizer, model):
    before = model.table.detach().clone()
    embed_texts(["a b c", "d"], tokenizer, model, "cpu")
    assert torch.equal(before, model.table)


def test_prepare_texts():
    assert prepare_texts([" a ", "", "  ", None, float("nan")]) == ["a", "", "", "", ""]


# ============================================================
# MODEL STATE AND DEVICE
# ============================================================

def test_freeze_model_sets_eval_and_no_grad():
    layer = torch.nn.Linear(3, 3)
    layer.train()
    freeze_model(layer)
    assert layer.training is False
    assert not any(p.requires_grad for p in layer.parameters())


def test_parameter_fingerprint_is_stable_and_detects_changes(model):
    before = parameter_fingerprint(model)
    assert parameter_fingerprint(model) == before
    with torch.no_grad():
        model.table[5, 0] += 1e-6
    assert parameter_fingerprint(model) != before


def test_select_device_cpu_and_auto():
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in {"cpu", "mps"}


# ============================================================
# PILOT SELECTION
# ============================================================

@pytest.fixture
def posts():
    n = 500
    return pd.DataFrame({
        "post_id": [f"55{i:016d}" for i in range(n)],
        "root_id": [f"55{(i // 10):016d}" for i in range(n)],
        "text_bertweet": [f"text {i}" for i in range(n)],
    })


def test_pilot_selection_is_deterministic(posts):
    a = select_pilot_posts(posts, 100, seed=42)
    b = select_pilot_posts(posts, 100, seed=42)
    assert a["post_id"].tolist() == b["post_id"].tolist()


def test_pilot_selection_size_and_uniqueness(posts):
    selected = select_pilot_posts(posts, 100, seed=42)
    assert len(selected) == 100
    assert selected["post_id"].is_unique


def test_pilot_selection_changes_with_seed(posts):
    a = select_pilot_posts(posts, 100, seed=1)
    b = select_pilot_posts(posts, 100, seed=2)
    assert a["post_id"].tolist() != b["post_id"].tolist()


def test_pilot_selection_keeps_ids_as_strings_and_file_order(posts):
    selected = select_pilot_posts(posts, 100, seed=42)
    assert all(isinstance(i, str) for i in selected["post_id"])
    assert selected["post_id"].tolist() == sorted(selected["post_id"])


def test_pilot_selection_with_fewer_posts_returns_all(posts):
    selected = select_pilot_posts(posts.head(30), 100, seed=42)
    assert len(selected) == 30


# ============================================================
# SAVING AND ID ALIGNMENT
# ============================================================

def test_save_and_load_round_trip(tmp_path, posts):
    subset = posts.head(5)
    embeddings = np.random.default_rng(0).normal(size=(5, EMBEDDING_DIM)).astype(np.float32)

    save_embeddings(tmp_path, embeddings, subset, {"stage": "test"})
    loaded, ids = load_embeddings(tmp_path)

    assert np.array_equal(loaded, embeddings)
    assert ids["post_id"].tolist() == subset["post_id"].tolist()
    assert ids["root_id"].tolist() == subset["root_id"].tolist()
    assert ids["row_index"].tolist() == [str(i) for i in range(5)]


def test_save_rejects_wrong_shape(tmp_path, posts):
    with pytest.raises(ValueError, match="shape"):
        save_embeddings(tmp_path, np.zeros((4, EMBEDDING_DIM)), posts.head(5), {})
    with pytest.raises(ValueError, match="shape"):
        save_embeddings(tmp_path, np.zeros((5, 10)), posts.head(5), {})


def test_save_rejects_duplicate_ids(tmp_path, posts):
    subset = pd.concat([posts.head(2), posts.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        save_embeddings(tmp_path, np.zeros((3, EMBEDDING_DIM)), subset, {})


def test_load_detects_row_mismatch(tmp_path, posts):
    save_embeddings(tmp_path, np.zeros((5, EMBEDDING_DIM), np.float32), posts.head(5), {})
    ids = pd.read_csv(tmp_path / "ids.csv", dtype=str)
    ids.head(4).to_csv(tmp_path / "ids.csv", index=False)
    with pytest.raises(ValueError, match="rows"):
        load_embeddings(tmp_path)
