import numpy as np
import pandas as pd
import pytest

from nlp import tfidf_features
from nlp.tfidf_features import (
    DEFAULT_TFIDF_PARAMS,
    check_no_overlap,
    fit_tfidf,
    load_features,
    load_vectorizer,
    save_features,
    save_vectorizer,
    select_posts,
    transform_tfidf,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def posts():
    """Small stand-in for posts_clean.csv (all columns are strings)."""

    return pd.DataFrame({
        "post_id": ["101", "102", "103", "201", "202", "301", "302"],
        "root_id": ["101", "101", "101", "201", "201", "301", "301"],
        "post_type": ["source", "reaction", "reaction",
                      "source", "reaction", "source", "reaction"],
        "text_tfidf": [
            "police confirm 12 dead xxurl",
            "xxuser this is not true xxqmark",
            "xxuser do not believe it xxemoticon_sad",
            "police say the gunman is dead",
            "xxuser not confirmed yet xxqmark",
            "hay un soldado muerto",
            "",
        ],
    })


@pytest.fixture
def train(posts):
    return select_posts(posts, root_ids=["101", "201"])


@pytest.fixture
def test(posts):
    return select_posts(posts, root_ids=["301"])


# ============================================================
# SETTINGS
# ============================================================

def test_defaults_keep_stop_words_and_use_bigrams():
    assert DEFAULT_TFIDF_PARAMS["stop_words"] is None
    assert DEFAULT_TFIDF_PARAMS["ngram_range"] == (1, 2)
    assert DEFAULT_TFIDF_PARAMS["token_pattern"] == r"(?u)\S+"


# ============================================================
# SELECTING POSTS
# ============================================================

def test_select_by_root_ids_returns_whole_threads(posts):
    subset = select_posts(posts, root_ids=["201"])
    assert subset["post_id"].tolist() == ["201", "202"]


def test_select_by_post_ids_keeps_requested_order(posts):
    subset = select_posts(posts, post_ids=["202", "101", "301"])
    assert subset["post_id"].tolist() == ["202", "101", "301"]


def test_select_source_posts_only(posts):
    subset = select_posts(posts, root_ids=["101", "201"], post_type="source")
    assert subset["post_id"].tolist() == ["101", "201"]


def test_select_unknown_ids_raise(posts):
    with pytest.raises(ValueError, match="not found"):
        select_posts(posts, post_ids=["999"])
    with pytest.raises(ValueError, match="not found"):
        select_posts(posts, root_ids=["999"])


def test_select_duplicate_ids_raise(posts):
    with pytest.raises(ValueError, match="duplicate"):
        select_posts(posts, post_ids=["101", "101"])


def test_select_requires_exactly_one_id_list(posts):
    with pytest.raises(ValueError, match="exactly one"):
        select_posts(posts)


# ============================================================
# FIT / TRANSFORM
# ============================================================

def test_vocabulary_comes_from_training_only(train, test):
    vectorizer, _ = fit_tfidf(train, min_df=1)
    vocabulary = vectorizer.vocabulary_
    assert "soldado" not in vocabulary
    assert "muerto" not in vocabulary


def test_vocabulary_keeps_negations_bigrams_and_placeholders(train):
    vectorizer, _ = fit_tfidf(train, min_df=1)
    vocabulary = vectorizer.vocabulary_
    for term in ["not", "the", "is", "do not", "not true", "xxqmark",
                 "xxemoticon_sad", "12"]:
        assert term in vocabulary


def test_non_english_text_is_kept(posts):
    subset = select_posts(posts, root_ids=["301"])
    vectorizer, _ = fit_tfidf(subset, min_df=1)
    assert "soldado" in vectorizer.vocabulary_
    assert "un soldado" in vectorizer.vocabulary_


def test_feature_dimensions(train, test):
    vectorizer, X_train = fit_tfidf(train, min_df=1)
    X_test = transform_tfidf(vectorizer, test)
    vocabulary_size = len(vectorizer.vocabulary_)
    assert X_train.shape == (len(train), vocabulary_size)
    assert X_test.shape == (len(test), vocabulary_size)
    assert X_train.dtype == np.float32


def test_fit_and_transform_give_same_training_matrix(train):
    vectorizer, X_fit = fit_tfidf(train, min_df=1)
    X_again = transform_tfidf(vectorizer, train)
    assert np.allclose(X_fit.toarray(), X_again.toarray())


def test_min_df_two_drops_words_seen_in_one_post(train):
    vectorizer, _ = fit_tfidf(train)
    assert "police" in vectorizer.vocabulary_
    assert "gunman" not in vectorizer.vocabulary_


def test_rows_are_l2_normalised(train):
    _, X = fit_tfidf(train, min_df=1)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_row_order_follows_subset_order(posts):
    subset = select_posts(posts, root_ids=["101", "201"])
    shuffled = select_posts(posts, post_ids=subset["post_id"].tolist()[::-1])
    vectorizer, X = fit_tfidf(subset, min_df=1)
    X_shuffled = transform_tfidf(vectorizer, shuffled)
    assert np.allclose(X.toarray()[::-1], X_shuffled.toarray())


def test_empty_text_gives_zero_row(posts, train):
    vectorizer, _ = fit_tfidf(train, min_df=1)
    empty = select_posts(posts, post_ids=["302"])
    X = transform_tfidf(vectorizer, empty)
    assert X.shape[0] == 1
    assert X.nnz == 0


def test_all_empty_training_text_raises_clear_error(posts):
    empty = select_posts(posts, post_ids=["302"])
    with pytest.raises(ValueError, match="vocabulary is empty"):
        fit_tfidf(empty)


def test_fit_is_deterministic(train):
    v1, X1 = fit_tfidf(train, min_df=1)
    v2, X2 = fit_tfidf(train, min_df=1)
    assert v1.vocabulary_ == v2.vocabulary_
    assert (X1 != X2).nnz == 0


def test_duplicate_post_ids_in_subset_raise(train):
    doubled = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        fit_tfidf(doubled)


def test_overlap_between_train_and_test_raises(posts, train):
    leaky = select_posts(posts, post_ids=["101", "301"])
    with pytest.raises(ValueError, match="appear in both"):
        check_no_overlap(train, leaky, "test")


# ============================================================
# SERIALIZATION
# ============================================================

def test_features_round_trip(tmp_path, train):
    _, X = fit_tfidf(train, min_df=1)
    save_features(tmp_path, "train", X, train)

    X_loaded, ids = load_features(tmp_path, "train")

    assert (X != X_loaded).nnz == 0
    assert ids["post_id"].tolist() == train["post_id"].tolist()
    assert ids["root_id"].tolist() == train["root_id"].tolist()
    assert ids["row_index"].tolist() == [str(i) for i in range(len(train))]


def test_save_features_rejects_row_mismatch(tmp_path, train):
    _, X = fit_tfidf(train, min_df=1)
    with pytest.raises(ValueError, match="rows"):
        save_features(tmp_path, "train", X, train.iloc[:2])


def test_load_features_detects_mismatch(tmp_path, train):
    _, X = fit_tfidf(train, min_df=1)
    save_features(tmp_path, "train", X, train)
    ids = pd.read_csv(tmp_path / "train_ids.csv", dtype=str)
    ids.iloc[:-1].to_csv(tmp_path / "train_ids.csv", index=False)
    with pytest.raises(ValueError, match="rows"):
        load_features(tmp_path, "train")


def test_vectorizer_round_trip(tmp_path, train, test):
    vectorizer, _ = fit_tfidf(train, min_df=1)
    info = save_vectorizer(vectorizer, tmp_path, train)

    loaded = load_vectorizer(tmp_path)

    assert loaded.vocabulary_ == vectorizer.vocabulary_
    assert np.allclose(
        transform_tfidf(loaded, test).toarray(),
        transform_tfidf(vectorizer, test).toarray(),
    )
    assert info["vocabulary_size"] == len(vectorizer.vocabulary_)
    assert (tmp_path / "vectorizer_info.json").exists()


# ============================================================
# COMMAND LINE
# ============================================================

def test_cli_writes_outputs_and_refuses_overwrite(tmp_path, monkeypatch, posts):
    clean_file = tmp_path / "posts_clean.csv"
    posts.to_csv(clean_file, index=False)

    pd.DataFrame({"root_id": ["101", "201"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"root_id": ["301"]}).to_csv(tmp_path / "test.csv", index=False)

    monkeypatch.setattr(tfidf_features, "TFIDF_DIR", tmp_path / "tfidf")
    monkeypatch.setattr(tfidf_features, "POSTS_CLEAN_FILE", clean_file)
    monkeypatch.setattr(tfidf_features, "POSTS_CLEAN_INFO_FILE", tmp_path / "none.json")
    monkeypatch.setattr(
        tfidf_features, "load_clean_posts",
        lambda path=clean_file: pd.read_csv(path, dtype=str, keep_default_na=False),
    )

    argv = [
        "--run-name", "demo",
        "--train-split", str(tmp_path / "train.csv"),
        "--transform-split", f"test={tmp_path / 'test.csv'}",
        "--min-df", "1",
    ]
    tfidf_features.main(argv)

    run_dir = tmp_path / "tfidf" / "demo"
    for name in ["vectorizer.joblib", "vectorizer_info.json",
                 "train_matrix.npz", "train_ids.csv",
                 "test_matrix.npz", "test_ids.csv"]:
        assert (run_dir / name).exists()

    _, test_ids = load_features(run_dir, "test")
    assert test_ids["post_id"].tolist() == ["301", "302"]

    with pytest.raises(SystemExit):
        tfidf_features.main(argv)


def test_cli_rejects_leaky_split(tmp_path, monkeypatch, posts):
    clean_file = tmp_path / "posts_clean.csv"
    posts.to_csv(clean_file, index=False)
    pd.DataFrame({"root_id": ["101", "201"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"root_id": ["201"]}).to_csv(tmp_path / "test.csv", index=False)

    monkeypatch.setattr(tfidf_features, "TFIDF_DIR", tmp_path / "tfidf")
    monkeypatch.setattr(
        tfidf_features, "load_clean_posts",
        lambda path=clean_file: pd.read_csv(path, dtype=str, keep_default_na=False),
    )

    with pytest.raises(ValueError, match="appear in both"):
        tfidf_features.main([
            "--run-name", "leaky",
            "--train-split", str(tmp_path / "train.csv"),
            "--transform-split", f"test={tmp_path / 'test.csv'}",
        ])
