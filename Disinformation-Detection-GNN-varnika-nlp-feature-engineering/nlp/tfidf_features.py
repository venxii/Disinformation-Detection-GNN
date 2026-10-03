"""
TF-IDF features for PHEME posts.

The vectorizer is fitted ONLY on a training subset supplied by the
caller (the official split is defined by the evaluation protocol, not
here). Validation/test subsets are then transformed with the already
fitted vectorizer, so no information from them leaks into the
vocabulary or the IDF weights.

Python usage:
    from nlp.tfidf_features import (
        load_clean_posts, select_posts, fit_tfidf, transform_tfidf
    )
    posts = load_clean_posts()
    train = select_posts(posts, root_ids=train_root_ids)
    test = select_posts(posts, root_ids=test_root_ids)
    vectorizer, X_train = fit_tfidf(train)
    X_test = transform_tfidf(vectorizer, test)

Command-line usage (split files are CSVs with a root_id or post_id column):
    .venv/bin/python -m nlp.tfidf_features \\
        --run-name my_run \\
        --train-split splits/train.csv \\
        --transform-split test=splits/test.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.config import (
    POSTS_CLEAN_FILE,
    POSTS_CLEAN_INFO_FILE,
    PREPROCESSING_VERSION,
    TFIDF_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

TEXT_COLUMN = "text_tfidf"

# text_tfidf is already lower-cased and space-separated, so tokens are
# simply "anything between spaces". This keeps negations ("not"),
# placeholders ("xxqmark"), single characters and non-English words.
DEFAULT_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\S+",
    "lowercase": False,
    "stop_words": None,
    "strip_accents": None,
    "min_df": 2,
    "max_df": 1.0,
    "max_features": None,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}

VALID_POST_TYPES = {"all", "source", "reaction"}


# ============================================================
# LOADING AND SELECTING POSTS
# ============================================================

def load_clean_posts(path=POSTS_CLEAN_FILE):
    """Load posts_clean.csv with every column as an exact string."""

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: .venv/bin/python -m nlp.build_clean_text"
        )

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def select_posts(posts, post_ids=None, root_ids=None, post_type="all"):
    """
    Return the rows of `posts` for the given IDs.

    - post_ids: rows are returned in exactly this order.
    - root_ids: all posts of these threads, in posts_clean.csv order.
    - post_type: "all", "source" or "reaction".

    Raises ValueError for unknown or duplicated IDs instead of
    silently dropping them.
    """

    if (post_ids is None) == (root_ids is None):
        raise ValueError("Provide exactly one of post_ids or root_ids")

    if post_type not in VALID_POST_TYPES:
        raise ValueError(f"post_type must be one of {sorted(VALID_POST_TYPES)}")

    if post_ids is not None:
        selected = _select_by_post_ids(posts, post_ids)
    else:
        selected = _select_by_root_ids(posts, root_ids)

    if post_type != "all":
        selected = selected[selected["post_type"] == post_type]

    return selected.reset_index(drop=True)


def _select_by_post_ids(posts, post_ids):

    post_ids = [str(i) for i in post_ids]
    _raise_on_duplicates(post_ids, "post_ids")

    indexed = posts.set_index("post_id", drop=False)
    missing = [i for i in post_ids if i not in indexed.index]

    if missing:
        raise ValueError(
            f"{len(missing)} post_ids not found in posts_clean.csv, "
            f"e.g. {missing[:5]}"
        )

    return indexed.loc[post_ids]


def _select_by_root_ids(posts, root_ids):

    root_ids = [str(i) for i in root_ids]
    _raise_on_duplicates(root_ids, "root_ids")

    known = set(posts["root_id"])
    missing = [i for i in root_ids if i not in known]

    if missing:
        raise ValueError(
            f"{len(missing)} root_ids not found in posts_clean.csv, "
            f"e.g. {missing[:5]}"
        )

    return posts[posts["root_id"].isin(set(root_ids))]


def _raise_on_duplicates(ids, name):

    seen = set()
    duplicates = {i for i in ids if i in seen or seen.add(i)}

    if duplicates:
        raise ValueError(f"duplicate {name}: {sorted(duplicates)[:5]}")


def _validate_subset(subset):

    for column in ["post_id", TEXT_COLUMN]:
        if column not in subset.columns:
            raise ValueError(f"subset is missing column '{column}'")

    _raise_on_duplicates(subset["post_id"].tolist(), "post_id in subset")


def _texts(subset):
    return subset[TEXT_COLUMN].fillna("").astype(str).tolist()


# ============================================================
# FIT / TRANSFORM
# ============================================================

def make_vectorizer(**overrides):
    """TfidfVectorizer with project defaults; overrides change single settings."""

    params = {**DEFAULT_TFIDF_PARAMS, **overrides}

    return TfidfVectorizer(**params)


def fit_tfidf(train_subset, **overrides):
    """
    Fit on the training subset only.
    Returns (fitted vectorizer, sparse matrix with one row per train post,
    in the same order as train_subset).
    """

    _validate_subset(train_subset)
    vectorizer = make_vectorizer(**overrides)

    try:
        matrix = vectorizer.fit_transform(_texts(train_subset))
    except ValueError as error:
        raise ValueError(
            "TF-IDF vocabulary is empty: the training texts contain no "
            "tokens that pass min_df/max_df. "
            f"Original error: {error}"
        ) from error

    return vectorizer, matrix.tocsr()


def transform_tfidf(vectorizer, subset):
    """
    Transform validation/test posts with an already fitted vectorizer.
    Words not seen in training are ignored. Row order = subset order.
    """

    _validate_subset(subset)

    return vectorizer.transform(_texts(subset)).tocsr()


def check_no_overlap(train_subset, other_subset, name):
    """Raise if a post appears in both the training and another subset."""

    overlap = set(train_subset["post_id"]) & set(other_subset["post_id"])

    if overlap:
        raise ValueError(
            f"{len(overlap)} posts appear in both train and '{name}', "
            f"e.g. {sorted(overlap)[:5]}"
        )


# ============================================================
# SAVING AND LOADING
# ============================================================

def id_mapping(subset):
    """row_index -> post_id / root_id for a feature matrix."""

    return pd.DataFrame({
        "row_index": range(len(subset)),
        "post_id": subset["post_id"].tolist(),
        "root_id": subset["root_id"].tolist(),
    })


def save_features(output_dir, name, matrix, subset):
    """Save <name>_matrix.npz and <name>_ids.csv separately."""

    if matrix.shape[0] != len(subset):
        raise ValueError(
            f"matrix has {matrix.shape[0]} rows but subset has {len(subset)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    sp.save_npz(output_dir / f"{name}_matrix.npz", matrix.tocsr())
    id_mapping(subset).to_csv(output_dir / f"{name}_ids.csv", index=False)


def load_features(output_dir, name):
    """Load a matrix and its ID mapping, checking that they line up."""

    matrix = sp.load_npz(output_dir / f"{name}_matrix.npz").tocsr()
    ids = pd.read_csv(
        output_dir / f"{name}_ids.csv", dtype=str, keep_default_na=False
    )

    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"{name}: matrix has {matrix.shape[0]} rows but "
            f"ID mapping has {len(ids)}"
        )

    expected = [str(i) for i in range(len(ids))]

    if ids["row_index"].tolist() != expected:
        raise ValueError(f"{name}: row_index column is not 0..n-1")

    return matrix, ids


def save_vectorizer(vectorizer, output_dir, train_subset, extra_info=None):
    """Save the fitted vectorizer (joblib) and a readable JSON description."""

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")

    params = vectorizer.get_params()
    params["ngram_range"] = list(params["ngram_range"])
    params["dtype"] = np.dtype(params["dtype"]).name

    info = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "preprocessing_version": PREPROCESSING_VERSION,
        "text_column": TEXT_COLUMN,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "num_train_posts": len(train_subset),
        "num_train_threads": int(train_subset["root_id"].nunique()),
        "params": params,
        **(extra_info or {}),
    }

    with open(output_dir / "vectorizer_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, default=str)

    return info


def load_vectorizer(output_dir):
    """Load a fitted vectorizer. Only load files created by this project."""

    return joblib.load(output_dir / "vectorizer.joblib")


# ============================================================
# COMMAND LINE
# ============================================================

def read_split_file(path):
    """A split file is a CSV with a 'post_id' or a 'root_id' column."""

    split = pd.read_csv(path, dtype=str, keep_default_na=False)

    if "post_id" in split.columns:
        return {"post_ids": split["post_id"].tolist()}

    if "root_id" in split.columns:
        return {"root_ids": split["root_id"].tolist()}

    raise ValueError(f"{path} needs a 'post_id' or 'root_id' column")


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Fit TF-IDF on a supplied training split and "
                    "transform other splits."
    )
    parser.add_argument("--run-name", required=True,
                        help="Output folder name under data/nlp/tfidf/")
    parser.add_argument("--train-split", required=True,
                        help="CSV with post_id or root_id column")
    parser.add_argument("--transform-split", action="append", default=[],
                        metavar="NAME=PATH",
                        help="Extra split to transform, e.g. test=test.csv")
    parser.add_argument("--post-type", default="all",
                        choices=sorted(VALID_POST_TYPES))
    parser.add_argument("--min-df", type=int,
                        default=DEFAULT_TFIDF_PARAMS["min_df"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing run folder")

    return parser.parse_args(argv)


def main(argv=None):

    args = parse_args(argv)
    output_dir = TFIDF_DIR / args.run_name

    print("=" * 60)
    print("TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    if output_dir.exists() and not args.overwrite:
        print(f"\nERROR: {output_dir} already exists. Use --overwrite.")
        sys.exit(1)

    posts = load_clean_posts()

    train = select_posts(
        posts, post_type=args.post_type, **read_split_file(args.train_split)
    )

    others = {}

    for item in args.transform_split:
        name, _, path = item.partition("=")

        if not name or not path or name == "train":
            print(f"\nERROR: invalid --transform-split '{item}'")
            sys.exit(1)

        others[name] = select_posts(
            posts, post_type=args.post_type, **read_split_file(path)
        )
        check_no_overlap(train, others[name], name)

    vectorizer, train_matrix = fit_tfidf(train, min_df=args.min_df)
    save_features(output_dir, "train", train_matrix, train)

    print(f"\nTrain posts     : {train_matrix.shape[0]}")
    print(f"Vocabulary size : {train_matrix.shape[1]}")

    for name, subset in others.items():
        matrix = transform_tfidf(vectorizer, subset)
        save_features(output_dir, name, matrix, subset)
        print(f"{name:<15} : {matrix.shape[0]} posts")

    source_info = {}

    if POSTS_CLEAN_INFO_FILE.exists():
        with open(POSTS_CLEAN_INFO_FILE, encoding="utf-8") as file:
            source_info = json.load(file)

    save_vectorizer(
        vectorizer,
        output_dir,
        train,
        extra_info={
            "run_name": args.run_name,
            "post_type": args.post_type,
            "train_split_file": args.train_split,
            "transformed_splits": {
                name: len(subset) for name, subset in others.items()
            },
            "posts_clean_created_at": source_info.get("created_at"),
        },
    )

    print(f"\nSaved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
