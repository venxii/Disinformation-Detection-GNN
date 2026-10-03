"""
Stage 7: end-to-end checks for the NLP feature pipeline.

This module only reads existing data and writes two reports. It does not
rebuild cleaned text, refit TF-IDF on PHEME, re-extract BERTweet embeddings,
or edit Person 1's files.

Run from the project root:

    .venv/bin/python -m nlp.stage7_validation
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nlp.build_clean_text import OUTPUT_COLUMNS, REQUIRED_COLUMNS
from nlp.config import (
    BERTWEET_MAX_LENGTH,
    BERTWEET_MODEL_NAME,
    BERTWEET_REVISION,
    CLAIMS_FILE,
    EMBEDDING_DIM,
    FULL_EMBEDDINGS_DIR,
    GRAPH_METADATA_FILE,
    GRAPHS_DIR,
    NLP_DATA,
    POSTS_CLEAN_FILE,
    POSTS_CLEAN_INFO_FILE,
    POSTS_FILE,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
)
from nlp.embedding_node_helper import (
    EmbeddingLookupError,
    get_graph_node_embeddings,
    get_node_embeddings,
    load_embedding_store,
    load_graph,
)
from nlp.text_preprocessing import (
    URL_TOKEN,
    USER_TOKEN,
    normalize_for_bertweet,
    normalize_for_tfidf,
    text_flags,
)
from nlp.tfidf_features import (
    DEFAULT_TFIDF_PARAMS,
    check_no_overlap,
    fit_tfidf,
    load_features,
    load_vectorizer,
    make_vectorizer,
    save_features,
    save_vectorizer,
    select_posts,
    transform_tfidf,
)


EXPECTED_POSTS = 104582
EXPECTED_GRAPHS = 6425
EXPECTED_EDGES = 95074
EXPECTED_LABELS = {"non-rumour", "true", "false", "unverified"}
TFIDF_PLACEHOLDERS = {URL_TOKEN, USER_TOKEN}

REPORT_JSON = NLP_DATA / "stage7_validation_report.json"
REPORT_MD = NLP_DATA / "STAGE7_VALIDATION_REPORT.md"
STAGE6_REPORT = NLP_DATA / "node_helper_check.json"

PERSON1_PATHS = [
    "data/processed",
    "data/graphs",
    "dataset",
    "docs/data_pipeline.md",
]

LARGE_IGNORED_PATHS = [
    "data/nlp/embeddings/full/embeddings.npy",
    "data/nlp/embeddings/full/ids.csv",
    "data/nlp/embeddings/pilot/embeddings.npy",
    "data/nlp/embeddings/pilot/ids.csv",
    "data/nlp/posts_clean.csv",
    ".cache",
    ".venv",
]

SMALL_TRACKED_PATHS = [
    "data/nlp/embeddings/full/metadata.json",
    "data/nlp/embeddings/pilot/metadata.json",
    "data/nlp/posts_clean_info.json",
    "data/nlp/node_helper_check.json",
    "data/nlp/bertweet_tokenizer_check.json",
    "data/nlp/stage7_validation_report.json",
    "data/nlp/STAGE7_VALIDATION_REPORT.md",
    "nlp/stage7_validation.py",
    "requirements-nlp.txt",
]

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(
        r"(?i)(api[_-]?key|app[_-]?secret|password|access[_-]?token)"
        r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
]
HOME_PATH_PATTERN = re.compile(r"/Users/[A-Za-z0-9._-]+")
SOURCE_SCAN_DIRS = ["nlp", "docs"]
SOURCE_SCAN_FILES = [".gitignore", "requirements-nlp.txt"]


# ============================================================
# SMALL HELPERS
# ============================================================

def check(name, passed, detail, critical=True):
    """One validation result. `detail` must be plain text, not a matrix."""

    return {
        "name": name,
        "passed": bool(passed),
        "critical": bool(critical),
        "detail": str(detail),
    }


def overall_status(checks, warnings):
    """FAILED if any check failed. Warnings do not override a failure."""

    if any(not item["passed"] for item in checks):
        return "FAILED"

    if warnings:
        return "PASSED_WITH_WARNINGS"

    return "PASSED"


def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def snapshot_files(paths):
    """Size, mtime and hash of files this stage must not overwrite."""

    recorded = {}

    for path in paths:

        if not path.exists():
            recorded[str(path.relative_to(PROJECT_ROOT))] = {"exists": False}
            continue

        stat = path.stat()
        recorded[str(path.relative_to(PROJECT_ROOT))] = {
            "exists": True,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": file_sha256(path),
        }

    return recorded


def stored_bool(value):

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"

    raise ValueError(f"invalid boolean flag {value!r}")


def stored_int(value):

    if isinstance(value, bool):
        raise ValueError(f"invalid count {value!r}")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)

    raise ValueError(f"invalid count {value!r}")


def parse_pytest_summary(output):
    """Read pytest's final summary line. Missing counts are zero."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    summary = lines[-1] if lines else ""

    def grab(word):
        match = re.search(rf"(\d+) {word}", summary)
        return int(match.group(1)) if match else 0

    return {
        "summary_line": summary,
        "passed": grab("passed"),
        "failed": grab("failed"),
        "skipped": grab("skipped"),
        "errors": grab("errors?"),
    }


def edges_missing_from_posts(posts, edges):
    """
    Rows in edges.csv whose parent or child is not a post in that thread.

    Graph export skips those rows, so the graph files can contain fewer
    edges than edges.csv. This does not rebuild any graph.
    """

    ids_by_root = {}

    for post_id, root_id in zip(posts["post_id"].tolist(), posts["root_id"].tolist()):
        ids_by_root.setdefault(root_id, set()).add(post_id)

    missing = []

    for root_id, parent_id, child_id in zip(
        edges["root_id"].tolist(),
        edges["parent_id"].tolist(),
        edges["child_id"].tolist(),
    ):
        known = ids_by_root.get(root_id, set())
        parent_ok = parent_id in known
        child_ok = child_id in known

        if parent_ok and child_ok:
            continue

        missing.append({
            "root_id": root_id,
            "parent_id": parent_id,
            "child_id": child_id if len(child_id) <= 40 else child_id[:40] + "...",
            "child_id_length": len(child_id),
            "child_id_is_digits": bool(re.fullmatch(r"\d+", child_id)),
            "parent_in_thread": parent_ok,
            "child_in_thread": child_ok,
        })

    return missing


def compare_id_sets(named_sets):
    """
    Compare ID sets by value.

    The first set is the reference. Row order is ignored. A set comparison
    cannot see duplicates, so callers check uniqueness separately.
    """

    names = list(named_sets)
    reference_name = names[0]
    reference = named_sets[reference_name]
    rows = []
    passed = True

    for name in names[1:]:
        other = named_sets[name]
        missing = reference - other
        extra = other - reference

        if missing or extra:
            passed = False

        rows.append({
            "name": name,
            "count": len(other),
            "missing_from_this_set": len(missing),
            "extra_not_in_reference": len(extra),
            "missing_examples": sorted(missing)[:5],
            "extra_examples": sorted(extra)[:5],
        })

    return {
        "passed": passed,
        "reference": reference_name,
        "reference_count": len(reference),
        "sets": rows,
    }


def id_round_trip_through_float(post_id):
    """What this ID becomes if it is stored as a JSON number and read back."""

    return str(int(float(post_id)))


# ============================================================
# CLEANED TEXT
# ============================================================

def _finish_cleaned_scan(
    checks, warnings, counts, originals, bert_texts, tfidf_texts,
    post_id_list, flag_columns, progress,
):
    """Exact counters for the cleaned-text scan. Examples are capped at 5."""

    bert_mismatch_ids = []
    tfidf_mismatch_ids = []
    flag_mismatch_ids = []
    empty_ids = []
    mention_ids = []
    invalid_ids = []
    bert_mismatch = tfidf_mismatch = flag_mismatch = 0
    empty_bad = mention_bad = invalid = punctuation_only = 0

    for index, text in enumerate(originals):

        if progress and index and index % 20000 == 0:
            progress(f"rechecked cleaned text for {index} / {len(originals)} posts")

        post_id = post_id_list[index]
        bertweet = bert_texts[index]
        tfidf = tfidf_texts[index]

        if not isinstance(bertweet, str) or not isinstance(tfidf, str):
            invalid += 1
            if len(invalid_ids) < 5:
                invalid_ids.append(post_id)
            continue

        if normalize_for_bertweet(text) != bertweet:
            bert_mismatch += 1
            if len(bert_mismatch_ids) < 5:
                bert_mismatch_ids.append(post_id)

        if normalize_for_tfidf(text) != tfidf:
            tfidf_mismatch += 1
            if len(tfidf_mismatch_ids) < 5:
                tfidf_mismatch_ids.append(post_id)

        computed = text_flags(text)

        try:
            stored = {
                "num_urls": stored_int(flag_columns["num_urls"][index]),
                "num_mentions": stored_int(flag_columns["num_mentions"][index]),
                "num_hashtags": stored_int(flag_columns["num_hashtags"][index]),
                "num_emojis": stored_int(flag_columns["num_emojis"][index]),
                "starts_with_rt": stored_bool(flag_columns["starts_with_rt"][index]),
                "is_empty_text": stored_bool(flag_columns["is_empty_text"][index]),
                "only_mentions_urls": stored_bool(
                    flag_columns["only_mentions_urls"][index]
                ),
            }
        except ValueError:
            invalid += 1
            if len(invalid_ids) < 5:
                invalid_ids.append(post_id)
            continue

        if stored != computed:
            flag_mismatch += 1
            if len(flag_mismatch_ids) < 5:
                flag_mismatch_ids.append(post_id)

        is_empty = stored["is_empty_text"]
        mentions_only = stored["only_mentions_urls"]

        if is_empty and (bertweet != "" or tfidf != ""):
            empty_bad += 1
            if len(empty_ids) < 5:
                empty_ids.append(post_id)

        if (not is_empty) and bertweet == "":
            empty_bad += 1
            if len(empty_ids) < 5:
                empty_ids.append(post_id)

        if mentions_only:
            tokens = tfidf.split()
            tokens_ok = bool(tokens) and all(token in TFIDF_PLACEHOLDERS for token in tokens)
            if is_empty or not tokens_ok:
                mention_bad += 1
                if len(mention_ids) < 5:
                    mention_ids.append(post_id)

        if (not is_empty) and tfidf == "" and bertweet != "":
            punctuation_only += 1

    checks.append(check(
        "preprocessing flags match the original text",
        flag_mismatch == 0 and invalid == 0,
        f"rows whose stored flags differ from text_flags(text_original): {flag_mismatch}"
        f"{_examples(flag_mismatch_ids)}; invalid flag values: {invalid}"
        f"{_examples(invalid_ids)}",
    ))
    checks.append(check(
        "empty-text flags agree with cleaned text",
        empty_bad == 0,
        "rows where is_empty_text does not agree with text_bertweet / text_tfidf: "
        f"{empty_bad}{_examples(empty_ids)}",
    ))
    checks.append(check(
        "mentions/URL-only flags agree with TF-IDF tokens",
        mention_bad == 0,
        "only_mentions_urls rows whose text_tfidf is not made only of "
        f"{sorted(TFIDF_PLACEHOLDERS)}: {mention_bad}{_examples(mention_ids)}",
    ))
    checks.append(check(
        "text_bertweet and text_tfidf are reproducible",
        bert_mismatch == 0 and tfidf_mismatch == 0,
        "Reran the Stage 2 normalisers on text_original. "
        f"text_bertweet mismatches: {bert_mismatch}{_examples(bert_mismatch_ids)}; "
        f"text_tfidf mismatches: {tfidf_mismatch}{_examples(tfidf_mismatch_ids)}",
    ))

    counts["punctuation_only_tfidf"] = punctuation_only
    counts["flag_mismatches"] = flag_mismatch
    counts["bertweet_mismatches"] = bert_mismatch
    counts["tfidf_mismatches"] = tfidf_mismatch

    if punctuation_only:
        noun = "post" if punctuation_only == 1 else "posts"
        verb = "has" if punctuation_only == 1 else "have"
        warnings.append(
            f"{punctuation_only} {noun} {verb} non-empty BERTweet text but empty "
            "TF-IDF text. Punctuation-only tweets become an empty bag of words "
            "after TF-IDF cleaning, so those rows will be all zeros once a "
            "vectorizer is fit. This was left unchanged."
        )

    return {"checks": checks, "warnings": warnings, "counts": counts}


def _examples(ids):

    if not ids:
        return ""

    return f", e.g. {ids}"


def validate_cleaned_posts(posts, clean, expected_posts=None, progress=None):
    """
    Compare cleaned posts with the original posts by post_id.

    The Stage 2 normalisers are rerun on every original text. Nothing is written.
    """

    checks = []
    warnings = []
    counts = {
        "posts": int(len(posts)),
        "cleaned_posts": int(len(clean)),
    }

    missing_posts_columns = [c for c in REQUIRED_COLUMNS if c not in posts.columns]
    missing_clean_columns = [c for c in OUTPUT_COLUMNS if c not in clean.columns]
    extra_clean_columns = [c for c in clean.columns if c not in OUTPUT_COLUMNS]
    checks.append(check(
        "cleaned text columns are present",
        not missing_posts_columns and not missing_clean_columns and not extra_clean_columns,
        "missing from posts.csv: "
        f"{missing_posts_columns or 'none'}; "
        "missing from posts_clean.csv: "
        f"{missing_clean_columns or 'none'}; "
        f"unexpected cleaned columns: {extra_clean_columns or 'none'}",
    ))

    if missing_posts_columns or missing_clean_columns:
        return {"checks": checks, "warnings": warnings, "counts": counts}

    same_rows = len(posts) == len(clean)
    expected_ok = expected_posts is None or len(posts) == expected_posts
    checks.append(check(
        "posts and cleaned posts have the same number of rows",
        same_rows and expected_ok,
        f"posts.csv has {len(posts)} rows, posts_clean.csv has {len(clean)} rows"
        + (f", expected {expected_posts}" if expected_posts is not None else ""),
    ))

    duplicate_posts = int(posts["post_id"].duplicated().sum())
    duplicate_clean = int(clean["post_id"].duplicated().sum())
    empty_posts = int((posts["post_id"] == "").sum())
    empty_clean = int((clean["post_id"] == "").sum())
    non_digit = [
        post_id for post_id in clean["post_id"].tolist()
        if not re.fullmatch(r"\d+", post_id)
    ]
    checks.append(check(
        "post_id values are unique, non-empty digit strings",
        duplicate_posts == 0 and duplicate_clean == 0
        and empty_posts == 0 and empty_clean == 0 and not non_digit,
        f"duplicate original IDs: {duplicate_posts}; "
        f"duplicate cleaned IDs: {duplicate_clean}; "
        f"empty original IDs: {empty_posts}; empty cleaned IDs: {empty_clean}; "
        f"non-digit cleaned IDs: {len(non_digit)}"
        + (f", e.g. {non_digit[:5]}" if non_digit else ""),
    ))

    lengths = {}
    for post_id in clean["post_id"].tolist():
        if re.fullmatch(r"\d+", post_id):
            lengths[len(post_id)] = lengths.get(len(post_id), 0) + 1
    counts["post_id_length_counts"] = {str(k): lengths[k] for k in sorted(lengths)}

    if any(length != 18 for length in lengths):
        warnings.append(
            "Not every post_id is 18 digits. Length counts: "
            f"{counts['post_id_length_counts']}. IDs were still compared as "
            "exact strings, which is what keeps Twitter IDs aligned."
        )

    if duplicate_posts or duplicate_clean:
        return {"checks": checks, "warnings": warnings, "counts": counts}

    posts_ids = set(posts["post_id"])
    clean_ids = set(clean["post_id"])
    id_report = compare_id_sets({
        "posts.csv": posts_ids,
        "posts_clean.csv": clean_ids,
    })
    checks.append(check(
        "cleaned post_id set equals the original post_id set",
        id_report["passed"],
        json.dumps(id_report["sets"]),
    ))

    if id_report["passed"]:
        posts_by_id = posts.set_index("post_id", drop=False)
        shared_ids = clean["post_id"].tolist()
        root_mismatch = sum(
            a != b for a, b in zip(
                posts_by_id.loc[shared_ids, "root_id"].tolist(),
                clean["root_id"].tolist(),
            )
        )
        type_mismatch = sum(
            a != b for a, b in zip(
                posts_by_id.loc[shared_ids, "post_type"].tolist(),
                clean["post_type"].tolist(),
            )
        )
        text_mismatch = sum(
            a != b for a, b in zip(
                posts_by_id.loc[shared_ids, "text"].tolist(),
                clean["text_original"].tolist(),
            )
        )
        same_order = posts["post_id"].tolist() == clean["post_id"].tolist()
        counts["same_row_order_as_posts_csv"] = same_order
        checks.append(check(
            "root_id, post_type, and original text match by post_id",
            root_mismatch == 0 and type_mismatch == 0 and text_mismatch == 0,
            f"root_id mismatches: {root_mismatch}; "
            f"post_type mismatches: {type_mismatch}; "
            f"text mismatches: {text_mismatch}. "
            "Compared by post_id, not by row position. "
            f"Row order happens to match: {same_order}.",
        ))
    else:
        checks.append(check(
            "root_id, post_type, and original text match by post_id",
            False,
            "Not compared because the post_id sets differ.",
        ))

    flag_columns = {column: clean[column].tolist() for column in [
        "num_urls", "num_mentions", "num_hashtags", "num_emojis",
        "starts_with_rt", "is_empty_text", "only_mentions_urls",
    ]}

    return _finish_cleaned_scan(
        checks, warnings, counts,
        clean["text_original"].tolist(),
        clean["text_bertweet"].tolist(),
        clean["text_tfidf"].tolist(),
        clean["post_id"].tolist(),
        flag_columns,
        progress,
    )


# ============================================================
# TF-IDF
# ============================================================

def _synthetic_tfidf_posts():
    """Small stand-in for posts_clean.csv. Not a real PHEME split."""

    return pd.DataFrame({
        "post_id": ["101", "102", "103", "201", "202", "301", "302"],
        "root_id": ["101", "101", "101", "201", "201", "301", "301"],
        "post_type": [
            "source", "reaction", "reaction",
            "source", "reaction", "source", "reaction",
        ],
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


def _raises(func, text):

    try:
        func()
    except ValueError as error:
        return text in str(error), str(error)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"

    return False, "no error was raised"


def validate_tfidf_guarantees():
    """
    Fit TF-IDF on a tiny synthetic training set.

    The full PHEME collection is never vectorised here, and no official
    train/validation/test split is created.
    """

    checks = []
    warnings = []
    posts = _synthetic_tfidf_posts()
    train = select_posts(posts, root_ids=["101", "201"])
    test = select_posts(posts, root_ids=["301"])

    vectorizer = make_vectorizer()
    params = vectorizer.get_params()
    design_ok = (
        params["ngram_range"] == (1, 2)
        and params["token_pattern"] == r"(?u)\S+"
        and params["lowercase"] is False
        and params["stop_words"] is None
        and params["strip_accents"] is None
        and params["min_df"] == 2
        and params["max_df"] == 1.0
        and params["max_features"] is None
        and params["sublinear_tf"] is True
        and params["norm"] == "l2"
        and np.dtype(params["dtype"]) == np.dtype(np.float32)
        and DEFAULT_TFIDF_PARAMS["stop_words"] is None
    )
    checks.append(check(
        "TF-IDF parameters match the Stage 3 design",
        design_ok,
        "unigrams and bigrams, token pattern (?u)\\S+, no stop-word list, "
        "no accent stripping, min_df 2, sublinear term frequency, L2 norm, "
        f"float32. lowercase={params['lowercase']}.",
    ))

    fitted, _ = fit_tfidf(train, min_df=1)
    vocabulary = fitted.vocabulary_
    checks.append(check(
        "vocabulary is fit on training text only",
        "soldado" not in vocabulary and "muerto" not in vocabulary and "police" in vocabulary,
        "The test-only words 'soldado' and 'muerto' are absent from the "
        "vocabulary. 'police', which appears in training, is present.",
    ))

    idf_before = np.array(fitted.idf_, copy=True)
    vocabulary_before = dict(vocabulary)
    transform_tfidf(fitted, test)
    idf_unchanged = np.array_equal(fitted.idf_, idf_before)
    vocabulary_unchanged = fitted.vocabulary_ == vocabulary_before
    checks.append(check(
        "transform does not change vocabulary or IDF",
        idf_unchanged and vocabulary_unchanged,
        "After transforming the held-out posts, vocabulary and idf_ were "
        f"unchanged (vocabulary {vocabulary_unchanged}, IDF {idf_unchanged}).",
    ))

    combined = pd.concat([train, test], ignore_index=True)
    refit, _ = fit_tfidf(combined, min_df=1)
    checks.append(check(
        "refitting on train plus test would change the vocabulary",
        "soldado" in refit.vocabulary_ and "soldado" not in fitted.vocabulary_,
        "Fitting on train+test adds 'soldado'. Transforming the test set "
        "with the training vectorizer does not. Validation and test text "
        "must go through transform, not fit.",
    ))

    repeated = pd.DataFrame({
        "post_id": ["1", "2"],
        "root_id": ["1", "1"],
        "text_tfidf": ["alpha alpha alpha beta", "alpha beta"],
    })
    _, sublinear = fit_tfidf(repeated, min_df=1, sublinear_tf=True)
    _, linear = fit_tfidf(repeated, min_df=1, sublinear_tf=False)
    checks.append(check(
        "sublinear term frequency is active",
        not np.allclose(sublinear.toarray(), linear.toarray()),
        "The same repeated-word documents produce different weights with "
        "sublinear_tf True and False. The project default is True.",
    ))

    default_fit, _ = fit_tfidf(train)
    checks.append(check(
        "default min_df drops terms seen in only one training post",
        "police" in default_fit.vocabulary_ and "gunman" not in default_fit.vocabulary_,
        "'police' appears in two training posts and is kept. 'gunman' appears "
        "in one and is dropped by min_df=2.",
    ))

    _, dense_ready = fit_tfidf(train, min_df=1)
    norms = np.sqrt(dense_ready.multiply(dense_ready).sum(axis=1)).A1
    checks.append(check(
        "training rows are L2 normalised",
        bool(np.allclose(norms, 1.0, atol=1e-5)),
        f"L2 norms of the {len(norms)} synthetic training rows are all 1 "
        "within 1e-5.",
    ))

    checks.append(check(
        "unigrams and bigrams are both created",
        "not" in vocabulary and "not true" in vocabulary,
        "With min_df=1 the training vocabulary contains the unigram 'not' "
        "and the bigram 'not true'.",
    ))
    checks.append(check(
        "stop words are kept",
        "the" in vocabulary and "is" in vocabulary,
        "With min_df=1 the words 'the' and 'is' stay in the vocabulary. "
        "stop_words is None.",
    ))

    overlap_ok, overlap_detail = _raises(
        lambda: check_no_overlap(
            train, select_posts(posts, post_ids=["101", "301"]), "test"
        ),
        "appear in both",
    )
    checks.append(check(
        "train/test post_id overlap is rejected",
        overlap_ok,
        overlap_detail,
    ))

    unknown_ok, unknown_detail = _raises(
        lambda: select_posts(posts, post_ids=["999"]),
        "not found",
    )
    checks.append(check(
        "unknown post_ids are rejected",
        unknown_ok,
        unknown_detail,
    ))

    duplicate_ok, duplicate_detail = _raises(
        lambda: select_posts(posts, post_ids=["101", "101"]),
        "duplicate",
    )
    doubled = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    fit_duplicate_ok, fit_duplicate_detail = _raises(
        lambda: fit_tfidf(doubled, min_df=1),
        "duplicate",
    )
    checks.append(check(
        "duplicate post_ids are rejected",
        duplicate_ok and fit_duplicate_ok,
        f"select_posts: {duplicate_detail}; fit_tfidf: {fit_duplicate_detail}",
    ))

    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        fitted_again, matrix = fit_tfidf(train, min_df=1)
        save_features(output_dir, "train", matrix, train)
        info = save_vectorizer(fitted_again, output_dir, train)
        loaded_matrix, loaded_ids = load_features(output_dir, "train")
        loaded_vectorizer = load_vectorizer(output_dir)
        consistent = (
            loaded_matrix.shape == matrix.shape
            and (loaded_matrix != matrix).nnz == 0
            and loaded_ids["post_id"].tolist() == train["post_id"].tolist()
            and loaded_ids["root_id"].tolist() == train["root_id"].tolist()
            and loaded_ids["row_index"].tolist() == [str(i) for i in range(len(train))]
            and loaded_vectorizer.vocabulary_ == fitted_again.vocabulary_
            and info["vocabulary_size"] == matrix.shape[1] == len(fitted_again.vocabulary_)
            and np.allclose(
                transform_tfidf(loaded_vectorizer, test).toarray(),
                transform_tfidf(fitted_again, test).toarray(),
            )
        )
        edited = pd.read_csv(output_dir / "train_ids.csv", dtype=str, keep_default_na=False)
        edited.loc[len(edited) - 1, "post_id"] = edited.loc[0, "post_id"]
        edited.to_csv(output_dir / "train_ids.csv", index=False)
        load_rejected_duplicate = False
        try:
            load_features(output_dir, "train")
        except ValueError:
            load_rejected_duplicate = True

    checks.append(check(
        "saved matrix, ID mapping, vectorizer, and metadata agree",
        consistent,
        "On the synthetic training set, the reloaded sparse matrix, row IDs, "
        "vocabulary size, and transformed test rows match the objects that "
        "were saved.",
    ))

    if not load_rejected_duplicate:
        warnings.append(
            "load_features checks that the matrix row count matches the ID "
            "file, but it does not reject a duplicated post_id inside that "
            "file. select_posts and fit_tfidf already reject duplicates "
            "before anything is saved. Do not hand-edit saved ID files."
        )

    return {"checks": checks, "warnings": warnings}


# ============================================================
# EMBEDDINGS
# ============================================================

def find_nonfinite(embeddings, chunk_size=8192):
    """Count NaN and infinite values without loading the whole matrix at once."""

    count = 0
    first = None
    n_rows = int(embeddings.shape[0])

    for start in range(0, n_rows, chunk_size):
        block = np.asarray(embeddings[start:start + chunk_size])
        bad = ~np.isfinite(block)
        found = int(bad.sum())

        if found and first is None:
            coords = np.argwhere(bad)[0]
            first = {
                "row": int(start + coords[0]),
                "column": int(coords[1]),
            }

        count += found

    return {"count": count, "first": first}


def validate_embedding_metadata(metadata, n_rows, n_ids):

    checks = []
    required = {
        "status": "complete",
        "model_name": BERTWEET_MODEL_NAME,
        "model_revision": BERTWEET_REVISION,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "max_length": BERTWEET_MAX_LENGTH,
        "text_column": "text_bertweet",
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    missing = [key for key, value in required.items() if metadata.get(key) != value]
    pooling = str(metadata.get("pooling", ""))
    pooling_ok = "mean" in pooling.lower()
    checks.append(check(
        "extraction metadata records a completed run",
        metadata.get("status") == "complete" and not missing and pooling_ok
        and metadata.get("truncation") is True
        and metadata.get("num_posts") == n_rows == n_ids,
        "Mismatched settings: "
        f"{missing or 'none'}. Pooling mentions mean: {pooling_ok}. "
        f"num_posts={metadata.get('num_posts')}, matrix rows={n_rows}, "
        f"ID rows={n_ids}.",
    ))
    checks.append(check(
        "embedding settings match the project configuration",
        not missing and pooling_ok and metadata.get("max_length") == BERTWEET_MAX_LENGTH,
        f"model {metadata.get('model_name')} @ {metadata.get('model_revision')}; "
        f"dim {metadata.get('embedding_dim')}; max_length {metadata.get('max_length')}; "
        f"pooling: {pooling}",
    ))

    return checks


def check_identical_texts(store, ids, clean):
    """Posts with the same cleaned BERTweet text must share one embedding."""

    known_text = dict(zip(clean["post_id"].tolist(), clean["text_bertweet"].tolist()))
    missing = [post_id for post_id in ids["post_id"].tolist() if post_id not in known_text]

    if missing:
        return check(
            "identical cleaned texts have identical embeddings",
            False,
            f"{len(missing)} embedding IDs are not in the cleaned posts, e.g. {missing[:5]}.",
        ), 0, 0

    text_by_id = known_text
    rows_by_text = {}

    for row, post_id in enumerate(ids["post_id"].tolist()):
        rows_by_text.setdefault(text_by_id[post_id], []).append((row, post_id))

    groups = [rows for rows in rows_by_text.values() if len(rows) > 1]
    mismatched_groups = 0
    examples = []

    for rows in groups:
        base = np.asarray(store.embeddings[rows[0][0]])

        for row, post_id in rows[1:]:
            if not np.array_equal(base, np.asarray(store.embeddings[row])):
                mismatched_groups += 1
                if len(examples) < 5:
                    examples.append([rows[0][1], post_id])
                break

    posts_in_groups = sum(len(rows) for rows in groups)

    return check(
        "identical cleaned texts have identical embeddings",
        mismatched_groups == 0,
        f"{len(groups)} cleaned texts are shared by {posts_in_groups} posts. "
        f"Groups whose embedding rows differ: {mismatched_groups}"
        + (f", e.g. post_ids {examples}" if examples else "")
        + ". Matching vectors are expected. The ID map still has one row per post.",
    ), len(groups), posts_in_groups


# ============================================================
# GRAPHS
# ============================================================

def build_edge_index(node_ids, edges):
    """
    Build a (2, num_edges) index from the edges list.

    `parent_id` is intentionally not an argument. Endpoints that are not
    exact string node IDs produce an error and no index.
    """

    errors = []
    index = {}

    for position, post_id in enumerate(node_ids):
        if not isinstance(post_id, str) or post_id == "" or post_id != post_id.strip():
            errors.append(f"node {position} post_id is not a clean string")
            continue
        if post_id in index:
            errors.append(f"duplicate node post_id {post_id}")
            continue
        index[post_id] = position

    if not isinstance(edges, list):
        return None, errors + ["edges is not a list"]

    sources = []
    targets = []

    for edge_number, edge in enumerate(edges):

        if not isinstance(edge, dict):
            errors.append(f"edge {edge_number} is not an object")
            continue

        source = edge.get("source")
        target = edge.get("target")

        for role, value in (("source", source), ("target", target)):
            if not isinstance(value, str) or value == "" or value != value.strip():
                errors.append(
                    f"edge {edge_number} {role} is {type(value).__name__}, not a clean string"
                )
            elif value not in index:
                errors.append(
                    f"edge {edge_number} {role} {value!r} is not a node post_id"
                )

        if (
            isinstance(source, str) and isinstance(target, str)
            and source in index and target in index
        ):
            sources.append(index[source])
            targets.append(index[target])

    if errors:
        return None, errors

    edge_index = np.empty((2, len(sources)), dtype=np.int64)

    if sources:
        edge_index[0] = sources
        edge_index[1] = targets

        if (
            int(edge_index.min()) < 0
            or int(edge_index.max()) >= len(node_ids)
        ):
            return None, ["edge index is outside the node list"]

    return edge_index, []


def _parent_as_id(parent):
    """Diagnostic only. Not used to build edges."""

    if parent is None or isinstance(parent, bool):
        return None

    if isinstance(parent, float):
        if not math.isfinite(parent):
            return None
        return str(int(parent))

    if isinstance(parent, int):
        return str(parent)

    if isinstance(parent, str):
        return parent.strip()

    return None


def _parent_stats(nodes, edges):

    non_null = float_count = int_count = string_count = other = 0
    parent_of = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue

        parent = node.get("parent_id", None)
        post_id = node.get("post_id")

        if isinstance(post_id, str):
            parent_of[post_id] = parent

        if parent is None:
            continue

        non_null += 1

        if isinstance(parent, float):
            float_count += 1
        elif isinstance(parent, bool):
            other += 1
        elif isinstance(parent, int):
            int_count += 1
        elif isinstance(parent, str):
            string_count += 1
        else:
            other += 1

    compared = disagreements = 0

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        source = edge.get("source")
        target = edge.get("target")

        if not isinstance(source, str) or not isinstance(target, str):
            continue

        if target not in parent_of:
            continue

        compared += 1

        if _parent_as_id(parent_of[target]) != source:
            disagreements += 1

    return {
        "parent_id_non_null": non_null,
        "parent_id_float": float_count,
        "parent_id_int": int_count,
        "parent_id_string": string_count,
        "parent_id_other": other,
        "parent_edges_compared": compared,
        "parent_edge_disagreements": disagreements,
    }


def inspect_graph(graph):
    """Read nodes and edges. Connectivity comes only from the edges list."""

    schema_errors = []
    nodes = graph.get("nodes")
    edges = graph.get("edges")

    if not isinstance(nodes, list):
        return {
            "node_ids": None,
            "num_nodes": 0,
            "num_edges": 0,
            "empty": False,
            "single_node": False,
            "no_edges": False,
            "unsorted": False,
            "edge_index": None,
            "schema_errors": ["schema: graph has no nodes list"],
            "edge_errors": [],
            **_parent_stats([], []),
        }

    if not isinstance(edges, list):
        schema_errors.append("schema: graph has no edges list")
        edges = []

    node_ids = []
    bad_node = False

    for position, node in enumerate(nodes):
        post_id = node.get("post_id") if isinstance(node, dict) else None

        if not isinstance(post_id, str) or post_id == "" or post_id != post_id.strip():
            schema_errors.append(f"schema: node {position} post_id is not a clean string")
            bad_node = True
        else:
            node_ids.append(post_id)

    if not bad_node and len(node_ids) != len(set(node_ids)):
        schema_errors.append("schema: duplicate node post_ids")

    if "num_nodes" in graph and graph["num_nodes"] != len(nodes):
        schema_errors.append(
            f"schema: num_nodes is {graph.get('num_nodes')} but the nodes list has {len(nodes)}"
        )

    if "num_edges" in graph and graph["num_edges"] != len(graph.get("edges", [])):
        schema_errors.append(
            f"schema: num_edges is {graph.get('num_edges')} but the edges list has "
            f"{len(graph.get('edges', []))}"
        )

    edge_index = None
    edge_errors = []

    if not bad_node and not any("duplicate node" in msg for msg in schema_errors):
        edge_index, built_errors = build_edge_index(node_ids, edges)
        edge_errors = [f"edge: {msg}" for msg in built_errors]

    return {
        "node_ids": None if bad_node else node_ids,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "empty": len(nodes) == 0,
        "single_node": len(nodes) == 1,
        "no_edges": len(nodes) > 0 and len(edges) == 0,
        "unsorted": (not bad_node) and node_ids != sorted(node_ids),
        "edge_index": edge_index,
        "schema_errors": schema_errors,
        "edge_errors": edge_errors,
        **_parent_stats(nodes, edges),
    }


def validate_graph_against_store(graph, store, root_by_post=None):
    """Check node order, embedding rows, and edges for one graph dict."""

    inspected = inspect_graph(graph)
    alignment_errors = []
    root_errors = []
    node_ids = inspected["node_ids"]
    ids_usable = node_ids is not None and not any(
        msg.startswith("schema: node") or "duplicate node" in msg
        for msg in inspected["schema_errors"]
    )

    if ids_usable:
        try:
            returned, features = get_graph_node_embeddings(graph, store)
        except EmbeddingLookupError as error:
            alignment_errors.append(f"alignment: {error}".split("\n")[0][:400])
        else:
            if returned != node_ids:
                alignment_errors.append(
                    "alignment: helper node order differs from the nodes list"
                )

            if features.dtype != np.float32 or features.shape != (len(node_ids), store.dim):
                alignment_errors.append(
                    f"alignment: features are shape {tuple(features.shape)} "
                    f"dtype {features.dtype}"
                )
            elif len(node_ids):
                rows = np.fromiter(
                    (store.index[post_id] for post_id in node_ids),
                    dtype=np.int64,
                    count=len(node_ids),
                )

                if not np.array_equal(features, np.asarray(store.embeddings[rows])):
                    alignment_errors.append(
                        "alignment: a returned row is not the embedding stored for that post_id"
                    )
    elif node_ids is None:
        alignment_errors.append("alignment: not checked because a node post_id is invalid")

    if root_by_post is not None and ids_usable:
        root_id = graph.get("root_id")

        if not isinstance(root_id, str):
            root_errors.append("root: graph root_id is not a string")
        else:
            for post_id in node_ids:
                other = root_by_post.get(post_id)

                if other != root_id:
                    root_errors.append(
                        f"root: post {post_id} has root_id {other!r}, "
                        f"graph root_id is {root_id!r}"
                    )
                    break

    inspected["alignment_errors"] = alignment_errors
    inspected["root_errors"] = root_errors
    inspected["ok"] = not (
        inspected["schema_errors"]
        or inspected["edge_errors"]
        or alignment_errors
        or root_errors
    )

    return inspected


def partial_order_matches(store, node_ids):
    """A subset of nodes must come back in the requested order."""

    if len(node_ids) < 3:
        return None

    requested = [node_ids[2], node_ids[0], node_ids[1]]
    partial = get_node_embeddings(requested, store)
    full = get_node_embeddings(node_ids, store)

    return bool(np.array_equal(partial, full[[2, 0, 1]]))


def demo_graph_cases(store):
    """
    Small graphs, including ones with a misleading parent_id.

    These do not read Person 1's files. On the real store they only look up
    a few existing post_ids.
    """

    ids = list(store.index)
    empty = {
        "root_id": "unused",
        "num_nodes": 0,
        "num_edges": 0,
        "nodes": [],
        "edges": [],
    }
    single = {
        "root_id": "unused",
        "num_nodes": 1,
        "num_edges": 0,
        "nodes": [{"post_id": ids[0], "parent_id": None}],
        "edges": [],
    }
    no_edge = {
        "root_id": "unused",
        "num_nodes": 2,
        "num_edges": 0,
        "nodes": [
            {"post_id": ids[0], "parent_id": None},
            {"post_id": ids[1], "parent_id": 1.0},
        ],
        "edges": [],
    }
    misleading = {
        "root_id": "unused",
        "num_nodes": 2,
        "num_edges": 1,
        "nodes": [
            {"post_id": ids[0], "parent_id": None},
            {"post_id": ids[1], "parent_id": 123.0},
        ],
        "edges": [{"source": ids[0], "target": ids[1]}],
    }
    unknown = {
        "root_id": "unused",
        "num_nodes": 2,
        "num_edges": 1,
        "nodes": [{"post_id": ids[0]}, {"post_id": ids[1]}],
        "edges": [{"source": ids[0], "target": "999999999999999999"}],
    }
    numeric = {
        "root_id": "unused",
        "num_nodes": 2,
        "num_edges": 1,
        "nodes": [{"post_id": ids[0]}, {"post_id": ids[1]}],
        "edges": [{"source": ids[0], "target": 5}],
    }

    misleading_result = validate_graph_against_store(misleading, store)
    edge_index = misleading_result["edge_index"]
    full = get_node_embeddings(ids[:4], store)
    reversed_features = get_node_embeddings(list(reversed(ids[:4])), store)

    return {
        "empty": validate_graph_against_store(empty, store)["ok"],
        "single": validate_graph_against_store(single, store)["ok"],
        "no_edge": validate_graph_against_store(no_edge, store)["ok"],
        "misleading_parent_still_aligned": (
            misleading_result["ok"]
            and misleading_result["parent_edge_disagreements"] == 1
            and edge_index is not None
            and edge_index.shape == (2, 1)
            and int(edge_index[0, 0]) == 0
            and int(edge_index[1, 0]) == 1
        ),
        "unknown_endpoint_rejected": not validate_graph_against_store(unknown, store)["ok"],
        "numeric_endpoint_rejected": not validate_graph_against_store(numeric, store)["ok"],
        "partial_order": partial_order_matches(store, ids[:4]) is True,
        "reversal_follows_request": bool(np.array_equal(full, reversed_features[::-1])),
    }


def validate_all_graphs(store, root_by_post, metadata, progress=None):
    """Walk every graph JSON once. Embeddings are looked up, not recomputed."""

    files = sorted(GRAPHS_DIR.glob("graph_*.json"))
    seen = {}
    problems = []
    labels = {}
    totals = {
        "graph_files": len(files),
        "graphs_read": 0,
        "schema_error_graphs": 0,
        "edge_error_graphs": 0,
        "alignment_error_graphs": 0,
        "root_error_graphs": 0,
        "metadata_mismatch_graphs": 0,
        "missing_metadata": 0,
        "empty_graphs": 0,
        "single_node_graphs": 0,
        "no_edge_graphs": 0,
        "single_node_error_graphs": 0,
        "no_edge_error_graphs": 0,
        "unsorted_graphs": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "parent_id_non_null": 0,
        "parent_id_float": 0,
        "parent_id_int": 0,
        "parent_id_string": 0,
        "parent_edges_compared": 0,
        "parent_edge_disagreements": 0,
    }
    partial = {"graph": None, "passed": None}

    def note(name, message):
        if len(problems) < 8:
            problems.append({"graph": name, "error": message})

    for number, path in enumerate(files, start=1):

        if progress and number % 1000 == 0:
            progress(f"checked {number} / {len(files)} graphs")

        name = path.name
        meta = metadata.get(name)

        if meta is None:
            totals["missing_metadata"] += 1
            note(name, "no graph_metadata.csv row")

        try:
            graph = load_graph(path)
        except (OSError, json.JSONDecodeError) as error:
            totals["schema_error_graphs"] += 1
            note(name, f"unreadable JSON: {error}")
            continue

        totals["graphs_read"] += 1
        result = validate_graph_against_store(graph, store, root_by_post)
        totals["total_nodes"] += result["num_nodes"]
        totals["total_edges"] += result["num_edges"]
        totals["parent_id_non_null"] += result["parent_id_non_null"]
        totals["parent_id_float"] += result["parent_id_float"]
        totals["parent_id_int"] += result["parent_id_int"]
        totals["parent_id_string"] += result["parent_id_string"]
        totals["parent_edges_compared"] += result["parent_edges_compared"]
        totals["parent_edge_disagreements"] += result["parent_edge_disagreements"]

        if result["empty"]:
            totals["empty_graphs"] += 1
        if result["single_node"]:
            totals["single_node_graphs"] += 1
        if result["no_edges"]:
            totals["no_edge_graphs"] += 1
        if result["unsorted"]:
            totals["unsorted_graphs"] += 1

        if result["schema_errors"]:
            totals["schema_error_graphs"] += 1
            note(name, result["schema_errors"][0])
        if result["edge_errors"]:
            totals["edge_error_graphs"] += 1
            note(name, result["edge_errors"][0])
        if result["alignment_errors"]:
            totals["alignment_error_graphs"] += 1
            note(name, result["alignment_errors"][0])
            if result["single_node"]:
                totals["single_node_error_graphs"] += 1
            if result["no_edges"]:
                totals["no_edge_error_graphs"] += 1
        if result["root_errors"]:
            totals["root_error_graphs"] += 1
            note(name, result["root_errors"][0])

        if meta is not None:
            meta_ok = (
                meta["root_id"] == graph.get("root_id")
                and meta["num_nodes"] == result["num_nodes"]
                and meta["num_edges"] == result["num_edges"]
            )
            if not meta_ok:
                totals["metadata_mismatch_graphs"] += 1
                note(
                    name,
                    "metadata row disagrees with the JSON "
                    f"(meta nodes/edges {meta['num_nodes']}/{meta['num_edges']}, "
                    f"json {result['num_nodes']}/{result['num_edges']})",
                )
            labels[meta["label"]] = labels.get(meta["label"], 0) + 1

        if result["node_ids"] is not None and not result["alignment_errors"]:
            for post_id in result["node_ids"]:
                seen[post_id] = seen.get(post_id, 0) + 1

            if partial["graph"] is None and len(result["node_ids"]) >= 3:
                partial = {
                    "graph": name,
                    "passed": partial_order_matches(store, result["node_ids"]),
                }

    extra_metadata = sorted(set(metadata) - {path.name for path in files})
    multi = sum(1 for count in seen.values() if count > 1)
    unused = len(set(store.index) - set(seen))

    return {
        "totals": totals,
        "problems": problems,
        "labels": labels,
        "extra_metadata": extra_metadata[:8],
        "extra_metadata_count": len(extra_metadata),
        "distinct_post_ids": len(seen),
        "post_ids_in_more_than_one_graph": multi,
        "embedding_rows_not_used": unused,
        "partial_order": partial,
        "seen_ids": set(seen),
    }


# ============================================================
# SAFETY
# ============================================================

def git_output(args):

    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return completed.returncode, completed.stdout, completed.stderr


def is_git_ignored(relative):

    code, _, _ = git_output(["check-ignore", "-q", "--", relative])

    return code == 0


def scan_source_for_secrets():

    files = []

    for directory in SOURCE_SCAN_DIRS:
        root = PROJECT_ROOT / directory

        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".txt"}:
                files.append(path)

    for name in SOURCE_SCAN_FILES:
        files.append(PROJECT_ROOT / name)

    if NLP_DATA.exists():
        for path in NLP_DATA.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md"}:
                files.append(path)

    findings = []

    for path in files:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(PROJECT_ROOT))

        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{relative} matched a credential pattern")

        if HOME_PATH_PATTERN.search(text):
            findings.append(f"{relative} contains an absolute /Users path")

    return sorted(set(findings))


def dependency_check():

    import emoji
    import joblib
    import pytest
    import scipy
    import sklearn
    import torch
    import tqdm
    import transformers

    installed = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "emoji": emoji.__version__,
        "tqdm": tqdm.__version__,
        "pytest": pytest.__version__,
    }
    pinned = {}

    for line in (PROJECT_ROOT / "requirements-nlp.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        pinned[name.strip()] = version.strip()

    mismatched = [
        f"{name} installed {installed.get(name)} pinned {version}"
        for name, version in pinned.items()
        if installed.get(name) != version
    ]

    return installed, mismatched


def repository_checks():

    checks = []
    notes = []
    code, status_out, status_err = git_output(
        ["status", "--porcelain", "--", *PERSON1_PATHS]
    )
    checks.append(check(
        "Person 1 data files are unchanged in git",
        code == 0 and status_out.strip() == "",
        status_out.strip() or status_err.strip() or "no changes under data/processed, data/graphs, dataset, or docs/data_pipeline.md",
    ))

    ignored_failures = [
        path for path in LARGE_IGNORED_PATHS if not is_git_ignored(path)
    ]
    checks.append(check(
        "large generated files are git-ignored",
        not ignored_failures,
        "These large paths are not ignored: "
        f"{ignored_failures or 'none'}.",
    ))

    tracked_failures = [
        path for path in SMALL_TRACKED_PATHS if is_git_ignored(path)
    ]
    checks.append(check(
        "small validation metadata is not git-ignored",
        not tracked_failures,
        "Ignored even though they are small reports or source: "
        f"{tracked_failures or 'none'}.",
    ))

    findings = scan_source_for_secrets()
    checks.append(check(
        "project source contains no credentials or absolute home paths",
        not findings,
        "Scanned NLP source, docs, requirements, and data/nlp JSON/Markdown. "
        f"Findings: {findings or 'none'}. Graph JSON tweet text was not scanned.",
    ))

    _, mismatched = dependency_check()
    checks.append(check(
        "installed package versions match requirements-nlp.txt",
        not mismatched,
        "; ".join(mismatched) if mismatched else "All pinned NLP packages match the environment.",
    ))

    code, porcelain, _ = git_output(["status", "--porcelain", "--untracked-files=normal"])
    interesting = []

    for line in porcelain.splitlines():
        path = line[3:] if len(line) > 3 else line
        if path.startswith(".cache") or path.startswith(".venv"):
            continue
        interesting.append(line)

    if interesting:
        notes.append(
            "Git working tree notes (Stage 7 did not commit anything): "
            + "; ".join(interesting[:30])
            + (" ..." if len(interesting) > 30 else "")
        )

    return {"checks": checks, "notes": notes}


# ============================================================
# REPORT
# ============================================================

def to_plain(value):

    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]

    if isinstance(value, set):
        raise TypeError("report contains a set; store counts and a few examples instead")

    if isinstance(value, np.ndarray):
        raise TypeError("report contains an array; that does not belong in the summary")

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def render_markdown(report):

    lines = [
        "# Stage 7: End-to-End NLP Feature Validation",
        "",
        f"**Overall status: {report['overall_status']}**",
        "",
        report["status_meaning"],
        "",
        f"Validated at: {report['validated_at']}",
        "",
        "This stage only checked the NLP features that already exist. It did not train a classifier, train a graph neural network, fine-tune BERTweet, create an official train/validation/test split, or edit Person 1's posts, graphs, or preprocessing scripts.",
        "",
        "## Counts",
        "",
    ]

    for key, value in report["counts"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Checks", ""])
    lines.append("| Check | Result | What was found |")
    lines.append("| --- | --- | --- |")

    for item in report["checks"]:
        result = "passed" if item["passed"] else "FAILED"
        detail = item["detail"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {item['name']} | {result} | {detail} |")

    lines.extend(["", "## Failed checks", ""])

    if report["failed_checks"]:
        for item in report["failed_checks"]:
            lines.append(f"- **{item['name']}.** {item['detail']}")
    else:
        lines.append("No critical check failed.")

    lines.extend(["", "## Cross-stage ID consistency", ""])
    cross = report["cross_stage_ids"]
    lines.append(
        f"Reference set: {cross.get('reference')} ({cross.get('reference_count')} IDs). "
        "IDs were compared as exact strings. Row order was not assumed to match."
    )
    lines.append("")

    for row in cross.get("sets", []):
        lines.append(
            f"- {row['name']}: count {row['count']}, "
            f"missing {row['missing_from_this_set']}, "
            f"extra {row['extra_not_in_reference']}"
        )

    lines.extend(["", "## Reused checks", ""])

    for item in report["reused_checks"]:
        lines.append(
            f"- Stage {item['stage']}: {item['name']}. "
            f"Rerun: {item['rerun']}. Passed in the earlier report: {item['passed']}. "
            f"{item['justification']}"
        )

    lines.extend(["", "## Warnings and known limitations", ""])

    for warning in report["warnings"]:
        lines.append(f"- {warning}")

    if report["repository_notes"]:
        lines.extend(["", "## Repository notes", ""])
        for note in report["repository_notes"]:
            lines.append(f"- {note}")

    suite = report["test_suite"]
    lines.extend([
        "",
        "## Test suite",
        "",
        f"Command: `{suite.get('command', 'not run')}`",
        "",
        f"Passed: {suite.get('passed', 'not run')}. "
        f"Failed: {suite.get('failed', 'not run')}. "
        f"Skipped: {suite.get('skipped', 'not run')}. "
        f"Errors: {suite.get('errors', 'not run')}.",
        "",
        f"Summary: {suite.get('summary_line', '')}",
        "",
        "## Readiness",
        "",
        report["readiness"]["explanation"],
        "",
        f"Runtime: {report['runtime_seconds']} seconds.",
        "",
    ])

    return "\n".join(lines)


def run_pytest_suite():

    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "nlp/tests", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = parse_pytest_summary(completed.stdout + "\n" + completed.stderr)
    parsed["exit_code"] = completed.returncode
    parsed["seconds"] = round(time.perf_counter() - started, 1)
    parsed["command"] = ".venv/bin/python -m pytest nlp/tests -q"

    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
        parsed["failure_tail"] = tail[-30:]

    return parsed


def _load_table(path):

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _print(message):

    print(message, flush=True)


def run_validation(run_pytest=True):
    """Run every Stage 7 check and write the JSON and Markdown reports."""

    started_at = datetime.now().astimezone()
    t0 = time.perf_counter()
    checks = []
    warnings = []
    _print("Stage 7 validation started. Existing artifacts are only read.")

    artifact_paths = [
        POSTS_FILE,
        POSTS_CLEAN_FILE,
        POSTS_CLEAN_INFO_FILE,
        FULL_EMBEDDINGS_DIR / "embeddings.npy",
        FULL_EMBEDDINGS_DIR / "ids.csv",
        FULL_EMBEDDINGS_DIR / "metadata.json",
        STAGE6_REPORT,
    ]
    before = snapshot_files(artifact_paths)

    posts = _load_table(POSTS_FILE)
    clean = _load_table(POSTS_CLEAN_FILE)
    _print(f"Loaded {len(posts)} original posts and {len(clean)} cleaned posts.")
    cleaned = validate_cleaned_posts(
        posts, clean, expected_posts=EXPECTED_POSTS, progress=_print
    )
    checks.extend(cleaned["checks"])
    warnings.extend(cleaned["warnings"])

    _print("Checking TF-IDF behaviour on synthetic posts.")
    tfidf = validate_tfidf_guarantees()
    checks.extend(tfidf["checks"])
    warnings.extend(tfidf["warnings"])

    _print("Loading the embedding store with memory mapping.")
    store = load_embedding_store(FULL_EMBEDDINGS_DIR, mmap=True)
    ids = _load_table(FULL_EMBEDDINGS_DIR / "ids.csv")
    metadata = store.metadata
    mapped = isinstance(store.embeddings, np.memmap)
    checks.append(check(
        "embeddings are memory-mapped",
        mapped and store.embeddings.shape[1] == EMBEDDING_DIM,
        f"Array type: {type(store.embeddings).__name__}. "
        f"Shape: {tuple(store.embeddings.shape)}. Dtype: {store.embeddings.dtype}.",
    ))
    checks.extend(validate_embedding_metadata(metadata, store.embeddings.shape[0], len(ids)))

    index_ok = (
        ids["embedding_index"].tolist() == [str(i) for i in range(len(ids))]
        and not ids["post_id"].duplicated().any()
        and all(isinstance(post_id, str) and re.fullmatch(r"\d+", post_id) for post_id in ids["post_id"])
        and all(store.index[post_id] == row for row, post_id in enumerate(ids["post_id"].tolist()))
    )
    finite = find_nonfinite(store.embeddings)
    checks.append(check(
        "matrix shape, dtype, and ID mapping agree",
        store.embeddings.shape == (len(ids), EMBEDDING_DIM)
        and store.embeddings.dtype == np.float32
        and index_ok,
        f"shape {tuple(store.embeddings.shape)}, dtype {store.embeddings.dtype}, "
        f"ID rows {len(ids)}, embedding_index is 0..n-1 and matches the store index: {index_ok}.",
    ))
    checks.append(check(
        "embedding values are finite",
        finite["count"] == 0,
        "NaN/Inf scan was rerun on the memory-mapped matrix. "
        f"Non-finite values: {finite['count']}. First location: {finite['first']}.",
    ))

    embedding_ids = set(ids["post_id"])
    clean_ids = set(clean["post_id"])
    id_match = compare_id_sets({
        "posts_clean.csv": clean_ids,
        "embedding ids.csv": embedding_ids,
    })
    checks.append(check(
        "embedding post_ids match the cleaned posts",
        id_match["passed"],
        json.dumps(id_match["sets"]),
    ))

    if id_match["passed"]:
        clean_root = dict(zip(clean["post_id"].tolist(), clean["root_id"].tolist()))
        root_mismatch = sum(
            clean_root[post_id] != root_id
            for post_id, root_id in zip(ids["post_id"].tolist(), ids["root_id"].tolist())
        )
        checks.append(check(
            "embedding root_id values match the cleaned posts by post_id",
            root_mismatch == 0,
            f"root_id mismatches: {root_mismatch}.",
        ))
    else:
        checks.append(check(
            "embedding root_id values match the cleaned posts by post_id",
            False,
            "Not compared because the post_id sets differ.",
        ))

    identical_check, identical_groups, identical_posts = check_identical_texts(store, ids, clean)
    checks.append(identical_check)

    if identical_groups:
        warnings.append(
            f"{identical_posts} posts share a cleaned BERTweet text with at least "
            f"one other post ({identical_groups} distinct texts). Their embeddings "
            "match. That is expected for identical input and is not an alignment error."
        )

    truncated = metadata.get("warnings", {}).get("truncated_post_ids", [])
    warnings.append(
        "BERTweet (vinai/bertweet-base) is an English Twitter model. Posts in "
        "other languages were still embedded and were not removed. Those vectors "
        "are a weaker representation of non-English text. This was not changed."
    )

    if truncated:
        present = [post_id for post_id in truncated if post_id in store]
        warnings.append(
            f"{len(truncated)} posts were truncated at {BERTWEET_MAX_LENGTH} tokens "
            f"during the completed extraction: {truncated}. "
            f"Present in the embedding store: {present}. "
            "Their vectors only represent the first 128 tokens. "
            "The extraction was not rerun."
        )

    _print("Checking graph files against the embedding store.")
    meta_table = _load_table(GRAPH_METADATA_FILE)
    metadata_by_file = {
        row.graph_id: {
            "root_id": row.root_id,
            "label": row.label,
            "num_nodes": int(row.num_nodes),
            "num_edges": int(row.num_edges),
        }
        for row in meta_table.itertuples(index=False)
    }
    root_by_post = dict(zip(clean["post_id"].tolist(), clean["root_id"].tolist()))
    demo = demo_graph_cases(store)
    graphs = validate_all_graphs(store, root_by_post, metadata_by_file, progress=_print)
    totals = graphs["totals"]

    checks.append(check(
        "every graph file is readable and has a metadata row",
        totals["graph_files"] == EXPECTED_GRAPHS
        and totals["graphs_read"] == EXPECTED_GRAPHS
        and totals["missing_metadata"] == 0
        and graphs["extra_metadata_count"] == 0
        and len(meta_table) == EXPECTED_GRAPHS,
        f"graph files {totals['graph_files']}, readable {totals['graphs_read']}, "
        f"metadata rows {len(meta_table)}, missing metadata {totals['missing_metadata']}, "
        f"metadata rows without a file {graphs['extra_metadata_count']}.",
    ))
    edge_table = _load_table(POSTS_FILE.parent / "edges.csv")
    missing_edges = edges_missing_from_posts(posts, edge_table)
    edges_accounted_for = (
        totals["total_edges"] + len(missing_edges) == len(edge_table)
    )
    checks.append(check(
        "graph node counts and edge counts match the metadata",
        totals["metadata_mismatch_graphs"] == 0
        and totals["schema_error_graphs"] == 0
        and totals["total_nodes"] == EXPECTED_POSTS
        and edges_accounted_for,
        f"metadata mismatches: {totals['metadata_mismatch_graphs']}; "
        f"schema problems: {totals['schema_error_graphs']}; "
        f"nodes {totals['total_nodes']}; graph edges {totals['total_edges']}; "
        f"edges.csv rows {len(edge_table)}; "
        f"edges.csv rows that are not in a graph: {len(missing_edges)}. "
        f"Examples: {graphs['problems'] or 'none'}.",
    ))

    if missing_edges and edges_accounted_for:
        warnings.append(
            f"docs/data_pipeline.md says {EXPECTED_EDGES} extracted edges. "
            f"edges.csv has {len(edge_table)} rows, and the graph JSON files "
            f"contain {totals['total_edges']} edges. The difference is "
            f"{len(missing_edges)} edges.csv row(s) whose endpoint is not a "
            f"post in that thread, so graph export skipped them: {missing_edges}. "
            "Graph metadata matches the graph files. Embedding alignment uses "
            "those graph files, not the documentation count. Person 1's files "
            "were not changed."
        )
    checks.append(check(
        "every graph node has exactly one embedding in node order",
        totals["alignment_error_graphs"] == 0
        and graphs["post_ids_in_more_than_one_graph"] == 0
        and graphs["embedding_rows_not_used"] == 0
        and graphs["distinct_post_ids"] == len(store),
        "Each returned row was compared with the embedding indexed by that "
        f"post_id. Alignment problems: {totals['alignment_error_graphs']}. "
        f"Distinct graph post_ids: {graphs['distinct_post_ids']}. "
        f"IDs used in more than one graph: {graphs['post_ids_in_more_than_one_graph']}. "
        f"Embedding rows unused by any graph: {graphs['embedding_rows_not_used']}.",
    ))
    checks.append(check(
        "edge endpoints are valid string post_ids and edge_index is in range",
        totals["edge_error_graphs"] == 0 and all(demo.values()),
        "Edges were read from the edges list only. parent_id was not used to "
        f"build any index. Graphs with an edge problem: {totals['edge_error_graphs']}. "
        f"Synthetic empty, single-node, no-edge, misleading-parent, and bad-endpoint "
        f"cases: {demo}.",
    ))
    checks.append(check(
        "empty, single-node, and no-edge graphs are handled",
        demo["empty"] and demo["single"] and demo["no_edge"]
        and totals["single_node_error_graphs"] == 0
        and totals["no_edge_error_graphs"] == 0,
        f"Real graphs with no nodes: {totals['empty_graphs']} "
        "(none are expected; the empty case was checked with a synthetic graph). "
        f"Single-node graphs: {totals['single_node_graphs']} "
        f"({totals['single_node_error_graphs']} alignment problems). "
        f"Graphs with nodes but no edges: {totals['no_edge_graphs']} "
        f"({totals['no_edge_error_graphs']} alignment problems).",
    ))
    checks.append(check(
        "partial node lists and unsorted node lists keep the requested order",
        demo["partial_order"] and demo["reversal_follows_request"]
        and graphs["partial_order"]["passed"] is True,
        "A three-node subset was requested as positions 2, 0, 1 and the "
        f"feature rows followed that order on {graphs['partial_order']['graph']}. "
        f"Graphs whose node list is not sorted by post_id: {totals['unsorted_graphs']}. "
        "The helper was not asked to sort nodes.",
    ))
    checks.append(check(
        "every node belongs to the graph's root_id",
        totals["root_error_graphs"] == 0,
        f"Graphs whose node root_id disagrees with the graph root_id: {totals['root_error_graphs']}.",
    ))

    claims = _load_table(CLAIMS_FILE)
    thread_sets = compare_id_sets({
        "unique root_id in posts.csv": set(posts["root_id"]),
        "claims.csv": set(claims["root_id"]),
        "graph_metadata.csv": set(meta_table["root_id"]),
    })
    checks.append(check(
        "thread IDs agree across posts, claims, and graphs",
        thread_sets["passed"] and len(set(posts["root_id"])) == EXPECTED_GRAPHS,
        json.dumps(thread_sets),
    ))

    graph_id_sets = compare_id_sets({
        "posts.csv": set(posts["post_id"]),
        "posts_clean.csv": clean_ids,
        "embedding ids.csv": embedding_ids,
        "graph nodes": graphs["seen_ids"],
    })
    checks.append(check(
        "posts, cleaned posts, embedding IDs, and graph nodes are the same set",
        graph_id_sets["passed"] and len(set(posts["post_id"])) == EXPECTED_POSTS,
        json.dumps({"reference_count": graph_id_sets["reference_count"], "sets": graph_id_sets["sets"]}),
    ))

    stage6 = None

    if STAGE6_REPORT.exists():
        with open(STAGE6_REPORT, encoding="utf-8") as file:
            stage6 = json.load(file)

    stage6_ok = bool(stage6 and stage6.get("passed"))
    coverage = (stage6 or {}).get("coverage", {})
    coverage_agrees = stage6_ok and (
        coverage.get("graphs") == totals["graph_files"]
        and coverage.get("total_nodes") == totals["total_nodes"]
        and coverage.get("distinct_post_ids") == graphs["distinct_post_ids"]
        and coverage.get("post_ids_in_more_than_one_graph") == graphs["post_ids_in_more_than_one_graph"]
        and coverage.get("embedding_rows_not_used_by_any_graph") == graphs["embedding_rows_not_used"]
        and coverage.get("graphs_with_problems") == []
    )
    checks.append(check(
        "fresh graph coverage agrees with the Stage 6 report",
        coverage_agrees,
        "Stage 6 coverage was not the only evidence. This run walked the graphs "
        f"again. Stage 6 passed={stage6_ok}. Counts agree={coverage_agrees}. "
        f"Stage 6 nodes={coverage.get('total_nodes')}, this run={totals['total_nodes']}.",
    ))

    independent = (stage6 or {}).get("independent_check", [])
    independent_ok = stage6_ok and bool(independent) and all(
        item.get("nodes_within_1e-3") == item.get("num_nodes") for item in independent
    )
    reused = [{
        "stage": 6,
        "name": "Independent BERTweet re-embedding of selected graphs",
        "source": "data/nlp/node_helper_check.json",
        "rerun": False,
        "passed": independent_ok,
        "justification": (
            "Re-embedding would load BERTweet and recompute vectors. Stage 7 does "
            "not re-extract embeddings. The Stage 6 report records this check on "
            f"{len(independent)} graphs. This run repeated post_id alignment for "
            "all graphs by comparing helper rows with the stored matrix."
        ),
    }]

    if not independent_ok:
        checks.append(check(
            "Stage 6 re-embedding evidence is present",
            False,
            "The earlier re-embedding report is missing or did not pass, and "
            "Stage 7 did not recompute embeddings.",
        ))

    changed_ids = []
    changed_count = 0

    for post_id in clean["post_id"].tolist():
        if id_round_trip_through_float(post_id) != post_id:
            changed_count += 1
            if len(changed_ids) < 5:
                changed_ids.append(post_id)

    warnings.append(
        "Graph parent_id values are JSON numbers, not strings. Stage 7 built "
        "every edge index from the edges list and the exact string post_ids. "
        f"parent_id was a float on {totals['parent_id_float']} nodes, an int on "
        f"{totals['parent_id_int']} nodes, and a string on {totals['parent_id_string']} "
        f"nodes ({totals['parent_id_non_null']} non-null values). "
        f"After converting parent_id back to text, {totals['parent_edge_disagreements']} "
        f"of {totals['parent_edges_compared']} edges disagreed with that parent. "
        f"Passing the post_id itself through float changes {changed_count} IDs"
        + (f", e.g. {changed_ids}" if changed_ids else "")
        + ". Do not rebuild connectivity from parent_id."
    )

    unexpected_labels = sorted(set(graphs["labels"]) - EXPECTED_LABELS)
    if unexpected_labels:
        warnings.append(
            "One or more graphs are outside the four project classes "
            f"{sorted(EXPECTED_LABELS)}: {graphs['labels']}. "
            "Person 1's data notes say the single 'unknown' thread is a PHEME "
            "annotation whose rumour value was unclear, and it was kept rather "
            "than given one of the four labels. The NLP features still include it."
        )

    _print("Checking git ignore rules, hashes, and package versions.")
    safety = repository_checks()
    checks.extend(safety["checks"])
    notes = safety["notes"]

    info = json.loads(POSTS_CLEAN_INFO_FILE.read_text(encoding="utf-8"))
    posts_hash = before["data/processed/posts.csv"]["sha256"]
    clean_hash = before["data/nlp/posts_clean.csv"]["sha256"]
    checks.append(check(
        "posts.csv hash matches the cleaned-text record",
        posts_hash == info.get("source_sha256"),
        "The current posts.csv hash matches source_sha256 stored when "
        "posts_clean.csv was built."
        if posts_hash == info.get("source_sha256")
        else "posts.csv has changed since posts_clean.csv was built. It was not rebuilt.",
    ))
    checks.append(check(
        "posts_clean.csv hash matches the embedding metadata",
        clean_hash == metadata.get("input_sha256"),
        "The current posts_clean.csv hash matches input_sha256 stored with "
        "the full embeddings."
        if clean_hash == metadata.get("input_sha256")
        else "posts_clean.csv has changed since the embeddings were built. Embeddings were not regenerated.",
    ))

    after = snapshot_files(artifact_paths)
    changed_artifacts = [
        path for path in before
        if before[path] != after.get(path)
    ]
    checks.append(check(
        "embedding artifacts and source tables were not modified during validation",
        not changed_artifacts,
        "SHA-256, size, and modification time were recorded before and after. "
        f"Changed paths: {changed_artifacts or 'none'}.",
    ))

    suite = {
        "command": ".venv/bin/python -m pytest nlp/tests -q",
        "passed": None,
        "failed": None,
        "skipped": None,
        "errors": None,
        "exit_code": None,
        "summary_line": "not run",
    }

    if run_pytest:
        _print("Running the NLP test suite.")
        suite = run_pytest_suite()
        checks.append(check(
            "existing NLP test suite passes",
            suite["exit_code"] == 0 and suite["failed"] == 0 and suite["errors"] == 0,
            suite["summary_line"],
        ))
    else:
        checks.append(check(
            "existing NLP test suite passes",
            False,
            "Pytest was skipped, so this required check has no result.",
        ))

    status = overall_status(checks, warnings)
    failed = [item for item in checks if not item["passed"]]
    counts = {
        "posts": len(posts),
        "cleaned_posts": len(clean),
        "embeddings": len(store),
        "embedding_dimension": int(store.dim),
        "graphs": totals["graph_files"],
        "graph_nodes": totals["total_nodes"],
        "graph_edges": totals["total_edges"],
        "claims": len(claims),
        "empty_text_flag_true": int((clean["is_empty_text"] == "True").sum()),
        "only_mentions_urls_flag_true": int((clean["only_mentions_urls"] == "True").sum()),
        "punctuation_only_tfidf": cleaned["counts"].get("punctuation_only_tfidf", 0),
        "single_node_graphs": totals["single_node_graphs"],
        "no_edge_graphs": totals["no_edge_graphs"],
        "empty_graphs": totals["empty_graphs"],
        "identical_text_groups": identical_groups,
        "posts_sharing_cleaned_text": identical_posts,
        "parent_edge_disagreements": totals["parent_edge_disagreements"],
        "post_ids_changed_by_float_round_trip": changed_count,
        "label_counts": graphs["labels"],
        "post_id_length_counts": cleaned["counts"].get("post_id_length_counts", {}),
    }

    if status == "FAILED":
        explanation = (
            "The NLP feature pipeline is not ready for Person 3 or Person 4 "
            "until the failed checks are understood. No cleaned text, TF-IDF "
            "matrix, or BERTweet embedding was regenerated."
        )
        ready = False
    else:
        explanation = (
            "Person 3 can attach these embeddings to graph nodes with "
            "load_embedding_store() and get_graph_node_embeddings(). Row i of "
            "the returned matrix is the embedding of node i in the graph's "
            "nodes list. Person 4 should fit TF-IDF only after the official "
            "split exists; this stage did not create that split and did not "
            "train a model. The warnings about truncation, language, empty "
            "TF-IDF rows, and parent_id should be written up. They are not a "
            "reason to re-extract the embeddings."
        )
        ready = True

    report = {
        "stage": 7,
        "title": "End-to-end NLP feature validation",
        "validated_at": started_at.isoformat(timespec="seconds"),
        "overall_status": status,
        "status_meaning": (
            "PASSED means every critical check passed and there were no warnings. "
            "PASSED_WITH_WARNINGS means every critical check passed, and the "
            "warnings are limitations downstream work should account for. "
            "FAILED means at least one critical check failed."
        ),
        "counts": counts,
        "checks": checks,
        "failed_checks": failed,
        "warnings": warnings,
        "reused_checks": reused,
        "cross_stage_ids": {
            "reference": graph_id_sets["reference"],
            "reference_count": graph_id_sets["reference_count"],
            "passed": graph_id_sets["passed"],
            "sets": graph_id_sets["sets"],
            "threads": thread_sets,
        },
        "repository_notes": notes,
        "test_suite": {key: value for key, value in suite.items() if key != "failure_tail"},
        "readiness": {
            "ready_for_person3_and_person4": ready,
            "official_split_created": False,
            "tfidf_fit_on_full_pheme": False,
            "classifiers_trained": False,
            "embeddings_reextracted": False,
            "explanation": explanation,
        },
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    report = to_plain(report)

    NLP_DATA.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    _print(f"Overall status: {status}")
    _print(f"Wrote {REPORT_JSON}")
    _print(f"Wrote {REPORT_MD}")

    return report


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip the test suite. The report will then be FAILED for that check.",
    )
    args = parser.parse_args(argv)
    report = run_validation(run_pytest=not args.skip_pytest)

    if report["overall_status"] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
