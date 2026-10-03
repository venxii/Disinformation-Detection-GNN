"""
Tests for Stage 7 validation logic.

These use tiny tables and matrices. They do not load the 321 MB embedding
file or refit TF-IDF on PHEME.
"""

import numpy as np
import pandas as pd

from nlp.build_clean_text import build_clean_dataframe
from nlp.embedding_node_helper import build_embedding_store
from nlp.stage7_validation import (
    build_edge_index,
    compare_id_sets,
    demo_graph_cases,
    find_nonfinite,
    overall_status,
    parse_pytest_summary,
    render_markdown,
    validate_cleaned_posts,
    validate_graph_against_store,
    validate_tfidf_guarantees,
)


def _posts():
    return pd.DataFrame({
        "post_id": ["498235547685756928", "498243332204949504"],
        "root_id": ["498235547685756928", "498235547685756928"],
        "text": [
            "Hello @bob http://t.co/abc",
            "Only punctuation?!",
        ],
        "post_type": ["source", "reaction"],
    })


def _check(result, name):
    matches = [item for item in result["checks"] if item["name"] == name]
    assert len(matches) == 1, name
    return matches[0]


def _store(post_ids):
    matrix = np.arange(len(post_ids) * 3, dtype=np.float32).reshape(len(post_ids), 3)
    return build_embedding_store(matrix, post_ids)


# ============================================================
# CLEANED TEXT
# ============================================================

def test_cleaned_posts_match_by_id_when_row_order_differs():
    posts = _posts()
    clean = build_clean_dataframe(posts)
    shuffled = posts.iloc[::-1].reset_index(drop=True)

    result = validate_cleaned_posts(shuffled, clean)

    assert _check(result, "cleaned post_id set equals the original post_id set")["passed"]
    assert _check(result, "root_id, post_type, and original text match by post_id")["passed"]
    assert _check(result, "text_bertweet and text_tfidf are reproducible")["passed"]
    assert result["counts"]["same_row_order_as_posts_csv"] is False


def test_cleaned_posts_detect_text_and_flag_errors():
    posts = _posts()
    clean = build_clean_dataframe(posts)
    broken = clean.copy()
    broken.loc[0, "text_bertweet"] = "this was edited"
    broken.loc[0, "is_empty_text"] = True

    result = validate_cleaned_posts(posts, broken)

    assert _check(result, "text_bertweet and text_tfidf are reproducible")["passed"] is False
    assert _check(result, "preprocessing flags match the original text")["passed"] is False


def test_cleaned_posts_detect_duplicate_ids():
    posts = _posts()
    clean = build_clean_dataframe(posts)
    clean.loc[1, "post_id"] = clean.loc[0, "post_id"]

    result = validate_cleaned_posts(posts, clean)

    assert _check(result, "post_id values are unique, non-empty digit strings")["passed"] is False


def test_mentions_only_flag_matches_tfidf_placeholders():
    posts = pd.DataFrame({
        "post_id": ["11", "22"],
        "root_id": ["11", "11"],
        "text": ["@bob http://t.co/abc", "hello"],
        "post_type": ["source", "reaction"],
    })
    clean = build_clean_dataframe(posts)
    result = validate_cleaned_posts(posts, clean)
    mentions = clean.loc[clean["post_id"] == "11", "only_mentions_urls"].iloc[0]

    assert bool(mentions) is True
    assert _check(result, "mentions/URL-only flags agree with TF-IDF tokens")["passed"]
    assert _check(result, "empty-text flags agree with cleaned text")["passed"]


def test_expected_row_count_is_enforced():
    posts = _posts()
    clean = build_clean_dataframe(posts)
    result = validate_cleaned_posts(posts, clean, expected_posts=99)

    assert _check(result, "posts and cleaned posts have the same number of rows")["passed"] is False


# ============================================================
# TF-IDF
# ============================================================

def test_tfidf_guarantees_hold_on_synthetic_posts():
    result = validate_tfidf_guarantees()

    failed = [item["name"] for item in result["checks"] if not item["passed"]]
    assert failed == []
    assert any("load_features" in warning for warning in result["warnings"])


# ============================================================
# EMBEDDINGS AND GRAPHS
# ============================================================

def test_nonfinite_values_are_found_in_the_last_chunk():
    values = np.zeros((10, 3), dtype=np.float32)
    values[9, 2] = np.nan
    found = find_nonfinite(values, chunk_size=4)

    assert found["count"] == 1
    assert found["first"] == {"row": 9, "column": 2}
    assert find_nonfinite(np.zeros((5, 2), dtype=np.float32), chunk_size=4)["count"] == 0

    values[3, 0] = np.inf
    assert find_nonfinite(values, chunk_size=4)["count"] == 2


def test_edge_index_uses_edges_and_ignores_parent_id():
    node_ids = ["100", "200"]
    edge_index, errors = build_edge_index(
        node_ids,
        [{"source": "100", "target": "200", "parent_id": 999.0}],
    )

    assert errors == []
    assert edge_index.shape == (2, 1)
    assert edge_index.dtype == np.int64
    assert edge_index.tolist() == [[0], [1]]

    _, bad_edges = build_edge_index(node_ids, [{"source": "100", "target": "200"}])
    assert bad_edges == []

    _, unknown = build_edge_index(node_ids, [{"source": "100", "target": "999"}])
    assert unknown

    _, numeric = build_edge_index(node_ids, [{"source": "100", "target": 200}])
    assert any("not a clean string" in message for message in numeric)

    empty_index, empty_errors = build_edge_index(["100"], [])
    assert empty_errors == []
    assert empty_index.shape == (2, 0)


def test_misleading_parent_id_does_not_change_alignment():
    post_ids = ["30", "10", "40", "20"]
    store = _store(post_ids)
    cases = demo_graph_cases(store)

    assert all(cases.values()), cases

    graph = {
        "root_id": "30",
        "num_nodes": 2,
        "num_edges": 1,
        "nodes": [
            {"post_id": "30", "parent_id": None},
            {"post_id": "10", "parent_id": 999.0},
        ],
        "edges": [{"source": "30", "target": "10"}],
    }
    result = validate_graph_against_store(
        graph,
        store,
        root_by_post={"30": "30", "10": "30"},
    )

    assert result["ok"]
    assert result["parent_edge_disagreements"] == 1
    assert result["edge_index"].tolist() == [[0], [1]]
    assert result["node_ids"] == ["30", "10"]


def test_graph_root_mismatch_is_reported():
    store = _store(["30", "10"])
    graph = {
        "root_id": "30",
        "num_nodes": 1,
        "num_edges": 0,
        "nodes": [{"post_id": "30", "parent_id": None}],
        "edges": [],
    }
    result = validate_graph_against_store(
        graph, store, root_by_post={"30": "999"}
    )

    assert result["ok"] is False
    assert result["root_errors"]


# ============================================================
# STATUS AND REPORT
# ============================================================

def test_edges_missing_from_posts_are_reported():
    from nlp.stage7_validation import edges_missing_from_posts

    posts = pd.DataFrame({
        "post_id": ["1", "2"],
        "root_id": ["1", "1"],
    })
    edges = pd.DataFrame({
        "root_id": ["1", "1"],
        "parent_id": ["1", "1"],
        "child_id": ["2", "not-a-real-post-id"],
    })

    missing = edges_missing_from_posts(posts, edges)

    assert len(missing) == 1
    assert missing[0]["parent_in_thread"] is True
    assert missing[0]["child_in_thread"] is False
    assert missing[0]["child_id_is_digits"] is False


def test_id_sets_ignore_row_order():
    report = compare_id_sets({
        "posts": {"2", "1"},
        "embeddings": {"1", "2"},
        "graphs": {"1"},
    })

    assert report["passed"] is False
    assert report["sets"][0]["missing_from_this_set"] == 0
    assert report["sets"][1]["missing_from_this_set"] == 1
    assert report["sets"][1]["missing_examples"] == ["2"]


def test_overall_status():
    passed = {"name": "a", "passed": True, "critical": True, "detail": ""}
    failed = {"name": "b", "passed": False, "critical": True, "detail": ""}

    assert overall_status([passed], []) == "PASSED"
    assert overall_status([passed], ["a limitation"]) == "PASSED_WITH_WARNINGS"
    assert overall_status([passed, failed], ["a limitation"]) == "FAILED"


def test_pytest_summary_parser():
    parsed = parse_pytest_summary("..............\n170 passed in 12.50s\n")
    assert parsed == {
        "summary_line": "170 passed in 12.50s",
        "passed": 170,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }

    failed = parse_pytest_summary("2 failed, 3 passed, 1 skipped, 1 error in 4s")
    assert failed["failed"] == 2
    assert failed["passed"] == 3
    assert failed["skipped"] == 1
    assert failed["errors"] == 1


def test_markdown_report_states_status_and_omits_vectors():
    report = {
        "overall_status": "PASSED_WITH_WARNINGS",
        "status_meaning": "Warnings are limitations.",
        "validated_at": "2026-10-02T22:00:00+05:30",
        "counts": {"posts": 2, "embeddings": 2},
        "checks": [{
            "name": "post ids match",
            "passed": True,
            "critical": True,
            "detail": "0 mismatches",
        }],
        "failed_checks": [],
        "cross_stage_ids": {
            "reference": "posts.csv",
            "reference_count": 2,
            "sets": [{
                "name": "embeddings",
                "count": 2,
                "missing_from_this_set": 0,
                "extra_not_in_reference": 0,
            }],
        },
        "reused_checks": [{
            "stage": 6,
            "name": "re-embedding",
            "rerun": False,
            "passed": True,
            "justification": "The model was not reloaded.",
        }],
        "warnings": ["Two posts were truncated."],
        "repository_notes": [],
        "test_suite": {
            "command": ".venv/bin/python -m pytest nlp/tests -q",
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "summary_line": "1 passed in 0.1s",
        },
        "readiness": {"explanation": "Ready for the graph model, with the warnings above."},
        "runtime_seconds": 1.2,
    }

    markdown = render_markdown(report)

    assert "PASSED_WITH_WARNINGS" in markdown
    assert "post ids match" in markdown
    assert "embeddings.npy" not in markdown
    assert "0.1234" not in markdown
