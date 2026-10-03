"""
Build data/nlp/posts_clean.csv from data/processed/posts.csv.

Run from the project root:
    .venv/bin/python -m nlp.build_clean_text
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from nlp.config import (
    NLP_DATA,
    POSTS_CLEAN_FILE,
    POSTS_CLEAN_INFO_FILE,
    POSTS_FILE,
    PREPROCESSING_VERSION,
)
from nlp.text_preprocessing import (
    normalize_for_bertweet,
    normalize_for_tfidf,
    text_flags,
)


REQUIRED_COLUMNS = ["post_id", "root_id", "text", "post_type"]

OUTPUT_COLUMNS = [
    "post_id",
    "root_id",
    "post_type",
    "text_original",
    "text_bertweet",
    "text_tfidf",
    "num_urls",
    "num_mentions",
    "num_hashtags",
    "num_emojis",
    "starts_with_rt",
    "is_empty_text",
    "only_mentions_urls",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_posts_as_text(path):
    """
    Read every column as a string.
    keep_default_na=False keeps tweets such as "NA" or "null" as text
    and keeps 18-digit IDs exact (no float conversion).
    """

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_clean_dataframe(posts):
    """Return a new DataFrame with cleaned text; `posts` is not modified."""

    missing = [c for c in REQUIRED_COLUMNS if c not in posts.columns]

    if missing:
        raise ValueError(f"posts is missing columns: {missing}")

    if (posts["post_id"] == "").any():
        raise ValueError("posts contains empty post_id values")

    duplicated = posts["post_id"].duplicated()

    if duplicated.any():
        examples = posts.loc[duplicated, "post_id"].head(5).tolist()
        raise ValueError(f"posts contains duplicate post_id values: {examples}")

    texts = posts["text"].tolist()

    clean = pd.DataFrame({
        "post_id": posts["post_id"].tolist(),
        "root_id": posts["root_id"].tolist(),
        "post_type": posts["post_type"].tolist(),
        "text_original": texts,
        "text_bertweet": [normalize_for_bertweet(t) for t in texts],
        "text_tfidf": [normalize_for_tfidf(t) for t in texts],
    })

    flags = pd.DataFrame([text_flags(t) for t in texts])

    return pd.concat([clean, flags], axis=1)[OUTPUT_COLUMNS]


def verify_round_trip(posts, path):
    """Re-read the written file and check IDs and original text are exact."""

    written = load_posts_as_text(path)

    checks = {
        "row count": len(written) == len(posts),
        "post_id exact": written["post_id"].tolist() == posts["post_id"].tolist(),
        "root_id exact": written["root_id"].tolist() == posts["root_id"].tolist(),
        "text exact": written["text_original"].tolist() == posts["text"].tolist(),
    }

    return checks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLP TEXT PREPROCESSING")
    print("=" * 60)

    if not POSTS_FILE.exists():
        print(f"\nERROR: {POSTS_FILE} not found.")
        sys.exit(1)

    hash_before = file_sha256(POSTS_FILE)

    posts = load_posts_as_text(POSTS_FILE)
    print(f"\nPosts loaded : {len(posts)}")

    clean = build_clean_dataframe(posts)

    NLP_DATA.mkdir(parents=True, exist_ok=True)
    clean.to_csv(POSTS_CLEAN_FILE, index=False)

    checks = verify_round_trip(posts, POSTS_CLEAN_FILE)
    checks["posts.csv unchanged"] = file_sha256(POSTS_FILE) == hash_before

    print("\nVerification:")
    for name, passed in checks.items():
        print(f"  {name:<22}: {'OK' if passed else 'FAILED'}")

    summary = {
        "posts": len(clean),
        "empty_text": int(clean["is_empty_text"].sum()),
        "only_mentions_urls": int(clean["only_mentions_urls"].sum()),
        "empty_bertweet_text": int((clean["text_bertweet"] == "").sum()),
        "empty_tfidf_text": int((clean["text_tfidf"] == "").sum()),
        "with_urls": int((clean["num_urls"] > 0).sum()),
        "with_mentions": int((clean["num_mentions"] > 0).sum()),
        "with_hashtags": int((clean["num_hashtags"] > 0).sum()),
        "with_emojis": int((clean["num_emojis"] > 0).sum()),
        "starts_with_rt": int(clean["starts_with_rt"].sum()),
    }

    print("\nSummary:")
    for name, value in summary.items():
        print(f"  {name:<22}: {value}")

    info = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_file": str(POSTS_FILE.relative_to(POSTS_FILE.parents[2])),
        "source_sha256": hash_before,
        "output_file": str(POSTS_CLEAN_FILE.relative_to(POSTS_FILE.parents[2])),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": OUTPUT_COLUMNS,
        "summary": summary,
        "verification": checks,
    }

    with open(POSTS_CLEAN_INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("\nOutput files:")
    print(f"  {POSTS_CLEAN_FILE}")
    print(f"  {POSTS_CLEAN_INFO_FILE}")

    print("\n" + "=" * 60)

    if all(checks.values()):
        print("TEXT PREPROCESSING COMPLETE")
    else:
        print("TEXT PREPROCESSING FAILED VERIFICATION")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
