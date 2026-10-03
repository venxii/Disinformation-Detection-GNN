"""
Compare the official BERTweet normalisation with our text_bertweet column.

Two routes are compared for every post:
  official : raw text -> shared cleaning -> BertweetTokenizer(normalization=True)
  ours     : text_bertweet -> BertweetTokenizer(normalization=False)

A difference is classified as a cleaning difference when the normalised
strings differ, and as a tokenization difference when the strings are
equal but the token IDs are not.

Run from the project root:
    .venv/bin/python -m nlp.check_bertweet_tokenizer
"""

import json
import re
from collections import Counter

import emoji
import numpy as np
from transformers import AutoTokenizer

from nlp.bertweet_embeddings import load_clean_posts
from nlp.config import (
    BERTWEET_MAX_LENGTH,
    BERTWEET_MODEL_NAME,
    BERTWEET_REVISION,
    HF_CACHE_DIR,
    PREPROCESSING_VERSION,
    TOKENIZER_CHECK_FILE,
)
from nlp.text_preprocessing import basic_clean


EXAMPLE_CATEGORIES = {
    "ordinary English": lambda t: re.fullmatch(r"[A-Za-z ,.']{40,}", t),
    "mentions": lambda t: t.startswith("@") and t.count("@") >= 2,
    "URLs": lambda t: "http" in t,
    "hashtags": lambda t: t.count("#") >= 2,
    "punctuation": lambda t: re.search(r"[!?]{2,}|\.\.\.", t),
    "emojis": lambda t: emoji.emoji_count(t) > 0,
    "contractions": lambda t: re.search(r"\w(n't|'re|'ll|'m)\b", t),
    "RT prefix": lambda t: t.startswith("RT @"),
    "non-English": lambda t: re.search(r"[а-яА-Яぁ-んァ-ン一-龥]", t),
}


def load_tokenizer(normalization):

    return AutoTokenizer.from_pretrained(
        BERTWEET_MODEL_NAME,
        revision=BERTWEET_REVISION,
        cache_dir=HF_CACHE_DIR,
        normalization=normalization,
    )


REGIONAL_INDICATOR_RE = re.compile("[\U0001F1E6-\U0001F1FF]")


def classify_cleaning_difference(official, ours):

    if REGIONAL_INDICATOR_RE.search(official):
        return "flag emoji split into letters by official rule"

    if emoji.emoji_count(official) > 0:
        return "emoji left unconverted by official rule"

    if "\ufe0f" in official:
        return "stray variation selector (U+FE0F) in official output"

    return "other"


def main():

    print("=" * 60)
    print("BERTWEET TOKENIZER COMPATIBILITY CHECK")
    print("=" * 60)

    official_tokenizer = load_tokenizer(normalization=True)
    plain_tokenizer = load_tokenizer(normalization=False)

    posts = load_clean_posts()
    print(f"\nPosts: {len(posts)}")

    unk_id = plain_tokenizer.unk_token_id

    cleaning_diff = []
    token_only_diff = []
    raw_route_diff = 0
    lengths = []
    unk_counts = []
    unk_pieces = Counter()

    for raw, ours in zip(posts["text_original"], posts["text_bertweet"]):

        cleaned = basic_clean(raw)
        official_text = " ".join(
            official_tokenizer.normalizeTweet(cleaned).split()
        )

        ours_ids = plain_tokenizer(ours)["input_ids"]
        official_ids = official_tokenizer(cleaned)["input_ids"]

        if official_text != ours:
            cleaning_diff.append((raw, official_text, ours))
        elif ours_ids != official_ids:
            token_only_diff.append((raw, official_text, ours))

        if official_tokenizer(raw)["input_ids"] != ours_ids:
            raw_route_diff += 1

        lengths.append(len(ours_ids))
        unk = [i for i, token_id in enumerate(ours_ids) if token_id == unk_id]
        unk_counts.append(len(unk))

        if unk:
            for piece in plain_tokenizer.tokenize(ours):
                if plain_tokenizer.convert_tokens_to_ids(piece) == unk_id:
                    unk_pieces[piece] += 1

    lengths = np.array(lengths)
    unk_counts = np.array(unk_counts)

    reasons = Counter(
        classify_cleaning_difference(official, ours)
        for _, official, ours in cleaning_diff
    )

    examples = {}

    for category, matches in EXAMPLE_CATEGORIES.items():

        for raw, ours in zip(posts["text_original"], posts["text_bertweet"]):

            if len(raw) < 140 and matches(raw):
                examples[category] = {
                    "original": raw,
                    "text_bertweet": ours,
                    "official_normalization": " ".join(
                        official_tokenizer.normalizeTweet(basic_clean(raw)).split()
                    ),
                    "tokens": plain_tokenizer.tokenize(ours),
                }
                break

    report = {
        "model_name": BERTWEET_MODEL_NAME,
        "model_revision": BERTWEET_REVISION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "posts": int(len(posts)),
        "identical_normalization": int(len(posts) - len(cleaning_diff)),
        "cleaning_differences": int(len(cleaning_diff)),
        "cleaning_difference_reasons": dict(reasons),
        "tokenization_only_differences": int(len(token_only_diff)),
        "raw_text_route_differences": int(raw_route_diff),
        "token_length": {
            "min": int(lengths.min()),
            "mean": round(float(lengths.mean()), 2),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            "over_max_length": int((lengths > BERTWEET_MAX_LENGTH).sum()),
        },
        "unk": {
            "posts_with_unk": int((unk_counts > 0).sum()),
            "total_unk_tokens": int(unk_counts.sum()),
            "most_common_unk_pieces": unk_pieces.most_common(15),
        },
        "cleaning_difference_examples": [
            {"original": r, "official": o, "ours": t}
            for r, o, t in cleaning_diff[:6]
        ],
        "category_examples": examples,
    }

    with open(TOKENIZER_CHECK_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Identical normalisation       : {report['identical_normalization']}")
    print(f"Cleaning differences          : {report['cleaning_differences']}")
    for reason, count in reasons.items():
        print(f"    {reason}: {count}")
    print(f"Tokenization-only differences : {report['tokenization_only_differences']}")
    print(f"Raw-text route differences    : {raw_route_diff}")
    print(f"Token length                  : {report['token_length']}")
    print(f"Unknown tokens                : {report['unk']['posts_with_unk']} posts, "
          f"{report['unk']['total_unk_tokens']} tokens")
    print(f"\nReport saved to: {TOKENIZER_CHECK_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
