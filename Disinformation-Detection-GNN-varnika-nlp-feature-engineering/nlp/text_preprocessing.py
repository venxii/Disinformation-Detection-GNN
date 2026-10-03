"""
Text cleaning for PHEME tweets.

Two output formats are produced from the same raw text:

- normalize_for_bertweet: reproduces the official BERTweet tweet
  normalisation (vinai/BERTweet), so the text looks like the data
  BERTweet was pre-trained on. Case, punctuation and hashtags are kept.

- normalize_for_tfidf: lower-cased, space-separated tokens for a
  bag-of-words model. Negations, question marks and exclamation marks
  are turned into explicit tokens so they survive tokenisation.

All functions are pure and deterministic: the same input always
gives the same output.
"""

import html
import math
import re

import emoji
from transformers.models.bertweet.tokenization_bertweet import (
    TweetTokenizer,
)


# ============================================================
# SHARED PATTERNS
# ============================================================

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# (?<!\w) avoids matching e-mail addresses such as name@site.com
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")

RETWEET_PREFIX_RE = re.compile(r"^RT\b")
WHITESPACE_RE = re.compile(r"\s+")


# ============================================================
# SHARED CLEANING
# ============================================================

def to_text(value):
    """Return value as a string; missing values become ''."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value)


def basic_clean(text):
    """
    Steps shared by both formats:
    1. missing -> ''
    2. HTML entities decoded (&amp; -> &, &gt; -> >, &lt; -> <)
    3. newlines/tabs/repeated spaces collapsed to one space
    """

    text = to_text(text)
    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


# ============================================================
# BERTWEET FORMAT
# ============================================================

# Same character mapping and tokenizer as BertweetTokenizer.normalizeTweet
BERTWEET_SPECIAL_PUNCTS = {"’": "'", "…": "..."}

_TWEET_TOKENIZER = TweetTokenizer()


def _bertweet_normalize_token(token):
    """Token rule from BertweetTokenizer.normalizeToken."""

    if token.startswith("@"):
        return "@USER"

    lowered = token.lower()

    if lowered.startswith("http") or lowered.startswith("www"):
        return "HTTPURL"

    if len(token) == 1:

        if token in BERTWEET_SPECIAL_PUNCTS:
            return BERTWEET_SPECIAL_PUNCTS[token]

        return emoji.demojize(token)

    return token


# Unicode private-use characters never occur in the tweets; each one
# temporarily stands in for a multi-character emoji during tokenisation.
_PLACEHOLDER_START = 0xE000


def _protect_multi_codepoint_emojis(text):
    """
    Replace flags, ZWJ sequences and emoji+variation-selector pairs
    with single placeholder characters. Returns (text, placeholder->name).
    """

    names = {}

    def replace(chars, data):

        if len(chars) == 1:
            return chars

        placeholder = chr(_PLACEHOLDER_START + len(names))
        names[placeholder] = data["en"]

        return f" {placeholder} "

    return emoji.replace_emoji(text, replace=replace), names


def normalize_for_bertweet(text):
    """
    Official BERTweet normalisation:
    - @mentions -> @USER
    - URLs -> HTTPURL
    - emojis -> text names, e.g. 😂 -> :face_with_tears_of_joy:
    - ’ -> '   and   … -> ...
    - contractions split as in pre-training: "don't " -> "do n't "
    Hashtags, capital letters, punctuation and "RT" are kept.

    Extra step (not in the official code): multi-character emojis
    such as flags (🇫🇷), ❤️ or family sequences are converted to names
    before tokenisation. The official tokenizer splits them into
    single code points that cannot be converted afterwards.
    """

    text = basic_clean(text)

    if not text:
        return ""

    for punct, replacement in BERTWEET_SPECIAL_PUNCTS.items():
        text = text.replace(punct, replacement)

    text, emoji_names = _protect_multi_codepoint_emojis(text)

    tokens = _TWEET_TOKENIZER.tokenize(text)

    normalized = " ".join(
        emoji_names.get(token) or _bertweet_normalize_token(token)
        for token in tokens
    )

    normalized = (
        normalized.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normalized = (
        normalized.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normalized = (
        normalized.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normalized = emoji.demojize(normalized)

    return " ".join(normalized.split())


# ============================================================
# TF-IDF FORMAT
# ============================================================

URL_TOKEN = "xxurl"
USER_TOKEN = "xxuser"
QUESTION_TOKEN = "xxqmark"
EXCLAMATION_TOKEN = "xxexcl"
EMOJI_PREFIX = "xxemoji_"

APOSTROPHES_RE = re.compile(r"[’‘`´]")

# Irregular negative contractions, applied before the generic n't rule
IRREGULAR_NEGATIONS = {
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "is not",
    "cannot": "can not",
}

# Negative contractions commonly written without an apostrophe on Twitter
APOSTROPHE_FREE_NEGATIONS = {
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "isnt": "is not",
    "wasnt": "was not",
    "arent": "are not",
    "werent": "were not",
    "cant": "can not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "aint": "is not",
}

OTHER_CONTRACTIONS = [
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'ve\b"), " have"),
    # 's (is/has/possessive) and 'd (would/had) are ambiguous: dropped
    (re.compile(r"'[sd]\b"), ""),
]

NON_WORD_RE = re.compile(r"[^\w\s]")

# Emoticons are matched only as standalone tokens (start/end of text,
# surrounded by spaces, or followed by . , ! ?) so that times such as
# "10:30" or text like "(see:" are not mistaken for faces.
EMOTICON_PREFIX = "xxemoticon_"

EMOTICONS = [
    ("brokenheart", r"</3+"),
    ("heart", r"<3+"),
    ("cry", r":'\(+"),
    ("smile", r"[:=]-?\)+|\(-?[:=]|[:=]\]"),
    ("sad", r"[:=]-?\(+|\)-?[:=]|;\("),
    ("wink", r";-?\)+|\(-?;"),
    ("laugh", r"[:=]-?D+|[xX]D+"),
    ("tongue", r"[:=]-?[pP]"),
    ("skeptical", r"[:=]-?[/\\]"),
]

EMOTICON_RES = [
    (
        re.compile(rf"(?<!\S)(?:{pattern})(?=$|\s|[.,!?])"),
        f" {EMOTICON_PREFIX}{name} ",
    )
    for name, pattern in EMOTICONS
]


def _replace_emoticons(text):
    """Must run after URL removal and before lower-casing."""

    for pattern, token in EMOTICON_RES:
        text = pattern.sub(token, text)

    return text


def _emoji_to_token(chars, data):
    name = data["en"].strip(":").lower()
    name = re.sub(r"\W+", "_", name).strip("_")

    return f" {EMOJI_PREFIX}{name} "


def _expand_negations(text):

    for contraction, expansion in IRREGULAR_NEGATIONS.items():
        text = re.sub(
            rf"\b{re.escape(contraction)}\b", expansion, text
        )

    for contraction, expansion in APOSTROPHE_FREE_NEGATIONS.items():
        text = re.sub(rf"\b{contraction}\b", expansion, text)

    for pattern, replacement in OTHER_CONTRACTIONS:
        text = pattern.sub(replacement, text)

    return text


def normalize_for_tfidf(text):
    """
    Bag-of-words normalisation:
    1. shared cleaning (HTML entities, whitespace)
    2. URLs -> xxurl, @mentions -> xxuser
    3. emojis -> one token each, e.g. 😂 -> xxemoji_face_with_tears_of_joy;
       standalone text emoticons -> e.g. :) -> xxemoticon_smile
    4. lower-case
    5. contractions expanded, negations kept as "not"
       (don't / dont / can't / won't / cannot -> ... not)
    6. runs of ?/？ -> xxqmark, runs of !/！ -> xxexcl
    7. other punctuation (including the # of hashtags) -> space
    No stop words are removed. Non-English letters are kept.
    """

    text = basic_clean(text)

    if not text:
        return ""

    text = URL_RE.sub(f" {URL_TOKEN} ", text)
    text = MENTION_RE.sub(f" {USER_TOKEN} ", text)
    text = emoji.replace_emoji(text, replace=_emoji_to_token)
    text = _replace_emoticons(text)

    text = text.lower()
    text = APOSTROPHES_RE.sub("'", text)
    text = _expand_negations(text)

    # Full-width ？ and ！ appear in Japanese/Chinese tweets
    text = re.sub(r"[?？]+", f" {QUESTION_TOKEN} ", text)
    text = re.sub(r"[!！]+", f" {EXCLAMATION_TOKEN} ", text)

    text = NON_WORD_RE.sub(" ", text)

    return " ".join(text.split())


# ============================================================
# TEXT FLAGS
# ============================================================

def text_flags(text):
    """
    Simple descriptive counts computed on the shared-cleaned text.
    These record what the placeholders replaced.
    """

    raw = to_text(text)
    cleaned = basic_clean(raw)

    without_mentions_urls = MENTION_RE.sub("", URL_RE.sub("", cleaned))

    return {
        "num_urls": len(URL_RE.findall(cleaned)),
        "num_mentions": len(MENTION_RE.findall(cleaned)),
        "num_hashtags": len(HASHTAG_RE.findall(cleaned)),
        "num_emojis": emoji.emoji_count(cleaned),
        "starts_with_rt": bool(RETWEET_PREFIX_RE.match(cleaned)),
        "is_empty_text": cleaned == "",
        # True only when nothing at all remains; "@user ?" is content
        "only_mentions_urls": (
            cleaned != "" and without_mentions_urls.strip() == ""
        ),
    }
