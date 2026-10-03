import inspect

import pandas as pd
import pytest
from transformers.models.bertweet.tokenization_bertweet import (
    BertweetTokenizer,
)

from nlp.text_preprocessing import (
    BERTWEET_SPECIAL_PUNCTS,
    basic_clean,
    normalize_for_bertweet,
    normalize_for_tfidf,
    text_flags,
    to_text,
)
from nlp.build_clean_text import build_clean_dataframe


# ============================================================
# SHARED CLEANING
# ============================================================

@pytest.mark.parametrize("value", [None, float("nan"), ""])
def test_missing_values_become_empty_string(value):
    assert to_text(value) == ""
    assert basic_clean(value) == ""
    assert normalize_for_bertweet(value) == ""
    assert normalize_for_tfidf(value) == ""


def test_html_entities_are_decoded():
    assert basic_clean("safe &amp; strong") == "safe & strong"
    assert basic_clean("-&gt; look &lt;3") == "-> look <3"


def test_whitespace_and_newlines_are_collapsed():
    assert basic_clean("  a\n\nb\t c  ") == "a b c"


def test_special_puncts_match_official_bertweet():
    official_source = inspect.getsource(BertweetTokenizer.__init__)
    assert "self.special_puncts = {\"’\": \"'\", \"…\": \"...\"}" in official_source
    assert BERTWEET_SPECIAL_PUNCTS == {"’": "'", "…": "..."}


# ============================================================
# BERTWEET FORMAT
# ============================================================

def test_bertweet_mentions_and_urls():
    text = "@BBCBreaking 12 dead http://t.co/T8FFDcjMdi"
    assert normalize_for_bertweet(text) == "@USER 12 dead HTTPURL"


def test_bertweet_keeps_case_hashtags_and_question_marks():
    text = "Is this TRUE? #JeSuisCharlie"
    assert normalize_for_bertweet(text) == "Is this TRUE ? #JeSuisCharlie"


def test_bertweet_emojis_become_names():
    out = normalize_for_bertweet("@cnni so sad 😞😞")
    assert out == "@USER so sad :disappointed_face: :disappointed_face:"


def test_bertweet_multi_codepoint_emoji_is_converted():
    assert "🇫🇷" not in normalize_for_bertweet("with you 🇫🇷")
    assert ":France:" in normalize_for_bertweet("with you 🇫🇷")


def test_bertweet_contractions_split_like_pretraining():
    out = normalize_for_bertweet("You don’t understand science")
    assert out == "You do n't understand science"


def test_bertweet_keeps_negations():
    out = normalize_for_bertweet("not true, no evidence, never happened")
    assert out.split()[0] == "not"
    assert "no" in out.split()
    assert "never" in out.split()


def test_bertweet_ellipsis_and_html():
    out = normalize_for_bertweet("Shocking… stay safe &amp; strong")
    assert out == "Shocking ... stay safe & strong"


def test_bertweet_mention_only_tweet():
    assert normalize_for_bertweet("@ABC") == "@USER"


# ============================================================
# TF-IDF FORMAT
# ============================================================

def test_tfidf_placeholders_and_lowercase():
    out = normalize_for_tfidf("@AFP Police CONFIRM http://t.co/abc")
    assert out == "xxuser police confirm xxurl"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I don't believe it", "i do not believe it"),
        ("I don’t believe it", "i do not believe it"),
        ("I dont believe it", "i do not believe it"),
        ("we can't know", "we can not know"),
        ("we cannot know", "we can not know"),
        ("it won't stop", "it will not stop"),
        ("Isn't this true", "is not this true"),
    ],
)
def test_tfidf_negative_contractions_keep_not(text, expected):
    assert normalize_for_tfidf(text) == expected


def test_tfidf_never_removes_negation_or_stop_words():
    out = normalize_for_tfidf("No, the police have not confirmed. Never.")
    tokens = out.split()
    for word in ["no", "the", "not", "never"]:
        assert word in tokens


def test_tfidf_question_and_exclamation_tokens():
    out = normalize_for_tfidf("Is it true??? Wake up!!!")
    assert out == "is it true xxqmark wake up xxexcl"


def test_tfidf_full_width_question_and_exclamation():
    out = normalize_for_tfidf("見えるか？ 言い返さないだろ！")
    assert out == "見えるか xxqmark 言い返さないだろ xxexcl"


def test_tfidf_hashtag_word_is_kept():
    out = normalize_for_tfidf("#JeSuisCharlie #Ferguson")
    assert out == "jesuischarlie ferguson"


def test_tfidf_emoji_tokens():
    out = normalize_for_tfidf("so sad 😞")
    assert out == "so sad xxemoji_disappointed_face"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so true :)", "so true xxemoticon_smile"),
        ("so sad :(", "so sad xxemoticon_sad"),
        ("great :-) and :)))", "great xxemoticon_smile and xxemoticon_smile"),
        ("terrible :-( :(( =(", "terrible xxemoticon_sad xxemoticon_sad xxemoticon_sad"),
        ("RIP :'(", "rip xxemoticon_cry"),
        ("sure ;) ;-)", "sure xxemoticon_wink xxemoticon_wink"),
        ("lol :D xD", "lol xxemoticon_laugh xxemoticon_laugh"),
        ("hmm :/ :-/", "hmm xxemoticon_skeptical xxemoticon_skeptical"),
        ("joke :P", "joke xxemoticon_tongue"),
        ("paris <3 </3", "paris xxemoticon_heart xxemoticon_brokenheart"),
        (":) at start", "xxemoticon_smile at start"),
        ("ears... :)!", "ears xxemoticon_smile xxexcl"),
    ],
)
def test_tfidf_emoticons(text, expected):
    assert normalize_for_tfidf(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "meet at 10:30 today",
        "see http://t.co/abc:/x",
        "note (see: page 2)",
        "scary:(",
        "word:D",
    ],
)
def test_tfidf_non_standalone_emoticons_are_not_converted(text):
    assert "xxemoticon" not in normalize_for_tfidf(text)


def test_bertweet_keeps_emoticons_unchanged():
    assert normalize_for_bertweet("so sad :( ok :)") == "so sad :( ok :)"


def test_tfidf_other_contractions():
    out = normalize_for_tfidf("they're here, I'm sure, we'll see")
    assert out == "they are here i am sure we will see"


def test_tfidf_keeps_non_english_text():
    out = normalize_for_tfidf("Hay un soldado muerto. Каклы перед")
    assert out == "hay un soldado muerto каклы перед"


def test_tfidf_email_is_not_a_mention():
    assert "xxuser" not in normalize_for_tfidf("mail me@site.com")


def test_tfidf_url_question_mark_is_not_a_question():
    assert normalize_for_tfidf("see http://x.com/a?b=1") == "see xxurl"


# ============================================================
# FLAGS
# ============================================================

def test_flags_counts():
    flags = text_flags("RT @a @b: look #x http://t.co/1 😂")
    assert flags["num_mentions"] == 2
    assert flags["num_hashtags"] == 1
    assert flags["num_urls"] == 1
    assert flags["num_emojis"] == 1
    assert flags["starts_with_rt"] is True
    assert flags["only_mentions_urls"] is False


def test_flags_only_mentions_urls():
    assert text_flags("@a @b http://t.co/1")["only_mentions_urls"] is True
    assert text_flags("@MarcusButler 💗💗")["only_mentions_urls"] is False
    assert text_flags("@Phil_R_Upp ?")["only_mentions_urls"] is False
    assert text_flags("")["is_empty_text"] is True


def test_rt_word_is_not_retweet_prefix():
    assert text_flags("Rt if you respect")["starts_with_rt"] is False


# ============================================================
# DETERMINISM
# ============================================================

def test_functions_are_deterministic():
    text = "@a Don’t panic… 😂 #Sydney?! http://t.co/x &amp; more"
    assert normalize_for_bertweet(text) == normalize_for_bertweet(text)
    assert normalize_for_tfidf(text) == normalize_for_tfidf(text)


# ============================================================
# DATAFRAME BUILDER
# ============================================================

def test_build_clean_dataframe_preserves_ids_and_text():
    posts = pd.DataFrame({
        "post_id": ["552783238415265792", "000123"],
        "root_id": ["552783238415265792", "552783238415265792"],
        "parent_id": ["", "552783238415265792"],
        "text": ["Breaking: 10 dead &amp; more\nhttp://t.co/x", "NA"],
        "timestamp": ["t1", "t2"],
        "post_type": ["source", "reaction"],
        "event": ["e", "e"],
        "thread_type": ["rumours", "rumours"],
    })
    original = posts.copy()

    clean = build_clean_dataframe(posts)

    pd.testing.assert_frame_equal(posts, original)
    assert list(clean["post_id"]) == ["552783238415265792", "000123"]
    assert list(clean["root_id"]) == list(posts["root_id"])
    assert list(clean["text_original"]) == list(posts["text"])
    assert clean.loc[1, "text_tfidf"] == "na"


def test_build_clean_dataframe_rejects_duplicate_ids():
    posts = pd.DataFrame({
        "post_id": ["1", "1"],
        "root_id": ["1", "1"],
        "text": ["a", "b"],
        "post_type": ["source", "reaction"],
    })
    with pytest.raises(ValueError, match="duplicate"):
        build_clean_dataframe(posts)
