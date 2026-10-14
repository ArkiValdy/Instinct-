"""Title feature extraction.

`features.extract` serves two callers that must agree: training, which sends
thousands of titles through `nlp.pipe`, and scoring, which sends one title at
a time. These tests pin the feature values for two hand-checked titles, the
documented decisions that are easy to undo by accident (lowercasing for the
tagger, cleaning for VADER, CJK brackets, the two-letter floor on all caps),
and the equivalence of the batch and single-title paths.

Expected values are counted by hand, not computed by the code under test. The
two statistical features, sentiment_compound and has_superlative, are tested
by the decision that produces them rather than by a pinned float.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.title.features import FEATURES, extract, vader_text

PROJECT = Path(__file__).resolve().parents[1]

# everything computed from the string alone, so it can be counted by hand.
# sentiment_compound comes from the VADER lexicon and has_superlative from
# the spaCy tagger; both get their own tests below.
DETERMINISTIC = [f for f in FEATURES if f not in ("sentiment_compound", "has_superlative")]

CALM = "how i fixed my sleep"
SHOUTY = "The EASIEST Sleep Fix EVER!! (10 Days) 😴"

# hand-counted. CALM: 20 characters, 5 words, no uppercase letter.
# SHOUTY: 40 characters (the emoji is one code point), 7 words (the bare
# emoji has no alphanumeric character so it is not a word), 26 letters of
# which 15 are uppercase.
WANT = {
    CALM: {
        "char_length": 20,
        "word_count": 5,
        "uppercase_ratio": 0.0,
        "has_all_caps_word": 0,
        "has_number": 0,
        "ends_with_question": 0,
        "has_question": 0,
        "has_brackets_or_pipe": 0,
        "emoji_count": 0,
        "exclamation_count": 0,
    },
    SHOUTY: {
        "char_length": 40,
        "word_count": 7,
        "uppercase_ratio": 15 / 26,
        "has_all_caps_word": 1,
        "has_number": 1,
        "ends_with_question": 0,
        "has_question": 0,
        "has_brackets_or_pipe": 1,
        "emoji_count": 1,
        "exclamation_count": 2,
    },
}


@pytest.fixture(scope="module")
def table():
    return extract([CALM, SHOUTY])


@pytest.mark.parametrize("title", [CALM, SHOUTY])
@pytest.mark.parametrize("feature", DETERMINISTIC)
def test_feature_matches_the_hand_count(table, title, feature):
    row = table.iloc[[CALM, SHOUTY].index(title)]
    assert row[feature] == pytest.approx(WANT[title][feature])


def test_every_listed_feature_is_actually_computed(table):
    """`extract` builds its frame with `columns=FEATURES`, which silently
    drops a computed feature missing from the list and silently fills an
    all-NaN column for a listed feature nothing computes."""
    assert list(table.columns) == FEATURES
    assert not table.isna().any().any()


def test_rows_come_back_in_the_order_the_titles_went_in():
    """`extract` zips the titles against a lazy `nlp.pipe` generator. If the
    two ever fall out of step, every title gets another title's spaCy
    features while the string features still look right."""
    titles = ["aaa", "bb?", "c!"]
    out = extract(titles)
    assert out["char_length"].tolist() == [3, 3, 2]
    assert out["has_question"].tolist() == [0, 1, 0]
    assert out["exclamation_count"].tolist() == [0, 0, 1]


def test_scoring_one_title_matches_that_title_in_a_batch():
    """The docstring promises the same code serves training and scoring. A
    feature normalised across the batch, or state held between titles, would
    make a title score differently alone than it did in training."""
    batch = extract([CALM, SHOUTY, "another sleep title"])
    alone = extract([SHOUTY])
    pd.testing.assert_series_equal(
        batch.iloc[1].rename(None), alone.iloc[0].rename(None)
    )


def test_a_title_with_no_letters_scores_zero_uppercase():
    """uppercase_ratio divides by the letter count, so an emoji-only or
    numeric title has to be guarded."""
    out = extract(["", "   ", "2026 🎉"])
    assert out["uppercase_ratio"].tolist() == [0.0, 0.0, 0.0]
    # "2026 🎉" is six code points; the emoji is one
    assert out["char_length"].tolist() == [0, 3, 6]
    assert out["word_count"].tolist() == [0, 0, 1]


def test_one_letter_words_are_not_all_caps_words():
    """Documented decision: a word needs two letters to count, so "I" and
    "A" do not make every title an all-caps title."""
    assert extract(["I Fixed A Sleep Problem"])["has_all_caps_word"].iloc[0] == 0
    assert extract(["I Fixed MY Sleep Problem"])["has_all_caps_word"].iloc[0] == 1


def test_cjk_brackets_count_as_brackets():
    """A bracketed qualifier is the same signal whether it is written
    "[ENG DUB]" or "【ENG DUB】"."""
    out = extract(["Sleep Fix [ENG DUB]", "Sleep Fix 【ENG DUB】", "Sleep Fix ENG DUB"])
    assert out["has_brackets_or_pipe"].tolist() == [1, 1, 0]


def test_vader_text_clears_what_hides_a_sentiment_word():
    """VADER splits on whitespace and strips ASCII punctuation only, so the
    lexicon never sees a word glued to a CJK bracket, and the emoji
    variation selector hides the heart."""
    assert vader_text("Better【ENG】") == "Better ENG "
    assert vader_text("❤️") == "❤"
    assert vader_text("don’t") == "don't"


def test_sentiment_survives_a_bracketed_title():
    """The consequence of the cleaning above: drop it and this title scores
    a flat zero, because "Better【ENG】" matches nothing in the lexicon."""
    assert extract(["Better【ENG】"])["sentiment_compound"].iloc[0] > 0


def test_negation_is_read_through_a_curly_apostrophe():
    """VADER matches its negation list by exact word, and "don’t" is not on
    it. Drop the curly-quote translation and a negated title reads as a
    positive one, which is the wrong sign rather than a smaller number."""
    plain = extract(["i love this"])["sentiment_compound"].iloc[0]
    straight = extract(["i don't love this"])["sentiment_compound"].iloc[0]
    curly = extract(["i don’t love this"])["sentiment_compound"].iloc[0]
    assert plain > 0
    assert straight < 0
    assert curly == pytest.approx(straight)


def test_superlatives_are_found_despite_title_casing():
    """The tagger is trained on prose and reads "Best" in Title Case as a
    proper noun, so `extract` tags a lowercased copy. Remove the `.lower()`
    and the superlative features in the trained model go quiet."""
    titles = ["The EASIEST Sleep Fix", "The Best Sleep Fix", "A Sleep Fix"]
    assert extract(titles)["has_superlative"].tolist() == [1, 1, 0]


# a fresh interpreter with the training-only dependencies made unimportable.
# "Sleep Fix" is nine characters, so printing that proves extraction ran
# rather than merely that the import succeeded.
IMPORT_PROBE = """
import sys

TRAINING_ONLY = ("matplotlib", "dotenv", "googleapiclient", "src.collection")


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.startswith(TRAINING_ONLY):
            raise ImportError("not available when scoring: " + name)
        return None


sys.meta_path.insert(0, Blocker())

from src.title.features import extract

print(int(extract(["Sleep Fix"])["char_length"].iloc[0]))
"""


def test_scoring_a_title_needs_neither_plots_nor_the_collection_stack():
    """`extract` is the scoring path: the app calls it on one title typed by
    a creator. Anything it imports at module scope, the app has to install
    and the app has to configure, and a module-level import of `target`
    reaches `src.collection.youtube_api`, which calls `load_dotenv` and
    wants a YouTube key to score a string.
    """
    probe = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=PROJECT, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "9"
