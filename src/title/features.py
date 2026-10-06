"""Title feature extraction.

Takes the collected dataset and returns one row per video, one column per
feature. Every feature is computed from the title string alone, so the same
code serves training (thousands of titles through nlp.pipe) and scoring (one
title at a time through score_title).

Which features depend on what:

    pure string operations     char_length, word_count, uppercase_ratio,
                               has_all_caps_word, ends_with_question,
                               has_question, has_brackets_or_pipe,
                               emoji_count, exclamation_count
    spaCy lexical attribute    has_number (Token.like_num, a lookup on the
                               token text with no model involved)
    VADER lexicon and rules    sentiment_compound
    spaCy statistical model    has_superlative (Token.tag_ from the tagger)

has_superlative is the only feature that rests on a statistical prediction.
The tagger is trained on edited prose, and titles are short, Title Cased and
often ungrammatical, so it degrades here: on the original casing it tags
"Best" and "EASIEST" as proper nouns. Tagging a lowercased copy of the title
recovered 10 of 10 superlatives in a hand check, against 4 of 10 on the
original, so the tagger sees lowercased text. Expect it to stay the noisiest
feature.

Usage:
    python -m src.title.features cooking
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import regex
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import TITLE, VIDEO_ID  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"

SPACY_MODEL = "en_core_web_sm"

FEATURES = [
    "char_length",
    "word_count",
    "uppercase_ratio",
    "has_all_caps_word",
    "has_number",
    "ends_with_question",
    "has_question",
    "has_brackets_or_pipe",
    "emoji_count",
    "exclamation_count",
    "sentiment_compound",
    "has_superlative",
]

BRACKETS_OR_PIPE = set("()[]{}|")
SUPERLATIVE_TAGS = {"JJS", "RBS"}  # "best", "easiest"; "most"

# One grapheme cluster is one emoji as a reader sees it, so a skin tone
# modifier or a family joined by zero-width joiners counts once. A cluster
# counts when it contains a character drawn as emoji by default, or carries
# the emoji variation selector U+FE0F. That keeps (c), (R) and TM, which are
# pictographic but drawn as text, out of the count.
GRAPHEME = regex.compile(r"\X")
EMOJI = regex.compile(r"\p{Emoji_Presentation}|️")


@lru_cache(maxsize=1)
def load_nlp():
    """The tagger needs tok2vec; the parser, NER and lemmatiser are unused."""
    return spacy.load(SPACY_MODEL, disable=["parser", "ner", "lemmatizer"])


@lru_cache(maxsize=1)
def load_vader():
    return SentimentIntensityAnalyzer()


def emoji_count(title):
    return sum(1 for g in GRAPHEME.findall(title) if EMOJI.search(g))


def string_features(title):
    """Everything that needs neither spaCy nor VADER."""
    words = [w for w in title.split() if any(c.isalnum() for c in w)]
    letters = [c for c in title if c.isalpha()]
    # a word counts as all caps with two or more letters, all uppercase, so
    # "I" and "A" do not trigger it. Acronyms like "BBQ" do; see the dictionary.
    caps_words = [
        w for w in words
        if sum(c.isalpha() for c in w) >= 2
        and all(c.isupper() for c in w if c.isalpha())
    ]
    stripped = title.rstrip()
    return {
        "char_length": len(title),
        "word_count": len(words),
        "uppercase_ratio": (
            sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        ),
        "has_all_caps_word": int(bool(caps_words)),
        "ends_with_question": int(stripped.endswith("?")),
        "has_question": int("?" in title),
        "has_brackets_or_pipe": int(any(c in BRACKETS_OR_PIPE for c in title)),
        "emoji_count": emoji_count(title),
        "exclamation_count": title.count("!"),
    }


def extract(titles):
    """Feature table for a sequence of titles, one row per title, in order."""
    titles = [str(t) for t in titles]
    vader = load_vader()
    # lowercased for the tagger only; see the module docstring
    docs = load_nlp().pipe(t.lower() for t in titles)

    rows = []
    for title, doc in zip(titles, docs):
        row = string_features(title)
        row["has_number"] = int(any(token.like_num for token in doc))
        row["sentiment_compound"] = vader.polarity_scores(title)["compound"]
        row["has_superlative"] = int(
            any(token.tag_ in SUPERLATIVE_TAGS for token in doc)
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def build_feature_table(dataset):
    """The dataset in, one row per video out, keyed by video_id."""
    features = extract(dataset[TITLE].tolist())
    features.insert(0, VIDEO_ID, dataset[VIDEO_ID].to_numpy())
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("content_type")
    args = parser.parse_args()

    source = RAW / f"youtube_api_{args.content_type}.parquet"
    dataset = pd.read_parquet(source)
    table = build_feature_table(dataset)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    target = PROCESSED / f"title_features_{args.content_type}.parquet"
    table.to_parquet(target, index=False)
    print(f"{len(table):,} rows x {len(FEATURES)} features -> "
          f"{target.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
