"""Parsing Claude's reply, and the rules a candidate title must pass.

What this does not do is rank. The app's purpose is to suggest titles that
reach further, and the thing that would know which of five candidates does
that is the title model. `docs/validation_protocol.md` records that the title
model does not predict reach on held-out channels: Spearman 0.006 against a
channel's own typical video, negative mean R squared, and no configuration
meeting the three conditions. Ranking candidates by a score that is
indistinguishable from noise would produce an ordered list, which reads as an
answer, which is worse than no list at all. So generation stops at "these are
title-shaped, distinct, and about this video" and says nothing about reach
until a model earns the right to.

The rules, in the order they are applied:

    EMPTY       nothing, or whitespace.
    TOO_LONG    over TITLE_MAX_CHARS. YouTube truncates at 100 characters,
                so a longer candidate is a title with its ending missing.
    UNCHANGED   the title the creator already typed, ignoring case and
                spacing. The one suggestion the feature cannot be.
    DUPLICATE   the same suggestion as an earlier accepted one, ignoring
                case and spacing.
    OFF_TOPIC   shares no content word with the original. A crude check and
                deliberately so: it is here to catch a well-formed title for
                somebody else's video, not to judge how well a candidate
                covers the subject.

Batch-level collapse, five fills of one template, is reported by
`opening_diversity` rather than rejected, because no single candidate is at
fault.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List

# YouTube truncates a title in search and on the watch page past this.
TITLE_MAX_CHARS = 100

# a word this long carries the subject. "how", "my", "in" and "30" do not, so
# a candidate is not credited with staying on topic for reusing them.
MIN_TOPIC_WORD = 4

# how many leading words define a candidate's opening, for diversity
OPENING_WORDS = 2

EMPTY = "empty"
TOO_LONG = "over the YouTube title limit"
UNCHANGED = "the original title"
DUPLICATE = "already suggested"
OFF_TOPIC = "not about this video"

# "1. ", "2) ", "- ", "* ", "• "
MARKER = re.compile(r"^(?:\d+[.)]|[-*•])\s+")
QUOTES = "\"'“”‘’"
WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Rejection:
    """One candidate and the rule it broke."""

    candidate: str
    reason: str


@dataclass(frozen=True)
class Review:
    """The candidates a creator should see, and what was dropped.

    The rejections are kept rather than discarded: a batch where most
    candidates were dropped is a signal about the prompt, and silently
    returning two titles out of ten hides it.
    """

    accepted: List[str]
    rejected: List[Rejection]


def normalise(title):
    """Case and spacing folded away, punctuation kept. Two candidates that
    differ only in spacing are the same suggestion; two that differ in
    punctuation are not, since that is a choice a creator makes."""
    return " ".join(title.casefold().split())


def topic_words(title):
    """The content words long enough to carry the subject."""
    return {w for w in WORD.findall(title.casefold()) if len(w) >= MIN_TOPIC_WORD}


def parse_candidates(text):
    """The titles in a reply, one per entry, in the order Claude gave them.

    Claude wraps a list in a preamble and a closing offer more often than
    not, so once anything is marked up as a list item, only the marked lines
    count. With nothing marked up the reply is taken at face value, which
    means a refusal arrives as a candidate and `review` throws it out.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    marked = [MARKER.sub("", line) for line in lines if MARKER.match(line)]
    return [line.strip().strip(QUOTES).strip() for line in (marked or lines)]


def review(original, candidates: Iterable[str]) -> Review:
    """Apply the rules in the order documented above. Accepted candidates
    keep Claude's ordering, which carries no claim about quality."""
    original_normalised = normalise(original)
    subject = topic_words(original)
    accepted: List[str] = []
    rejected: List[Rejection] = []
    seen = set()

    for candidate in candidates:
        normalised = normalise(candidate)
        if not normalised:
            reason = EMPTY
        elif len(candidate) > TITLE_MAX_CHARS:
            reason = TOO_LONG
        elif normalised == original_normalised:
            reason = UNCHANGED
        elif normalised in seen:
            reason = DUPLICATE
        elif subject and not (topic_words(candidate) & subject):
            reason = OFF_TOPIC
        else:
            accepted.append(candidate)
            seen.add(normalised)
            continue
        rejected.append(Rejection(candidate, reason))

    return Review(accepted=accepted, rejected=rejected)


def opening_diversity(candidates):
    """Share of distinct openings in a batch, 0 to 1.

    One template filled in five times is the cheapest way for generation to
    look productive while saying one thing, and it is invisible per
    candidate: each of "How I Fixed My Sleep", "How I Fixed My Diet" is a
    fine title on its own. Five of them score 0.2.
    """
    candidates = list(candidates)
    if not candidates:
        return 0.0
    openings = {tuple(normalise(c).split()[:OPENING_WORDS]) for c in candidates}
    return len(openings) / len(candidates)
