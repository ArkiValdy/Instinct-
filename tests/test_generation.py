"""Title generation: parsing what Claude returns, and the rules a candidate
must pass before a creator sees it.

None of these tests say a generated title will reach further than the
original. They cannot: `docs/validation_protocol.md` records that the title
model does not predict reach on held-out channels, so there is nothing here
qualified to rank candidates. What is testable is everything around that
judgement — that a candidate is parsed out of the reply intact, is a title
YouTube will accept, is not the title the creator already typed, is not the
same suggestion twice, and has not wandered off the subject of the video.

The reply shape is the Anthropic SDK's own `Message`, built from the real
types so the stand-in cannot drift from what the API returns.
"""

import pytest
from anthropic.types import Message, TextBlock, Usage

from src.generation.candidates import (
    DUPLICATE,
    EMPTY,
    OFF_TOPIC,
    TITLE_MAX_CHARS,
    TOO_LONG,
    UNCHANGED,
    opening_diversity,
    parse_candidates,
    review,
)
from src.generation.claude import build_prompt, generate

ORIGINAL = "How I Fixed My Sleep In 30 Days"
CONTENT_TYPE = "education"


# --- parsing ----------------------------------------------------------------


def test_a_numbered_list_loses_its_numbering_and_preamble():
    reply = (
        "Here are five options:\n"
        "\n"
        "1. A Sleep Fix That Stuck\n"
        '2. "Better Sleep In A Month"\n'
        "3. Sleep, Fixed\n"
    )
    assert parse_candidates(reply) == [
        "A Sleep Fix That Stuck",
        "Better Sleep In A Month",
        "Sleep, Fixed",
    ]


def test_bare_lines_are_candidates_when_nothing_is_marked_up():
    assert parse_candidates("A Sleep Fix\n\nBetter Sleep\n") == [
        "A Sleep Fix",
        "Better Sleep",
    ]


def test_bullets_are_markers_too():
    assert parse_candidates("- A Sleep Fix\n* Better Sleep\n") == [
        "A Sleep Fix",
        "Better Sleep",
    ]


def test_unmarked_commentary_is_dropped_once_anything_is_marked():
    """Claude adds a closing line more often than not. A trailing note must
    not arrive as a sixth suggestion."""
    reply = (
        "1. A Sleep Fix\n"
        "2. Better Sleep\n"
        "\n"
        "Let me know if you want a shorter angle.\n"
    )
    assert parse_candidates(reply) == ["A Sleep Fix", "Better Sleep"]


def test_an_empty_reply_parses_to_nothing():
    assert parse_candidates("") == []
    assert parse_candidates("\n  \n") == []


def test_an_unmarked_refusal_comes_through_for_the_rules_to_catch():
    """Parsing does not judge; with no markers to go by it takes the reply at
    face value and leaves `review` to throw it out."""
    assert parse_candidates("I can't help with that.") == ["I can't help with that."]
    assert review(ORIGINAL, parse_candidates("I can't help with that.")).accepted == []


# --- the rules --------------------------------------------------------------


def test_a_clean_batch_passes_in_order():
    out = review(ORIGINAL, ["Sleep, Fixed", "The Sleep Fix That Stuck"])
    assert out.accepted == ["Sleep, Fixed", "The Sleep Fix That Stuck"]
    assert out.rejected == []


def test_a_title_youtube_would_truncate_is_rejected_at_the_boundary():
    """YouTube cuts a title off at 100 characters. 100 is a title; 101 is a
    title with its ending missing."""
    at_limit = ("Sleep " + "x" * TITLE_MAX_CHARS)[:TITLE_MAX_CHARS]
    over_limit = ("Sleep " + "x" * TITLE_MAX_CHARS)[: TITLE_MAX_CHARS + 1]

    out = review(ORIGINAL, [at_limit, over_limit])
    assert out.accepted == [at_limit]
    assert [(r.candidate, r.reason) for r in out.rejected] == [(over_limit, TOO_LONG)]


def test_the_creators_own_title_is_not_a_suggestion():
    """A suggestion identical to the input is the one outcome the feature
    cannot deliver, so it must never reach the app as a candidate."""
    out = review(ORIGINAL, [ORIGINAL, "  how i FIXED my sleep in 30 days  "])
    assert out.accepted == []
    assert [r.reason for r in out.rejected] == [UNCHANGED, UNCHANGED]


def test_the_same_suggestion_twice_is_kept_once():
    out = review(ORIGINAL, ["Sleep, Fixed", "sleep,   fixed", "Sleep Fixed"])
    assert out.accepted == ["Sleep, Fixed", "Sleep Fixed"]
    assert [(r.candidate, r.reason) for r in out.rejected] == [
        ("sleep,   fixed", DUPLICATE)
    ]


def test_a_title_for_a_different_video_is_rejected():
    """Nothing downstream checks that a candidate is about this video, and a
    plausible, well-formed title for someone else's video is the failure a
    creator would notice first."""
    out = review(ORIGINAL, ["The Best Banana Bread Recipe", "My Sleep Experiment"])
    assert out.accepted == ["My Sleep Experiment"]
    assert [(r.candidate, r.reason) for r in out.rejected] == [
        ("The Best Banana Bread Recipe", OFF_TOPIC)
    ]


def test_a_blank_candidate_is_rejected_before_anything_else():
    out = review(ORIGINAL, ["", "   "])
    assert out.accepted == []
    assert [r.reason for r in out.rejected] == [EMPTY, EMPTY]


# --- diversity --------------------------------------------------------------


def test_one_template_filled_in_five_times_scores_low():
    """The cheapest way for generation to look productive while saying one
    thing. It is a property of the batch, not of any single candidate, so it
    is reported rather than used to reject."""
    collapsed = [
        "How I Fixed My Sleep In 10 Days",
        "How I Fixed My Diet In 20 Days",
        "How I Fixed My Focus In 30 Days",
        "How I Fixed My Mornings In 40 Days",
    ]
    assert opening_diversity(collapsed) == pytest.approx(0.25)


def test_four_different_openings_score_one():
    varied = [
        "Sleep, Fixed",
        "The Sleep Experiment That Worked",
        "I Stopped Guessing About Sleep",
        "Thirty Days To Better Sleep",
    ]
    assert opening_diversity(varied) == pytest.approx(1.0)


def test_diversity_of_nothing_is_zero():
    assert opening_diversity([]) == 0.0


# --- the Claude call --------------------------------------------------------


def test_the_prompt_carries_the_inputs_the_model_needs():
    prompt = build_prompt(ORIGINAL, CONTENT_TYPE, 5)
    assert ORIGINAL in prompt
    assert CONTENT_TYPE in prompt
    assert "5" in prompt
    assert str(TITLE_MAX_CHARS) in prompt


class RecordingClient:
    """Stands in for `anthropic.Anthropic` at the network boundary only. The
    reply is a real `Message`, so a change to the SDK's response shape breaks
    this the same way it would break production."""

    def __init__(self, text):
        self.text = text
        self.request = None
        self.messages = self

    def create(self, **kwargs):
        self.request = kwargs
        return Message(
            id="msg_test",
            content=[TextBlock(text=self.text, type="text")],
            model="claude-sonnet-5",
            role="assistant",
            type="message",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def test_generate_returns_the_reviewed_candidates_not_the_raw_reply():
    client = RecordingClient(
        "1. Sleep, Fixed\n"
        f"2. {ORIGINAL}\n"
        "3. The Best Banana Bread Recipe\n"
        "4. My Sleep Experiment\n"
    )

    out = generate(ORIGINAL, CONTENT_TYPE, n=4, client=client)

    assert out.accepted == ["Sleep, Fixed", "My Sleep Experiment"]
    assert [r.reason for r in out.rejected] == [UNCHANGED, OFF_TOPIC]


def test_generate_sends_the_creators_title_to_the_model():
    client = RecordingClient("1. Sleep, Fixed\n")

    generate(ORIGINAL, CONTENT_TYPE, n=1, client=client)

    sent = str(client.request["messages"])
    assert ORIGINAL in sent
