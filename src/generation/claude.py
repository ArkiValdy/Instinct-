"""Asking Claude for alternative titles.

One call, one reply, straight through `candidates.review`. The client is a
parameter so the parsing and the rules can be tested without the network;
nothing here retries, batches or caches, because nothing yet knows whether a
second opinion is worth asking for.

The prompt asks for plain lines rather than JSON on purpose: a malformed line
costs one candidate, while a malformed JSON document costs the whole reply,
and `parse_candidates` already copes with the numbering and preamble Claude
adds anyway.

Usage:
    python -m src.generation.claude "How I Fixed My Sleep In 30 Days" education
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import CONTENT_TYPES  # noqa: E402
from src.generation.candidates import (  # noqa: E402
    TITLE_MAX_CHARS,
    opening_diversity,
    parse_candidates,
    review,
)

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
DEFAULT_N = 10

PROMPT = """\
A creator is about to publish a {content_type} video on YouTube. Their working \
title is:

{original}

Write {n} alternative titles for the same video.

Rules:
- Each title is about this video. Do not invent a different subject.
- Each title is at most {limit} characters, because YouTube truncates there.
- Vary the structure. Do not fill one template {n} times.
- No title is a restatement of another.

Reply with the {n} titles, one per line, and nothing else."""


def build_prompt(original, content_type, n):
    """The whole instruction, as one user turn."""
    return PROMPT.format(
        original=original, content_type=content_type, n=n, limit=TITLE_MAX_CHARS
    )


def default_client():
    """Imported here so the pure parts of this package stay importable
    without the SDK or a key."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; copy .env.example to .env")
    return anthropic.Anthropic()


def reply_text(message):
    """The text of a Messages API reply, ignoring any non-text block."""
    return "".join(block.text for block in message.content if block.type == "text")


def generate(original, content_type, n=DEFAULT_N, client=None, model=MODEL):
    """Ask for n alternatives and return them reviewed.

    The result is a `Review`, not a ranked list: see the note at the top of
    `candidates.py` on why nothing here claims one candidate will reach
    further than another.
    """
    client = client or default_client()
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": build_prompt(original, content_type, n)}],
    )
    return review(original, parse_candidates(reply_text(message)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("title")
    parser.add_argument("content_type", choices=CONTENT_TYPES)
    parser.add_argument("-n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    out = generate(args.title, args.content_type, n=args.n)

    print(f"{len(out.accepted)} of {args.n} candidates passed the rules")
    for candidate in out.accepted:
        print(f"  {len(candidate):>3}  {candidate}")
    for rejection in out.rejected:
        print(f"  dropped: {rejection.candidate!r} ({rejection.reason})")
    print(f"\nopening diversity {opening_diversity(out.accepted):.2f} "
          f"(1.0 is every candidate opening differently)")
    print("No reach score is shown: see docs/validation_protocol.md.")


if __name__ == "__main__":
    main()
