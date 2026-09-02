"""Fake implementations of the scoring contract.

This exists so the interface pair can build and test the app before either
model is trained. The numbers mean nothing.

Delete this file once both real models are working.
"""

import random

from src.scoring.interface import ScoreResult

PLACEHOLDER_NOTE = "This output is a placeholder, not a real prediction."


def score_title(title: str, content_type: str) -> ScoreResult:
    """Return a random score for a title, in the shape of the real thing."""
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={
            "title_length": random.uniform(-10, 10),
            "has_number": random.uniform(-10, 10),
            "sentiment": random.uniform(-10, 10),
        },
        notes=[PLACEHOLDER_NOTE],
    )


def score_thumbnail(image_path: str, content_type: str) -> ScoreResult:
    """Return a random score for a thumbnail, in the shape of the real thing."""
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={
            "face_present": random.uniform(-10, 10),
            "brightness": random.uniform(-10, 10),
            "text_area": random.uniform(-10, 10),
        },
        notes=[PLACEHOLDER_NOTE],
    )
