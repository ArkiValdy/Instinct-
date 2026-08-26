"""Fake scorer so the app can be built before the models exist. Delete at week 8."""

import random
from src.scoring.interface import ScoreResult


def score_title(title: str, content_type: str) -> ScoreResult:
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={"length": 0.3, "has_number": 0.2},
        notes=["placeholder output"],
    )


def score_thumbnail(image_path: str, content_type: str) -> ScoreResult:
    return ScoreResult(
        score=random.uniform(20, 90),
        drivers={"face_present": 0.4, "contrast": 0.1},
        notes=["placeholder output"],
    )
