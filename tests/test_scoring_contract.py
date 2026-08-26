"""Both models must return the same shape. Run: pytest"""

from src.scoring import placeholder
from src.scoring.interface import ScoreResult


def test_title_returns_score_result():
    r = placeholder.score_title("How I Built This in 7 Days", "gaming")
    assert isinstance(r, ScoreResult)
    assert 0 <= r.score <= 100
    assert isinstance(r.drivers, dict)


def test_thumbnail_returns_score_result():
    r = placeholder.score_thumbnail("data/thumbnails/example.jpg", "gaming")
    assert isinstance(r, ScoreResult)
    assert 0 <= r.score <= 100
