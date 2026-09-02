"""Both models must satisfy the scoring contract.

These run against the placeholder today. When a real model lands, point the
import at it and the same tests must still pass.
"""

import pytest

from src.scoring import placeholder
from src.scoring.interface import ScoreResult

CONTENT_TYPE = "gaming"


@pytest.fixture
def results():
    return [
        placeholder.score_title("How I Fixed My Sleep In 30 Days", CONTENT_TYPE),
        placeholder.score_thumbnail("data/thumbnails/example.jpg", CONTENT_TYPE),
    ]


def test_returns_score_result(results):
    for result in results:
        assert isinstance(result, ScoreResult)


def test_score_is_within_range(results):
    for result in results:
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 100


def test_drivers_is_a_dict(results):
    for result in results:
        assert isinstance(result.drivers, dict)


def test_notes_is_a_list_of_strings(results):
    for result in results:
        assert isinstance(result.notes, list)
        assert all(isinstance(note, str) for note in result.notes)


def test_interface_is_not_implemented():
    from src.scoring import interface

    with pytest.raises(NotImplementedError):
        interface.score_title("anything", CONTENT_TYPE)
    with pytest.raises(NotImplementedError):
        interface.score_thumbnail("anything.jpg", CONTENT_TYPE)
