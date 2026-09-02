"""The scoring contract.

The title model and the thumbnail model are built separately by different
pairs, and the app consumes both. Both must return a ScoreResult, so the app
can be built and tested against this interface before either model exists.

Changes to this file require agreement from all three pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScoreResult:
    """The result of scoring one title or one thumbnail.

    score: predicted reach, 0 to 100.
    drivers: feature name -> its contribution to the score.
    notes: human readable suggestions for the creator.
    """

    score: float = 0.0
    drivers: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def score_title(title: str, content_type: str) -> ScoreResult:
    """Score a proposed video title. Implemented by the title pair."""
    raise NotImplementedError


def score_thumbnail(image_path: str, content_type: str) -> ScoreResult:
    """Score a proposed thumbnail image. Implemented by the thumbnail pair."""
    raise NotImplementedError
