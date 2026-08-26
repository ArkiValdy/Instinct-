"""The contract. Both models implement score(); the app only calls this."""

from dataclasses import dataclass, field


@dataclass
class ScoreResult:
    score: float                      # 0 to 100
    drivers: dict = field(default_factory=dict)   # feature name -> contribution
    notes: list = field(default_factory=list)     # suggestion strings


def score_title(title: str, content_type: str) -> ScoreResult:
    raise NotImplementedError


def score_thumbnail(image_path: str, content_type: str) -> ScoreResult:
    raise NotImplementedError
