"""Combined-score calculation for Phase 2.

Formula (defaults, configurable in ``config.py`` / ``backend/.env``)::

    combined = 0.70 * semantic_similarity + 0.30 * keyword_coverage

Both components are already on a 0-100 scale. The weighted result is clamped to
0-100 and rounded to two decimals so stored and displayed values agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import EVALUATION_WEIGHTS


@dataclass(frozen=True)
class CombinedScore:
    semantic_score: float
    keyword_score: float
    combined_score: float


def _normalize_component(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")

    return max(0.0, min(100.0, number))


def combine_scores(
    semantic_score: float,
    keyword_score: float,
) -> CombinedScore:
    """Combine component scores with the configured weights."""
    semantic = _normalize_component(semantic_score, "semantic_score")
    keyword = _normalize_component(keyword_score, "keyword_score")

    semantic_weight = EVALUATION_WEIGHTS["semantic"]
    keyword_weight = EVALUATION_WEIGHTS["keyword"]

    combined = semantic_weight * semantic + keyword_weight * keyword
    combined = max(0.0, min(100.0, combined))

    return CombinedScore(
        semantic_score=round(semantic, 2),
        keyword_score=round(keyword, 2),
        combined_score=round(combined, 2),
    )
