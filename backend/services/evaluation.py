"""Phase 2 evaluation pipeline: usability -> keyword + semantic -> combined.

The transcript is only read. Nothing in this module updates the saved answer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from config import EVALUATION_MIN_MEANINGFUL_TOKENS
from services.keyword_evaluation import KeywordCoverage, keyword_coverage
from services.nlp_preprocessing import (
    assess_answer_usability,
    normalize_text,
)
from services.scoring import combine_scores
from services.semantic_similarity import semantic_similarity

STATUS_EVALUATED = "evaluated"
STATUS_UNEVALUABLE = "unevaluable"


class MissingReferenceAnswerError(Exception):
    """The question has no reference answer, so no score can be produced."""

    status_code = 409

    def __init__(self) -> None:
        self.message = (
            "This question has no reference answer, so it cannot be evaluated. "
            "Ask an administrator to add one to interview_questions."
        )
        super().__init__(self.message)


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    semantic_score: float
    keyword_score: float
    combined_score: float
    matched_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    status_detail: str | None = None


def fingerprint(text: str | None) -> str:
    """Hash normalized text to detect stale evaluations safely."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def evaluate_transcript(
    transcript: str | None,
    reference_answer: str | None,
) -> EvaluationResult:
    """Evaluate one saved transcript against a reference answer.

    Empty, filler-only, and very short answers are persisted with status
    ``unevaluable`` and zero component/combined scores. They never reach the
    semantic model, so a short but superficially similar sentence cannot obtain
    a misleadingly high score.
    """
    reference = normalize_text(reference_answer)
    if not reference:
        raise MissingReferenceAnswerError()

    usability = assess_answer_usability(
        transcript,
        min_meaningful_tokens=EVALUATION_MIN_MEANINGFUL_TOKENS,
    )
    coverage: KeywordCoverage = keyword_coverage(transcript, reference)

    if not usability.usable:
        return EvaluationResult(
            status=STATUS_UNEVALUABLE,
            semantic_score=0.0,
            keyword_score=0.0,
            combined_score=0.0,
            matched_keywords=coverage.matched,
            missing_keywords=coverage.missing,
            status_detail=usability.reason,
        )

    semantic_score = semantic_similarity(transcript, reference)
    combined = combine_scores(semantic_score, coverage.score)

    return EvaluationResult(
        status=STATUS_EVALUATED,
        semantic_score=combined.semantic_score,
        keyword_score=combined.keyword_score,
        combined_score=combined.combined_score,
        matched_keywords=coverage.matched,
        missing_keywords=coverage.missing,
        status_detail=None,
    )
