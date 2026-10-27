"""Phase 5 deterministic, evidence-based interview feedback.

Feedback is generated only from persisted, current sources:

- a Phase 2 ``AnswerEvaluation`` with matching transcript/reference hashes and
  status ``evaluated``;
- an optional Phase 3 ``ContentAnalysis`` with matching hashes and status
  ``completed``;
- an optional Phase 3 ``VoiceAnalysis`` with matching transcript/audio hashes,
  status ``completed``, and detected speech;
- an optional Phase 4 ``ScorePrediction`` that is current (not stale).

Nothing is invented. Missing, stale, unevaluable, or failed sources are excluded
and explained. Missing voice measurements are never treated as zero. The service
does not call an LLM, does not predict confidence/emotion/personality/honesty,
and never labels the rule-based baseline as a machine-learning prediction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from models import (
    AnswerEvaluation,
    AnswerFeedback,
    ContentAnalysis,
    InterviewAnswer,
    InterviewFeedback,
    InterviewQuestion,
    ScorePrediction,
    VoiceAnalysis,
)
from services.content_analysis import (
    STATUS_COMPLETED as CONTENT_STATUS_COMPLETED,
)
from services.evaluation import fingerprint
from services.ml_features import SourceSnapshot, build_source_snapshot
from services.ml_scoring import (
    METHOD_BASELINE,
    METHOD_ML,
    METHOD_NONE,
)
from services.voice_analysis import (
    STATUS_COMPLETED as VOICE_STATUS_COMPLETED,
)

logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

INTERVIEW_STATUS_COMPLETED = "completed"
INTERVIEW_STATUS_PARTIAL = "partial"
INTERVIEW_STATUS_UNAVAILABLE = "unavailable"

BAND_STRONG = "strong"
BAND_AVERAGE = "average"
BAND_WEAK = "weak"

FEEDBACK_STATUS_UNANSWERED = "unanswered"
FEEDBACK_STATUS_STALE = "stale"

ANSWER_LIMITATIONS = (
    "Feedback is generated deterministically from the saved Phase 2 evaluation, "
    "any current Phase 3 analyses, and any current Phase 4 prediction. No LLM or "
    "external feedback service is used.",
    "It is a practice aid, not a hiring decision or proof of knowledge.",
    "Content feedback measures lexical and semantic agreement with one reference "
    "answer, not factual correctness.",
    "Voice observations are objective acoustic measurements. They are not "
    "confidence, emotion, personality, honesty, intelligence, or communication-"
    "quality judgements.",
    "Missing or stale sources are excluded and explained. They are never treated "
    "as zero.",
)

INTERVIEW_LIMITATIONS = (
    "Interview feedback is an aggregate of deterministic per-answer feedback, "
    "not an independent assessment of a candidate.",
    "Unanswered questions and unavailable/stale sources are excluded from "
    "averages and are never scored as zero.",
    "The Phase 2 combined score compares one transcript with one reference answer; "
    "it is not factual correctness or overall ability.",
    "Voice measurements describe audio only. They are not confidence, emotion, "
    "personality, honesty, or communication quality.",
    "A rule-based baseline prediction is not a machine-learning prediction. An ML "
    "label appears only when the Phase 4 row records method 'ml_model'.",
)


class FeedbackValidationError(Exception):
    """A persisted source value is outside the documented range."""

    status_code = 422

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class AnswerFeedbackSource:
    fingerprint: str
    metadata: dict[str, object]
    snapshot: SourceSnapshot
    prediction: ScorePrediction | None
    prediction_stale: bool


@dataclass(frozen=True)
class InterviewFeedbackSource:
    fingerprint: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class AnswerFeedbackResult:
    status: str
    band: str | None
    status_detail: str | None
    summary: str
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    actionable_suggestions: list[str] = field(default_factory=list)
    matched_concepts: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    voice_observations: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)
    prediction_method: str = METHOD_NONE
    prediction_score: float | None = None
    prediction_is_ml: bool = False
    prediction_note: str | None = None
    voice_available: bool = False
    content_available: bool = False
    semantic_score: float | None = None
    keyword_score: float | None = None
    combined_score: float | None = None
    content_coverage_score: float | None = None
    source_fingerprint: str = ""
    source_versions: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InterviewFeedbackResult:
    status: str
    status_detail: str | None
    summary: str
    overall_strengths: list[str] = field(default_factory=list)
    overall_improvements: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    question_breakdown: list[dict[str, object]] = field(default_factory=list)
    questions_total: int = 0
    questions_answered: int = 0
    questions_unanswered: int = 0
    feedback_completed: int = 0
    feedback_unavailable: int = 0
    feedback_failed: int = 0
    feedback_stale: int = 0
    voice_available_count: int = 0
    ml_predictions: int = 0
    baseline_predictions: int = 0
    average_score: float | None = None
    average_predicted_score: float | None = None
    source_fingerprint: str = ""
    source_versions: dict[str, object] = field(default_factory=dict)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _round(value: float) -> float:
    return round(float(value), 2)


def _validate_score(value: float | int | None, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackValidationError(
            f"The {label} score is not a numeric value, so feedback was not "
            "generated from it."
        )
    number = float(value)
    if not math.isfinite(number):
        raise FeedbackValidationError(
            f"The {label} score is not finite, so feedback was not generated "
            "from it."
        )
    if not 0.0 <= number <= 100.0:
        raise FeedbackValidationError(
            f"The {label} score is outside the documented 0-100 range, so "
            "feedback was not generated from it."
        )
    return _round(number)


def _optional_score(value: float | int | None, label: str) -> float | None:
    if value is None:
        return None
    return _validate_score(value, label)


def _current_evaluation(
    db: Session, answer_id: int
) -> AnswerEvaluation | None:
    return db.scalar(
        select(AnswerEvaluation).where(
            AnswerEvaluation.answer_id == answer_id,
            AnswerEvaluation.methodology_version
            == config.EVALUATION_METHODOLOGY_VERSION,
        )
    )


def _current_content(db: Session, answer_id: int) -> ContentAnalysis | None:
    return db.scalar(
        select(ContentAnalysis).where(
            ContentAnalysis.answer_id == answer_id,
            ContentAnalysis.methodology_version
            == config.ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def _current_voice(db: Session, answer_id: int) -> VoiceAnalysis | None:
    return db.scalar(
        select(VoiceAnalysis).where(
            VoiceAnalysis.answer_id == answer_id,
            VoiceAnalysis.methodology_version
            == config.ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def _current_prediction(
    db: Session, answer_id: int
) -> ScorePrediction | None:
    return db.scalar(
        select(ScorePrediction).where(
            ScorePrediction.answer_id == answer_id,
            ScorePrediction.methodology_version
            == config.ML_PREDICTION_METHODOLOGY_VERSION,
        )
    )


def build_answer_feedback_source(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> AnswerFeedbackSource:
    """Fingerprint every source that can change an answer's feedback."""
    snapshot = build_source_snapshot(db, answer, question)
    prediction = _current_prediction(db, answer.id)

    prediction_stale = False
    prediction_metadata: dict[str, object] | None = None
    if prediction is not None:
        prediction_stale = (
            prediction.methodology_version
            != config.ML_PREDICTION_METHODOLOGY_VERSION
            or prediction.feature_schema_version
            != config.ML_FEATURE_SCHEMA_VERSION
            or prediction.source_fingerprint != snapshot.fingerprint
        )
        prediction_metadata = {
            "id": prediction.id,
            "status": prediction.status,
            "method": prediction.method,
            "score": prediction.score,
            "model_version": prediction.model_version,
            "feature_schema_version": prediction.feature_schema_version,
            "methodology_version": prediction.methodology_version,
            "source_fingerprint": prediction.source_fingerprint,
            "updated_at": _iso(prediction.updated_at),
            "is_stale": prediction_stale,
        }

    metadata: dict[str, object] = {
        **snapshot.metadata(),
        "feedback_methodology_version": config.FEEDBACK_METHODOLOGY_VERSION,
        "prediction": prediction_metadata,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return AnswerFeedbackSource(
        fingerprint=digest,
        metadata=metadata,
        snapshot=snapshot,
        prediction=prediction,
        prediction_stale=prediction_stale,
    )


def build_interview_feedback_source(
    db: Session,
    interview_id: int,
) -> InterviewFeedbackSource:
    """Fingerprint the interview questions and every answer source."""
    questions = list(
        db.scalars(
            select(InterviewQuestion).order_by(InterviewQuestion.id)
        ).all()
    )
    answers = list(
        db.scalars(
            select(InterviewAnswer).where(
                InterviewAnswer.interview_id == interview_id
            )
        ).all()
    )
    answer_by_question = {answer.question_id: answer for answer in answers}

    question_metadata: list[dict[str, object]] = []
    for question in questions:
        answer = answer_by_question.get(question.id)
        item: dict[str, object] = {
            "question_id": question.id,
            "question_text": question.question_text,
            "reference_hash": fingerprint(question.reference_answer or ""),
            "answer_id": answer.id if answer is not None else None,
        }
        if answer is not None:
            answer_source = build_answer_feedback_source(db, answer, question)
            item["feedback_source_fingerprint"] = answer_source.fingerprint
            item["transcript_hash"] = answer_source.snapshot.transcript_hash
            item["audio_fingerprint"] = answer_source.snapshot.audio_fingerprint
        question_metadata.append(item)

    metadata: dict[str, object] = {
        "interview_id": interview_id,
        "feedback_methodology_version": config.FEEDBACK_METHODOLOGY_VERSION,
        "questions": question_metadata,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return InterviewFeedbackSource(fingerprint=digest, metadata=metadata)


def answer_feedback_is_stale(
    feedback: AnswerFeedback,
    answer: InterviewAnswer,
    question: InterviewQuestion,
    db: Session,
) -> bool:
    """Read-time staleness for one persisted answer-feedback row."""
    if feedback.methodology_version != config.FEEDBACK_METHODOLOGY_VERSION:
        return True
    current = build_answer_feedback_source(db, answer, question)
    return feedback.source_fingerprint != current.fingerprint


def interview_feedback_is_stale(
    feedback: InterviewFeedback,
    db: Session,
) -> bool:
    """Read-time staleness for one persisted interview-feedback row."""
    if feedback.methodology_version != config.FEEDBACK_METHODOLOGY_VERSION:
        return True
    current = build_interview_feedback_source(db, feedback.interview_id)
    return feedback.source_fingerprint != current.fingerprint


def _terms_text(terms: list[str], limit: int = 8) -> str:
    shown = [f"'{term}'" for term in terms[:limit]]
    text = ", ".join(shown)
    if len(terms) > limit:
        text += f" and {len(terms) - limit} more"
    return text


def _band_for_score(score: float) -> str:
    if score >= config.FEEDBACK_STRONG_SCORE:
        return BAND_STRONG
    if score < config.FEEDBACK_WEAK_SCORE:
        return BAND_WEAK
    return BAND_AVERAGE


def _prediction_fields(
    prediction: ScorePrediction | None,
    stale: bool,
) -> tuple[str, float | None, bool, str]:
    """Return method, score, is-ML flag, and an explicit evidence note."""
    if prediction is None:
        return (
            METHOD_NONE,
            None,
            False,
            "No Phase 4 prediction is saved for this answer; feedback uses "
            "Phase 2/3 evidence only.",
        )
    if stale:
        return (
            METHOD_NONE,
            None,
            False,
            "The saved Phase 4 prediction is stale and was excluded from this "
            "feedback. Refresh the prediction before relying on it.",
        )
    if prediction.status != STATUS_COMPLETED or prediction.score is None:
        return (
            METHOD_NONE,
            None,
            False,
            f"The Phase 4 prediction status is '{prediction.status}', so no "
            "predicted score was used.",
        )
    if prediction.method not in (METHOD_ML, METHOD_BASELINE):
        return (
            METHOD_NONE,
            None,
            False,
            "The Phase 4 prediction has an unrecognized method and was excluded.",
        )
    try:
        score = _validate_score(prediction.score, "prediction")
    except FeedbackValidationError as exc:
        return METHOD_NONE, None, False, exc.message

    if prediction.method == METHOD_ML:
        version = prediction.model_version or "unknown version"
        return (
            METHOD_ML,
            score,
            True,
            f"The validated ML model ({version}) predicted {score:.2f}/100. "
            "This is an ML prediction, not a rule-based baseline.",
        )
    return (
        METHOD_BASELINE,
        score,
        False,
        f"The rule-based baseline predicted {score:.2f}/100. This is a "
        "rule-based baseline, not an ML prediction.",
    )


def _content_evidence(content: ContentAnalysis | None) -> dict[str, object] | None:
    if content is None:
        return None
    return {
        "id": content.id,
        "status": content.status,
        "status_detail": content.status_detail,
        "methodology_version": content.methodology_version,
        "coverage_score": content.coverage_score,
        "coverage_source": content.coverage_source,
        "matched_concepts": list(content.matched_concepts or []),
        "missing_concepts": list(content.missing_concepts or []),
        "filler_word_count": content.filler_word_count,
        "filler_rate_percent": content.filler_rate_percent,
        "repetition_rate_percent": content.repetition_rate_percent,
        "sentence_count": content.sentence_count,
        "average_sentence_words": content.average_sentence_words,
        "updated_at": _iso(content.updated_at),
    }


def _voice_evidence(voice: VoiceAnalysis | None) -> dict[str, object] | None:
    if voice is None:
        return None
    return {
        "id": voice.id,
        "status": voice.status,
        "status_detail": voice.status_detail,
        "methodology_version": voice.methodology_version,
        "speech_detected": voice.speech_detected,
        "duration_seconds": voice.duration_seconds,
        "speech_ratio_percent": voice.speech_ratio_percent,
        "speaking_rate_wpm": voice.speaking_rate_wpm,
        "pauses_per_minute": voice.pauses_per_minute,
        "rms_variation_percent": voice.rms_variation_percent,
        "clipping_percent": voice.clipping_percent,
        "truncated_for_analysis": voice.truncated_for_analysis,
        "quality_notes": list(voice.quality_notes or []),
        "updated_at": _iso(voice.updated_at),
    }


def _prediction_evidence(
    prediction: ScorePrediction | None,
    stale: bool,
) -> dict[str, object] | None:
    if prediction is None:
        return None
    return {
        "id": prediction.id,
        "status": prediction.status,
        "method": prediction.method,
        "score": prediction.score,
        "is_ml_prediction": prediction.method == METHOD_ML,
        "model_version": prediction.model_version,
        "feature_schema_version": prediction.feature_schema_version,
        "methodology_version": prediction.methodology_version,
        "source_fingerprint": prediction.source_fingerprint,
        "is_stale": stale,
        "updated_at": _iso(prediction.updated_at),
    }


def _evaluation_evidence(
    evaluation: AnswerEvaluation | None,
) -> dict[str, object] | None:
    if evaluation is None:
        return None
    return {
        "id": evaluation.id,
        "status": evaluation.status,
        "status_detail": evaluation.status_detail,
        "semantic_score": evaluation.semantic_score,
        "keyword_score": evaluation.keyword_score,
        "combined_score": evaluation.combined_score,
        "matched_keywords": list(evaluation.matched_keywords or []),
        "missing_keywords": list(evaluation.missing_keywords or []),
        "methodology_version": evaluation.methodology_version,
        "updated_at": _iso(evaluation.updated_at),
    }


def _evidence(
    question: InterviewQuestion,
    snapshot: AnswerFeedbackSource,
    evaluation: AnswerEvaluation | None,
    content: ContentAnalysis | None,
    voice: VoiceAnalysis | None,
    prediction_note: str,
) -> dict[str, object]:
    return {
        "question": {
            "id": question.id,
            "question_text": question.question_text,
            "category": question.category,
            "has_reference_answer": bool(
                (question.reference_answer or "").strip()
            ),
        },
        "phase2_evaluation": _evaluation_evidence(evaluation),
        "content_analysis": _content_evidence(content),
        "voice_analysis": _voice_evidence(voice),
        "prediction": _prediction_evidence(
            snapshot.prediction,
            snapshot.prediction_stale,
        ),
        "prediction_note": prediction_note,
        "band_thresholds": {
            "strong_at_or_above": config.FEEDBACK_STRONG_SCORE,
            "weak_below": config.FEEDBACK_WEAK_SCORE,
        },
        "source_fingerprint": snapshot.fingerprint,
    }


def _base_result(
    snapshot: AnswerFeedbackSource,
    *,
    status: str,
    band: str | None,
    status_detail: str | None,
    summary: str,
    prediction_method: str = METHOD_NONE,
    prediction_score: float | None = None,
    prediction_is_ml: bool = False,
    prediction_note: str | None = None,
    notices: list[str] | None = None,
) -> AnswerFeedbackResult:
    return AnswerFeedbackResult(
        status=status,
        band=band,
        status_detail=status_detail,
        summary=summary,
        notices=list(notices or []),
        limitations=list(ANSWER_LIMITATIONS),
        prediction_method=prediction_method,
        prediction_score=prediction_score,
        prediction_is_ml=prediction_is_ml,
        prediction_note=prediction_note,
        source_fingerprint=snapshot.fingerprint,
        source_versions=snapshot.metadata,
    )


def _voice_observations(voice: VoiceAnalysis) -> list[str]:
    observations: list[str] = []
    if voice.speaking_rate_wpm is not None:
        observations.append(
            "Measured speaking rate: "
            f"{float(voice.speaking_rate_wpm):.2f} words per minute (saved "
            "transcript words divided by detected speech time)."
        )
    if voice.pauses_per_minute is not None:
        observations.append(
            "Detected silent pauses: "
            f"{float(voice.pauses_per_minute):.2f} per minute."
        )
    if voice.speech_ratio_percent is not None:
        observations.append(
            "Detected speech covered "
            f"{float(voice.speech_ratio_percent):.2f}% of the analysed "
            "recording."
        )
    if voice.rms_variation_percent is not None:
        observations.append(
            "RMS loudness variation: "
            f"{float(voice.rms_variation_percent):.2f}% of the mean level."
        )
    if voice.clipping_percent is not None and float(voice.clipping_percent) > 0:
        observations.append(
            "Clipping affected "
            f"{float(voice.clipping_percent):.2f}% of the analysed samples."
        )
    if voice.truncated_for_analysis:
        observations.append(
            "Only part of the recording was analysed because it exceeded the "
            "configured analysis limit; the metrics describe the analysed "
            "portion."
        )
    for note in voice.quality_notes or []:
        observations.append(str(note))
    return observations


def _content_statistics(
    content: ContentAnalysis,
) -> tuple[dict[str, float | int | None], list[str]]:
    stats: dict[str, float | int | None] = {
        "filler_word_count": content.filler_word_count,
        "filler_rate_percent": content.filler_rate_percent,
        "repetition_rate_percent": content.repetition_rate_percent,
        "sentence_count": content.sentence_count,
        "average_sentence_words": content.average_sentence_words,
    }
    missing = [name for name, value in stats.items() if value is None]
    return stats, missing


def generate_answer_feedback(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> AnswerFeedbackResult:
    """Generate deterministic, question-specific feedback for one answer."""
    snapshot = build_answer_feedback_source(db, answer, question)
    evaluation = _current_evaluation(db, answer.id)
    content = _current_content(db, answer.id)
    voice = _current_voice(db, answer.id)

    prediction_method, prediction_score, prediction_is_ml, prediction_note = (
        _prediction_fields(snapshot.prediction, snapshot.prediction_stale)
    )
    prediction_notices: list[str] = []
    if snapshot.prediction is None:
        prediction_notices.append(prediction_note)
    elif prediction_method == METHOD_NONE:
        prediction_notices.append(prediction_note)
    elif prediction_is_ml:
        prediction_notices.append(
            "The prediction below is an ML prediction from a validated model "
            "artifact; it is not a rule-based baseline."
        )
    else:
        prediction_notices.append(
            "The prediction below is a rule-based baseline, not an ML prediction."
        )

    evidence = _evidence(
        question,
        snapshot,
        evaluation,
        content,
        voice,
        prediction_note,
    )

    reference = (question.reference_answer or "").strip()
    if not reference:
        detail = (
            "This question has no reference answer, so answer-specific content "
            "feedback cannot be generated. Add a reference answer to the "
            "question and evaluate the answer first."
        )
        return _base_result(
            snapshot,
            status=STATUS_UNAVAILABLE,
            band=None,
            status_detail=detail,
            summary=f"Feedback unavailable: {detail}",
            prediction_method=prediction_method,
            prediction_score=prediction_score,
            prediction_is_ml=prediction_is_ml,
            prediction_note=prediction_note,
            notices=[detail, *prediction_notices],
        )

    if evaluation is None:
        detail = (
            "No Phase 2 evaluation exists for this answer. Evaluate the answer "
            "before generating feedback."
        )
        return _base_result(
            snapshot,
            status=STATUS_UNAVAILABLE,
            band=None,
            status_detail=detail,
            summary=f"Feedback unavailable: {detail}",
            prediction_method=prediction_method,
            prediction_score=prediction_score,
            prediction_is_ml=prediction_is_ml,
            prediction_note=prediction_note,
            notices=[detail, *prediction_notices],
        )

    if (
        evaluation.transcript_hash != snapshot.snapshot.transcript_hash
        or evaluation.reference_hash != snapshot.snapshot.reference_hash
    ):
        detail = (
            "The saved Phase 2 evaluation is stale because the transcript or "
            "reference answer changed. Re-evaluate the answer before generating "
            "feedback. The old score was not reused."
        )
        return _base_result(
            snapshot,
            status=STATUS_UNAVAILABLE,
            band=None,
            status_detail=detail,
            summary=f"Feedback unavailable: {detail}",
            prediction_method=prediction_method,
            prediction_score=prediction_score,
            prediction_is_ml=prediction_is_ml,
            prediction_note=prediction_note,
            notices=[detail, *prediction_notices],
        )

    if evaluation.status != "evaluated":
        detail = (
            evaluation.status_detail
            or "The Phase 2 evaluation did not produce a usable score."
        )
        detail = f"{detail} Feedback was not generated from an unevaluable answer."
        return _base_result(
            snapshot,
            status=STATUS_UNAVAILABLE,
            band=None,
            status_detail=detail,
            summary=f"Feedback unavailable: {detail}",
            prediction_method=prediction_method,
            prediction_score=prediction_score,
            prediction_is_ml=prediction_is_ml,
            prediction_note=prediction_note,
            notices=[detail, *prediction_notices],
        )

    try:
        semantic = _validate_score(evaluation.semantic_score, "semantic")
        keyword = _validate_score(evaluation.keyword_score, "keyword")
        combined = _validate_score(evaluation.combined_score, "combined")
    except FeedbackValidationError as exc:
        detail = (
            f"{exc.message} The invalid Phase 2 value was not converted into "
            "feedback."
        )
        return _base_result(
            snapshot,
            status=STATUS_FAILED,
            band=None,
            status_detail=detail,
            summary=f"Feedback generation failed: {detail}",
            prediction_method=prediction_method,
            prediction_score=prediction_score,
            prediction_is_ml=prediction_is_ml,
            prediction_note=prediction_note,
            notices=[detail, *prediction_notices],
        )

    notices: list[str] = list(prediction_notices)
    matched = list(evaluation.matched_keywords or [])
    missing = list(evaluation.missing_keywords or [])
    content_available = False
    content_stats: dict[str, float | int | None] = {}
    content_coverage: float | None = None

    if content is None:
        notices.append(
            "Content analysis is missing. Content statistics were excluded and "
            "were not treated as zero; coverage below comes from the Phase 2 "
            "evaluation."
        )
    elif (
        content.transcript_hash != snapshot.snapshot.transcript_hash
        or content.reference_hash != snapshot.snapshot.reference_hash
    ):
        notices.append(
            "The saved content analysis is stale because the transcript or "
            "reference answer changed. Its statistics were excluded, not "
            "treated as zero."
        )
    elif content.status != CONTENT_STATUS_COMPLETED:
        notices.append(
            f"The content analysis status is '{content.status}', so its "
            "statistics were excluded, not treated as zero."
        )
    else:
        content_available = True
        if content.matched_concepts:
            matched = list(content.matched_concepts)
        if content.missing_concepts:
            missing = list(content.missing_concepts)
        try:
            content_coverage = _optional_score(
                content.coverage_score,
                "content coverage",
            )
        except FeedbackValidationError as exc:
            notices.append(f"{exc.message} Coverage was excluded.")
        content_stats, missing_stats = _content_statistics(content)
        if missing_stats:
            notices.append(
                "Some content statistics were not stored and were omitted: "
                + ", ".join(missing_stats)
                + "."
            )

    voice_available = False
    voice_observations: list[str] = []
    if voice is None:
        notices.append(
            "Voice analysis is missing. No delivery measurements are available; "
            "missing voice values were not treated as zero."
        )
    elif (
        voice.transcript_hash != snapshot.snapshot.transcript_hash
        or voice.audio_sha256 != snapshot.snapshot.audio_fingerprint
    ):
        notices.append(
            "The saved voice analysis is stale because the recording or "
            "transcript changed. Its measurements were excluded, not treated as "
            "zero."
        )
    elif voice.status != VOICE_STATUS_COMPLETED or not voice.speech_detected:
        notices.append(
            f"The voice analysis status is '{voice.status}' with "
            f"speech_detected={voice.speech_detected}. No reliable delivery "
            "measurements were included, and missing values were not treated as "
            "zero."
        )
    else:
        voice_observations = _voice_observations(voice)
        voice_available = bool(voice_observations)
        if not voice_available:
            notices.append(
                "The voice analysis completed but stored no reliable delivery "
                "measurements."
            )

    band = _band_for_score(combined)
    strengths: list[str] = []
    improvements: list[str] = []
    suggestions: list[str] = []

    if band == BAND_STRONG:
        strengths.append(
            f"Strong content match: {combined:.2f}/100 on the Phase 2 "
            f"reference comparison (semantic {semantic:.2f}, keyword "
            f"{keyword:.2f})."
        )
    elif semantic >= config.FEEDBACK_STRONG_SCORE:
        strengths.append(
            f"Semantic similarity with the reference answer was "
            f"{semantic:.2f}/100."
        )
    if keyword >= config.FEEDBACK_STRONG_SCORE:
        strengths.append(
            f"Keyword coverage of reference terminology was {keyword:.2f}/100."
        )
    if matched:
        strengths.append(
            "Reference concepts found in the transcript: "
            f"{_terms_text(matched)}."
        )
    if content_available:
        filler_rate = content_stats.get("filler_rate_percent")
        repetition_rate = content_stats.get("repetition_rate_percent")
        average_words = content_stats.get("average_sentence_words")
        filler_count = content_stats.get("filler_word_count")
        if filler_rate is not None and float(filler_rate) == 0.0:
            strengths.append("No filler words were detected in the transcript.")
        elif (
            filler_rate is not None
            and float(filler_rate) <= config.FEEDBACK_FILLER_RATE_MAX_PERCENT
        ):
            strengths.append(
                f"Filler-word share was low at {float(filler_rate):.2f}% "
                f"({filler_count} filler word(s))."
            )
        if (
            repetition_rate is not None
            and float(repetition_rate)
            <= config.FEEDBACK_REPETITION_RATE_MAX_PERCENT
        ):
            strengths.append(
                "Repeated meaningful terms accounted for "
                f"{float(repetition_rate):.2f}% of meaningful words."
            )
        if (
            average_words is not None
            and config.FEEDBACK_SENTENCE_WORDS_MIN
            <= float(average_words)
            <= config.FEEDBACK_SENTENCE_WORDS_MAX
        ):
            strengths.append(
                f"Average sentence length was {float(average_words):.2f} words."
            )
    if voice_available and voice is not None and voice.speaking_rate_wpm is not None:
        rate = float(voice.speaking_rate_wpm)
        if (
            config.FEEDBACK_SPEAKING_RATE_MIN_WPM
            <= rate
            <= config.FEEDBACK_SPEAKING_RATE_MAX_WPM
        ):
            strengths.append(
                f"Measured speaking rate was {rate:.2f} words per minute, inside "
                "the baseline's "
                f"{config.FEEDBACK_SPEAKING_RATE_MIN_WPM:.0f}-"
                f"{config.FEEDBACK_SPEAKING_RATE_MAX_WPM:.0f} words-per-minute "
                "reference range."
            )

    if missing:
        improvements.append(
            "The reference answer contains concepts that were not found in the "
            f"transcript: {_terms_text(missing)}."
        )
        suggestions.append(
            "Add one sentence that explicitly defines or explains: "
            f"{_terms_text(missing)}."
        )
    if semantic < config.FEEDBACK_STRONG_SCORE:
        improvements.append(
            f"Semantic similarity was {semantic:.2f}/100. The explanation did "
            "not closely match the reference answer's meaning."
        )
        suggestions.append(
            "Restate the core definition in the reference answer's terms before "
            "adding examples or background."
        )
    if keyword < config.FEEDBACK_STRONG_SCORE:
        improvements.append(
            f"Keyword coverage was {keyword:.2f}/100. Some expected reference "
            "terminology is absent."
        )
        suggestions.append(
            "Use the missing reference terms where they genuinely apply instead "
            "of adding them as an unrelated list."
        )
    if combined < config.FEEDBACK_WEAK_SCORE:
        improvements.append(
            f"The combined Phase 2 score was {combined:.2f}/100, below the "
            f"documented weak threshold of {config.FEEDBACK_WEAK_SCORE:.2f}."
        )
        suggestions.append(
            "Review the reference answer, then re-answer this question with the "
            "core definition and the missing concepts."
        )
    if content_available:
        filler_rate = content_stats.get("filler_rate_percent")
        repetition_rate = content_stats.get("repetition_rate_percent")
        average_words = content_stats.get("average_sentence_words")
        filler_count = content_stats.get("filler_word_count")
        if (
            filler_rate is not None
            and float(filler_rate) > config.FEEDBACK_FILLER_RATE_MAX_PERCENT
        ):
            improvements.append(
                f"Filler-word share was {float(filler_rate):.2f}% "
                f"({filler_count} filler word(s)), above the documented "
                f"{config.FEEDBACK_FILLER_RATE_MAX_PERCENT:.2f}% guideline."
            )
            suggestions.append(
                "Replace filler words with a brief pause so the substantive "
                "content is easier to follow."
            )
        if (
            repetition_rate is not None
            and float(repetition_rate)
            > config.FEEDBACK_REPETITION_RATE_MAX_PERCENT
        ):
            improvements.append(
                "Repeated meaningful terms accounted for "
                f"{float(repetition_rate):.2f}% of meaningful words, above the "
                f"documented {config.FEEDBACK_REPETITION_RATE_MAX_PERCENT:.2f}% "
                "guideline."
            )
            suggestions.append(
                "Remove repeated wording and use the space for the missing "
                "reference concepts."
            )
        if (
            average_words is not None
            and float(average_words) < config.FEEDBACK_SENTENCE_WORDS_MIN
        ):
            improvements.append(
                f"Average sentence length was {float(average_words):.2f} words, "
                "below the documented "
                f"{config.FEEDBACK_SENTENCE_WORDS_MIN:.0f}-word guideline."
            )
            suggestions.append(
                "Expand very short statements into complete explanations with a "
                "subject, mechanism, and result."
            )
        elif (
            average_words is not None
            and float(average_words) > config.FEEDBACK_SENTENCE_WORDS_MAX
        ):
            improvements.append(
                f"Average sentence length was {float(average_words):.2f} words, "
                "above the documented "
                f"{config.FEEDBACK_SENTENCE_WORDS_MAX:.0f}-word guideline."
            )
            suggestions.append(
                "Split long sentences so each one makes a single clear point."
            )
    if voice is None:
        suggestions.append(
            "Record and analyse the original audio if you want objective delivery "
            "measurements. Do not infer confidence or communication quality from "
            "the transcript."
        )
    elif voice_available and voice.speaking_rate_wpm is not None:
        rate = float(voice.speaking_rate_wpm)
        if (
            rate < config.FEEDBACK_SPEAKING_RATE_MIN_WPM
            or rate > config.FEEDBACK_SPEAKING_RATE_MAX_WPM
        ):
            improvements.append(
                f"Measured speaking rate was {rate:.2f} words per minute, outside "
                "the baseline's "
                f"{config.FEEDBACK_SPEAKING_RATE_MIN_WPM:.0f}-"
                f"{config.FEEDBACK_SPEAKING_RATE_MAX_WPM:.0f} words-per-minute "
                "reference range."
            )
            suggestions.append(
                "Compare the measured speaking rate with the baseline reference "
                "range and adjust pacing if that target is useful."
            )
    if content is None:
        suggestions.append(
            "Run content analysis for this answer to include filler, repetition, "
            "and sentence-length evidence."
        )
    if snapshot.prediction is None:
        suggestions.append(
            "Optionally generate a Phase 4 prediction to see a separately "
            "labelled baseline or ML score. It is not required for this feedback."
        )
    elif snapshot.prediction_stale:
        suggestions.append(
            "Refresh the stale Phase 4 prediction before relying on its score."
        )

    if not improvements:
        improvements.append(
            "No specific content weakness was identified from the available "
            "evidence; review the reference answer for any nuance it includes "
            "that your transcript does not."
        )
    if not suggestions:
        suggestions.append(
            "No additional action is required from the available evidence. "
            "Re-read the reference answer before your next attempt."
        )

    concept_sentence = ""
    if matched:
        concept_sentence += (
            f" Reference concepts found: {_terms_text(matched)}."
        )
    if missing:
        concept_sentence += (
            f" Reference concepts not found: {_terms_text(missing)}."
        )

    if band == BAND_STRONG:
        summary = (
            f"Strong content match: {combined:.2f}/100 on the Phase 2 reference "
            f"comparison (semantic {semantic:.2f}, keyword {keyword:.2f})."
        )
    elif band == BAND_WEAK:
        summary = (
            f"This answer needs revision: {combined:.2f}/100 on the Phase 2 "
            f"reference comparison (semantic {semantic:.2f}, keyword "
            f"{keyword:.2f}). The available evidence shows a substantial gap "
            "from the reference answer."
        )
    else:
        summary = (
            f"Mixed content match: {combined:.2f}/100 on the Phase 2 reference "
            f"comparison (semantic {semantic:.2f}, keyword {keyword:.2f}). The "
            "answer covers part of the reference but has identifiable gaps."
        )
    summary += concept_sentence
    if content_available:
        available_stats: list[str] = []
        if content_stats.get("filler_rate_percent") is not None:
            available_stats.append(
                f"filler rate {float(content_stats['filler_rate_percent']):.2f}%"
            )
        if content_stats.get("repetition_rate_percent") is not None:
            available_stats.append(
                "repetition rate "
                f"{float(content_stats['repetition_rate_percent']):.2f}%"
            )
        if content_stats.get("average_sentence_words") is not None:
            available_stats.append(
                "average sentence length "
                f"{float(content_stats['average_sentence_words']):.2f} words"
            )
        if available_stats:
            summary += (
                " A current content analysis was included ("
                + ", ".join(available_stats)
                + ")."
            )
    else:
        summary += " No current content analysis was included."
    if voice_available:
        summary += (
            f" {len(voice_observations)} objective voice measurement(s) were "
            "included."
        )
    else:
        summary += (
            " Voice measurements were unavailable and were not treated as zero."
        )

    return AnswerFeedbackResult(
        status=STATUS_COMPLETED,
        band=band,
        status_detail=None,
        summary=summary,
        strengths=strengths,
        improvements=improvements,
        actionable_suggestions=suggestions,
        matched_concepts=matched,
        missing_concepts=missing,
        voice_observations=voice_observations,
        notices=notices,
        limitations=list(ANSWER_LIMITATIONS),
        evidence=evidence,
        prediction_method=prediction_method,
        prediction_score=prediction_score,
        prediction_is_ml=prediction_is_ml,
        prediction_note=prediction_note,
        voice_available=voice_available,
        content_available=content_available,
        semantic_score=semantic,
        keyword_score=keyword,
        combined_score=combined,
        content_coverage_score=content_coverage,
        source_fingerprint=snapshot.fingerprint,
        source_versions=snapshot.metadata,
    )


def answer_feedback_is_reusable(
    feedback: AnswerFeedback,
    result: AnswerFeedbackResult,
) -> bool:
    """True when regenerating would not change the persisted result."""
    return (
        feedback.source_fingerprint == result.source_fingerprint
        and feedback.methodology_version == config.FEEDBACK_METHODOLOGY_VERSION
        and feedback.status == result.status
        and feedback.band == result.band
        and feedback.summary == result.summary
    )


def apply_answer_feedback(
    row: AnswerFeedback,
    result: AnswerFeedbackResult,
) -> None:
    """Copy a result onto its persisted row (upsert in place)."""
    row.status = result.status
    row.band = result.band
    row.status_detail = result.status_detail
    row.summary = result.summary
    row.strengths = list(result.strengths)
    row.improvements = list(result.improvements)
    row.actionable_suggestions = list(result.actionable_suggestions)
    row.matched_concepts = list(result.matched_concepts)
    row.missing_concepts = list(result.missing_concepts)
    row.voice_observations = list(result.voice_observations)
    row.notices = list(result.notices)
    row.limitations = list(result.limitations)
    row.evidence = dict(result.evidence)
    row.prediction_method = result.prediction_method
    row.prediction_score = result.prediction_score
    row.prediction_is_ml = result.prediction_is_ml
    row.prediction_note = result.prediction_note
    row.voice_available = result.voice_available
    row.content_available = result.content_available
    row.semantic_score = result.semantic_score
    row.keyword_score = result.keyword_score
    row.combined_score = result.combined_score
    row.content_coverage_score = result.content_coverage_score
    row.source_fingerprint = result.source_fingerprint
    row.source_versions = dict(result.source_versions)
    row.methodology_version = config.FEEDBACK_METHODOLOGY_VERSION


def _completed_pairs(
    questions: list[InterviewQuestion],
    answer_by_question: dict[int, InterviewAnswer],
    feedback_by_answer: dict[int, AnswerFeedback],
    db: Session,
) -> list[tuple[int, InterviewQuestion, InterviewAnswer, AnswerFeedback]]:
    """Completed, current feedback in question order with question numbers."""
    pairs: list[tuple[int, InterviewQuestion, InterviewAnswer, AnswerFeedback]] = []
    for index, question in enumerate(questions, start=1):
        answer = answer_by_question.get(question.id)
        if answer is None:
            continue
        feedback = feedback_by_answer.get(answer.id)
        if feedback is None:
            continue
        if feedback.status != STATUS_COMPLETED:
            continue
        if answer_feedback_is_stale(feedback, answer, question, db):
            continue
        pairs.append((index, question, answer, feedback))
    return pairs


def _collect_notes(
    pairs: list[tuple[int, InterviewQuestion, InterviewAnswer, AnswerFeedback]],
    field_name: str,
    *,
    limit: int = 6,
    per_question: int = 1,
    reverse: bool = False,
) -> list[str]:
    ordered = sorted(
        pairs,
        key=lambda item: (
            item[3].combined_score
            if item[3].combined_score is not None
            else -1.0
        ),
        reverse=reverse,
    )
    notes: list[str] = []
    for index, _question, _answer, feedback in ordered:
        values = getattr(feedback, field_name) or []
        for value in list(values)[:per_question]:
            note = f"Q{index}: {value}"
            if note not in notes:
                notes.append(note)
            if len(notes) >= limit:
                return notes
    return notes


def _dedupe(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def build_interview_feedback_result(
    db: Session,
    interview_id: int,
) -> InterviewFeedbackResult:
    """Aggregate current per-answer feedback into one interview result."""
    source = build_interview_feedback_source(db, interview_id)
    questions = list(
        db.scalars(
            select(InterviewQuestion).order_by(InterviewQuestion.id)
        ).all()
    )
    answers = list(
        db.scalars(
            select(InterviewAnswer).where(
                InterviewAnswer.interview_id == interview_id
            )
        ).all()
    )
    answer_by_question = {answer.question_id: answer for answer in answers}
    feedback_rows = list(
        db.scalars(
            select(AnswerFeedback).where(
                AnswerFeedback.interview_id == interview_id,
                AnswerFeedback.methodology_version
                == config.FEEDBACK_METHODOLOGY_VERSION,
            )
        ).all()
    )
    feedback_by_answer = {row.answer_id: row for row in feedback_rows}

    breakdown: list[dict[str, object]] = []
    questions_answered = 0
    questions_unanswered = 0
    feedback_completed = 0
    feedback_unavailable = 0
    feedback_failed = 0
    feedback_stale = 0
    voice_available_count = 0
    ml_predictions = 0
    baseline_predictions = 0
    scores: list[float] = []
    predicted_scores: list[float] = []

    for index, question in enumerate(questions, start=1):
        answer = answer_by_question.get(question.id)
        if answer is None:
            questions_unanswered += 1
            breakdown.append(
                {
                    "question_id": question.id,
                    "question_text": question.question_text,
                    "category": question.category,
                    "question_number": index,
                    "answer_id": None,
                    "feedback_id": None,
                    "status": FEEDBACK_STATUS_UNANSWERED,
                    "band": None,
                    "combined_score": None,
                    "summary": "No transcript has been saved for this question.",
                    "strengths": [],
                    "improvements": [],
                    "missing_concepts": [],
                    "voice_available": False,
                    "prediction_method": None,
                    "prediction_is_ml": False,
                    "status_detail": (
                        "Unanswered questions are excluded from feedback and "
                        "averages; they are not scored as zero."
                    ),
                }
            )
            continue

        questions_answered += 1
        feedback = feedback_by_answer.get(answer.id)
        if feedback is None:
            feedback_unavailable += 1
            breakdown.append(
                {
                    "question_id": question.id,
                    "question_text": question.question_text,
                    "category": question.category,
                    "question_number": index,
                    "answer_id": answer.id,
                    "feedback_id": None,
                    "status": STATUS_UNAVAILABLE,
                    "band": None,
                    "combined_score": None,
                    "summary": "Feedback has not been generated for this answer.",
                    "strengths": [],
                    "improvements": [],
                    "missing_concepts": [],
                    "voice_available": False,
                    "prediction_method": None,
                    "prediction_is_ml": False,
                    "status_detail": (
                        "Generate per-answer feedback for this saved answer."
                    ),
                }
            )
            continue

        stale = answer_feedback_is_stale(feedback, answer, question, db)
        item_status = FEEDBACK_STATUS_STALE if stale else feedback.status
        if stale:
            feedback_stale += 1
        elif feedback.status == STATUS_COMPLETED:
            feedback_completed += 1
        elif feedback.status == STATUS_FAILED:
            feedback_failed += 1
        else:
            feedback_unavailable += 1

        if not stale and feedback.status == STATUS_COMPLETED:
            if feedback.combined_score is not None:
                scores.append(float(feedback.combined_score))
            if feedback.voice_available:
                voice_available_count += 1
            if (
                feedback.prediction_method == METHOD_ML
                and feedback.prediction_score is not None
            ):
                ml_predictions += 1
            elif (
                feedback.prediction_method == METHOD_BASELINE
                and feedback.prediction_score is not None
            ):
                baseline_predictions += 1
            if feedback.prediction_score is not None:
                predicted_scores.append(float(feedback.prediction_score))

        breakdown.append(
            {
                "question_id": question.id,
                "question_text": question.question_text,
                "category": question.category,
                "question_number": index,
                "answer_id": answer.id,
                "feedback_id": feedback.id,
                "status": item_status,
                "band": feedback.band,
                "combined_score": feedback.combined_score,
                "summary": feedback.summary,
                "strengths": list(feedback.strengths or []),
                "improvements": list(feedback.improvements or []),
                "missing_concepts": list(feedback.missing_concepts or []),
                "voice_available": bool(feedback.voice_available),
                "prediction_method": feedback.prediction_method,
                "prediction_is_ml": bool(feedback.prediction_is_ml),
                "status_detail": feedback.status_detail,
            }
        )

    completed_pairs = _completed_pairs(
        questions,
        answer_by_question,
        feedback_by_answer,
        db,
    )
    average_score = (
        _round(sum(scores) / len(scores)) if scores else None
    )
    average_predicted_score = (
        _round(sum(predicted_scores) / len(predicted_scores))
        if predicted_scores
        else None
    )

    unavailable_total = feedback_unavailable + feedback_failed + feedback_stale
    if questions_answered == 0:
        status = INTERVIEW_STATUS_UNAVAILABLE
        status_detail = (
            "No answer has been saved in this interview, so no interview-level "
            "feedback can be produced."
        )
    elif feedback_completed == 0:
        status = INTERVIEW_STATUS_UNAVAILABLE
        status_detail = (
            "None of the saved answers has current completed feedback. "
            f"{unavailable_total} answer(s) are unavailable, failed, or stale."
        )
    elif unavailable_total > 0 or questions_unanswered > 0:
        status = INTERVIEW_STATUS_PARTIAL
        status_detail = (
            f"{feedback_completed} of {len(questions)} question(s) have current "
            f"completed feedback; {questions_unanswered} unanswered; "
            f"{unavailable_total} answered question(s) unavailable, failed, or "
            "stale."
        )
    else:
        status = INTERVIEW_STATUS_COMPLETED
        status_detail = (
            f"All {len(questions)} question(s) have current completed feedback."
        )

    total_questions = len(questions)
    if status == INTERVIEW_STATUS_UNAVAILABLE:
        summary = (
            "No completed answer feedback is available for this interview. "
            "Save at least one answer, evaluate it, and generate answer feedback "
            "before relying on an interview-level summary."
        )
        if questions_unanswered:
            summary += (
                f" {questions_unanswered} of {total_questions} question(s) are "
                "unanswered and are not scored as zero."
            )
    else:
        summary = (
            f"Completed feedback covers {feedback_completed} of {total_questions} "
            "questions."
        )
        if average_score is not None:
            summary += (
                " Across those questions, the average Phase 2 combined score was "
                f"{average_score:.2f}/100."
            )
        if completed_pairs:
            highest = max(
                completed_pairs,
                key=lambda item: (
                    item[3].combined_score
                    if item[3].combined_score is not None
                    else -1.0
                ),
            )
            lowest = min(
                completed_pairs,
                key=lambda item: (
                    item[3].combined_score
                    if item[3].combined_score is not None
                    else 101.0
                ),
            )
            highest_score = highest[3].combined_score
            lowest_score = lowest[3].combined_score
            if highest[0] == lowest[0]:
                summary += (
                    f" Highest combined score: Q{highest[0]} at "
                    f"{float(highest_score):.2f}/100."
                )
            else:
                summary += (
                    f" Highest combined score: Q{highest[0]} at "
                    f"{float(highest_score):.2f}/100; lowest: Q{lowest[0]} at "
                    f"{float(lowest_score):.2f}/100."
                )
        if questions_unanswered:
            summary += (
                f" {questions_unanswered} question(s) were unanswered and are "
                "excluded, not scored as zero."
            )
        if unavailable_total:
            summary += (
                f" {unavailable_total} answered question(s) lacked current "
                "completed feedback and are excluded from the average."
            )
        summary += (
            f" Objective voice measurements were available for "
            f"{voice_available_count} completed answer(s)."
        )
        if ml_predictions == 0 and baseline_predictions > 0:
            summary += (
                " No ML prediction was available; completed predictions are "
                "rule-based baseline scores, not ML predictions."
            )
        elif ml_predictions or baseline_predictions:
            summary += (
                f" Completed predictions: {ml_predictions} ML and "
                f"{baseline_predictions} rule-based baseline."
            )
        else:
            summary += " No completed Phase 4 prediction was included."

    overall_strengths = _collect_notes(
        completed_pairs,
        "strengths",
        limit=6,
        per_question=1,
        reverse=True,
    )
    overall_improvements = _collect_notes(
        completed_pairs,
        "improvements",
        limit=6,
        per_question=1,
        reverse=False,
    )
    recommendations = _collect_notes(
        completed_pairs,
        "actionable_suggestions",
        limit=6,
        per_question=1,
        reverse=False,
    )
    if questions_unanswered:
        recommendations.append(
            f"Answer the {questions_unanswered} unanswered question(s) to make "
            "the interview summary complete."
        )
    if feedback_stale:
        recommendations.append(
            f"Regenerate stale feedback for {feedback_stale} answer(s) before "
            "using it."
        )
    if feedback_unavailable or feedback_failed:
        recommendations.append(
            "Re-evaluate answers with unavailable or failed feedback, then "
            "regenerate their feedback."
        )
    if voice_available_count < feedback_completed:
        recommendations.append(
            "Record and analyse audio for answers without voice measurements if "
            "delivery evidence is useful; no voice-based claim is made without "
            "measurements."
        )
    if baseline_predictions > 0 and ml_predictions == 0:
        recommendations.append(
            "Treat the saved rule-based baseline predictions as a transparent "
            "aid, not as ML predictions."
        )
    recommendations = _dedupe(recommendations, limit=10)

    notices: list[str] = [
        "Feedback is deterministic and evidence-based; no LLM was used.",
    ]
    if questions_unanswered:
        notices.append(
            f"{questions_unanswered} question(s) were unanswered and excluded "
            "from feedback and averages; they were not treated as zero."
        )
    if unavailable_total:
        notices.append(
            f"{unavailable_total} answered question(s) were unavailable, failed, "
            "or stale and were excluded from averages."
        )
    if voice_available_count < feedback_completed:
        notices.append(
            f"Voice measurements were unavailable for "
            f"{feedback_completed - voice_available_count} completed answer(s). "
            "No delivery value was inferred in their place."
        )
    if baseline_predictions > 0 and ml_predictions == 0:
        notices.append(
            "All included predictions use the rule-based baseline. The baseline "
            "is not a machine-learning prediction."
        )
    elif ml_predictions:
        notices.append(
            f"{ml_predictions} included prediction(s) are ML predictions and "
            f"{baseline_predictions} are rule-based baseline scores."
        )
    else:
        notices.append(
            "No current completed Phase 4 prediction was included."
        )
    notices.append(
        "Per-answer notices record every excluded or missing source."
    )

    return InterviewFeedbackResult(
        status=status,
        status_detail=status_detail,
        summary=summary,
        overall_strengths=overall_strengths,
        overall_improvements=overall_improvements,
        recommendations=recommendations,
        notices=notices,
        limitations=list(INTERVIEW_LIMITATIONS),
        question_breakdown=breakdown,
        questions_total=total_questions,
        questions_answered=questions_answered,
        questions_unanswered=questions_unanswered,
        feedback_completed=feedback_completed,
        feedback_unavailable=feedback_unavailable,
        feedback_failed=feedback_failed,
        feedback_stale=feedback_stale,
        voice_available_count=voice_available_count,
        ml_predictions=ml_predictions,
        baseline_predictions=baseline_predictions,
        average_score=average_score,
        average_predicted_score=average_predicted_score,
        source_fingerprint=source.fingerprint,
        source_versions=source.metadata,
    )


def interview_feedback_is_reusable(
    feedback: InterviewFeedback,
    result: InterviewFeedbackResult,
) -> bool:
    return (
        feedback.source_fingerprint == result.source_fingerprint
        and feedback.methodology_version == config.FEEDBACK_METHODOLOGY_VERSION
        and feedback.status == result.status
        and feedback.summary == result.summary
    )


def apply_interview_feedback(
    row: InterviewFeedback,
    result: InterviewFeedbackResult,
) -> None:
    row.status = result.status
    row.status_detail = result.status_detail
    row.summary = result.summary
    row.overall_strengths = list(result.overall_strengths)
    row.overall_improvements = list(result.overall_improvements)
    row.recommendations = list(result.recommendations)
    row.notices = list(result.notices)
    row.limitations = list(result.limitations)
    row.question_breakdown = [dict(item) for item in result.question_breakdown]
    row.questions_total = result.questions_total
    row.questions_answered = result.questions_answered
    row.questions_unanswered = result.questions_unanswered
    row.feedback_completed = result.feedback_completed
    row.feedback_unavailable = result.feedback_unavailable
    row.feedback_failed = result.feedback_failed
    row.feedback_stale = result.feedback_stale
    row.voice_available_count = result.voice_available_count
    row.ml_predictions = result.ml_predictions
    row.baseline_predictions = result.baseline_predictions
    row.average_score = result.average_score
    row.average_predicted_score = result.average_predicted_score
    row.source_fingerprint = result.source_fingerprint
    row.source_versions = dict(result.source_versions)
    row.methodology_version = config.FEEDBACK_METHODOLOGY_VERSION
