"""Phase 4 feature construction and versioned feature schema.

Only current, persisted results are consumed:

- a Phase 2 ``AnswerEvaluation`` whose transcript/reference hashes match;
- a Phase 3 ``ContentAnalysis`` whose hashes match and whose status is
  ``completed``;
- a Phase 3 ``VoiceAnalysis`` whose transcript/audio hashes match, whose status
  is ``completed``, and which detected speech.

Stale, unevaluable, failed, or unavailable sources are excluded and explained.
Missing voice features are represented as ``None`` (plus an availability
indicator), never as a zero score. Pitch is intentionally not a scoring feature:
it describes the speaker rather than answer quality and would add demographic
bias without a validated target relationship.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import (
    ANALYSIS_METHODOLOGY_VERSION,
    EVALUATION_METHODOLOGY_VERSION,
    ML_FEATURE_SCHEMA_VERSION,
)
from models import (
    AnswerEvaluation,
    ContentAnalysis,
    InterviewAnswer,
    InterviewQuestion,
    VoiceAnalysis,
)
from services.evaluation import fingerprint
from services.voice_analysis import (
    STATUS_COMPLETED as VOICE_STATUS_COMPLETED,
)


class FeatureValidationError(Exception):
    """A source value or feature vector is outside the documented schema."""

    status_code = 422

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    minimum: float
    maximum: float
    required: bool
    description: str


# Stable order matters: model artifacts record this order and inference refuses
# an artifact whose feature order differs.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "semantic_score",
        "score_0_100",
        0.0,
        100.0,
        True,
        "Phase 2 semantic similarity between the transcript and reference.",
    ),
    FeatureSpec(
        "keyword_score",
        "score_0_100",
        0.0,
        100.0,
        True,
        "Phase 2 weighted lexical coverage of reference terminology.",
    ),
    FeatureSpec(
        "filler_rate_percent",
        "percent_0_100",
        0.0,
        100.0,
        False,
        "Share of transcript tokens that are filler words.",
    ),
    FeatureSpec(
        "repetition_rate_percent",
        "percent_0_100",
        0.0,
        100.0,
        False,
        "Share of meaningful words occupied by repeated occurrences.",
    ),
    FeatureSpec(
        "average_sentence_words",
        "words",
        0.0,
        500.0,
        False,
        "Mean words per detected sentence in the transcript.",
    ),
    FeatureSpec(
        "speaking_rate_wpm",
        "words_per_minute",
        0.0,
        600.0,
        False,
        "Transcript words divided by detected speech time (silence excluded).",
    ),
    FeatureSpec(
        "pauses_per_minute",
        "pauses_per_minute",
        0.0,
        300.0,
        False,
        "Detected silent pauses of at least the configured minimum, per minute.",
    ),
    FeatureSpec(
        "speech_ratio_percent",
        "percent_0_100",
        0.0,
        100.0,
        False,
        "Detected speech time as a share of the analysed recording.",
    ),
    FeatureSpec(
        "rms_variation_percent",
        "percent_0_1000",
        0.0,
        1000.0,
        False,
        "Variation of RMS loudness relative to the mean level.",
    ),
    FeatureSpec(
        "has_content_features",
        "flag_0_1",
        0.0,
        1.0,
        False,
        "1 when current content statistics were available, else 0.",
    ),
    FeatureSpec(
        "has_voice_features",
        "flag_0_1",
        0.0,
        1.0,
        False,
        "1 when a reliable completed voice analysis was available, else 0.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}
REQUIRED_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.required
)
OPTIONAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.required
)

ELIGIBLE = "eligible"
MISSING_REFERENCE = "missing_reference"
NOT_EVALUATED = "not_evaluated"
UNEVALUABLE = "unevaluable"
STALE_EVALUATION = "stale_evaluation"
INVALID_EVALUATION = "invalid_evaluation"

# Used in the source fingerprint when an answer has no stored recording, so the
# fingerprint is still deterministic and later audio correctly makes it stale.
NO_AUDIO_FINGERPRINT = hashlib.sha256(b"phase4:no-audio").hexdigest()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            "A source feature was not a valid number."
        ) from exc
    if not math.isfinite(number):
        return None
    return number


def validate_feature_value(name: str, value: float | int | None) -> float | None:
    """Validate one feature against its documented type and range."""
    spec = FEATURE_BY_NAME.get(name)
    if spec is None:
        raise FeatureValidationError(
            f"Feature '{name}' is not part of {ML_FEATURE_SCHEMA_VERSION}."
        )
    if value is None:
        if spec.required:
            raise FeatureValidationError(
                f"Required feature '{name}' is unavailable."
            )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureValidationError(
            f"Feature '{name}' must be numeric (got {type(value).__name__})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise FeatureValidationError(
            f"Feature '{name}' must be a finite number."
        )
    if not spec.minimum <= number <= spec.maximum:
        raise FeatureValidationError(
            f"Feature '{name}' is outside its documented range "
            f"[{spec.minimum:g}, {spec.maximum:g}]."
        )
    return number


def validate_feature_dict(features: dict[str, float | None]) -> dict[str, float | None]:
    """Validate a complete feature dictionary and reject unknown/missing keys."""
    unknown = set(features) - set(FEATURE_NAMES)
    if unknown:
        raise FeatureValidationError(
            "Unknown feature(s): " + ", ".join(sorted(unknown))
        )
    missing = set(FEATURE_NAMES) - set(features)
    if missing:
        raise FeatureValidationError(
            "Missing feature(s): " + ", ".join(sorted(missing))
        )
    return {
        name: validate_feature_value(name, features[name])
        for name in FEATURE_NAMES
    }


@dataclass(frozen=True)
class SourceSnapshot:
    """The exact source rows/versions a prediction was computed from."""

    fingerprint: str
    transcript_hash: str
    reference_hash: str
    audio_fingerprint: str
    evaluation_version: str | None
    content_version: str | None
    voice_version: str | None
    evaluation_updated_at: str | None
    content_updated_at: str | None
    voice_updated_at: str | None
    evaluation_id: int | None
    content_id: int | None
    voice_id: int | None

    def metadata(self) -> dict[str, object]:
        return {
            "transcript_hash": self.transcript_hash,
            "reference_hash": self.reference_hash,
            "audio_fingerprint": self.audio_fingerprint,
            "evaluation_version": self.evaluation_version,
            "content_version": self.content_version,
            "voice_version": self.voice_version,
            "evaluation_id": self.evaluation_id,
            "content_id": self.content_id,
            "voice_id": self.voice_id,
            "evaluation_updated_at": self.evaluation_updated_at,
            "content_updated_at": self.content_updated_at,
            "voice_updated_at": self.voice_updated_at,
        }


@dataclass(frozen=True)
class FeatureRecord:
    answer_id: int
    interview_id: int
    question_id: int
    feature_schema_version: str
    eligibility_status: str
    eligibility_detail: str | None
    features: dict[str, float | None]
    missing_features: tuple[str, ...]
    warnings: tuple[str, ...]
    snapshot: SourceSnapshot
    # Reused unchanged from the current Phase 2 evaluation. The baseline uses it
    # instead of recomputing an answer-quality score.
    combined_score: float | None = None

    @property
    def eligible(self) -> bool:
        return self.eligibility_status == ELIGIBLE

    def feature_vector(self) -> list[float]:
        """Return values for an already-fitted pipeline (NaN = unavailable)."""
        if not self.eligible:
            raise FeatureValidationError(
                "This answer does not have eligible Phase 2/3 sources."
            )
        return [
            math.nan if self.features[name] is None else float(self.features[name])
            for name in FEATURE_NAMES
        ]


def _current_evaluation(
    db: Session, answer_id: int
) -> AnswerEvaluation | None:
    return db.scalar(
        select(AnswerEvaluation).where(
            AnswerEvaluation.answer_id == answer_id,
            AnswerEvaluation.methodology_version
            == EVALUATION_METHODOLOGY_VERSION,
        )
    )


def _current_content(db: Session, answer_id: int) -> ContentAnalysis | None:
    return db.scalar(
        select(ContentAnalysis).where(
            ContentAnalysis.answer_id == answer_id,
            ContentAnalysis.methodology_version == ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def _current_voice(db: Session, answer_id: int) -> VoiceAnalysis | None:
    return db.scalar(
        select(VoiceAnalysis).where(
            VoiceAnalysis.answer_id == answer_id,
            VoiceAnalysis.methodology_version == ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def build_source_snapshot(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> SourceSnapshot:
    """Fingerprint the current Phase 2/3 sources for staleness detection."""
    transcript_hash = fingerprint(answer.transcript)
    reference_hash = fingerprint(question.reference_answer or "")
    audio_fingerprint = answer.audio_sha256 or NO_AUDIO_FINGERPRINT

    evaluation = _current_evaluation(db, answer.id)
    content = _current_content(db, answer.id)
    voice = _current_voice(db, answer.id)

    metadata = {
        "transcript_hash": transcript_hash,
        "reference_hash": reference_hash,
        "audio_fingerprint": audio_fingerprint,
        "evaluation_version": (
            evaluation.methodology_version if evaluation is not None else None
        ),
        "content_version": (
            content.methodology_version if content is not None else None
        ),
        "voice_version": voice.methodology_version if voice is not None else None,
        "evaluation_id": evaluation.id if evaluation is not None else None,
        "content_id": content.id if content is not None else None,
        "voice_id": voice.id if voice is not None else None,
        "evaluation_updated_at": (
            _iso(evaluation.updated_at) if evaluation is not None else None
        ),
        "content_updated_at": (
            _iso(content.updated_at) if content is not None else None
        ),
        "voice_updated_at": (
            _iso(voice.updated_at) if voice is not None else None
        ),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    return SourceSnapshot(
        fingerprint=digest,
        transcript_hash=transcript_hash,
        reference_hash=reference_hash,
        audio_fingerprint=audio_fingerprint,
        evaluation_version=metadata["evaluation_version"],
        content_version=metadata["content_version"],
        voice_version=metadata["voice_version"],
        evaluation_updated_at=metadata["evaluation_updated_at"],
        content_updated_at=metadata["content_updated_at"],
        voice_updated_at=metadata["voice_updated_at"],
        evaluation_id=metadata["evaluation_id"],
        content_id=metadata["content_id"],
        voice_id=metadata["voice_id"],
    )


def _content_values(
    content: ContentAnalysis | None,
    transcript_hash: str,
    reference_hash: str,
) -> tuple[dict[str, float | None], str | None, bool]:
    """Return content feature values, a warning reason, and availability."""

    if content is None:
        return {}, "Content analysis is missing.", False
    if (
        content.transcript_hash != transcript_hash
        or content.reference_hash != reference_hash
    ):
        return {}, "Content analysis is stale.", False
    if content.status != "completed":
        return (
            {},
            f"Content analysis status is '{content.status}', so its features "
            "were excluded.",
            False,
        )

    values = {
        "filler_rate_percent": _finite_or_none(content.filler_rate_percent),
        "repetition_rate_percent": _finite_or_none(
            content.repetition_rate_percent
        ),
        "average_sentence_words": _finite_or_none(
            content.average_sentence_words
        ),
    }
    available = any(value is not None for value in values.values())
    if not available:
        return {}, "Content analysis has no usable statistics.", False
    return values, None, True


def _voice_values(
    voice: VoiceAnalysis | None,
    transcript_hash: str,
    audio_fingerprint: str,
) -> tuple[dict[str, float | None], str | None, bool]:
    """Return reliable voice feature values only when the row is usable."""

    if voice is None:
        return {}, "Voice analysis is missing.", False
    if (
        voice.transcript_hash != transcript_hash
        or voice.audio_sha256 != audio_fingerprint
    ):
        return {}, "Voice analysis is stale.", False
    if voice.status != VOICE_STATUS_COMPLETED or not voice.speech_detected:
        return (
            {},
            f"Voice analysis status is '{voice.status}' with "
            f"speech_detected={voice.speech_detected}, so its features were "
            "excluded.",
            False,
        )

    values = {
        "speaking_rate_wpm": _finite_or_none(voice.speaking_rate_wpm),
        "pauses_per_minute": _finite_or_none(voice.pauses_per_minute),
        "speech_ratio_percent": _finite_or_none(voice.speech_ratio_percent),
        "rms_variation_percent": _finite_or_none(voice.rms_variation_percent),
    }
    available = any(value is not None for value in values.values())
    if not available:
        return {}, "Voice analysis has no reliable delivery metrics.", False
    return values, None, True


def build_feature_record(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> FeatureRecord:
    """Build and validate a versioned feature record for one saved answer."""
    snapshot = build_source_snapshot(db, answer, question)
    evaluation = _current_evaluation(db, answer.id)
    content = _current_content(db, answer.id)
    voice = _current_voice(db, answer.id)

    if not (question.reference_answer or "").strip():
        return FeatureRecord(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            feature_schema_version=ML_FEATURE_SCHEMA_VERSION,
            eligibility_status=MISSING_REFERENCE,
            eligibility_detail=(
                "This question has no reference answer, so there is no Phase 2 "
                "evaluation to score."
            ),
            features={name: None for name in FEATURE_NAMES},
            missing_features=FEATURE_NAMES,
            warnings=(),
            snapshot=snapshot,
        )

    if evaluation is None:
        status = NOT_EVALUATED
        detail = (
            "No Phase 2 evaluation exists for this answer. Evaluate the answer "
            "before predicting an interview score."
        )
    elif (
        evaluation.transcript_hash != snapshot.transcript_hash
        or evaluation.reference_hash != snapshot.reference_hash
    ):
        status = STALE_EVALUATION
        detail = (
            "The saved Phase 2 evaluation is stale because the transcript or "
            "reference answer changed. Re-evaluate it before predicting."
        )
    elif evaluation.status != "evaluated":
        status = UNEVALUABLE
        detail = (
            evaluation.status_detail
            or "The Phase 2 evaluation did not produce a usable score."
        )
    else:
        status = ELIGIBLE
        detail = None

    if status != ELIGIBLE:
        return FeatureRecord(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            feature_schema_version=ML_FEATURE_SCHEMA_VERSION,
            eligibility_status=status,
            eligibility_detail=detail,
            features={name: None for name in FEATURE_NAMES},
            missing_features=FEATURE_NAMES,
            warnings=(),
            snapshot=snapshot,
        )

    warnings: list[str] = []
    content_values, content_reason, content_available = _content_values(
        content,
        snapshot.transcript_hash,
        snapshot.reference_hash,
    )
    if content_reason:
        warnings.append(
            f"{content_reason} Content features are omitted, not treated as "
            "zero."
        )

    voice_values, voice_reason, voice_available = _voice_values(
        voice,
        snapshot.transcript_hash,
        snapshot.audio_fingerprint,
    )
    if voice_reason:
        warnings.append(
            f"{voice_reason} Voice features are omitted, not treated as zero."
        )

    features: dict[str, float | None] = {
        "semantic_score": validate_feature_value(
            "semantic_score", evaluation.semantic_score
        ),
        "keyword_score": validate_feature_value(
            "keyword_score", evaluation.keyword_score
        ),
        "filler_rate_percent": None,
        "repetition_rate_percent": None,
        "average_sentence_words": None,
        "speaking_rate_wpm": None,
        "pauses_per_minute": None,
        "speech_ratio_percent": None,
        "rms_variation_percent": None,
        "has_content_features": 1.0 if content_available else 0.0,
        "has_voice_features": 1.0 if voice_available else 0.0,
    }
    for name, value in content_values.items():
        features[name] = validate_feature_value(name, value)
    for name, value in voice_values.items():
        features[name] = validate_feature_value(name, value)

    missing = tuple(
        name for name in FEATURE_NAMES if features[name] is None
    )

    return FeatureRecord(
        answer_id=answer.id,
        interview_id=answer.interview_id,
        question_id=answer.question_id,
        feature_schema_version=ML_FEATURE_SCHEMA_VERSION,
        eligibility_status=ELIGIBLE,
        eligibility_detail=None,
        features=features,
        missing_features=missing,
        warnings=tuple(warnings),
        snapshot=snapshot,
        combined_score=_finite_or_none(evaluation.combined_score),
    )


def feature_schema_descriptor() -> dict[str, object]:
    """Serializable description used by the API and training metadata."""
    return {
        "feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "target": {
            "name": "target_score",
            "unit": "score_0_100",
            "minimum": 0.0,
            "maximum": 100.0,
            "description": (
                "Human-provided overall interview score. No labelled dataset is "
                "included with this repository."
            ),
        },
        "features": [
            {
                "name": spec.name,
                "unit": spec.unit,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
                "required": spec.required,
                "description": spec.description,
            }
            for spec in FEATURE_SPECS
        ],
        "excluded_by_design": [
            {
                "name": "pitch",
                "reason": (
                    "Pitch describes the speaker, not answer quality, and adds "
                    "demographic bias without a validated target relationship."
                ),
            },
            {
                "name": "candidate_or_interview_id",
                "reason": "Identity/group keys must not become model features.",
            },
            {
                "name": "content_coverage_duplicate",
                "reason": (
                    "Content coverage duplicates the Phase 2 keyword score when "
                    "reused, so coverage appears only once as keyword_score."
                ),
            },
        ],
    }
