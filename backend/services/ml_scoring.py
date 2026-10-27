"""Phase 4 scoring service: validated ML inference or an explicit baseline.

The service never invents a model output. An ML score is returned only when a
local artifact exists, was trained with the current feature schema, is marked
``validated`` by the training pipeline, and is compatible with the installed
scikit-learn version. Otherwise a transparent rule-based baseline is returned
and labelled as such.

Predictions are additive to Phase 2/3 results: the Phase 2 semantic, keyword,
and combined scores are read but never overwritten.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from models import (
    InterviewAnswer,
    InterviewQuestion,
    ScorePrediction,
)
from services.ml_features import (
    ELIGIBLE,
    FEATURE_NAMES,
    FeatureRecord,
    FeatureValidationError,
    build_feature_record,
    build_source_snapshot,
)

logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

METHOD_ML = "ml_model"
METHOD_BASELINE = "rule_based_baseline"
METHOD_NONE = "none"

VALIDATION_VALIDATED = "validated"

BASELINE_WEIGHTS = {
    "answer": 0.60,
    "content_style": 0.20,
    "delivery": 0.20,
}

BASELINE_LIMITATIONS = (
    "This is a transparent rule-based baseline, not a machine-learning "
    "prediction.",
    "The fixed weights and target ranges below are design choices. They have "
    "not been learned from or validated against human interview scores.",
    "The answer component reuses the Phase 2 combined score, which measures "
    "lexical/semantic agreement with a reference answer, not factual truth.",
    "Unavailable content or voice features are excluded and the remaining "
    "weights are renormalized. They are never treated as zero.",
)

# Every key the training pipeline writes. An artifact missing any of them is
# rejected instead of being partially trusted.
_REQUIRED_BUNDLE_KEYS = (
    "model",
    "feature_names",
    "feature_schema_version",
    "model_version",
    "trained_at",
    "sklearn_version",
    "validation_status",
    "metrics",
    "limitations",
    "target_min",
    "target_max",
)


@dataclass(frozen=True)
class ModelStatus:
    """Safe, non-secret description of the installed scoring artifact."""

    available: bool
    state: str
    reason: str
    model_version: str | None = None
    feature_schema_version: str | None = None
    validation_status: str | None = None
    trained_at: str | None = None
    sklearn_version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    artifact_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "state": self.state,
            "reason": self.reason,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "validation_status": self.validation_status,
            "trained_at": self.trained_at,
            "sklearn_version": self.sklearn_version,
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "artifact_name": self.artifact_name,
        }


@dataclass(frozen=True)
class _LoadedArtifact:
    status: ModelStatus
    model: Any | None


_CACHE: dict[tuple[str, int, int], _LoadedArtifact] = {}


class ModelInferenceError(Exception):
    """A loaded model failed to produce a usable score."""


def clear_model_cache() -> None:
    """Drop cached artifacts. Used by tests and after installing a new model."""
    _CACHE.clear()


def _major_minor(version: str) -> tuple[str, str] | None:
    parts = str(version).strip().split(".")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _sklearn_compatible(artifact_version: str) -> bool:
    expected = _major_minor(sklearn.__version__)
    actual = _major_minor(artifact_version)
    return expected is not None and actual is not None and expected == actual


def _unavailable(
    reason: str,
    *,
    state: str = "unavailable",
    artifact_name: str | None = None,
    **values: Any,
) -> ModelStatus:
    return ModelStatus(
        available=False,
        state=state,
        reason=reason,
        artifact_name=artifact_name,
        **values,
    )


def _inspect_bundle(bundle: Any, artifact_name: str) -> _LoadedArtifact:
    if not isinstance(bundle, dict):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact is not a recognized bundle.",
                state="invalid",
                artifact_name=artifact_name,
            ),
            None,
        )

    missing = [key for key in _REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing:
        return _LoadedArtifact(
            _unavailable(
                "The model artifact is missing required metadata: "
                + ", ".join(missing)
                + ".",
                state="invalid",
                artifact_name=artifact_name,
            ),
            None,
        )

    feature_names = bundle.get("feature_names")
    if list(feature_names or []) != list(FEATURE_NAMES):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact was trained with a different feature order "
                "or feature set.",
                state="incompatible",
                artifact_name=artifact_name,
            ),
            None,
        )

    if bundle.get("feature_schema_version") != config.ML_FEATURE_SCHEMA_VERSION:
        return _LoadedArtifact(
            _unavailable(
                "The model artifact uses feature schema "
                f"'{bundle.get('feature_schema_version')}', but this server "
                f"expects '{config.ML_FEATURE_SCHEMA_VERSION}'.",
                state="incompatible",
                artifact_name=artifact_name,
            ),
            None,
        )

    validation_status = str(bundle.get("validation_status") or "")
    if validation_status != VALIDATION_VALIDATED:
        return _LoadedArtifact(
            _unavailable(
                "The model artifact is marked "
                f"'{validation_status or 'unknown'}'. Only an artifact "
                "validated by the training pipeline is used for inference.",
                state="unvalidated",
                artifact_name=artifact_name,
                validation_status=validation_status or None,
                model_version=bundle.get("model_version"),
                feature_schema_version=bundle.get("feature_schema_version"),
                trained_at=bundle.get("trained_at"),
                sklearn_version=bundle.get("sklearn_version"),
                metrics=bundle.get("metrics")
                if isinstance(bundle.get("metrics"), dict)
                else {},
                limitations=tuple(bundle.get("limitations") or ()),
            ),
            None,
        )

    artifact_sklearn = str(bundle.get("sklearn_version") or "")
    if not _sklearn_compatible(artifact_sklearn):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact was trained with scikit-learn "
                f"{artifact_sklearn or 'unknown'}, which is not compatible "
                f"with installed version {sklearn.__version__}.",
                state="incompatible",
                artifact_name=artifact_name,
                validation_status=validation_status,
                model_version=bundle.get("model_version"),
                feature_schema_version=bundle.get("feature_schema_version"),
                trained_at=bundle.get("trained_at"),
                sklearn_version=artifact_sklearn,
            ),
            None,
        )

    try:
        target_min = float(bundle.get("target_min"))
        target_max = float(bundle.get("target_max"))
    except (TypeError, ValueError):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact has invalid target bounds.",
                state="invalid",
                artifact_name=artifact_name,
            ),
            None,
        )
    if (
        not math.isfinite(target_min)
        or not math.isfinite(target_max)
        or abs(target_min - config.ML_TARGET_MIN) > 1e-9
        or abs(target_max - config.ML_TARGET_MAX) > 1e-9
        or target_min >= target_max
    ):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact targets a different score range than this "
                "server.",
                state="incompatible",
                artifact_name=artifact_name,
            ),
            None,
        )

    model = bundle.get("model")
    if model is None or not callable(getattr(model, "predict", None)):
        return _LoadedArtifact(
            _unavailable(
                "The model artifact does not contain a usable estimator.",
                state="invalid",
                artifact_name=artifact_name,
            ),
            None,
        )

    metrics = bundle.get("metrics")
    limitations = bundle.get("limitations")
    return _LoadedArtifact(
        ModelStatus(
            available=True,
            state="available",
            reason=(
                "A validated model artifact compatible with this server is "
                "installed."
            ),
            model_version=str(bundle.get("model_version") or "") or None,
            feature_schema_version=str(
                bundle.get("feature_schema_version") or ""
            )
            or None,
            validation_status=validation_status,
            trained_at=str(bundle.get("trained_at") or "") or None,
            sklearn_version=artifact_sklearn or None,
            metrics=metrics if isinstance(metrics, dict) else {},
            limitations=tuple(
                str(item) for item in limitations
            )
            if isinstance(limitations, list | tuple)
            else (),
            artifact_name=artifact_name,
        ),
        model,
    )


def _load_artifact(path: Path) -> _LoadedArtifact:
    try:
        stat = path.stat()
    except OSError:
        return _LoadedArtifact(
            _unavailable(
                "No ML model artifact is installed. Predictions use the "
                "clearly labelled rule-based baseline.",
                artifact_name=path.name,
            ),
            None,
        )

    if not path.is_file():
        return _LoadedArtifact(
            _unavailable(
                "The configured ML model path is not a file. Predictions use "
                "the clearly labelled rule-based baseline.",
                artifact_name=path.name,
            ),
            None,
        )

    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - a bad artifact must not crash
        logger.warning(
            "Could not load ML model artifact %s (%s).",
            path.name,
            type(exc).__name__,
        )
        loaded = _LoadedArtifact(
            _unavailable(
                "The installed ML model artifact could not be loaded. "
                "Predictions use the clearly labelled rule-based baseline.",
                state="invalid",
                artifact_name=path.name,
            ),
            None,
        )
    else:
        loaded = _inspect_bundle(bundle, path.name)

    _CACHE[key] = loaded
    return loaded


def get_model_status() -> ModelStatus:
    """Return the current, cached artifact status (never a secret path)."""
    return _load_artifact(Path(config.ML_MODEL_PATH)).status


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _available(values: dict[str, float | None]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if value is not None and math.isfinite(float(value))
    }


def _weighted_average(
    values: dict[str, float],
    weights: dict[str, float],
) -> float | None:
    usable = {
        name: value for name, value in values.items() if name in weights
    }
    total_weight = sum(weights[name] for name in usable)
    if total_weight <= 0:
        return None
    return sum(usable[name] * weights[name] for name in usable) / total_weight


def _content_component(
    features: dict[str, float | None],
) -> tuple[float | None, dict[str, float]]:
    filler = features.get("filler_rate_percent")
    repetition = features.get("repetition_rate_percent")
    sentence_words = features.get("average_sentence_words")

    subscores: dict[str, float] = {}
    if filler is not None:
        subscores["filler"] = _clamp(100.0 - min(100.0, float(filler) * 4.0))
    if repetition is not None:
        subscores["repetition"] = _clamp(
            100.0 - min(100.0, float(repetition) * 2.5)
        )
    if sentence_words is not None:
        subscores["sentence_length"] = _clamp(
            100.0 - min(100.0, abs(float(sentence_words) - 16.0) * 5.0)
        )

    score = _weighted_average(
        subscores,
        {"filler": 0.40, "repetition": 0.40, "sentence_length": 0.20},
    )
    return result_round(score), {key: result_round(value) for key, value in subscores.items()}


def _delivery_component(
    features: dict[str, float | None],
) -> tuple[float | None, dict[str, float]]:
    rate = features.get("speaking_rate_wpm")
    pauses = features.get("pauses_per_minute")
    speech_ratio = features.get("speech_ratio_percent")
    variation = features.get("rms_variation_percent")

    subscores: dict[str, float] = {}
    if rate is not None:
        value = float(rate)
        if 110.0 <= value <= 170.0:
            subscores["speaking_rate"] = 100.0
        else:
            distance = 110.0 - value if value < 110.0 else value - 170.0
            subscores["speaking_rate"] = _clamp(100.0 - distance * 1.5)
    if pauses is not None:
        subscores["pauses"] = _clamp(100.0 - min(100.0, float(pauses) * 5.0))
    if speech_ratio is not None:
        subscores["speech_ratio"] = _clamp(float(speech_ratio))
    if variation is not None:
        subscores["loudness_variation"] = _clamp(
            100.0 - min(100.0, abs(float(variation) - 30.0) * 2.0)
        )

    score = _weighted_average(
        subscores,
        {
            "speaking_rate": 0.35,
            "pauses": 0.25,
            "speech_ratio": 0.20,
            "loudness_variation": 0.20,
        },
    )
    return result_round(score), {key: result_round(value) for key, value in subscores.items()}


def result_round(value: float | None) -> float | None:
    """Round public scores consistently, preserving ``None``."""
    return None if value is None else round(float(value), 2)


def _baseline_prediction(
    record: FeatureRecord,
    model_status: ModelStatus,
) -> tuple[float | None, dict[str, Any], str | None]:
    """Return the baseline score, its explainable components, and a detail."""

    if record.combined_score is None:
        return (
            None,
            {},
            "The Phase 2 combined score is unavailable, so the baseline "
            "cannot be computed.",
        )

    components: list[dict[str, Any]] = []
    answer_score = _clamp(float(record.combined_score))
    components.append(
        {
            "name": "answer",
            "score": result_round(answer_score),
            "weight": BASELINE_WEIGHTS["answer"],
            "source": "phase2_combined_score",
        }
    )

    content_score, content_details = _content_component(record.features)
    if content_score is not None:
        components.append(
            {
                "name": "content_style",
                "score": content_score,
                "weight": BASELINE_WEIGHTS["content_style"],
                "source": "phase3_content_analysis",
                "subscores": content_details,
            }
        )

    delivery_score, delivery_details = _delivery_component(record.features)
    if delivery_score is not None:
        components.append(
            {
                "name": "delivery",
                "score": delivery_score,
                "weight": BASELINE_WEIGHTS["delivery"],
                "source": "phase3_voice_analysis",
                "subscores": delivery_details,
            }
        )

    total_weight = sum(float(item["weight"]) for item in components)
    if total_weight <= 0:
        return None, {}, "The baseline has no usable components."

    score = sum(
        float(item["score"]) * float(item["weight"]) for item in components
    ) / total_weight

    explained = {
        "components": components,
        "weights_renormalized": abs(total_weight - 1.0) > 1e-9,
        "effective_weight_total": result_round(total_weight),
        "formula": (
            "score = sum(component_score * component_weight) / "
            "sum(available component_weight)"
        ),
        "answer_component_source": "Phase 2 combined_score (unchanged)",
        "content_targets": {
            "filler_rate_percent": "lower is better",
            "repetition_rate_percent": "lower is better",
            "average_sentence_words": "target 16 words",
        },
        "delivery_targets": {
            "speaking_rate_wpm": "target 110-170",
            "pauses_per_minute": "lower is better",
            "speech_ratio_percent": "higher is better",
            "rms_variation_percent": "target 30",
        },
    }

    detail = None
    if not model_status.available:
        detail = (
            "ML inference is unavailable, so this score is the rule-based "
            "baseline."
        )
    return result_round(score), explained, detail


def _base_result(
    record: FeatureRecord,
    *,
    status: str,
    method: str,
    score: float | None,
    status_detail: str | None,
    model_version: str | None,
    components: dict[str, Any],
    warnings: list[str],
    limitations: list[str],
) -> "PredictionResult":
    return PredictionResult(
        status=status,
        method=method,
        score=result_round(score),
        status_detail=status_detail,
        model_version=model_version,
        feature_schema_version=config.ML_FEATURE_SCHEMA_VERSION,
        methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
        features={
            name: result_round(value) for name, value in record.features.items()
        },
        missing_features=list(record.missing_features),
        components=components,
        warnings=warnings,
        limitations=limitations,
        source_fingerprint=record.snapshot.fingerprint,
        source_versions=record.snapshot.metadata(),
    )


@dataclass(frozen=True)
class PredictionResult:
    status: str
    method: str
    score: float | None
    status_detail: str | None
    model_version: str | None
    feature_schema_version: str
    methodology_version: str
    features: dict[str, float | None]
    missing_features: list[str]
    components: dict[str, Any]
    warnings: list[str]
    limitations: list[str]
    source_fingerprint: str
    source_versions: dict[str, Any]

    @property
    def is_completed(self) -> bool:
        return self.status == STATUS_COMPLETED and self.score is not None


def _failed_result(
    record: FeatureRecord,
    detail: str,
    warnings: list[str] | None = None,
) -> PredictionResult:
    return PredictionResult(
        status=STATUS_FAILED,
        method=METHOD_NONE,
        score=None,
        status_detail=detail,
        model_version=None,
        feature_schema_version=config.ML_FEATURE_SCHEMA_VERSION,
        methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
        features={name: None for name in FEATURE_NAMES},
        missing_features=list(FEATURE_NAMES),
        components={},
        warnings=list(warnings or ()),
        limitations=[],
        source_fingerprint=record.snapshot.fingerprint,
        source_versions=record.snapshot.metadata(),
    )


def _ml_prediction(
    record: FeatureRecord,
    model_status: ModelStatus,
    model: Any,
) -> PredictionResult:
    vector = np.asarray([record.feature_vector()], dtype=float)
    try:
        raw = model.predict(vector)
        value = float(np.asarray(raw).reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001 - report, never hide, a model failure
        logger.warning(
            "ML model inference failed (%s).",
            type(exc).__name__,
        )
        return _failed_result(
            record,
            "The installed ML model could not score these features. The "
            "prediction was not replaced with a heuristic result.",
            list(record.warnings),
        )

    if not math.isfinite(value):
        return _failed_result(
            record,
            "The installed ML model returned a non-finite score. The "
            "prediction was not replaced with a heuristic result.",
            list(record.warnings),
        )

    score = result_round(_clamp(value, config.ML_TARGET_MIN, config.ML_TARGET_MAX))
    warnings = list(record.warnings)
    if record.missing_features:
        warnings.append(
            "Optional feature(s) "
            + ", ".join(record.missing_features)
            + " were missing and were imputed by the trained pipeline. They "
            "were not set to zero."
        )

    limitations = list(model_status.limitations)
    if not limitations:
        limitations = [
            "This score is a statistical prediction from a validated model, "
            "not a human hiring decision.",
            "Prediction quality is bounded by the labelled training data and "
            "the recorded feature coverage.",
        ]

    return _base_result(
        record,
        status=STATUS_COMPLETED,
        method=METHOD_ML,
        score=score,
        status_detail=None,
        model_version=model_status.model_version,
        components={
            "model": {
                "model_version": model_status.model_version,
                "feature_schema_version": model_status.feature_schema_version,
                "trained_at": model_status.trained_at,
                "validation_status": model_status.validation_status,
                "score": score,
            }
        },
        warnings=warnings,
        limitations=limitations,
    )


def compute_prediction(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> PredictionResult:
    """Build eligible features and return an ML or baseline prediction."""
    try:
        record = build_feature_record(db, answer, question)
    except FeatureValidationError as exc:
        snapshot = build_source_snapshot(db, answer, question)
        fallback = FeatureRecord(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            feature_schema_version=config.ML_FEATURE_SCHEMA_VERSION,
            eligibility_status="invalid_evaluation",
            eligibility_detail=exc.message,
            features={name: None for name in FEATURE_NAMES},
            missing_features=FEATURE_NAMES,
            warnings=(),
            snapshot=snapshot,
        )
        return _failed_result(fallback, exc.message)

    model_status = get_model_status()

    if not record.eligible:
        status = STATUS_UNAVAILABLE
        if record.eligibility_status == "invalid_evaluation":
            status = STATUS_FAILED
        warnings = list(record.warnings)
        if not model_status.available:
            warnings.append(model_status.reason)
        return PredictionResult(
            status=status,
            method=METHOD_NONE,
            score=None,
            status_detail=record.eligibility_detail,
            model_version=None,
            feature_schema_version=config.ML_FEATURE_SCHEMA_VERSION,
            methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
            features={name: None for name in FEATURE_NAMES},
            missing_features=list(record.missing_features),
            components={},
            warnings=warnings,
            limitations=[],
            source_fingerprint=record.snapshot.fingerprint,
            source_versions=record.snapshot.metadata(),
        )

    if model_status.available:
        loaded = _load_artifact(Path(config.ML_MODEL_PATH))
        if loaded.model is None:
            # The artifact changed between the status read and the load.
            model_status = loaded.status
        else:
            return _ml_prediction(record, model_status, loaded.model)

    score, components, detail = _baseline_prediction(record, model_status)
    if score is None:
        return _failed_result(
            record,
            detail or "The baseline could not be computed.",
            list(record.warnings),
        )

    warnings = list(record.warnings)
    if not model_status.available:
        warnings.append(model_status.reason)
    if record.missing_features:
        warnings.append(
            "Optional feature(s) "
            + ", ".join(record.missing_features)
            + " were unavailable and excluded from the baseline. They were "
            "not treated as zero."
        )

    return _base_result(
        record,
        status=STATUS_COMPLETED,
        method=METHOD_BASELINE,
        score=score,
        status_detail=detail,
        model_version=None,
        components=components,
        warnings=warnings,
        limitations=list(BASELINE_LIMITATIONS),
    )


def apply_prediction_result(row: ScorePrediction, result: PredictionResult) -> None:
    """Copy a result onto its persisted row (upsert in place)."""
    row.status = result.status
    row.method = result.method
    row.score = result.score
    row.status_detail = result.status_detail
    row.model_version = result.model_version
    row.feature_schema_version = result.feature_schema_version
    row.methodology_version = result.methodology_version
    row.features = dict(result.features)
    row.missing_features = list(result.missing_features)
    row.components = dict(result.components)
    row.warnings = list(result.warnings)
    row.limitations = list(result.limitations)
    row.source_fingerprint = result.source_fingerprint
    row.source_versions = dict(result.source_versions)


def prediction_is_stale(
    prediction: ScorePrediction,
    answer: InterviewAnswer,
    question: InterviewQuestion,
    db: Session,
) -> bool:
    """Read-time staleness: source changes and version changes only."""
    if (
        prediction.methodology_version
        != config.ML_PREDICTION_METHODOLOGY_VERSION
        or prediction.feature_schema_version
        != config.ML_FEATURE_SCHEMA_VERSION
    ):
        return True
    current = build_source_snapshot(db, answer, question)
    return prediction.source_fingerprint != current.fingerprint


def is_reusable(
    prediction: ScorePrediction,
    result: PredictionResult,
) -> bool:
    """True when rerunning would not change the persisted result."""
    if prediction.source_fingerprint != result.source_fingerprint:
        return False
    if prediction.methodology_version != result.methodology_version:
        return False
    if prediction.feature_schema_version != result.feature_schema_version:
        return False
    if prediction.method != result.method:
        return False
    if prediction.model_version != result.model_version:
        return False
    if prediction.status != result.status:
        return False
    if prediction.score != result.score:
        return False
    return True


@dataclass(frozen=True)
class PredictionSummary:
    methodology_version: str
    predictions_completed: int = 0
    predictions_ml: int = 0
    predictions_baseline: int = 0
    predictions_stale: int = 0
    predictions_unavailable: int = 0
    average_predicted_score: float | None = None
    prediction_method: str | None = None
    ml_model_available: bool = False


def build_prediction_summary(
    db: Session,
    interview_id: int,
) -> PredictionSummary:
    """Aggregate only current, completed answers for the summary card."""
    rows = db.execute(
        select(ScorePrediction, InterviewAnswer, InterviewQuestion)
        .join(InterviewAnswer, ScorePrediction.answer_id == InterviewAnswer.id)
        .join(
            InterviewQuestion,
            ScorePrediction.question_id == InterviewQuestion.id,
        )
        .where(ScorePrediction.interview_id == interview_id)
        .order_by(ScorePrediction.id)
    ).all()

    completed_scores: list[float] = []
    ml_count = 0
    baseline_count = 0
    stale_count = 0
    unavailable_count = 0

    for prediction, answer, question in rows:
        if prediction_is_stale(prediction, answer, question, db):
            stale_count += 1
            continue
        if (
            prediction.status == STATUS_COMPLETED
            and prediction.score is not None
        ):
            completed_scores.append(float(prediction.score))
            if prediction.method == METHOD_ML:
                ml_count += 1
            elif prediction.method == METHOD_BASELINE:
                baseline_count += 1
        else:
            unavailable_count += 1

    if ml_count and baseline_count:
        method = "mixed"
    elif ml_count:
        method = METHOD_ML
    elif baseline_count:
        method = METHOD_BASELINE
    else:
        method = None

    return PredictionSummary(
        methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
        predictions_completed=len(completed_scores),
        predictions_ml=ml_count,
        predictions_baseline=baseline_count,
        predictions_stale=stale_count,
        predictions_unavailable=unavailable_count,
        average_predicted_score=(
            round(sum(completed_scores) / len(completed_scores), 2)
            if completed_scores
            else None
        ),
        prediction_method=method,
        ml_model_available=get_model_status().available,
    )
