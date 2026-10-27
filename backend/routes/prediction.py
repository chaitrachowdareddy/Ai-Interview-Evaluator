"""Phase 4 prediction API: versioned ML/baseline scores for saved answers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import config
from database import get_db
from models import (
    InterviewAnswer,
    InterviewQuestion,
    InterviewSession,
    ScorePrediction,
)
from schemas import (
    FeatureSchemaResponse,
    ModelStatusOut,
    PredictionListResponse,
    PredictionResponse,
    ScorePredictionOut,
)
from services.ml_features import feature_schema_descriptor
from services.ml_scoring import (
    BASELINE_LIMITATIONS,
    BASELINE_WEIGHTS,
    METHOD_BASELINE,
    METHOD_ML,
    ModelStatus,
    apply_prediction_result,
    compute_prediction,
    get_model_status,
    is_reusable,
    prediction_is_stale,
)

router = APIRouter(tags=["Prediction"])

METHOD_LABELS = {
    METHOD_ML: "ML prediction",
    METHOD_BASELINE: "Rule-based baseline",
}


def _get_session_or_404(db: Session, interview_id: int) -> InterviewSession:
    session = db.get(InterviewSession, interview_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Interview session not found.")
    return session


def _get_answer_or_404(db: Session, answer_id: int) -> InterviewAnswer:
    answer = db.get(InterviewAnswer, answer_id)
    if answer is None:
        raise HTTPException(status_code=404, detail="Saved answer not found.")
    return answer


def _get_question_or_404(db: Session, question_id: int) -> InterviewQuestion:
    question = db.get(InterviewQuestion, question_id)
    if question is None:
        raise HTTPException(
            status_code=404,
            detail="Interview question not found.",
        )
    return question


def _find_prediction(
    db: Session,
    answer_id: int,
) -> ScorePrediction | None:
    return db.scalar(
        select(ScorePrediction).where(
            ScorePrediction.answer_id == answer_id,
            ScorePrediction.methodology_version
            == config.ML_PREDICTION_METHODOLOGY_VERSION,
        )
    )


def _to_model_status_out(status: ModelStatus) -> ModelStatusOut:
    return ModelStatusOut(**status.as_dict())


def _to_prediction_out(
    prediction: ScorePrediction,
    answer: InterviewAnswer,
    question: InterviewQuestion,
    db: Session,
) -> ScorePredictionOut:
    method_label = METHOD_LABELS.get(
        prediction.method,
        "No prediction",
    )
    return ScorePredictionOut(
        id=prediction.id,
        answer_id=prediction.answer_id,
        interview_id=prediction.interview_id,
        question_id=prediction.question_id,
        status=prediction.status,
        method=prediction.method,
        method_label=method_label,
        is_ml_prediction=prediction.method == METHOD_ML,
        score=prediction.score,
        status_detail=prediction.status_detail,
        model_version=prediction.model_version,
        feature_schema_version=prediction.feature_schema_version,
        methodology_version=prediction.methodology_version,
        features=dict(prediction.features or {}),
        missing_features=list(prediction.missing_features or []),
        components=dict(prediction.components or {}),
        warnings=list(prediction.warnings or []),
        limitations=list(prediction.limitations or []),
        source_versions=dict(prediction.source_versions or {}),
        is_stale=prediction_is_stale(prediction, answer, question, db),
        created_at=prediction.created_at,
        updated_at=prediction.updated_at,
    )


@router.get("/prediction/schema", response_model=FeatureSchemaResponse)
def get_prediction_schema() -> FeatureSchemaResponse:
    """Describe the stable feature schema and currently installed model."""
    return FeatureSchemaResponse(
        feature_schema=feature_schema_descriptor(),
        model_status=_to_model_status_out(get_model_status()),
        prediction_methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
        baseline_methodology={
            "method": METHOD_BASELINE,
            "label": "Rule-based baseline (not an ML prediction)",
            "weights": dict(BASELINE_WEIGHTS),
            "answer_source": "Phase 2 combined_score, reused unchanged",
            "detail": (
                "Available components are combined with their fixed weights "
                "renormalized. Missing voice/content features are excluded, "
                "never treated as zero."
            ),
            "limitations": list(BASELINE_LIMITATIONS),
        },
    )


@router.post("/prediction/{answer_id}", response_model=PredictionResponse)
def predict_saved_answer(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> PredictionResponse:
    """Create or refresh one answer prediction in place.

    The row is unique per ``(answer_id, methodology_version)``. Rerunning with
    unchanged sources returns the saved row instead of inserting a duplicate.
    A change to the transcript, reference answer, Phase 2 evaluation, or Phase 3
    analysis updates the same row and its source fingerprint.
    """
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)

    if db.get(InterviewSession, answer.interview_id) is None:
        raise HTTPException(
            status_code=409,
            detail="The saved answer is not linked to a valid interview session.",
        )

    result = compute_prediction(db, answer, question)
    existing = _find_prediction(db, answer.id)

    if existing is not None and is_reusable(existing, result):
        response.status_code = 200
        return PredictionResponse(
            prediction=_to_prediction_out(existing, answer, question, db),
            model_status=_to_model_status_out(get_model_status()),
            reused=True,
        )

    if existing is None:
        existing = ScorePrediction(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            methodology_version=config.ML_PREDICTION_METHODOLOGY_VERSION,
            feature_schema_version=config.ML_FEATURE_SCHEMA_VERSION,
            source_fingerprint=result.source_fingerprint,
        )
        db.add(existing)
        response.status_code = 201
    else:
        response.status_code = 200

    apply_prediction_result(existing, result)

    try:
        db.commit()
    except IntegrityError:
        # A concurrent request inserted the same (answer, version) row first.
        db.rollback()
        existing = _find_prediction(db, answer.id)
        if existing is None:
            raise
        response.status_code = 200
        return PredictionResponse(
            prediction=_to_prediction_out(existing, answer, question, db),
            model_status=_to_model_status_out(get_model_status()),
            reused=True,
        )

    db.refresh(existing)
    return PredictionResponse(
        prediction=_to_prediction_out(existing, answer, question, db),
        model_status=_to_model_status_out(get_model_status()),
        reused=False,
    )


@router.get("/prediction/{answer_id}", response_model=ScorePredictionOut)
def get_saved_prediction(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> ScorePredictionOut:
    """Return the saved prediction for an answer, if one exists."""
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)
    prediction = _find_prediction(db, answer.id)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction has been saved for this answer yet.",
        )
    return _to_prediction_out(prediction, answer, question, db)


@router.get(
    "/interview/{interview_id}/predictions",
    response_model=PredictionListResponse,
)
def get_interview_predictions(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionListResponse:
    """Return every current-version prediction for one interview."""
    _get_session_or_404(db, interview_id)

    rows = db.execute(
        select(ScorePrediction, InterviewAnswer, InterviewQuestion)
        .join(InterviewAnswer, ScorePrediction.answer_id == InterviewAnswer.id)
        .join(
            InterviewQuestion,
            ScorePrediction.question_id == InterviewQuestion.id,
        )
        .where(
            ScorePrediction.interview_id == interview_id,
            ScorePrediction.methodology_version
            == config.ML_PREDICTION_METHODOLOGY_VERSION,
        )
        .order_by(ScorePrediction.id)
    ).all()

    return PredictionListResponse(
        interview_id=interview_id,
        predictions=[
            _to_prediction_out(prediction, answer, question, db)
            for prediction, answer, question in rows
        ],
        model_status=_to_model_status_out(get_model_status()),
    )
