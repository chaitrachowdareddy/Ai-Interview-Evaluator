"""Phase 5 feedback API: deterministic, evidence-based answer feedback.

Every response is built from persisted sources. Nothing is invented, no LLM is
called, and missing/stale measurements are reported as unavailable instead of
being scored as zero.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import config
from database import get_db
from models import (
    AnswerFeedback,
    InterviewAnswer,
    InterviewFeedback,
    InterviewQuestion,
    InterviewSession,
)
from schemas import (
    AnswerFeedbackOut,
    AnswerFeedbackResponse,
    InterviewFeedbackOut,
    InterviewFeedbackResponse,
)
from services.feedback import (
    answer_feedback_is_reusable,
    answer_feedback_is_stale,
    apply_answer_feedback,
    apply_interview_feedback,
    build_interview_feedback_result,
    generate_answer_feedback,
    interview_feedback_is_reusable,
    interview_feedback_is_stale,
)

router = APIRouter(tags=["Feedback"])


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


def _find_answer_feedback(
    db: Session,
    answer_id: int,
) -> AnswerFeedback | None:
    return db.scalar(
        select(AnswerFeedback).where(
            AnswerFeedback.answer_id == answer_id,
            AnswerFeedback.methodology_version
            == config.FEEDBACK_METHODOLOGY_VERSION,
        )
    )


def _find_interview_feedback(
    db: Session,
    interview_id: int,
) -> InterviewFeedback | None:
    return db.scalar(
        select(InterviewFeedback).where(
            InterviewFeedback.interview_id == interview_id,
            InterviewFeedback.methodology_version
            == config.FEEDBACK_METHODOLOGY_VERSION,
        )
    )


def _to_answer_feedback_out(
    feedback: AnswerFeedback,
    answer: InterviewAnswer,
    question: InterviewQuestion,
    db: Session,
) -> AnswerFeedbackOut:
    return AnswerFeedbackOut(
        id=feedback.id,
        answer_id=feedback.answer_id,
        interview_id=feedback.interview_id,
        question_id=feedback.question_id,
        status=feedback.status,
        band=feedback.band,
        status_detail=feedback.status_detail,
        summary=feedback.summary,
        strengths=list(feedback.strengths or []),
        improvements=list(feedback.improvements or []),
        actionable_suggestions=list(feedback.actionable_suggestions or []),
        matched_concepts=list(feedback.matched_concepts or []),
        missing_concepts=list(feedback.missing_concepts or []),
        voice_observations=list(feedback.voice_observations or []),
        notices=list(feedback.notices or []),
        limitations=list(feedback.limitations or []),
        evidence=dict(feedback.evidence or {}),
        prediction_method=feedback.prediction_method,
        prediction_score=feedback.prediction_score,
        prediction_is_ml=feedback.prediction_is_ml,
        prediction_note=feedback.prediction_note,
        voice_available=feedback.voice_available,
        content_available=feedback.content_available,
        semantic_score=feedback.semantic_score,
        keyword_score=feedback.keyword_score,
        combined_score=feedback.combined_score,
        content_coverage_score=feedback.content_coverage_score,
        methodology_version=feedback.methodology_version,
        source_versions=dict(feedback.source_versions or {}),
        is_stale=answer_feedback_is_stale(feedback, answer, question, db),
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def _to_interview_feedback_out(
    feedback: InterviewFeedback,
    db: Session,
) -> InterviewFeedbackOut:
    return InterviewFeedbackOut(
        id=feedback.id,
        interview_id=feedback.interview_id,
        status=feedback.status,
        status_detail=feedback.status_detail,
        summary=feedback.summary,
        overall_strengths=list(feedback.overall_strengths or []),
        overall_improvements=list(feedback.overall_improvements or []),
        recommendations=list(feedback.recommendations or []),
        notices=list(feedback.notices or []),
        limitations=list(feedback.limitations or []),
        question_breakdown=list(feedback.question_breakdown or []),
        questions_total=feedback.questions_total,
        questions_answered=feedback.questions_answered,
        questions_unanswered=feedback.questions_unanswered,
        feedback_completed=feedback.feedback_completed,
        feedback_unavailable=feedback.feedback_unavailable,
        feedback_failed=feedback.feedback_failed,
        feedback_stale=feedback.feedback_stale,
        voice_available_count=feedback.voice_available_count,
        ml_predictions=feedback.ml_predictions,
        baseline_predictions=feedback.baseline_predictions,
        average_score=feedback.average_score,
        average_predicted_score=feedback.average_predicted_score,
        methodology_version=feedback.methodology_version,
        source_versions=dict(feedback.source_versions or {}),
        is_stale=interview_feedback_is_stale(feedback, db),
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def _upsert_answer_feedback(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> tuple[AnswerFeedback, bool]:
    """Create or refresh one answer-feedback row. Returns (row, reused)."""
    result = generate_answer_feedback(db, answer, question)
    existing = _find_answer_feedback(db, answer.id)

    if existing is not None and answer_feedback_is_reusable(existing, result):
        return existing, True

    if existing is None:
        existing = AnswerFeedback(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            methodology_version=config.FEEDBACK_METHODOLOGY_VERSION,
            source_fingerprint=result.source_fingerprint,
        )
        db.add(existing)

    apply_answer_feedback(existing, result)

    try:
        db.commit()
    except IntegrityError:
        # A concurrent request inserted the same (answer, version) row first.
        db.rollback()
        existing = _find_answer_feedback(db, answer.id)
        if existing is None:
            raise
        return existing, True

    db.refresh(existing)
    return existing, False


@router.post("/feedback/{answer_id}", response_model=AnswerFeedbackResponse)
def generate_saved_answer_feedback(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> AnswerFeedbackResponse:
    """Create or refresh evidence-based feedback for one saved answer.

    Feedback uses the current Phase 2 evaluation plus any current Phase 3
    content/voice analysis and Phase 4 prediction. When a required source is
    missing, stale, or unevaluable, an explicit unavailable/failed row is saved
    with the reason; no score is invented.
    """
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)

    if db.get(InterviewSession, answer.interview_id) is None:
        raise HTTPException(
            status_code=409,
            detail="The saved answer is not linked to a valid interview session.",
        )

    existing_before = _find_answer_feedback(db, answer.id)
    feedback, reused = _upsert_answer_feedback(db, answer, question)
    if reused:
        response.status_code = 200
    elif existing_before is None:
        response.status_code = 201
    else:
        response.status_code = 200

    return AnswerFeedbackResponse(
        feedback=_to_answer_feedback_out(feedback, answer, question, db),
        reused=reused,
    )


@router.get("/feedback/{answer_id}", response_model=AnswerFeedbackOut)
def get_saved_answer_feedback(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> AnswerFeedbackOut:
    """Return the saved feedback for one answer, if one exists."""
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)
    feedback = _find_answer_feedback(db, answer.id)
    if feedback is None:
        raise HTTPException(
            status_code=404,
            detail="No feedback has been generated for this answer yet.",
        )
    return _to_answer_feedback_out(feedback, answer, question, db)


@router.post(
    "/interview/{interview_id}/feedback",
    response_model=InterviewFeedbackResponse,
)
def generate_interview_feedback(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> InterviewFeedbackResponse:
    """Generate per-answer feedback, then aggregate one interview summary.

    Each saved answer is upserted in place. Unanswered questions are returned as
    explicit ``unanswered`` breakdown rows and are never scored as zero.
    """
    _get_session_or_404(db, interview_id)

    answers = list(
        db.scalars(
            select(InterviewAnswer)
            .where(InterviewAnswer.interview_id == interview_id)
            .order_by(InterviewAnswer.question_id)
        ).all()
    )
    questions = {
        question.id: question
        for question in db.scalars(select(InterviewQuestion)).all()
    }
    for answer in answers:
        question = questions.get(answer.question_id)
        if question is None:
            # A saved answer whose question was removed cannot be evaluated.
            # Leave it out of feedback instead of inventing a question.
            continue
        _upsert_answer_feedback(db, answer, question)

    result = build_interview_feedback_result(db, interview_id)
    existing = _find_interview_feedback(db, interview_id)
    reused = existing is not None and interview_feedback_is_reusable(
        existing, result
    )

    if not reused:
        if existing is None:
            existing = InterviewFeedback(
                interview_id=interview_id,
                methodology_version=config.FEEDBACK_METHODOLOGY_VERSION,
                source_fingerprint=result.source_fingerprint,
            )
            db.add(existing)
            response.status_code = 201
        else:
            response.status_code = 200
        apply_interview_feedback(existing, result)

        try:
            db.commit()
        except IntegrityError:
            # A concurrent request inserted the same (interview, version) row.
            db.rollback()
            existing = _find_interview_feedback(db, interview_id)
            if existing is None:
                raise
            reused = True

    if existing is None:  # pragma: no cover - defensive, commit creates it
        raise HTTPException(
            status_code=500,
            detail="Interview feedback could not be persisted.",
        )

    if reused:
        response.status_code = 200
        db.refresh(existing)

    rows = list(
        db.execute(
            select(AnswerFeedback, InterviewAnswer, InterviewQuestion)
            .join(
                InterviewAnswer,
                AnswerFeedback.answer_id == InterviewAnswer.id,
            )
            .join(
                InterviewQuestion,
                AnswerFeedback.question_id == InterviewQuestion.id,
            )
            .where(
                AnswerFeedback.interview_id == interview_id,
                AnswerFeedback.methodology_version
                == config.FEEDBACK_METHODOLOGY_VERSION,
            )
            .order_by(InterviewAnswer.question_id)
        ).all()
    )

    return InterviewFeedbackResponse(
        interview_id=interview_id,
        interview_feedback=_to_interview_feedback_out(existing, db),
        answer_feedback=[
            _to_answer_feedback_out(feedback, answer, question, db)
            for feedback, answer, question in rows
        ],
        reused=reused,
    )


@router.get(
    "/interview/{interview_id}/feedback",
    response_model=InterviewFeedbackResponse,
)
def get_interview_feedback(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> InterviewFeedbackResponse:
    """Return saved interview feedback, or an explicit empty dashboard shape.

    A session without generated feedback returns ``200`` with
    ``interview_feedback = null`` so the dashboard can render a dedicated empty
    state instead of treating it as an error.
    """
    _get_session_or_404(db, interview_id)

    existing = _find_interview_feedback(db, interview_id)
    rows = list(
        db.execute(
            select(AnswerFeedback, InterviewAnswer, InterviewQuestion)
            .join(
                InterviewAnswer,
                AnswerFeedback.answer_id == InterviewAnswer.id,
            )
            .join(
                InterviewQuestion,
                AnswerFeedback.question_id == InterviewQuestion.id,
            )
            .where(
                AnswerFeedback.interview_id == interview_id,
                AnswerFeedback.methodology_version
                == config.FEEDBACK_METHODOLOGY_VERSION,
            )
            .order_by(InterviewAnswer.question_id)
        ).all()
    )

    return InterviewFeedbackResponse(
        interview_id=interview_id,
        interview_feedback=(
            _to_interview_feedback_out(existing, db)
            if existing is not None
            else None
        ),
        answer_feedback=[
            _to_answer_feedback_out(feedback, answer, question, db)
            for feedback, answer, question in rows
        ],
        reused=True,
    )
