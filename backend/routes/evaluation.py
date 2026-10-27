"""Phase 2 evaluation API: score saved transcripts and summarize a session."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import (
    EVALUATION_METHODOLOGY_VERSION,
    ML_PREDICTION_METHODOLOGY_VERSION,
)
from database import get_db
from models import (
    AnswerEvaluation,
    InterviewAnswer,
    InterviewQuestion,
    InterviewSession,
    ScorePrediction,
)
from schemas import (
    EvaluationListResponse,
    EvaluationOut,
    EvaluationResponse,
    InterviewSummaryOut,
    QuestionSummaryOut,
)
from services.evaluation import (
    STATUS_EVALUATED,
    STATUS_UNEVALUABLE,
    MissingReferenceAnswerError,
    evaluate_transcript,
    fingerprint,
)
from services.ml_scoring import (
    build_prediction_summary,
    prediction_is_stale,
)
from services.semantic_similarity import SemanticModelError

router = APIRouter(tags=["Evaluation"])

STATUS_UNANSWERED = "unanswered"
STATUS_UNEVALUATED = "unevaluated"
STATUS_NOT_EVALUABLE = "not_evaluable"
STATUS_STALE = "stale"


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


def _reference_text(question: InterviewQuestion) -> str:
    return (question.reference_answer or "").strip()


def _evaluation_is_stale(
    evaluation: AnswerEvaluation,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> bool:
    return (
        evaluation.transcript_hash != fingerprint(answer.transcript)
        or evaluation.reference_hash
        != fingerprint(question.reference_answer or "")
    )


def _to_evaluation_out(
    evaluation: AnswerEvaluation,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> EvaluationOut:
    return EvaluationOut(
        id=evaluation.id,
        answer_id=evaluation.answer_id,
        interview_id=evaluation.interview_id,
        question_id=evaluation.question_id,
        semantic_score=evaluation.semantic_score,
        keyword_score=evaluation.keyword_score,
        combined_score=evaluation.combined_score,
        matched_keywords=list(evaluation.matched_keywords or []),
        missing_keywords=list(evaluation.missing_keywords or []),
        status=evaluation.status,
        status_detail=evaluation.status_detail,
        methodology_version=evaluation.methodology_version,
        is_stale=_evaluation_is_stale(evaluation, answer, question),
        created_at=evaluation.created_at,
        updated_at=evaluation.updated_at,
    )


def _prediction_fields(
    db: Session,
    prediction: ScorePrediction | None,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> dict[str, object]:
    """Backward-compatible prediction fields for one question summary."""
    if prediction is None:
        return {}
    return {
        "prediction_id": prediction.id,
        "prediction_status": prediction.status,
        "predicted_score": prediction.score,
        "prediction_method": prediction.method,
        "prediction_is_stale": prediction_is_stale(
            prediction,
            answer,
            question,
            db,
        ),
    }


@router.post("/evaluation/{answer_id}", response_model=EvaluationResponse)
def evaluate_saved_answer(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> EvaluationResponse:
    """Evaluate an already-saved transcript and persist the result.

    Repeat requests for the same transcript/scoring version return the saved
    evaluation instead of inserting duplicates. If the transcript or reference
    answer changed, the same row is recomputed.
    """
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)

    # The answer's own foreign keys define the association. No interview or
    # question IDs are accepted from the client, which prevents callers from
    # attaching an evaluation to a different session.
    if db.get(InterviewSession, answer.interview_id) is None:
        raise HTTPException(
            status_code=409,
            detail="The saved answer is not linked to a valid interview session.",
        )

    reference = _reference_text(question)
    if not reference:
        raise HTTPException(
            status_code=409,
            detail=(
                "This question has no reference answer, so it cannot be "
                "evaluated. Add a reference answer to interview_questions."
            ),
        )

    transcript_hash = fingerprint(answer.transcript)
    reference_hash = fingerprint(reference)

    evaluation = db.scalar(
        select(AnswerEvaluation).where(
            AnswerEvaluation.answer_id == answer.id,
            AnswerEvaluation.methodology_version
            == EVALUATION_METHODOLOGY_VERSION,
        )
    )

    if (
        evaluation is not None
        and evaluation.transcript_hash == transcript_hash
        and evaluation.reference_hash == reference_hash
    ):
        response.status_code = 200
        return EvaluationResponse(
            evaluation=_to_evaluation_out(evaluation, answer, question),
            reused=True,
        )

    try:
        result = evaluate_transcript(answer.transcript, reference)
    except MissingReferenceAnswerError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except SemanticModelError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    if evaluation is None:
        evaluation = AnswerEvaluation(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            methodology_version=EVALUATION_METHODOLOGY_VERSION,
            transcript_hash=transcript_hash,
            reference_hash=reference_hash,
        )
        db.add(evaluation)
        response.status_code = 201
    else:
        response.status_code = 200

    evaluation.semantic_score = result.semantic_score
    evaluation.keyword_score = result.keyword_score
    evaluation.combined_score = result.combined_score
    evaluation.matched_keywords = list(result.matched_keywords)
    evaluation.missing_keywords = list(result.missing_keywords)
    evaluation.status = result.status
    evaluation.status_detail = result.status_detail
    evaluation.transcript_hash = transcript_hash
    evaluation.reference_hash = reference_hash

    try:
        db.commit()
    except IntegrityError:
        # A concurrent request inserted the same (answer, version) row first.
        # Return that row rather than exposing a constraint error.
        db.rollback()
        evaluation = db.scalar(
            select(AnswerEvaluation).where(
                AnswerEvaluation.answer_id == answer.id,
                AnswerEvaluation.methodology_version
                == EVALUATION_METHODOLOGY_VERSION,
            )
        )
        if evaluation is None:
            raise
        response.status_code = 200
        return EvaluationResponse(
            evaluation=_to_evaluation_out(evaluation, answer, question),
            reused=True,
        )

    db.refresh(evaluation)
    return EvaluationResponse(
        evaluation=_to_evaluation_out(evaluation, answer, question),
        reused=False,
    )


@router.get("/evaluation/{answer_id}", response_model=EvaluationOut)
def get_saved_evaluation(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> EvaluationOut:
    """Return the saved evaluation for an answer, if one exists."""
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)

    evaluation = db.scalar(
        select(AnswerEvaluation).where(
            AnswerEvaluation.answer_id == answer.id,
            AnswerEvaluation.methodology_version
            == EVALUATION_METHODOLOGY_VERSION,
        )
    )
    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation has been saved for this answer yet.",
        )

    return _to_evaluation_out(evaluation, answer, question)


@router.get(
    "/interview/{interview_id}/evaluations",
    response_model=EvaluationListResponse,
)
def get_interview_evaluations(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> EvaluationListResponse:
    """Return every evaluation belonging to one interview session."""
    _get_session_or_404(db, interview_id)

    rows = db.execute(
        select(AnswerEvaluation, InterviewAnswer, InterviewQuestion)
        .join(
            InterviewAnswer,
            AnswerEvaluation.answer_id == InterviewAnswer.id,
        )
        .join(
            InterviewQuestion,
            AnswerEvaluation.question_id == InterviewQuestion.id,
        )
        .where(
            AnswerEvaluation.interview_id == interview_id,
            AnswerEvaluation.methodology_version
            == EVALUATION_METHODOLOGY_VERSION,
        )
        .order_by(AnswerEvaluation.id)
    ).all()

    return EvaluationListResponse(
        interview_id=interview_id,
        evaluations=[
            _to_evaluation_out(evaluation, answer, question)
            for evaluation, answer, question in rows
        ],
    )


@router.get(
    "/interview/{interview_id}/summary",
    response_model=InterviewSummaryOut,
)
def get_interview_summary(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> InterviewSummaryOut:
    """Return per-question results and the average of valid evaluations.

    Unanswered questions are not treated as zero. The average uses only
    currently valid (not stale) evaluations with status ``evaluated``.
    """
    session = _get_session_or_404(db, interview_id)

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
    evaluations = list(
        db.scalars(
            select(AnswerEvaluation).where(
                AnswerEvaluation.interview_id == interview_id,
                AnswerEvaluation.methodology_version
                == EVALUATION_METHODOLOGY_VERSION,
            )
        ).all()
    )
    predictions = list(
        db.scalars(
            select(ScorePrediction).where(
                ScorePrediction.interview_id == interview_id,
                ScorePrediction.methodology_version
                == ML_PREDICTION_METHODOLOGY_VERSION,
            )
        ).all()
    )

    answer_by_question = {answer.question_id: answer for answer in answers}
    evaluation_by_answer = {
        evaluation.answer_id: evaluation for evaluation in evaluations
    }
    prediction_by_answer = {
        prediction.answer_id: prediction for prediction in predictions
    }

    question_summaries: list[QuestionSummaryOut] = []
    evaluated_scores: list[float] = []
    unevaluable_count = 0

    for question in questions:
        reference = _reference_text(question)
        has_reference = bool(reference)
        answer = answer_by_question.get(question.id)

        if answer is None:
            question_summaries.append(
                QuestionSummaryOut(
                    question_id=question.id,
                    question_text=question.question_text,
                    category=question.category,
                    has_reference_answer=has_reference,
                    status=STATUS_UNANSWERED,
                    status_detail="No transcript has been saved for this question.",
                )
            )
            continue

        evaluation = evaluation_by_answer.get(answer.id)

        if not has_reference:
            question_summaries.append(
                QuestionSummaryOut(
                    question_id=question.id,
                    question_text=question.question_text,
                    category=question.category,
                    has_reference_answer=False,
                    answer_id=answer.id,
                    status=STATUS_NOT_EVALUABLE,
                    status_detail=(
                        "This question has no reference answer, so it cannot "
                        "be evaluated."
                    ),
                    **_prediction_fields(
                        db,
                        prediction_by_answer.get(answer.id),
                        answer,
                        question,
                    ),
                )
            )
            continue

        if evaluation is None:
            question_summaries.append(
                QuestionSummaryOut(
                    question_id=question.id,
                    question_text=question.question_text,
                    category=question.category,
                    has_reference_answer=True,
                    answer_id=answer.id,
                    status=STATUS_UNEVALUATED,
                    status_detail="A transcript is saved but not yet evaluated.",
                    **_prediction_fields(
                        db,
                        prediction_by_answer.get(answer.id),
                        answer,
                        question,
                    ),
                )
            )
            continue

        stale = _evaluation_is_stale(evaluation, answer, question)
        if stale:
            status = STATUS_STALE
            status_detail = (
                "The transcript or reference answer changed after this "
                "evaluation. Evaluate the answer again to refresh the score."
            )
        elif evaluation.status == STATUS_UNEVALUABLE:
            status = STATUS_UNEVALUABLE
            status_detail = evaluation.status_detail
            unevaluable_count += 1
        else:
            status = STATUS_EVALUATED
            status_detail = None
            evaluated_scores.append(evaluation.combined_score)

        question_summaries.append(
            QuestionSummaryOut(
                question_id=question.id,
                question_text=question.question_text,
                category=question.category,
                has_reference_answer=True,
                answer_id=answer.id,
                evaluation_id=evaluation.id,
                status=status,
                semantic_score=evaluation.semantic_score,
                keyword_score=evaluation.keyword_score,
                combined_score=evaluation.combined_score,
                matched_keywords=list(evaluation.matched_keywords or []),
                missing_keywords=list(evaluation.missing_keywords or []),
                status_detail=status_detail,
                **_prediction_fields(
                    db,
                    prediction_by_answer.get(answer.id),
                    answer,
                    question,
                ),
            )
        )

    average_score = (
        round(sum(evaluated_scores) / len(evaluated_scores), 2)
        if evaluated_scores
        else None
    )
    prediction_summary = build_prediction_summary(db, interview_id)

    return InterviewSummaryOut(
        interview_id=session.id,
        status=session.status,
        started_at=session.started_at,
        total_questions=len(questions),
        answers_submitted=len(answers),
        answers_evaluated=len(evaluated_scores),
        answers_unevaluable=unevaluable_count,
        questions_unanswered=sum(
            1
            for item in question_summaries
            if item.status == STATUS_UNANSWERED
        ),
        average_score=average_score,
        questions=question_summaries,
        prediction_methodology_version=prediction_summary.methodology_version,
        predictions_completed=prediction_summary.predictions_completed,
        predictions_ml=prediction_summary.predictions_ml,
        predictions_baseline=prediction_summary.predictions_baseline,
        predictions_stale=prediction_summary.predictions_stale,
        predictions_unavailable=prediction_summary.predictions_unavailable,
        average_predicted_score=prediction_summary.average_predicted_score,
        prediction_method=prediction_summary.prediction_method,
        ml_model_available=prediction_summary.ml_model_available,
    )
