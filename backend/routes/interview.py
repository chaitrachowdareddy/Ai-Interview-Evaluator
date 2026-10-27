"""Phase 1 interview API: questions, sessions, transcription, and answers."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import MAX_AUDIO_BYTES, MAX_AUDIO_MB
from database import get_db
from models import InterviewAnswer, InterviewQuestion, InterviewSession
from schemas import (
    AnswerOut,
    AnswersResponse,
    InterviewSessionOut,
    QuestionOut,
    TranscriptionResponse,
)
from services.audio_storage import store_audio
from services.transcription import (
    TranscriptionError,
    normalize_audio_type,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interview", tags=["Interview"])


def _get_interview_or_404(db: Session, interview_id: int) -> InterviewSession:
    interview = db.get(InterviewSession, interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Interview session not found.")
    return interview


def _get_question_or_404(db: Session, question_id: int) -> InterviewQuestion:
    question = db.get(InterviewQuestion, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Interview question not found.")
    return question


def _to_answer_out(
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> AnswerOut:
    return AnswerOut(
        id=answer.id,
        interview_id=answer.interview_id,
        question_id=answer.question_id,
        question_text=question.question_text,
        category=question.category,
        transcript=answer.transcript,
        has_audio=bool(answer.audio_path),
        created_at=answer.created_at,
        updated_at=answer.updated_at,
    )


@router.get("/questions", response_model=list[QuestionOut])
def get_questions(
    db: Annotated[Session, Depends(get_db)],
) -> list[QuestionOut]:
    """Return all interview questions stored in MySQL."""
    questions = list(
        db.scalars(
            select(InterviewQuestion).order_by(InterviewQuestion.id)
        ).all()
    )
    return [
        QuestionOut(
            id=question.id,
            question_text=question.question_text,
            category=question.category,
            has_reference_answer=bool(
                (question.reference_answer or "").strip()
            ),
        )
        for question in questions
    ]


@router.post(
    "/start",
    response_model=InterviewSessionOut,
    status_code=201,
)
def start_interview(
    db: Annotated[Session, Depends(get_db)],
) -> InterviewSession:
    """Create and persist a new interview session."""
    interview = InterviewSession(status="in_progress")
    db.add(interview)
    db.commit()
    db.refresh(interview)
    return interview


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_answer(
    file: Annotated[UploadFile, File(description="Recorded audio answer")],
    interview_id: Annotated[int, Form(gt=0)],
    question_id: Annotated[int, Form(gt=0)],
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> TranscriptionResponse:
    """Transcribe an uploaded recording and save it against a session/question."""
    _get_interview_or_404(db, interview_id)
    question = _get_question_or_404(db, question_id)

    try:
        mime_type, extension = normalize_audio_type(
            file.content_type,
            file.filename,
        )
    except TranscriptionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    # Read one byte past the limit so oversized uploads can be detected without
    # loading an arbitrarily large request into memory.
    audio_bytes = file.file.read(MAX_AUDIO_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The recording is empty. Please record your answer again.",
        )
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The recording is too large. Maximum upload size is "
                f"{MAX_AUDIO_MB} MB."
            ),
        )

    safe_filename = f"answer-{interview_id}-{question_id}.{extension}"

    try:
        transcript = transcribe_audio(
            audio_bytes,
            safe_filename,
            mime_type,
        )
    except TranscriptionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    answer = db.scalar(
        select(InterviewAnswer).where(
            InterviewAnswer.interview_id == interview_id,
            InterviewAnswer.question_id == question_id,
        )
    )

    updated = answer is not None
    if answer is None:
        answer = InterviewAnswer(
            interview_id=interview_id,
            question_id=question_id,
        )
        db.add(answer)

    # Keep the original bytes for voice analysis. A storage problem must not
    # discard the transcript that Groq already produced.
    try:
        stored = store_audio(
            audio_bytes,
            interview_id,
            question_id,
            extension,
        )
        answer.audio_path = stored.relative_path
        answer.audio_sha256 = stored.sha256
    except Exception as exc:  # noqa: BLE001 - transcription stays successful
        logger.warning(
            "Could not store the recording for voice analysis (%s).",
            type(exc).__name__,
        )
        answer.audio_path = None
        answer.audio_sha256 = None

    answer.transcript = transcript
    answer.audio_mime_type = mime_type
    answer.audio_size_bytes = len(audio_bytes)

    db.commit()
    db.refresh(answer)

    response.status_code = 200 if updated else 201
    return TranscriptionResponse(
        answer=_to_answer_out(answer, question),
        updated=updated,
    )


@router.get("/{interview_id}/answers", response_model=AnswersResponse)
def get_answers(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> AnswersResponse:
    """Return a session and all of its saved transcripts."""
    interview = _get_interview_or_404(db, interview_id)

    rows = db.execute(
        select(InterviewAnswer, InterviewQuestion)
        .join(
            InterviewQuestion,
            InterviewAnswer.question_id == InterviewQuestion.id,
        )
        .where(InterviewAnswer.interview_id == interview_id)
        .order_by(InterviewAnswer.id)
    ).all()

    return AnswersResponse(
        interview_id=interview.id,
        status=interview.status,
        started_at=interview.started_at,
        answers=[
            _to_answer_out(answer, question)
            for answer, question in rows
        ],
    )
