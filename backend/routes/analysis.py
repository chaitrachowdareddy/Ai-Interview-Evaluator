"""Phase 3 analysis API: content coverage/statistics and voice measurements."""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import (
    ANALYSIS_METHODOLOGY_VERSION,
    EVALUATION_METHODOLOGY_VERSION,
)
from database import get_db
from models import (
    AnswerEvaluation,
    ContentAnalysis,
    InterviewAnswer,
    InterviewQuestion,
    InterviewSession,
    VoiceAnalysis,
)
from schemas import (
    ContentAnalysisOut,
    ContentAnalysisResponse,
    InterviewAnalysesResponse,
    VoiceAnalysisOut,
    VoiceAnalysisResponse,
)
from services.audio_storage import (
    AudioStorageError,
    resolve_audio_path,
    sha256_file,
)
from services.content_analysis import (
    ExistingCoverage,
    analyze_content,
)
from services.evaluation import fingerprint
from services.voice_analysis import (
    STATUS_FAILED as VOICE_STATUS_FAILED,
    STATUS_UNAVAILABLE as VOICE_STATUS_UNAVAILABLE,
    VoiceAnalysisError,
    VoiceAnalysisResult,
    VoiceAnalysisUnavailable,
    analyze_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])

# A deterministic fingerprint for "this answer has no stored recording", so the
# non-null audio hash column stays meaningful and staleness still works.
_NO_AUDIO_FINGERPRINT = hashlib.sha256(b"phase3:no-audio").hexdigest()


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


def _validate_answer_session(db: Session, answer: InterviewAnswer) -> None:
    if db.get(InterviewSession, answer.interview_id) is None:
        raise HTTPException(
            status_code=409,
            detail="The saved answer is not linked to a valid interview session.",
        )


def _reference_text(question: InterviewQuestion) -> str:
    return (question.reference_answer or "").strip()


def _content_is_stale(
    analysis: ContentAnalysis,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> bool:
    return (
        analysis.transcript_hash != fingerprint(answer.transcript)
        or analysis.reference_hash
        != fingerprint(question.reference_answer or "")
    )


def _current_audio_fingerprint(answer: InterviewAnswer) -> str:
    return answer.audio_sha256 or _NO_AUDIO_FINGERPRINT


def _voice_is_stale(
    analysis: VoiceAnalysis,
    answer: InterviewAnswer,
) -> bool:
    return (
        analysis.audio_sha256 != _current_audio_fingerprint(answer)
        or analysis.transcript_hash != fingerprint(answer.transcript)
    )


def _filler_words(values: list[dict] | None) -> list[dict]:
    return [
        {"word": str(item.get("word", "")), "count": int(item.get("count", 0))}
        for item in (values or [])
        if isinstance(item, dict)
    ]


def _repeated_terms(values: list[dict] | None) -> list[dict]:
    return [
        {"term": str(item.get("term", "")), "count": int(item.get("count", 0))}
        for item in (values or [])
        if isinstance(item, dict)
    ]


def _to_content_out(
    analysis: ContentAnalysis,
    answer: InterviewAnswer,
    question: InterviewQuestion,
) -> ContentAnalysisOut:
    return ContentAnalysisOut(
        id=analysis.id,
        answer_id=analysis.answer_id,
        interview_id=analysis.interview_id,
        question_id=analysis.question_id,
        coverage_score=analysis.coverage_score,
        coverage_source=analysis.coverage_source,
        matched_concepts=list(analysis.matched_concepts or []),
        missing_concepts=list(analysis.missing_concepts or []),
        filler_word_count=analysis.filler_word_count,
        filler_rate_percent=analysis.filler_rate_percent,
        filler_words=_filler_words(analysis.filler_words),
        repeated_terms=_repeated_terms(analysis.repeated_terms),
        repetition_rate_percent=analysis.repetition_rate_percent,
        sentence_count=analysis.sentence_count,
        average_sentence_words=analysis.average_sentence_words,
        structure_notes=list(analysis.structure_notes or []),
        status=analysis.status,
        status_detail=analysis.status_detail,
        methodology_version=analysis.methodology_version,
        is_stale=_content_is_stale(analysis, answer, question),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
    )


def _to_voice_out(
    analysis: VoiceAnalysis,
    answer: InterviewAnswer,
) -> VoiceAnalysisOut:
    return VoiceAnalysisOut(
        id=analysis.id,
        answer_id=analysis.answer_id,
        interview_id=analysis.interview_id,
        question_id=analysis.question_id,
        status=analysis.status,
        status_detail=analysis.status_detail,
        duration_seconds=analysis.duration_seconds,
        analyzed_duration_seconds=analysis.analyzed_duration_seconds,
        sample_rate_hz=analysis.sample_rate_hz,
        channels=analysis.channels,
        truncated_for_analysis=analysis.truncated_for_analysis,
        speech_detected=analysis.speech_detected,
        speech_duration_seconds=analysis.speech_duration_seconds,
        silence_duration_seconds=analysis.silence_duration_seconds,
        speech_ratio_percent=analysis.speech_ratio_percent,
        pause_count=analysis.pause_count,
        total_pause_seconds=analysis.total_pause_seconds,
        mean_pause_seconds=analysis.mean_pause_seconds,
        longest_pause_seconds=analysis.longest_pause_seconds,
        pauses_per_minute=analysis.pauses_per_minute,
        word_count=analysis.word_count,
        speaking_rate_wpm=analysis.speaking_rate_wpm,
        rms_mean=analysis.rms_mean,
        rms_std=analysis.rms_std,
        rms_variation_percent=analysis.rms_variation_percent,
        rms_mean_dbfs=analysis.rms_mean_dbfs,
        peak_amplitude=analysis.peak_amplitude,
        clipping_percent=analysis.clipping_percent,
        pitch_median_hz=analysis.pitch_median_hz,
        pitch_mean_hz=analysis.pitch_mean_hz,
        pitch_std_hz=analysis.pitch_std_hz,
        pitch_p10_hz=analysis.pitch_p10_hz,
        pitch_p90_hz=analysis.pitch_p90_hz,
        pitch_range_hz=analysis.pitch_range_hz,
        pitch_voiced_seconds=analysis.pitch_voiced_seconds,
        pitch_reliable=analysis.pitch_reliable,
        quality_notes=list(analysis.quality_notes or []),
        methodology_version=analysis.methodology_version,
        is_stale=_voice_is_stale(analysis, answer),
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
    )


def _find_content_analysis(
    db: Session,
    answer_id: int,
) -> ContentAnalysis | None:
    return db.scalar(
        select(ContentAnalysis).where(
            ContentAnalysis.answer_id == answer_id,
            ContentAnalysis.methodology_version
            == ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def _find_voice_analysis(
    db: Session,
    answer_id: int,
) -> VoiceAnalysis | None:
    return db.scalar(
        select(VoiceAnalysis).where(
            VoiceAnalysis.answer_id == answer_id,
            VoiceAnalysis.methodology_version == ANALYSIS_METHODOLOGY_VERSION,
        )
    )


def _phase2_coverage(
    db: Session,
    answer: InterviewAnswer,
    question: InterviewQuestion,
    transcript_hash: str,
    reference_hash: str,
) -> ExistingCoverage | None:
    """Reuse a current Phase 2 evaluation instead of recomputing coverage."""

    evaluation = db.scalar(
        select(AnswerEvaluation).where(
            AnswerEvaluation.answer_id == answer.id,
            AnswerEvaluation.methodology_version
            == EVALUATION_METHODOLOGY_VERSION,
        )
    )
    if evaluation is None or evaluation.status != "evaluated":
        return None
    if (
        evaluation.transcript_hash != transcript_hash
        or evaluation.reference_hash != reference_hash
    ):
        return None

    return ExistingCoverage(
        score=float(evaluation.keyword_score),
        matched=tuple(evaluation.matched_keywords or []),
        missing=tuple(evaluation.missing_keywords or []),
    )


def _apply_content_result(
    row: ContentAnalysis,
    result,
    transcript_hash: str,
    reference_hash: str,
) -> None:
    row.coverage_score = result.coverage_score
    row.coverage_source = result.coverage_source
    row.matched_concepts = list(result.matched_concepts)
    row.missing_concepts = list(result.missing_concepts)
    row.filler_word_count = result.filler_word_count
    row.filler_rate_percent = result.filler_rate_percent
    row.filler_words = [
        {"word": word, "count": count}
        for word, count in result.filler_words
    ]
    row.repeated_terms = [
        {"term": term, "count": count}
        for term, count in result.repeated_terms
    ]
    row.repetition_rate_percent = result.repetition_rate_percent
    row.sentence_count = result.sentence_count
    row.average_sentence_words = result.average_sentence_words
    row.structure_notes = list(result.structure_notes)
    row.status = result.status
    row.status_detail = result.status_detail
    row.transcript_hash = transcript_hash
    row.reference_hash = reference_hash


def _empty_voice_result(
    status: str,
    detail: str,
) -> VoiceAnalysisResult:
    return VoiceAnalysisResult(
        status=status,
        status_detail=detail,
        duration_seconds=None,
        analyzed_duration_seconds=None,
        sample_rate_hz=None,
        channels=None,
        truncated_for_analysis=False,
        speech_detected=False,
        speech_duration_seconds=None,
        silence_duration_seconds=None,
        speech_ratio_percent=None,
        pause_count=None,
        total_pause_seconds=None,
        mean_pause_seconds=None,
        longest_pause_seconds=None,
        pauses_per_minute=None,
        word_count=None,
        speaking_rate_wpm=None,
        rms_mean=None,
        rms_std=None,
        rms_variation_percent=None,
        rms_mean_dbfs=None,
        peak_amplitude=None,
        clipping_percent=None,
        pitch_median_hz=None,
        pitch_mean_hz=None,
        pitch_std_hz=None,
        pitch_p10_hz=None,
        pitch_p90_hz=None,
        pitch_range_hz=None,
        pitch_voiced_seconds=None,
        pitch_reliable=False,
        quality_notes=(),
    )


def _apply_voice_result(
    row: VoiceAnalysis,
    result: VoiceAnalysisResult,
    audio_sha256: str,
    transcript_hash: str,
) -> None:
    row.status = result.status
    row.status_detail = result.status_detail
    row.duration_seconds = result.duration_seconds
    row.analyzed_duration_seconds = result.analyzed_duration_seconds
    row.sample_rate_hz = result.sample_rate_hz
    row.channels = result.channels
    row.truncated_for_analysis = result.truncated_for_analysis
    row.speech_detected = result.speech_detected
    row.speech_duration_seconds = result.speech_duration_seconds
    row.silence_duration_seconds = result.silence_duration_seconds
    row.speech_ratio_percent = result.speech_ratio_percent
    row.pause_count = result.pause_count
    row.total_pause_seconds = result.total_pause_seconds
    row.mean_pause_seconds = result.mean_pause_seconds
    row.longest_pause_seconds = result.longest_pause_seconds
    row.pauses_per_minute = result.pauses_per_minute
    row.word_count = result.word_count
    row.speaking_rate_wpm = result.speaking_rate_wpm
    row.rms_mean = result.rms_mean
    row.rms_std = result.rms_std
    row.rms_variation_percent = result.rms_variation_percent
    row.rms_mean_dbfs = result.rms_mean_dbfs
    row.peak_amplitude = result.peak_amplitude
    row.clipping_percent = result.clipping_percent
    row.pitch_median_hz = result.pitch_median_hz
    row.pitch_mean_hz = result.pitch_mean_hz
    row.pitch_std_hz = result.pitch_std_hz
    row.pitch_p10_hz = result.pitch_p10_hz
    row.pitch_p90_hz = result.pitch_p90_hz
    row.pitch_range_hz = result.pitch_range_hz
    row.pitch_voiced_seconds = result.pitch_voiced_seconds
    row.pitch_reliable = result.pitch_reliable
    row.quality_notes = list(result.quality_notes)
    row.audio_sha256 = audio_sha256
    row.transcript_hash = transcript_hash


@router.post(
    "/analysis/{answer_id}/content",
    response_model=ContentAnalysisResponse,
)
def analyze_saved_answer_content(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> ContentAnalysisResponse:
    """Analyse or re-analyse the saved transcript's content.

    Repeat requests for unchanged inputs return the saved row. A transcript or
    reference change marks the row stale and this endpoint updates the same row
    in place, never creating a duplicate.
    """

    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)
    _validate_answer_session(db, answer)

    transcript_hash = fingerprint(answer.transcript)
    reference_hash = fingerprint(question.reference_answer or "")
    existing = _find_content_analysis(db, answer.id)

    if (
        existing is not None
        and existing.transcript_hash == transcript_hash
        and existing.reference_hash == reference_hash
    ):
        response.status_code = 200
        return ContentAnalysisResponse(
            content_analysis=_to_content_out(existing, answer, question),
            reused=True,
        )

    existing_coverage = _phase2_coverage(
        db,
        answer,
        question,
        transcript_hash,
        reference_hash,
    )
    result = analyze_content(
        answer.transcript,
        _reference_text(question),
        existing_coverage,
    )

    if existing is None:
        existing = ContentAnalysis(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            methodology_version=ANALYSIS_METHODOLOGY_VERSION,
        )
        db.add(existing)
        response.status_code = 201
    else:
        response.status_code = 200

    _apply_content_result(
        existing,
        result,
        transcript_hash,
        reference_hash,
    )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = _find_content_analysis(db, answer.id)
        if existing is None:
            raise
        response.status_code = 200
        return ContentAnalysisResponse(
            content_analysis=_to_content_out(existing, answer, question),
            reused=True,
        )

    db.refresh(existing)
    return ContentAnalysisResponse(
        content_analysis=_to_content_out(existing, answer, question),
        reused=False,
    )


@router.get(
    "/analysis/{answer_id}/content",
    response_model=ContentAnalysisOut,
)
def get_saved_content_analysis(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> ContentAnalysisOut:
    """Return the saved content analysis for an answer, if one exists."""
    answer = _get_answer_or_404(db, answer_id)
    question = _get_question_or_404(db, answer.question_id)
    analysis = _find_content_analysis(db, answer.id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="No content analysis has been saved for this answer yet.",
        )
    return _to_content_out(analysis, answer, question)


@router.post(
    "/analysis/{answer_id}/voice",
    response_model=VoiceAnalysisResponse,
)
def analyze_saved_answer_voice(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
    response: Response,
) -> VoiceAnalysisResponse:
    """Measure the stored original recording or persist an explicit status.

    A missing recording is persisted as ``unavailable`` and an undecodable one
    as ``failed``; neither invents metric values. Missing analysis packages are
    a server problem and return 503 without writing a misleading row.
    """

    answer = _get_answer_or_404(db, answer_id)
    _get_question_or_404(db, answer.question_id)
    _validate_answer_session(db, answer)

    transcript_hash = fingerprint(answer.transcript)
    audio_fingerprint = _current_audio_fingerprint(answer)
    existing = _find_voice_analysis(db, answer.id)

    if (
        existing is not None
        and existing.audio_sha256 == audio_fingerprint
        and existing.transcript_hash == transcript_hash
    ):
        response.status_code = 200
        return VoiceAnalysisResponse(
            voice_analysis=_to_voice_out(existing, answer),
            reused=True,
        )

    if not answer.audio_path:
        result = _empty_voice_result(
            VOICE_STATUS_UNAVAILABLE,
            "No original recording is stored for this answer, so voice "
            "analysis is unavailable. Earlier recordings from before Phase 3 "
            "were not retained.",
        )
    else:
        try:
            path = resolve_audio_path(answer.audio_path)
        except AudioStorageError:
            result = _empty_voice_result(
                VOICE_STATUS_UNAVAILABLE,
                "The stored recording could not be located, so voice analysis "
                "is unavailable.",
            )
        else:
            if not path.is_file():
                result = _empty_voice_result(
                    VOICE_STATUS_UNAVAILABLE,
                    "The stored recording file is missing, so voice analysis "
                    "is unavailable.",
                )
            else:
                try:
                    actual_hash = sha256_file(path)
                except OSError:
                    result = _empty_voice_result(
                        VOICE_STATUS_FAILED,
                        "The stored recording could not be read, so voice "
                        "analysis failed.",
                    )
                else:
                    if (
                        answer.audio_sha256
                        and actual_hash != answer.audio_sha256
                    ):
                        result = _empty_voice_result(
                            VOICE_STATUS_FAILED,
                            "The stored recording does not match the "
                            "fingerprint saved with this answer, so analysis "
                            "was not run.",
                        )
                    else:
                        audio_fingerprint = (
                            answer.audio_sha256 or actual_hash
                        )
                        try:
                            result = analyze_audio(path, answer.transcript)
                        except VoiceAnalysisUnavailable as exc:
                            raise HTTPException(
                                status_code=exc.status_code,
                                detail=exc.message,
                            ) from exc
                        except VoiceAnalysisError as exc:
                            result = _empty_voice_result(
                                VOICE_STATUS_FAILED,
                                exc.message,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "Unexpected voice-analysis failure (%s).",
                                type(exc).__name__,
                            )
                            result = _empty_voice_result(
                                VOICE_STATUS_FAILED,
                                "Voice analysis failed unexpectedly. Check the "
                                "recording and try again.",
                            )

    if existing is None:
        existing = VoiceAnalysis(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            methodology_version=ANALYSIS_METHODOLOGY_VERSION,
        )
        db.add(existing)
        response.status_code = 201
    else:
        response.status_code = 200

    _apply_voice_result(
        existing,
        result,
        audio_fingerprint,
        transcript_hash,
    )

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = _find_voice_analysis(db, answer.id)
        if existing is None:
            raise
        response.status_code = 200
        return VoiceAnalysisResponse(
            voice_analysis=_to_voice_out(existing, answer),
            reused=True,
        )

    db.refresh(existing)
    return VoiceAnalysisResponse(
        voice_analysis=_to_voice_out(existing, answer),
        reused=False,
    )


@router.get(
    "/analysis/{answer_id}/voice",
    response_model=VoiceAnalysisOut,
)
def get_saved_voice_analysis(
    answer_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> VoiceAnalysisOut:
    """Return the saved voice analysis for an answer, if one exists."""
    answer = _get_answer_or_404(db, answer_id)
    analysis = _find_voice_analysis(db, answer.id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="No voice analysis has been saved for this answer yet.",
        )
    return _to_voice_out(analysis, answer)


@router.get(
    "/interview/{interview_id}/analyses",
    response_model=InterviewAnalysesResponse,
)
def get_interview_analyses(
    interview_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> InterviewAnalysesResponse:
    """Return both analysis types for one interview in one request."""
    _get_session_or_404(db, interview_id)

    content_rows = db.execute(
        select(ContentAnalysis, InterviewAnswer, InterviewQuestion)
        .join(
            InterviewAnswer,
            ContentAnalysis.answer_id == InterviewAnswer.id,
        )
        .join(
            InterviewQuestion,
            ContentAnalysis.question_id == InterviewQuestion.id,
        )
        .where(
            ContentAnalysis.interview_id == interview_id,
            ContentAnalysis.methodology_version
            == ANALYSIS_METHODOLOGY_VERSION,
        )
        .order_by(ContentAnalysis.id)
    ).all()

    voice_rows = db.execute(
        select(VoiceAnalysis, InterviewAnswer)
        .join(
            InterviewAnswer,
            VoiceAnalysis.answer_id == InterviewAnswer.id,
        )
        .where(
            VoiceAnalysis.interview_id == interview_id,
            VoiceAnalysis.methodology_version
            == ANALYSIS_METHODOLOGY_VERSION,
        )
        .order_by(VoiceAnalysis.id)
    ).all()

    return InterviewAnalysesResponse(
        interview_id=interview_id,
        content_analyses=[
            _to_content_out(analysis, answer, question)
            for analysis, answer, question in content_rows
        ],
        voice_analyses=[
            _to_voice_out(analysis, answer)
            for analysis, answer in voice_rows
        ],
    )
