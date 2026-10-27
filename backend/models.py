"""SQLAlchemy ORM models for the AI Interview Evaluator."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_text: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="General")
    # Nullable so existing installations can be upgraded without inventing an
    # answer. A question without a reference answer is not evaluable.
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )

    answers: Mapped[list["InterviewAnswer"]] = relationship(
        back_populates="question"
    )
    evaluations: Mapped[list["AnswerEvaluation"]] = relationship(
        back_populates="question"
    )
    content_analyses: Mapped[list["ContentAnalysis"]] = relationship(
        back_populates="question"
    )
    voice_analyses: Mapped[list["VoiceAnalysis"]] = relationship(
        back_populates="question"
    )
    predictions: Mapped[list["ScorePrediction"]] = relationship(
        back_populates="question"
    )
    answer_feedback: Mapped[list["AnswerFeedback"]] = relationship(
        back_populates="question"
    )

    def __repr__(self) -> str:
        return f"InterviewQuestion(id={self.id!r}, category={self.category!r})"


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="in_progress"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )

    answers: Mapped[list["InterviewAnswer"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    evaluations: Mapped[list["AnswerEvaluation"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    content_analyses: Mapped[list["ContentAnalysis"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    voice_analyses: Mapped[list["VoiceAnalysis"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["ScorePrediction"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    answer_feedback: Mapped[list["AnswerFeedback"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    interview_feedback: Mapped[list["InterviewFeedback"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"InterviewSession(id={self.id!r}, status={self.status!r})"


class InterviewAnswer(Base):
    __tablename__ = "interview_answers"
    __table_args__ = (
        UniqueConstraint(
            "interview_id",
            "question_id",
            name="uq_interview_answers_interview_question",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    audio_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    audio_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Phase 3 keeps the original recording. ``audio_path`` is relative to the
    # configured AUDIO_STORAGE_DIR so the database stays portable.
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    session: Mapped[InterviewSession] = relationship(back_populates="answers")
    question: Mapped[InterviewQuestion] = relationship(back_populates="answers")
    evaluations: Mapped[list["AnswerEvaluation"]] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
    )
    content_analyses: Mapped[list["ContentAnalysis"]] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
    )
    voice_analyses: Mapped[list["VoiceAnalysis"]] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["ScorePrediction"]] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
    )
    feedback: Mapped[list["AnswerFeedback"]] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            "InterviewAnswer("
            f"id={self.id!r}, interview_id={self.interview_id!r}, "
            f"question_id={self.question_id!r})"
        )


class AnswerEvaluation(Base):
    """A saved, explainable evaluation for one transcript."""
    __tablename__ = "answer_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "methodology_version",
            name="uq_answer_evaluations_answer_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("interview_answers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )

    semantic_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    keyword_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    combined_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    matched_keywords: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    missing_keywords: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="evaluated"
    )
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase2-v1"
    )

    # Hashes let the API detect when an answer was re-transcribed or a reference
    # answer was edited after an evaluation was saved. The original transcript
    # itself is never modified by evaluation.
    transcript_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    answer: Mapped[InterviewAnswer] = relationship(back_populates="evaluations")
    session: Mapped[InterviewSession] = relationship(back_populates="evaluations")
    question: Mapped[InterviewQuestion] = relationship(back_populates="evaluations")

    def __repr__(self) -> str:
        return (
            "AnswerEvaluation("
            f"id={self.id!r}, answer_id={self.answer_id!r}, "
            f"status={self.status!r}, combined_score={self.combined_score!r})"
        )


class ContentAnalysis(Base):
    """Explainable lexical/structure analysis of one saved transcript.

    The score reuses the Phase 2 weighted keyword coverage so the two features
    never display conflicting coverage numbers. It is lexical coverage of the
    reference terminology, not factual correctness.
    """

    __tablename__ = "content_analyses"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "methodology_version",
            name="uq_content_analyses_answer_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("interview_answers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )

    coverage_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_source: Mapped[str] = mapped_column(
        String(40), nullable=False, default="none"
    )
    matched_concepts: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    missing_concepts: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    filler_word_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    filler_rate_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    filler_words: Mapped[list[dict]] = mapped_column(
        JSON, nullable=False, default=list
    )

    repeated_terms: Mapped[list[dict]] = mapped_column(
        JSON, nullable=False, default=list
    )
    repetition_rate_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    sentence_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    average_sentence_words: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    structure_notes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="completed"
    )
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase3-v1"
    )

    transcript_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    answer: Mapped[InterviewAnswer] = relationship(
        back_populates="content_analyses"
    )
    session: Mapped[InterviewSession] = relationship(
        back_populates="content_analyses"
    )
    question: Mapped[InterviewQuestion] = relationship(
        back_populates="content_analyses"
    )

    def __repr__(self) -> str:
        return (
            "ContentAnalysis("
            f"id={self.id!r}, answer_id={self.answer_id!r}, "
            f"status={self.status!r}, coverage_score={self.coverage_score!r})"
        )


class VoiceAnalysis(Base):
    """Measured acoustic features for one saved recording.

    Only measurements are stored. Acoustic features are not interpreted as
    personality, confidence, honesty, intelligence, or emotion.
    """

    __tablename__ = "voice_analyses"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "methodology_version",
            name="uq_voice_analyses_answer_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("interview_answers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="completed"
    )
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)

    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    analyzed_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    truncated_for_analysis: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    speech_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    speech_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    silence_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    speech_ratio_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    pause_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_pause_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    mean_pause_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    longest_pause_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    pauses_per_minute: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    speaking_rate_wpm: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    rms_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    rms_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    rms_variation_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    rms_mean_dbfs: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_amplitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    clipping_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    pitch_median_hz: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    pitch_mean_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_std_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_p10_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_p90_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_range_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_voiced_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    pitch_reliable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    quality_notes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    transcript_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase3-v1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    answer: Mapped[InterviewAnswer] = relationship(
        back_populates="voice_analyses"
    )
    session: Mapped[InterviewSession] = relationship(
        back_populates="voice_analyses"
    )
    question: Mapped[InterviewQuestion] = relationship(
        back_populates="voice_analyses"
    )

    def __repr__(self) -> str:
        return (
            "VoiceAnalysis("
            f"id={self.id!r}, answer_id={self.answer_id!r}, "
            f"status={self.status!r}, "
            f"speaking_rate_wpm={self.speaking_rate_wpm!r})"
        )


class ScorePrediction(Base):
    """A versioned interview-score prediction for one saved answer.

    Two methods can produce a score:

    - ``ml_model``: a compatible, validated Random Forest artifact produced by
      the offline training pipeline. Only available when the operator trains and
      installs a model.
    - ``rule_based_baseline``: a transparent fixed-weight baseline built from
      the current Phase 2 evaluation plus available Phase 3 features. This is
      never labelled as an ML prediction.

    ``source_fingerprint`` fingerprints the exact source evaluations/analyses
    used. When any of them change, the saved prediction is reported as stale and
    can be refreshed in place.
    """

    __tablename__ = "score_predictions"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "methodology_version",
            name="uq_score_predictions_answer_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("interview_answers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )

    # completed -> a score exists; unavailable -> sources are not eligible;
    # failed -> feature/source data failed validation.
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="unavailable"
    )
    # ml_model | rule_based_baseline | none
    method: Mapped[str] = mapped_column(
        String(30), nullable=False, default="none"
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)

    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    feature_schema_version: Mapped[str] = mapped_column(
        String(40), nullable=False
    )
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase4-v1"
    )

    features: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    missing_features: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    components: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    warnings: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    limitations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_versions: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    answer: Mapped[InterviewAnswer] = relationship(back_populates="predictions")
    session: Mapped[InterviewSession] = relationship(back_populates="predictions")
    question: Mapped[InterviewQuestion] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            "ScorePrediction("
            f"id={self.id!r}, answer_id={self.answer_id!r}, "
            f"status={self.status!r}, method={self.method!r}, "
            f"score={self.score!r})"
        )


class AnswerFeedback(Base):
    """Evidence-based feedback for one saved, currently evaluated answer.

    Feedback is generated deterministically from the persisted Phase 2
    evaluation plus any current Phase 3 content/voice analyses and Phase 4
    prediction. Missing sources are recorded as excluded and are never treated
    as a zero score or as a negative measurement. The source fingerprint makes
    a saved row explicitly stale when any source changes.
    """

    __tablename__ = "answer_feedback"
    __table_args__ = (
        UniqueConstraint(
            "answer_id",
            "methodology_version",
            name="uq_answer_feedback_answer_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("interview_answers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("interview_questions.id"),
        nullable=False,
        index=True,
    )

    # completed -> evidence was sufficient; unavailable -> a required source is
    # missing/stale/unevaluable; failed -> a source value failed validation.
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="unavailable"
    )
    # strong | average | weak (only when a completed combined score exists).
    band: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    strengths: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    improvements: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    actionable_suggestions: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    matched_concepts: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    missing_concepts: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    voice_observations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    notices: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    limitations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # Structured evidence is stored for auditability. It is not used to render
    # hidden claims; every user-facing sentence is built from these values.
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    prediction_method: Mapped[str] = mapped_column(
        String(30), nullable=False, default="none"
    )
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_is_ml: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    prediction_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    voice_available: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    content_available: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    semantic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_coverage_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_versions: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase5-v1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    answer: Mapped[InterviewAnswer] = relationship(back_populates="feedback")
    session: Mapped[InterviewSession] = relationship(
        back_populates="answer_feedback"
    )
    question: Mapped[InterviewQuestion] = relationship(
        back_populates="answer_feedback"
    )

    def __repr__(self) -> str:
        return (
            "AnswerFeedback("
            f"id={self.id!r}, answer_id={self.answer_id!r}, "
            f"status={self.status!r}, band={self.band!r})"
        )


class InterviewFeedback(Base):
    """Aggregated, evidence-based feedback for one interview session."""

    __tablename__ = "interview_feedback"
    __table_args__ = (
        UniqueConstraint(
            "interview_id",
            "methodology_version",
            name="uq_interview_feedback_interview_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # completed -> every answered question has completed feedback and all
    # questions are answered; partial -> some submitted answers are unavailable
    # or some questions are unanswered; unavailable -> no completed feedback.
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="unavailable"
    )
    status_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    overall_strengths: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    overall_improvements: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    recommendations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    notices: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    limitations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # A self-contained snapshot of the question-wise breakdown at generation
    # time. Source fingerprints make the whole row stale when any answer,
    # reference, evaluation, analysis, or prediction changes.
    question_breakdown: Mapped[list[dict]] = mapped_column(
        JSON, nullable=False, default=list
    )

    questions_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    questions_answered: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    questions_unanswered: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    feedback_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    feedback_unavailable: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    feedback_failed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    feedback_stale: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    voice_available_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    ml_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_predictions: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    average_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_predicted_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_versions: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )
    methodology_version: Mapped[str] = mapped_column(
        String(30), nullable=False, default="phase5-v1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )

    session: Mapped[InterviewSession] = relationship(
        back_populates="interview_feedback"
    )

    def __repr__(self) -> str:
        return (
            "InterviewFeedback("
            f"id={self.id!r}, interview_id={self.interview_id!r}, "
            f"status={self.status!r}, average_score={self.average_score!r})"
        )
