"""Pydantic response schemas used by the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QuestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question_text: str
    category: str
    has_reference_answer: bool = False


class InterviewSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    started_at: datetime


class AnswerOut(BaseModel):
    id: int
    interview_id: int
    question_id: int
    question_text: str
    category: str
    transcript: str
    has_audio: bool = False
    created_at: datetime
    updated_at: datetime


class AnswersResponse(BaseModel):
    interview_id: int
    status: str
    started_at: datetime
    answers: list[AnswerOut]


class TranscriptionResponse(BaseModel):
    answer: AnswerOut
    updated: bool


class HealthResponse(BaseModel):
    status: str
    database: str
    transcription_configured: bool
    transcription_model: str
    transcription_provider: str = "groq"
    transcription_hint: str | None = None
    max_audio_mb: int = 25
    database_driver: str | None = None
    database_host: str | None = None
    database_port: int | None = None
    database_name: str | None = None
    database_user_configured: bool = False
    database_password_configured: bool = False
    database_error_category: str | None = None
    database_hint: str | None = None
    env_file: str = "backend/.env"
    env_file_found: bool = False
    database_url_source: str = "default"
    groq_api_key_source: str = "default"
    analysis_methodology_version: str = ""
    voice_analysis_available: bool = False
    prediction_methodology_version: str = ""
    feature_schema_version: str = ""
    ml_model_available: bool = False
    ml_model_state: str = "unavailable"
    ml_model_version: str | None = None
    ml_model_validation_status: str | None = None
    ml_model_reason: str | None = None
    feedback_methodology_version: str = ""


class EvaluationOut(BaseModel):
    id: int
    answer_id: int
    interview_id: int
    question_id: int
    semantic_score: float
    keyword_score: float
    combined_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    status: str
    status_detail: str | None = None
    methodology_version: str
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class EvaluationResponse(BaseModel):
    evaluation: EvaluationOut
    reused: bool


class EvaluationListResponse(BaseModel):
    interview_id: int
    evaluations: list[EvaluationOut]


class QuestionSummaryOut(BaseModel):
    question_id: int
    question_text: str
    category: str
    has_reference_answer: bool
    answer_id: int | None = None
    evaluation_id: int | None = None
    status: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    combined_score: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    status_detail: str | None = None
    prediction_id: int | None = None
    prediction_status: str | None = None
    predicted_score: float | None = None
    prediction_method: str | None = None
    prediction_is_stale: bool = False


class InterviewSummaryOut(BaseModel):
    interview_id: int
    status: str
    started_at: datetime
    total_questions: int
    answers_submitted: int
    answers_evaluated: int
    answers_unevaluable: int
    questions_unanswered: int
    average_score: float | None = None
    questions: list[QuestionSummaryOut]
    prediction_methodology_version: str | None = None
    predictions_completed: int = 0
    predictions_ml: int = 0
    predictions_baseline: int = 0
    predictions_stale: int = 0
    predictions_unavailable: int = 0
    average_predicted_score: float | None = None
    prediction_method: str | None = None
    ml_model_available: bool = False


class FillerWordOut(BaseModel):
    word: str
    count: int


class RepeatedTermOut(BaseModel):
    term: str
    count: int


class ContentAnalysisOut(BaseModel):
    id: int
    answer_id: int
    interview_id: int
    question_id: int
    coverage_score: float | None = None
    coverage_source: str
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    filler_word_count: int = 0
    filler_rate_percent: float | None = None
    filler_words: list[FillerWordOut] = Field(default_factory=list)
    repeated_terms: list[RepeatedTermOut] = Field(default_factory=list)
    repetition_rate_percent: float | None = None
    sentence_count: int = 0
    average_sentence_words: float | None = None
    structure_notes: list[str] = Field(default_factory=list)
    status: str
    status_detail: str | None = None
    methodology_version: str
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class ContentAnalysisResponse(BaseModel):
    content_analysis: ContentAnalysisOut
    reused: bool


class VoiceAnalysisOut(BaseModel):
    id: int
    answer_id: int
    interview_id: int
    question_id: int
    status: str
    status_detail: str | None = None
    duration_seconds: float | None = None
    analyzed_duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    truncated_for_analysis: bool = False
    speech_detected: bool = False
    speech_duration_seconds: float | None = None
    silence_duration_seconds: float | None = None
    speech_ratio_percent: float | None = None
    pause_count: int | None = None
    total_pause_seconds: float | None = None
    mean_pause_seconds: float | None = None
    longest_pause_seconds: float | None = None
    pauses_per_minute: float | None = None
    word_count: int | None = None
    speaking_rate_wpm: float | None = None
    rms_mean: float | None = None
    rms_std: float | None = None
    rms_variation_percent: float | None = None
    rms_mean_dbfs: float | None = None
    peak_amplitude: float | None = None
    clipping_percent: float | None = None
    pitch_median_hz: float | None = None
    pitch_mean_hz: float | None = None
    pitch_std_hz: float | None = None
    pitch_p10_hz: float | None = None
    pitch_p90_hz: float | None = None
    pitch_range_hz: float | None = None
    pitch_voiced_seconds: float | None = None
    pitch_reliable: bool = False
    quality_notes: list[str] = Field(default_factory=list)
    methodology_version: str
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class VoiceAnalysisResponse(BaseModel):
    voice_analysis: VoiceAnalysisOut
    reused: bool


class InterviewAnalysesResponse(BaseModel):
    interview_id: int
    content_analyses: list[ContentAnalysisOut] = Field(default_factory=list)
    voice_analyses: list[VoiceAnalysisOut] = Field(default_factory=list)


class ModelStatusOut(BaseModel):
    available: bool
    state: str
    reason: str
    model_version: str | None = None
    feature_schema_version: str | None = None
    validation_status: str | None = None
    trained_at: str | None = None
    sklearn_version: str | None = None
    metrics: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    artifact_name: str | None = None


class ScorePredictionOut(BaseModel):
    id: int
    answer_id: int
    interview_id: int
    question_id: int
    status: str
    method: str
    method_label: str
    is_ml_prediction: bool
    score: float | None = None
    status_detail: str | None = None
    model_version: str | None = None
    feature_schema_version: str
    methodology_version: str
    features: dict[str, float | None] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    components: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_versions: dict[str, object] = Field(default_factory=dict)
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class PredictionResponse(BaseModel):
    prediction: ScorePredictionOut
    model_status: ModelStatusOut
    reused: bool


class PredictionListResponse(BaseModel):
    interview_id: int
    predictions: list[ScorePredictionOut] = Field(default_factory=list)
    model_status: ModelStatusOut


class FeatureSchemaResponse(BaseModel):
    feature_schema: dict[str, object]
    model_status: ModelStatusOut
    prediction_methodology_version: str
    baseline_methodology: dict[str, object]


class AnswerFeedbackOut(BaseModel):
    id: int
    answer_id: int
    interview_id: int
    question_id: int
    status: str
    band: str | None = None
    status_detail: str | None = None
    summary: str | None = None
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    actionable_suggestions: list[str] = Field(default_factory=list)
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    voice_observations: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: dict[str, object] = Field(default_factory=dict)
    prediction_method: str = "none"
    prediction_score: float | None = None
    prediction_is_ml: bool = False
    prediction_note: str | None = None
    voice_available: bool = False
    content_available: bool = False
    semantic_score: float | None = None
    keyword_score: float | None = None
    combined_score: float | None = None
    content_coverage_score: float | None = None
    methodology_version: str
    source_versions: dict[str, object] = Field(default_factory=dict)
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class AnswerFeedbackResponse(BaseModel):
    feedback: AnswerFeedbackOut
    reused: bool


class FeedbackQuestionBreakdownOut(BaseModel):
    question_id: int
    question_text: str
    category: str
    question_number: int
    answer_id: int | None = None
    feedback_id: int | None = None
    status: str
    band: str | None = None
    combined_score: float | None = None
    summary: str | None = None
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    voice_available: bool = False
    prediction_method: str | None = None
    prediction_is_ml: bool = False
    status_detail: str | None = None


class InterviewFeedbackOut(BaseModel):
    id: int
    interview_id: int
    status: str
    status_detail: str | None = None
    summary: str | None = None
    overall_strengths: list[str] = Field(default_factory=list)
    overall_improvements: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    question_breakdown: list[FeedbackQuestionBreakdownOut] = Field(
        default_factory=list
    )
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
    methodology_version: str
    source_versions: dict[str, object] = Field(default_factory=dict)
    is_stale: bool = False
    created_at: datetime
    updated_at: datetime


class InterviewFeedbackResponse(BaseModel):
    interview_id: int
    interview_feedback: InterviewFeedbackOut | None = None
    answer_feedback: list[AnswerFeedbackOut] = Field(default_factory=list)
    reused: bool = False

