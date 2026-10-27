"""Phase 5 tests: deterministic, evidence-based answer and interview feedback.

These tests use the project's isolated SQLite database. No MySQL server, Groq
key, microphone, trained model, or labelled dataset is required. Persisted
Phase 2/3/4 rows are created directly where a precise score band is needed, and
at least one path runs through the real Phase 2 and Phase 4 APIs.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

import config
import routes.feedback as feedback_routes
import services.evaluation as evaluation_service
import services.ml_scoring as ml_scoring
from config import (
    ANALYSIS_METHODOLOGY_VERSION,
    EVALUATION_METHODOLOGY_VERSION,
    FEEDBACK_METHODOLOGY_VERSION,
    ML_FEATURE_SCHEMA_VERSION,
    ML_PREDICTION_METHODOLOGY_VERSION,
)
from database import SessionLocal
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
from services.evaluation import fingerprint
from services.ml_features import build_source_snapshot
from services.ml_scoring import METHOD_BASELINE, METHOD_ML

REFERENCE = (
    "Inheritance is an object-oriented mechanism where a child class reuses and "
    "extends the properties and methods of a parent class. It models an is-a "
    "relationship and enables code reuse."
)
STRONG_TRANSCRIPT = (
    "Inheritance is an object-oriented mechanism where a child class reuses and "
    "extends the properties and methods of a parent class. It models an is-a "
    "relationship and enables code reuse. This avoids duplicate code."
)
AVERAGE_TRANSCRIPT = (
    "Inheritance means a child class can reuse some methods from a parent class, "
    "so common code does not have to be written twice."
)
WEAK_TRANSCRIPT = (
    "I think it is when one thing gets stuff from another thing and that is "
    "useful when writing programs."
)


@pytest.fixture(autouse=True)
def isolated_model_artifact(tmp_path, monkeypatch):
    """Keep the real Phase 4 artifact out of every feedback test."""
    monkeypatch.setattr(
        config,
        "ML_MODEL_PATH",
        tmp_path / "no-installed-model.joblib",
    )
    ml_scoring.clear_model_cache()
    yield
    ml_scoring.clear_model_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_interview(client) -> int:
    response = client.post("/interview/start")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_question(reference: str | None = REFERENCE) -> int:
    with SessionLocal() as db:
        question = InterviewQuestion(
            question_text=f"Phase 5 test question {uuid.uuid4()}",
            category="Phase 5 Test",
            reference_answer=reference,
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        return question.id


def _create_answer(
    interview_id: int,
    question_id: int,
    transcript: str = STRONG_TRANSCRIPT,
) -> int:
    with SessionLocal() as db:
        answer = InterviewAnswer(
            interview_id=interview_id,
            question_id=question_id,
            transcript=transcript,
        )
        db.add(answer)
        db.commit()
        db.refresh(answer)
        return answer.id


def _new_answer(
    client,
    transcript: str = STRONG_TRANSCRIPT,
    reference: str | None = REFERENCE,
) -> tuple[int, int, int]:
    interview_id = _start_interview(client)
    question_id = _create_question(reference)
    answer_id = _create_answer(interview_id, question_id, transcript)
    return interview_id, question_id, answer_id


def _add_evaluation(
    answer_id: int,
    *,
    semantic: float = 62.0,
    keyword: float = 58.0,
    combined: float = 60.0,
    status: str = "evaluated",
    matched: list[str] | None = None,
    missing: list[str] | None = None,
    status_detail: str | None = None,
) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        question = db.get(InterviewQuestion, answer.question_id)
        assert question is not None
        db.add(
            AnswerEvaluation(
                answer_id=answer.id,
                interview_id=answer.interview_id,
                question_id=answer.question_id,
                semantic_score=semantic,
                keyword_score=keyword,
                combined_score=combined,
                matched_keywords=list(matched or []),
                missing_keywords=list(missing or []),
                status=status,
                status_detail=status_detail,
                methodology_version=EVALUATION_METHODOLOGY_VERSION,
                transcript_hash=fingerprint(answer.transcript),
                reference_hash=fingerprint(question.reference_answer or ""),
            )
        )
        db.commit()


def _add_content(
    answer_id: int,
    *,
    status: str = "completed",
    coverage: float | None = 58.0,
    matched: list[str] | None = None,
    missing: list[str] | None = None,
    filler_rate: float | None = 1.5,
    filler_count: int = 1,
    repetition_rate: float | None = 6.0,
    sentence_count: int = 3,
    average_words: float | None = 14.0,
    stale: bool = False,
) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        question = db.get(InterviewQuestion, answer.question_id)
        assert question is not None
        db.add(
            ContentAnalysis(
                answer_id=answer.id,
                interview_id=answer.interview_id,
                question_id=answer.question_id,
                coverage_score=coverage,
                coverage_source="phase2" if coverage is not None else "none",
                matched_concepts=list(matched or []),
                missing_concepts=list(missing or []),
                filler_word_count=filler_count,
                filler_rate_percent=filler_rate,
                repetition_rate_percent=repetition_rate,
                sentence_count=sentence_count,
                average_sentence_words=average_words,
                status=status,
                status_detail=None if status == "completed" else "Synthetic.",
                methodology_version=ANALYSIS_METHODOLOGY_VERSION,
                transcript_hash=fingerprint(
                    "stale transcript" if stale else answer.transcript
                ),
                reference_hash=fingerprint(question.reference_answer or ""),
            )
        )
        db.commit()


def _add_voice(
    answer_id: int,
    *,
    status: str = "completed",
    speech_detected: bool = True,
    speaking_rate: float | None = 142.0,
    pauses: float | None = 3.0,
    speech_ratio: float | None = 78.0,
    rms_variation: float | None = 32.0,
    stale: bool = False,
) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        snapshot = build_source_snapshot(
            db,
            answer,
            db.get(InterviewQuestion, answer.question_id),
        )
        db.add(
            VoiceAnalysis(
                answer_id=answer.id,
                interview_id=answer.interview_id,
                question_id=answer.question_id,
                status=status,
                status_detail=None if status == "completed" else "Synthetic.",
                speech_detected=speech_detected,
                speaking_rate_wpm=speaking_rate,
                pauses_per_minute=pauses,
                speech_ratio_percent=speech_ratio,
                rms_variation_percent=rms_variation,
                methodology_version=ANALYSIS_METHODOLOGY_VERSION,
                audio_sha256=(
                    "f" * 64 if stale else snapshot.audio_fingerprint
                ),
                transcript_hash=(
                    fingerprint("stale transcript")
                    if stale
                    else snapshot.transcript_hash
                ),
            )
        )
        db.commit()


def _add_prediction(
    answer_id: int,
    *,
    method: str = METHOD_BASELINE,
    score: float | None = 64.0,
    status: str = "completed",
    stale: bool = False,
) -> int:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        question = db.get(InterviewQuestion, answer.question_id)
        assert question is not None
        snapshot = build_source_snapshot(db, answer, question)
        row = ScorePrediction(
            answer_id=answer.id,
            interview_id=answer.interview_id,
            question_id=answer.question_id,
            status=status,
            method=method,
            score=score,
            status_detail=None if status == "completed" else "Synthetic.",
            model_version="test-rf-1" if method == METHOD_ML else None,
            feature_schema_version=ML_FEATURE_SCHEMA_VERSION,
            methodology_version=ML_PREDICTION_METHODOLOGY_VERSION,
            source_fingerprint=("0" * 64 if stale else snapshot.fingerprint),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id


def _post_answer_feedback(client, answer_id: int):
    return client.post(f"/feedback/{answer_id}")


def _feedback_count(answer_id: int) -> int:
    with SessionLocal() as db:
        return int(
            db.scalar(
                select(func.count())
                .select_from(AnswerFeedback)
                .where(AnswerFeedback.answer_id == answer_id)
            )
            or 0
        )


def _interview_feedback_count(interview_id: int) -> int:
    with SessionLocal() as db:
        return int(
            db.scalar(
                select(func.count())
                .select_from(InterviewFeedback)
                .where(InterviewFeedback.interview_id == interview_id)
            )
            or 0
        )


def _evaluate_via_api(client, monkeypatch, answer_id: int, score: float = 92.0):
    monkeypatch.setattr(
        evaluation_service,
        "semantic_similarity",
        lambda _candidate, _reference: score,
    )
    response = client.post(f"/evaluation/{answer_id}")
    assert response.status_code in {200, 201}, response.text
    return response.json()["evaluation"]


# ---------------------------------------------------------------------------
# Bands and evidence
# ---------------------------------------------------------------------------


def test_strong_feedback_from_real_evaluation_records(client, monkeypatch):
    interview_id, question_id, answer_id = _new_answer(client)
    evaluation = _evaluate_via_api(client, monkeypatch, answer_id, score=92.0)
    assert evaluation["status"] == "evaluated"
    assert evaluation["combined_score"] >= config.FEEDBACK_STRONG_SCORE

    content = client.post(f"/analysis/{answer_id}/content")
    assert content.status_code in {200, 201}, content.text

    voice = client.post(f"/analysis/{answer_id}/voice")
    assert voice.status_code in {200, 201}, voice.text

    prediction = client.post(f"/prediction/{answer_id}")
    assert prediction.status_code in {200, 201}, prediction.text
    assert prediction.json()["prediction"]["method"] == METHOD_BASELINE

    response = _post_answer_feedback(client, answer_id)
    assert response.status_code == 201, response.text
    body = response.json()
    feedback = body["feedback"]

    assert feedback["status"] == "completed"
    assert feedback["band"] == "strong"
    assert feedback["combined_score"] >= config.FEEDBACK_STRONG_SCORE
    assert feedback["semantic_score"] == 92.0
    assert feedback["content_available"] is True
    assert feedback["matched_concepts"]
    assert feedback["summary"].startswith("Strong content match")
    assert str(question_id) in str(feedback["evidence"]["question"]["id"])
    assert feedback["source_versions"]["transcript_hash"]
    assert feedback["methodology_version"] == FEEDBACK_METHODOLOGY_VERSION
    assert feedback["is_stale"] is False
    assert body["reused"] is False

    # The baseline must never be presented as an ML prediction.
    assert feedback["prediction_method"] == METHOD_BASELINE
    assert feedback["prediction_is_ml"] is False
    assert any(
        "not an ML prediction" in item for item in feedback["notices"]
    )

    # Phase 2/4 responses remain available and unchanged in shape.
    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200
    summary_body = summary.json()
    summary_question = next(
        item
        for item in summary_body["questions"]
        if item["question_id"] == question_id
    )
    assert summary_question["combined_score"] == feedback["combined_score"]
    assert summary_body["prediction_method"] == METHOD_BASELINE


@pytest.mark.parametrize(
    ("semantic", "keyword", "combined", "band"),
    [
        (88.0, 86.0, 87.0, "strong"),
        (60.0, 55.0, 58.0, "average"),
        (32.0, 25.0, 28.0, "weak"),
    ],
)
def test_bands_are_derived_from_persisted_scores(
    client,
    semantic,
    keyword,
    combined,
    band,
):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(
        answer_id,
        semantic=semantic,
        keyword=keyword,
        combined=combined,
        matched=["inheritance"],
        missing=["polymorphism"],
    )

    response = _post_answer_feedback(client, answer_id)
    assert response.status_code == 201, response.text
    feedback = response.json()["feedback"]
    assert feedback["status"] == "completed"
    assert feedback["band"] == band
    assert feedback["semantic_score"] == semantic
    assert feedback["keyword_score"] == keyword
    assert feedback["combined_score"] == combined

    if band == "strong":
        assert any("Strong content match" in item for item in feedback["strengths"])
    elif band == "average":
        assert "Mixed content match" in feedback["summary"]
        assert feedback["improvements"]
        assert feedback["actionable_suggestions"]
    else:
        assert "needs revision" in feedback["summary"]
        assert any(
            "weak threshold" in item for item in feedback["improvements"]
        )
        assert any(
            "missing concepts" in item.lower()
            or "reference answer" in item.lower()
            for item in feedback["actionable_suggestions"]
        )


def test_missing_reference_is_explicitly_unavailable(client):
    _interview_id, _question_id, answer_id = _new_answer(
        client,
        transcript=AVERAGE_TRANSCRIPT,
        reference=None,
    )
    response = _post_answer_feedback(client, answer_id)
    assert response.status_code == 201, response.text
    feedback = response.json()["feedback"]
    assert feedback["status"] == "unavailable"
    assert feedback["band"] is None
    assert feedback["combined_score"] is None
    assert "no reference answer" in feedback["status_detail"].lower()
    assert "cannot be generated" in feedback["summary"].lower()


def test_unevaluated_and_unevaluable_answers_are_not_scored(client):
    _interview_id, _question_id, unevaluated_id = _new_answer(client)
    unevaluated = _post_answer_feedback(client, unevaluated_id).json()["feedback"]
    assert unevaluated["status"] == "unavailable"
    assert "no phase 2 evaluation" in unevaluated["status_detail"].lower()
    assert unevaluated["combined_score"] is None

    _interview_id, _question_id, unevaluable_id = _new_answer(
        client,
        transcript=WEAK_TRANSCRIPT,
    )
    _add_evaluation(
        unevaluable_id,
        status="unevaluable",
        status_detail="The answer is too short to evaluate.",
        matched=[],
        missing=[],
    )
    feedback = _post_answer_feedback(client, unevaluable_id).json()["feedback"]
    assert feedback["status"] == "unavailable"
    assert "too short" in feedback["status_detail"].lower()
    assert feedback["combined_score"] is None


def test_stale_evaluation_is_rejected_and_saved_feedback_becomes_stale(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=80.0, semantic=82.0, keyword=76.0)

    first = _post_answer_feedback(client, answer_id)
    assert first.status_code == 201, first.text
    assert first.json()["feedback"]["is_stale"] is False
    feedback_id = first.json()["feedback"]["id"]

    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        answer.transcript = f"{answer.transcript} Updated after feedback."
        db.commit()

    saved = client.get(f"/feedback/{answer_id}")
    assert saved.status_code == 200, saved.text
    assert saved.json()["id"] == feedback_id
    assert saved.json()["is_stale"] is True

    regenerated = _post_answer_feedback(client, answer_id)
    assert regenerated.status_code == 200, regenerated.text
    stale_result = regenerated.json()["feedback"]
    assert stale_result["status"] == "unavailable"
    assert "stale" in stale_result["status_detail"].lower()
    assert stale_result["combined_score"] is None
    assert stale_result["is_stale"] is False
    assert _feedback_count(answer_id) == 1

    interview = client.post(f"/interview/{first.json()['feedback']['interview_id']}/feedback")
    assert interview.status_code == 201, interview.text
    breakdown = [
        item
        for item in interview.json()["interview_feedback"]["question_breakdown"]
        if item["answer_id"] == answer_id
    ]
    assert breakdown
    assert breakdown[0]["status"] == "unavailable"
    assert breakdown[0]["combined_score"] is None


def test_missing_voice_is_excluded_and_never_zero(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=66.0, semantic=68.0, keyword=61.0)

    feedback = _post_answer_feedback(client, answer_id).json()["feedback"]
    assert feedback["status"] == "completed"
    assert feedback["voice_available"] is False
    assert feedback["voice_observations"] == []
    assert any(
        "voice analysis is missing" in item.lower() for item in feedback["notices"]
    )
    assert any(
        "not treated as zero" in item.lower() for item in feedback["notices"]
    )
    assert not any("confidence" in item.lower() for item in feedback["notices"])


def test_unavailable_voice_row_is_excluded_without_zero_fill(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=66.0, semantic=68.0, keyword=61.0)
    _add_voice(answer_id, status="insufficient_speech", speech_detected=False)

    feedback = _post_answer_feedback(client, answer_id).json()["feedback"]
    assert feedback["voice_available"] is False
    assert feedback["voice_observations"] == []
    assert any(
        "insufficient_speech" in item for item in feedback["notices"]
    )


def test_current_voice_measurements_are_quoted_objectively(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=70.0, semantic=72.0, keyword=65.0)
    _add_voice(
        answer_id,
        speaking_rate=142.0,
        pauses=3.0,
        speech_ratio=78.0,
        rms_variation=32.0,
    )

    feedback = _post_answer_feedback(client, answer_id).json()["feedback"]
    assert feedback["status"] == "completed"
    assert feedback["voice_available"] is True
    joined = " ".join(feedback["voice_observations"])
    assert "142.00 words per minute" in joined
    assert "3.00 per minute" in joined
    assert "78.00%" in joined
    assert "confidence" not in joined.lower()
    assert "emotion" not in joined.lower()
    assert "personality" not in joined.lower()


def test_ml_prediction_is_labelled_separately_from_baseline(client):
    _interview_id, _question_id, ml_answer = _new_answer(client)
    _add_evaluation(ml_answer, combined=72.0, semantic=74.0, keyword=67.0)
    _add_prediction(ml_answer, method=METHOD_ML, score=74.5)

    ml_feedback = _post_answer_feedback(client, ml_answer).json()["feedback"]
    assert ml_feedback["prediction_method"] == METHOD_ML
    assert ml_feedback["prediction_is_ml"] is True
    assert ml_feedback["prediction_score"] == 74.5
    assert "ML prediction" in ml_feedback["prediction_note"]
    assert "not a rule-based baseline" in ml_feedback["prediction_note"]

    _interview_id, _question_id, baseline_answer = _new_answer(client)
    _add_evaluation(baseline_answer, combined=65.0, semantic=67.0, keyword=60.0)
    _add_prediction(baseline_answer, method=METHOD_BASELINE, score=63.2)

    baseline_feedback = _post_answer_feedback(client, baseline_answer).json()[
        "feedback"
    ]
    assert baseline_feedback["prediction_method"] == METHOD_BASELINE
    assert baseline_feedback["prediction_is_ml"] is False
    assert "rule-based baseline" in baseline_feedback["prediction_note"]
    assert "not an ML prediction" in baseline_feedback["prediction_note"]


def test_stale_prediction_is_excluded_with_notice(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=68.0, semantic=70.0, keyword=63.0)
    _add_prediction(answer_id, method=METHOD_BASELINE, score=66.0, stale=True)

    feedback = _post_answer_feedback(client, answer_id).json()["feedback"]
    assert feedback["status"] == "completed"
    assert feedback["prediction_method"] == "none"
    assert feedback["prediction_score"] is None
    assert feedback["prediction_is_ml"] is False
    assert any("stale" in item.lower() for item in feedback["notices"])


def test_invalid_source_values_fail_without_inventing_feedback(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(
        answer_id,
        semantic=120.0,
        keyword=-4.0,
        combined=150.0,
    )

    response = _post_answer_feedback(client, answer_id)
    assert response.status_code == 201, response.text
    feedback = response.json()["feedback"]
    assert feedback["status"] == "failed"
    assert feedback["band"] is None
    assert feedback["combined_score"] is None
    assert "0-100" in feedback["status_detail"]
    assert any("not generated" in item.lower() for item in feedback["notices"])


# ---------------------------------------------------------------------------
# Persistence, duplicates, and concurrency
# ---------------------------------------------------------------------------


def test_duplicate_requests_upsert_one_row(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=71.0, semantic=73.0, keyword=66.0)

    first = _post_answer_feedback(client, answer_id)
    second = _post_answer_feedback(client, answer_id)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["feedback"]["id"] == first.json()["feedback"]["id"]
    assert _feedback_count(answer_id) == 1

    saved = client.get(f"/feedback/{answer_id}")
    assert saved.status_code == 200
    assert saved.json()["id"] == first.json()["feedback"]["id"]


def test_concurrent_insert_conflict_is_re_read(
    client,
    monkeypatch,
):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=71.0, semantic=73.0, keyword=66.0)

    original_apply = feedback_routes.apply_answer_feedback

    def racing_apply(row, result):
        # Simulate another request committing the same unique row first.
        with SessionLocal() as other:
            other.add(
                AnswerFeedback(
                    answer_id=answer_id,
                    interview_id=row.interview_id,
                    question_id=row.question_id,
                    status="completed",
                    source_fingerprint=result.source_fingerprint,
                    methodology_version=FEEDBACK_METHODOLOGY_VERSION,
                )
            )
            other.commit()
        original_apply(row, result)

    monkeypatch.setattr(feedback_routes, "apply_answer_feedback", racing_apply)

    response = client.post(f"/feedback/{answer_id}")
    assert response.status_code == 200, response.text
    assert response.json()["reused"] is True
    assert _feedback_count(answer_id) == 1


# ---------------------------------------------------------------------------
# Interview-level feedback
# ---------------------------------------------------------------------------


def test_interview_feedback_marks_unanswered_and_partial(client):
    interview_id = _start_interview(client)
    answered_question = _create_question()
    unanswered_question = _create_question()
    answer_id = _create_answer(interview_id, answered_question)
    _add_evaluation(answer_id, combined=64.0, semantic=66.0, keyword=59.0)

    response = client.post(f"/interview/{interview_id}/feedback")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["reused"] is False
    interview_feedback = body["interview_feedback"]
    assert interview_feedback["status"] == "partial"
    assert interview_feedback["questions_answered"] == 1
    assert interview_feedback["feedback_completed"] == 1
    assert interview_feedback["questions_unanswered"] >= 1
    assert interview_feedback["average_score"] == 64.0
    assert any(
        "unanswered" in item.lower()
        for item in interview_feedback["recommendations"]
    )

    by_question = {
        item["question_id"]: item
        for item in interview_feedback["question_breakdown"]
    }
    assert by_question[answered_question]["status"] == "completed"
    assert by_question[unanswered_question]["status"] == "unanswered"
    assert by_question[unanswered_question]["combined_score"] is None
    assert by_question[unanswered_question]["answer_id"] is None
    assert any(
        "not scored as zero" in item["status_detail"].lower()
        for item in interview_feedback["question_breakdown"]
        if item["status"] == "unanswered"
    )

    listed = client.get(f"/interview/{interview_id}/feedback")
    assert listed.status_code == 200
    assert listed.json()["interview_feedback"]["id"] == interview_feedback["id"]
    assert len(listed.json()["answer_feedback"]) == 1
    assert _interview_feedback_count(interview_id) == 1


def test_interview_feedback_is_refreshed_when_answer_changes(client):
    interview_id = _start_interview(client)
    question_id = _create_question()
    answer_id = _create_answer(interview_id, question_id)
    _add_evaluation(answer_id, combined=62.0, semantic=64.0, keyword=57.0)

    first = client.post(f"/interview/{interview_id}/feedback")
    assert first.status_code == 201, first.text
    first_body = first.json()
    first_summary = first_body["interview_feedback"]["summary"]
    first_answer_summary = first_body["answer_feedback"][0]["summary"]

    _add_content(
        answer_id,
        coverage=61.0,
        matched=["inheritance", "class"],
        missing=["polymorphism"],
    )

    second = client.post(f"/interview/{interview_id}/feedback")
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["interview_feedback"]["id"] == first_body["interview_feedback"]["id"]
    assert body["interview_feedback"]["is_stale"] is False
    assert body["answer_feedback"][0]["content_available"] is True
    assert body["answer_feedback"][0]["summary"] != first_answer_summary
    assert body["interview_feedback"]["source_versions"] != first_body[
        "interview_feedback"
    ]["source_versions"]
    assert body["interview_feedback"]["summary"] == first_summary
    assert _interview_feedback_count(interview_id) == 1


def test_interview_feedback_empty_state_is_200_with_null(client):
    interview_id = _start_interview(client)
    response = client.get(f"/interview/{interview_id}/feedback")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interview_id"] == interview_id
    assert body["interview_feedback"] is None
    assert body["answer_feedback"] == []


def test_interview_with_no_answers_is_unavailable_not_zero(client):
    interview_id = _start_interview(client)
    response = client.post(f"/interview/{interview_id}/feedback")
    assert response.status_code == 201, response.text
    feedback = response.json()["interview_feedback"]
    assert feedback["status"] == "unavailable"
    assert feedback["questions_answered"] == 0
    assert feedback["feedback_completed"] == 0
    assert feedback["average_score"] is None
    assert "no completed answer feedback" in feedback["summary"].lower()


# ---------------------------------------------------------------------------
# API errors and backward compatibility
# ---------------------------------------------------------------------------


def test_missing_resources_return_404(client):
    assert client.get("/feedback/999999").status_code == 404
    assert client.post("/feedback/999999").status_code == 404
    assert client.get("/interview/999999/feedback").status_code == 404
    assert client.post("/interview/999999/feedback").status_code == 404


def test_get_feedback_before_generation_returns_404(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    response = client.get(f"/feedback/{answer_id}")
    assert response.status_code == 404
    assert "no feedback" in response.json()["detail"].lower()


def test_phase_1_to_4_response_shapes_are_unchanged(client):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _add_evaluation(answer_id, combined=75.0, semantic=77.0, keyword=70.0)
    _add_content(answer_id)
    _add_voice(answer_id)
    _add_prediction(answer_id)

    assert _post_answer_feedback(client, answer_id).status_code == 201

    evaluation = client.get(f"/evaluation/{answer_id}")
    assert evaluation.status_code == 200
    assert evaluation.json()["combined_score"] == 75.0
    assert evaluation.json()["status"] == "evaluated"

    content = client.get(f"/analysis/{answer_id}/content")
    assert content.status_code == 200
    assert content.json()["coverage_score"] == 58.0
    assert content.json()["status"] == "completed"

    voice = client.get(f"/analysis/{answer_id}/voice")
    assert voice.status_code == 200
    assert voice.json()["status"] == "completed"
    assert voice.json()["speech_detected"] is True

    prediction = client.get(f"/prediction/{answer_id}")
    assert prediction.status_code == 200
    assert "is_ml_prediction" in prediction.json()

    summary = client.get(f"/interview/{_interview_id}/summary")
    assert summary.status_code == 200
    assert {
        "total_questions",
        "questions",
        "average_score",
        "predictions_completed",
    } <= set(summary.json())
