"""Phase 2 tests: preprocessing, scoring, persistence, API, and summary.

Semantic inference is mocked in most tests so the suite is fast and does not
require a model download. The final test is an opt-in real-model integration
test (set ``RUN_SEMANTIC_MODEL_INTEGRATION=1``).
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import func, select

import routes.interview as interview_routes
import services.evaluation as evaluation_service
import services.semantic_similarity as semantic_service
from database import SessionLocal
from models import AnswerEvaluation, InterviewAnswer, InterviewQuestion
from services.evaluation import (
    MissingReferenceAnswerError,
    STATUS_UNEVALUABLE,
    evaluate_transcript,
)
from services.keyword_evaluation import keyword_coverage, term_is_present
from services.nlp_preprocessing import (
    assess_answer_usability,
    content_tokens,
    meaningful_tokens,
    normalize_text,
    tokenize,
)
from services.scoring import combine_scores
from services.semantic_similarity import SemanticModelError, semantic_similarity
from seed import SEED_QUESTIONS, seed_questions


def _start_interview(client) -> int:
    response = client.post("/interview/start")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _seed_question(index: int = 0) -> dict:
    with SessionLocal() as db:
        question = db.scalars(
            select(InterviewQuestion).order_by(InterviewQuestion.id)
        ).all()[index]
        return {
            "id": question.id,
            "question_text": question.question_text,
            "reference_answer": question.reference_answer,
        }


def _create_question(reference_answer: str | None) -> int:
    with SessionLocal() as db:
        question = InterviewQuestion(
            question_text=f"Phase 2 test question {uuid.uuid4()}",
            category="Phase 2 Test",
            reference_answer=reference_answer,
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        return question.id


def _create_answer(
    interview_id: int,
    question_id: int,
    transcript: str,
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


def _set_transcript(answer_id: int, transcript: str) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        answer.transcript = transcript
        db.commit()


def _set_reference(question_id: int, reference_answer: str) -> None:
    with SessionLocal() as db:
        question = db.get(InterviewQuestion, question_id)
        assert question is not None
        question.reference_answer = reference_answer
        db.commit()


def _evaluation_count(answer_id: int) -> int:
    with SessionLocal() as db:
        return db.scalar(
            select(func.count())
            .select_from(AnswerEvaluation)
            .where(AnswerEvaluation.answer_id == answer_id)
        )


def _patch_semantic(monkeypatch, score: float = 80.0) -> dict:
    calls = {"count": 0}

    def fake_semantic(candidate, reference):
        calls["count"] += 1
        return score

    monkeypatch.setattr(evaluation_service, "semantic_similarity", fake_semantic)
    return calls


# ---------------------------------------------------------------------------
# Pure NLP and scoring tests
# ---------------------------------------------------------------------------


def test_preprocessing_preserves_technical_terms_and_negation():
    assert normalize_text("  C++\n\tuses   .NET ") == "C++ uses .NET"

    tokens = tokenize("C++ C# .NET Node.js TCP/IP OOP don't 1NF")
    for expected in ("c++", "c#", ".net", "node.js", "tcp/ip", "oop", "don't", "1nf"):
        assert expected in tokens

    assert "not" in content_tokens("This is not a reliable protocol")
    assert "is" not in content_tokens("This is not a reliable protocol")

    assert meaningful_tokens("um uh hmm like basically") == []
    assert assess_answer_usability(" ").usable is False
    assert assess_answer_usability("um uh hmm").usable is False
    assert assess_answer_usability("Inheritance").usable is False
    assert (
        assess_answer_usability(
            "Inheritance lets a child class reuse parent methods."
        ).usable
        is True
    )


def test_keyword_coverage_is_weighted_deduplicated_and_explainable():
    reference = "TCP is connection-oriented and reliable. UDP is connectionless."
    coverage = keyword_coverage(
        "TCP is reliable and connection-oriented.",
        reference,
    )

    # TCP=1.75, connection-oriented=2.0 (technical punctuation), reliable=1.15
    # UDP=1.75, connectionless=1.3 (10+ chars).
    expected_score = round(
        ((1.75 + 2.0 + 1.15) / (1.75 + 2.0 + 1.15 + 1.75 + 1.3)) * 100,
        2,
    )
    assert coverage.score == expected_score
    assert coverage.matched == ("connection-oriented", "tcp", "reliable")
    assert set(coverage.missing) == {"udp", "connectionless"}

    assert keyword_coverage(reference, reference).score == 100.0
    assert keyword_coverage("TCP reliable", "TCP reliable TCP").score == 100.0

    empty = keyword_coverage("", reference)
    assert empty.score == 0.0
    assert empty.matched == ()
    assert set(empty.missing) == set(coverage.missing) | set(coverage.matched)

    assert term_is_present("tcp/ip", {"tcp", "ip"})
    assert term_is_present("classes", {"class"})
    assert not term_is_present("udp", {"tcp"})

    assert keyword_coverage("table", "TCP table").score < keyword_coverage(
        "tcp", "TCP table"
    ).score


def test_combined_score_formula_rounding_and_bounds():
    combined = combine_scores(80, 50)
    assert combined.semantic_score == 80.0
    assert combined.keyword_score == 50.0
    assert combined.combined_score == 71.0

    clamped = combine_scores(120, -5)
    assert clamped.semantic_score == 100.0
    assert clamped.keyword_score == 0.0
    assert clamped.combined_score == 70.0

    rounded = combine_scores(66.666, 33.333)
    assert rounded.combined_score == round(0.7 * 66.666 + 0.3 * 33.333, 2)


def test_unusable_answers_are_explicitly_unevaluable(monkeypatch):
    def fail_semantic(*_args, **_kwargs):
        raise AssertionError("Semantic model must not run for unusable answers")

    monkeypatch.setattr(
        evaluation_service,
        "semantic_similarity",
        fail_semantic,
    )
    reference = _seed_question(0)["reference_answer"]

    for transcript in ("", "   ", "um uh hmm", "Inheritance"):
        result = evaluate_transcript(transcript, reference)
        assert result.status == STATUS_UNEVALUABLE
        assert result.semantic_score == 0.0
        assert result.keyword_score == 0.0
        assert result.combined_score == 0.0
        assert result.status_detail

    with pytest.raises(MissingReferenceAnswerError):
        evaluate_transcript("A perfectly reasonable answer.", None)


def test_semantic_similarity_with_fake_model(monkeypatch):
    class FakeModel:
        def encode(self, texts, **_kwargs):
            assert len(texts) == 2
            return [[1.0, 0.0], [0.6, 0.8]]

    monkeypatch.setattr(
        semantic_service,
        "get_semantic_model",
        lambda: FakeModel(),
    )
    assert semantic_similarity("one answer", "another answer") == 60.0

    def fail_model():
        raise AssertionError("Model should not load for empty input")

    monkeypatch.setattr(semantic_service, "get_semantic_model", fail_model)
    assert semantic_similarity("", "reference") == 0.0

    class BrokenModel:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        semantic_service,
        "get_semantic_model",
        lambda: BrokenModel(),
    )
    with pytest.raises(SemanticModelError):
        semantic_similarity("one answer", "another answer")


# ---------------------------------------------------------------------------
# API, persistence, and summary tests
# ---------------------------------------------------------------------------


def test_transcribe_then_evaluate_returns_saved_scores(client, monkeypatch):
    question = _seed_question(0)
    interview_id = _start_interview(client)
    calls = _patch_semantic(monkeypatch, 80.0)

    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        lambda *_args, **_kwargs: question["reference_answer"],
    )
    transcribed = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(interview_id),
            "question_id": str(question["id"]),
        },
        files={
            "file": (
                "answer.webm",
                b"fake-webm-audio" * 20,
                "audio/webm",
            )
        },
    )
    assert transcribed.status_code == 201, transcribed.text
    answer_id = transcribed.json()["answer"]["id"]

    first = client.post(f"/evaluation/{answer_id}")
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["reused"] is False
    evaluation = body["evaluation"]
    assert evaluation["answer_id"] == answer_id
    assert evaluation["interview_id"] == interview_id
    assert evaluation["question_id"] == question["id"]
    assert evaluation["status"] == "evaluated"
    assert evaluation["semantic_score"] == 80.0
    assert evaluation["keyword_score"] == 100.0
    assert evaluation["combined_score"] == 86.0
    assert evaluation["is_stale"] is False
    assert evaluation["matched_keywords"]
    assert evaluation["missing_keywords"] == []

    second = client.post(f"/evaluation/{answer_id}")
    assert second.status_code == 200, second.text
    assert second.json()["reused"] is True
    assert second.json()["evaluation"]["id"] == evaluation["id"]
    assert calls["count"] == 1
    assert _evaluation_count(answer_id) == 1

    fetched = client.get(f"/evaluation/{answer_id}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == evaluation["id"]


def test_unevaluable_answer_is_persisted_and_explained(client, monkeypatch):
    question = _seed_question(0)
    interview_id = _start_interview(client)
    answer_id = _create_answer(interview_id, question["id"], "um uh hmm")

    def fail_semantic(*_args, **_kwargs):
        raise AssertionError("Semantic model must not run for filler-only input")

    monkeypatch.setattr(
        evaluation_service,
        "semantic_similarity",
        fail_semantic,
    )

    response = client.post(f"/evaluation/{answer_id}")
    assert response.status_code == 201, response.text
    evaluation = response.json()["evaluation"]
    assert evaluation["status"] == "unevaluable"
    assert evaluation["semantic_score"] == 0.0
    assert evaluation["keyword_score"] == 0.0
    assert evaluation["combined_score"] == 0.0
    assert evaluation["status_detail"]

    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200, summary.text
    body = summary.json()
    item = next(
        item for item in body["questions"] if item["question_id"] == question["id"]
    )
    assert item["status"] == "unevaluable"
    assert item["combined_score"] == 0.0
    assert body["answers_unevaluable"] == 1
    assert body["average_score"] is None

    assert _evaluation_count(answer_id) == 1


def test_missing_reference_and_invalid_ids_are_safe(client):
    no_reference_question_id = _create_question(None)
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        no_reference_question_id,
        "This answer has enough meaningful words to evaluate.",
    )

    response = client.post(f"/evaluation/{answer_id}")
    assert response.status_code == 409, response.text
    assert "reference answer" in response.json()["detail"]

    assert client.get(f"/evaluation/{answer_id}").status_code == 404

    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200, summary.text
    item = next(
        item
        for item in summary.json()["questions"]
        if item["question_id"] == no_reference_question_id
    )
    assert item["status"] == "not_evaluable"
    assert item["has_reference_answer"] is False

    questions = client.get("/interview/questions").json()
    question = next(
        item for item in questions if item["id"] == no_reference_question_id
    )
    assert question["has_reference_answer"] is False

    assert client.post("/evaluation/99999999").status_code == 404
    assert client.get("/evaluation/99999999").status_code == 404
    assert client.get("/interview/99999999/evaluations").status_code == 404
    assert client.get("/interview/99999999/summary").status_code == 404

    # Non-integer path values are rejected by FastAPI validation, not by a
    # database lookup or an unhandled exception.
    assert client.post("/evaluation/not-an-id").status_code == 422
    assert client.get("/interview/not-an-id/summary").status_code == 422


def test_seed_backfills_only_missing_reference_answers(client):
    seed = SEED_QUESTIONS[1]
    custom_answer = "A locally customized reference answer that must survive."

    with SessionLocal() as db:
        question = db.scalar(
            select(InterviewQuestion).where(
                InterviewQuestion.question_text == seed["question_text"]
            )
        )
        assert question is not None
        original_reference = question.reference_answer
        question.reference_answer = custom_answer
        db.commit()

        seed_questions(db)

        db.refresh(question)
        assert question.reference_answer == custom_answer

        question.reference_answer = None
        db.commit()

        seed_questions(db)

        db.refresh(question)
        assert question.reference_answer == original_reference

        # Restore the seed value for any later test.
        question.reference_answer = original_reference
        db.commit()


def test_semantic_model_failure_returns_503_without_fake_scores(
    client,
    monkeypatch,
):
    question = _seed_question(0)
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question["id"],
        "Inheritance lets a child class reuse the methods of a parent class.",
    )

    def unavailable(candidate, reference):
        raise SemanticModelError("The semantic model is unavailable.")

    monkeypatch.setattr(evaluation_service, "semantic_similarity", unavailable)

    response = client.post(f"/evaluation/{answer_id}")
    assert response.status_code == 503, response.text
    assert "semantic model" in response.json()["detail"].lower()

    # A failed evaluation must not persist a placeholder/zero result.
    assert client.get(f"/evaluation/{answer_id}").status_code == 404
    assert _evaluation_count(answer_id) == 0

    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200, summary.text
    item = next(
        item
        for item in summary.json()["questions"]
        if item["question_id"] == question["id"]
    )
    assert item["status"] == "unevaluated"
    assert summary.json()["average_score"] is None


def test_duplicate_requests_and_session_scoped_summary(client, monkeypatch):
    first_question = _seed_question(0)
    second_question = _seed_question(1)
    third_question = _seed_question(2)

    _patch_semantic(monkeypatch, 80.0)
    interview_id = _start_interview(client)

    first_answer_id = _create_answer(
        interview_id,
        first_question["id"],
        first_question["reference_answer"],
    )
    second_answer_id = _create_answer(
        interview_id,
        second_question["id"],
        second_question["reference_answer"],
    )
    third_answer_id = _create_answer(
        interview_id,
        third_question["id"],
        "um uh hmm",
    )

    assert client.post(f"/evaluation/{first_answer_id}").status_code == 201
    assert client.post(f"/evaluation/{second_answer_id}").status_code == 201
    assert client.post(f"/evaluation/{third_answer_id}").status_code == 201

    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200, summary.text
    body = summary.json()

    assert body["answers_submitted"] == 3
    assert body["answers_evaluated"] == 2
    assert body["answers_unevaluable"] == 1
    assert body["average_score"] == 86.0
    assert body["total_questions"] >= 3
    assert body["questions_unanswered"] == body["total_questions"] - 3

    statuses = {item["question_id"]: item["status"] for item in body["questions"]}
    assert statuses[first_question["id"]] == "evaluated"
    assert statuses[second_question["id"]] == "evaluated"
    assert statuses[third_question["id"]] == "unevaluable"

    evaluations = client.get(f"/interview/{interview_id}/evaluations")
    assert evaluations.status_code == 200, evaluations.text
    assert len(evaluations.json()["evaluations"]) == 3

    empty_session_id = _start_interview(client)
    empty_summary = client.get(f"/interview/{empty_session_id}/summary")
    assert empty_summary.status_code == 200, empty_summary.text
    assert empty_summary.json()["average_score"] is None
    assert empty_summary.json()["answers_submitted"] == 0
    assert empty_summary.json()["answers_evaluated"] == 0

    other_evaluations = client.get(
        f"/interview/{empty_session_id}/evaluations"
    )
    assert other_evaluations.status_code == 200
    assert other_evaluations.json()["evaluations"] == []


def test_stale_evaluation_is_detected_and_recomputed(client, monkeypatch):
    reference = (
        "Inheritance lets a child class reuse and extend the methods of a "
        "parent class."
    )
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer_id = _create_answer(interview_id, question_id, reference)

    _patch_semantic(monkeypatch, 80.0)
    created = client.post(f"/evaluation/{answer_id}")
    assert created.status_code == 201, created.text
    evaluation_id = created.json()["evaluation"]["id"]

    _set_reference(
        question_id,
        "Inheritance allows a derived class to acquire parent behavior and "
        "reuse code.",
    )

    stale = client.get(f"/evaluation/{answer_id}")
    assert stale.status_code == 200
    assert stale.json()["is_stale"] is True

    stale_summary = client.get(f"/interview/{interview_id}/summary")
    item = next(
        item
        for item in stale_summary.json()["questions"]
        if item["question_id"] == question_id
    )
    assert item["status"] == "stale"
    assert stale_summary.json()["average_score"] is None

    refreshed = client.post(f"/evaluation/{answer_id}")
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["reused"] is False
    assert refreshed.json()["evaluation"]["id"] == evaluation_id
    assert refreshed.json()["evaluation"]["is_stale"] is False

    _set_transcript(
        answer_id,
        "TCP is a reliable transport protocol and UDP is connectionless.",
    )
    assert client.get(f"/evaluation/{answer_id}").json()["is_stale"] is True

    refreshed_again = client.post(f"/evaluation/{answer_id}")
    assert refreshed_again.status_code == 200, refreshed_again.text
    assert refreshed_again.json()["reused"] is False
    assert refreshed_again.json()["evaluation"]["id"] == evaluation_id
    assert _evaluation_count(answer_id) == 1


def test_stale_answer_is_excluded_from_summary_average(client, monkeypatch):
    """A stale answer must drop out of the average while valid ones remain."""
    first_reference = (
        "Inheritance lets a child class reuse the methods and properties of a "
        "parent class."
    )
    second_reference = (
        "A primary key uniquely identifies each row in a database table."
    )
    first_question = _create_question(first_reference)
    second_question = _create_question(second_reference)
    interview_id = _start_interview(client)
    first_answer = _create_answer(interview_id, first_question, first_reference)
    second_answer = _create_answer(
        interview_id,
        second_question,
        second_reference,
    )

    # Both answers are identical to their reference, so keyword coverage is 100
    # and every combined score is 0.70 * 80 + 0.30 * 100 = 86.0.
    _patch_semantic(monkeypatch, 80.0)
    created = client.post(f"/evaluation/{first_answer}")
    assert created.status_code == 201, created.text
    first_evaluation_id = created.json()["evaluation"]["id"]
    assert client.post(f"/evaluation/{second_answer}").status_code == 201

    summary = client.get(f"/interview/{interview_id}/summary").json()
    assert summary["answers_evaluated"] == 2
    assert summary["average_score"] == 86.0

    # Editing the reference invalidates only the first question's evaluation.
    _set_reference(
        first_question,
        "Database indexes speed up lookups by reducing full table scans.",
    )

    summary = client.get(f"/interview/{interview_id}/summary").json()
    assert summary["answers_evaluated"] == 1
    assert summary["average_score"] == 86.0  # only the second answer remains
    statuses = {
        item["question_id"]: item["status"] for item in summary["questions"]
    }
    assert statuses[first_question] == "stale"
    assert statuses[second_question] == "evaluated"
    assert client.get(f"/evaluation/{first_answer}").json()["is_stale"] is True

    refreshed = client.post(f"/evaluation/{first_answer}")
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["evaluation"]["id"] == first_evaluation_id
    assert refreshed.json()["evaluation"]["is_stale"] is False

    summary = client.get(f"/interview/{interview_id}/summary").json()
    assert summary["answers_evaluated"] == 2
    assert summary["average_score"] is not None
    assert _evaluation_count(first_answer) == 1
    assert _evaluation_count(second_answer) == 1


@pytest.mark.skipif(
    os.getenv("RUN_SEMANTIC_MODEL_INTEGRATION") != "1",
    reason=(
        "Set RUN_SEMANTIC_MODEL_INTEGRATION=1 to download/load the real "
        "Sentence-Transformers model."
    ),
)
def test_real_semantic_model_integration():
    semantic_service.reset_semantic_model()
    score = semantic_similarity(
        "Inheritance lets a child class reuse the methods of a parent class.",
        "Inheritance allows a derived class to acquire properties and methods "
        "from a base class, enabling code reuse.",
    )
    assert semantic_service.model_is_loaded() is True
    assert 0.0 <= score <= 100.0
    assert score > 40.0
