"""Focused Phase 1 API and persistence tests (SQLite test database)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

import routes.interview as interview_routes
import services.transcription as transcription_service
from database import SessionLocal
from models import InterviewAnswer, InterviewQuestion
from seed import SEED_QUESTIONS, seed_questions


def _start_interview(client) -> int:
    response = client.post("/interview/start")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _questions(client) -> list[dict]:
    response = client.get("/interview/questions")
    assert response.status_code == 200, response.text
    return response.json()


def _audio_file(
    content: bytes = b"fake-webm-audio" * 40,
    filename: str = "answer.webm",
    content_type: str = "audio/webm",
):
    return {"file": (filename, content, content_type)}


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["phase"] == "5"

    health = client.get("/health")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["transcription_configured"] is False
    assert body["transcription_model"] == "whisper-large-v3-turbo"


def test_health_includes_safe_configuration_diagnostics(client):
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["database_driver"] == "sqlite+pysqlite"
    assert body["database_error_category"] is None
    assert body["database_hint"] is None
    assert body["transcription_provider"] == "groq"
    assert body["transcription_hint"]
    assert body["max_audio_mb"] == 1
    assert body["env_file"] == "backend/.env"
    assert isinstance(body["env_file_found"], bool)
    assert body["database_url_source"] in {
        "environment",
        "backend/.env",
        "default",
    }
    assert body["groq_api_key_source"] in {
        "environment",
        "backend/.env",
        "default",
    }
    assert body["prediction_methodology_version"] == "phase4-v1"
    assert body["feature_schema_version"] == "phase4-features-v1"
    assert body["feedback_methodology_version"] == "phase5-v1"
    assert body["ml_model_available"] is False
    assert body["ml_model_state"] in {
        "unavailable",
        "invalid",
        "incompatible",
        "unvalidated",
    }
    assert body["ml_model_reason"]


def test_health_failure_is_sanitized_and_actionable(client, monkeypatch):
    import main

    secret = "sup3r-secret-password-should-not-leak"
    driver_error = RuntimeError(
        f'(1045, "Access denied for user root@localhost '
        f'(using password: YES) {secret}")'
    )

    def fail_db_check():
        raise OperationalError("SELECT 1", {}, driver_error)

    monkeypatch.setattr(main, "ensure_db_ready", fail_db_check)

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()

    assert body["database"] == "unavailable"
    assert body["database_error_category"] == "authentication_failed"
    assert "DATABASE_URL" in body["database_hint"]
    assert secret not in response.text


@pytest.mark.parametrize(
    ("error_code", "expected_category"),
    [
        (1044, "permission_denied"),
        (1045, "authentication_failed"),
        (1049, "unknown_database"),
        (2002, "connection_failed"),
        (2003, "connection_failed"),
        (2005, "unknown_host"),
        (2059, "auth_plugin_error"),
    ],
)
def test_database_errors_are_classified_safely(
    error_code,
    expected_category,
):
    from database import describe_database_error

    result = describe_database_error(
        RuntimeError(error_code, "driver detail with a secret")
    )

    assert result["database_error_category"] == expected_category
    assert result["database_hint"]
    assert "driver detail" not in result["database_hint"]
    assert "secret" not in result["database_hint"]


def test_placeholder_groq_key_is_not_reported_as_configured(client, monkeypatch):
    import main

    monkeypatch.setattr(main, "GROQ_API_KEY", "your_groq_api_key_here")

    body = client.get("/health").json()
    assert body["transcription_configured"] is False
    assert body["transcription_model"] == "whisper-large-v3-turbo"
    assert "console.groq.com/keys" in body["transcription_hint"]


def test_questions_are_seeded_from_database(client):
    questions = _questions(client)
    assert len(questions) >= 3

    texts = {question["question_text"] for question in questions}
    for seed in SEED_QUESTIONS:
        assert seed["question_text"] in texts

    by_text = {question["question_text"]: question for question in questions}
    for seed in SEED_QUESTIONS:
        assert by_text[seed["question_text"]]["has_reference_answer"] is True

    for question in questions:
        assert isinstance(question["id"], int)
        assert question["category"]
        assert "created_at" not in question  # response deliberately stays small


def test_seeding_is_idempotent(client):
    with SessionLocal() as db:
        before = db.scalar(
            select(func.count()).select_from(InterviewQuestion)
        )

        assert seed_questions(db) == 0
        assert seed_questions(db) == 0

        after = db.scalar(
            select(func.count()).select_from(InterviewQuestion)
        )

    assert after == before
    assert len(_questions(client)) == after


def test_start_interview_and_empty_answers(client):
    interview_id = _start_interview(client)

    response = client.get(f"/interview/{interview_id}/answers")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interview_id"] == interview_id
    assert body["status"] == "in_progress"
    assert body["answers"] == []

    missing = client.get("/interview/99999999/answers")
    assert missing.status_code == 404


def test_transcribe_persists_and_updates_answer(client, monkeypatch):
    interview_id = _start_interview(client)
    question = _questions(client)[0]

    calls: list[tuple] = []

    def fake_transcribe(audio_bytes, filename, content_type):
        calls.append((audio_bytes, filename, content_type))
        return "Inheritance lets a child class reuse and extend a parent class."

    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        fake_transcribe,
    )

    response = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(interview_id),
            "question_id": str(question["id"]),
        },
        files=_audio_file(content_type="audio/webm;codecs=opus"),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["updated"] is False
    assert body["answer"]["interview_id"] == interview_id
    assert body["answer"]["question_id"] == question["id"]
    assert body["answer"]["question_text"] == question["question_text"]
    assert body["answer"]["category"] == question["category"]
    assert body["answer"]["transcript"].startswith("Inheritance")
    assert calls and calls[0][2] == "audio/webm"

    with SessionLocal() as db:
        saved = db.scalar(
            select(InterviewAnswer).where(
                InterviewAnswer.interview_id == interview_id,
                InterviewAnswer.question_id == question["id"],
            )
        )
        assert saved is not None
        assert saved.audio_size_bytes and saved.audio_size_bytes > 0

    answers = client.get(f"/interview/{interview_id}/answers")
    assert answers.status_code == 200
    assert len(answers.json()["answers"]) == 1

    # Re-submitting the same question updates the existing row instead of
    # creating a duplicate.
    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        lambda *_args, **_kwargs: "Updated transcript.",
    )
    second = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(interview_id),
            "question_id": str(question["id"]),
        },
        files=_audio_file(),
    )
    assert second.status_code == 200, second.text
    assert second.json()["updated"] is True
    assert second.json()["answer"]["transcript"] == "Updated transcript."

    answers = client.get(f"/interview/{interview_id}/answers")
    assert len(answers.json()["answers"]) == 1


def test_uploaded_wav_is_saved_and_retrievable(client, monkeypatch):
    interview_id = _start_interview(client)
    question = _questions(client)[0]
    transcript = "A WAV upload is transcribed and stored with its audio metadata."
    wav_bytes = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 32

    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        lambda *_args, **_kwargs: transcript,
    )

    response = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(interview_id),
            "question_id": str(question["id"]),
        },
        files={
            "file": (
                "answer.wav",
                wav_bytes,
                "audio/wav",
            )
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["answer"]["transcript"] == transcript

    with SessionLocal() as db:
        saved = db.scalar(
            select(InterviewAnswer).where(
                InterviewAnswer.interview_id == interview_id,
                InterviewAnswer.question_id == question["id"],
            )
        )
        assert saved is not None
        assert saved.transcript == transcript
        assert saved.audio_mime_type == "audio/wav"
        assert saved.audio_size_bytes == len(wav_bytes)

    # This is the same endpoint the React app uses after a page refresh.
    refreshed = client.get(f"/interview/{interview_id}/answers")
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["answers"][0]["transcript"] == transcript


def test_answers_are_scoped_to_their_session(client, monkeypatch):
    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        lambda *_args, **_kwargs: "Session-scoped answer.",
    )
    question = _questions(client)[0]

    first_session = _start_interview(client)
    second_session = _start_interview(client)

    for interview_id in (first_session, second_session):
        response = client.post(
            "/interview/transcribe",
            data={
                "interview_id": str(interview_id),
                "question_id": str(question["id"]),
            },
            files=_audio_file(),
        )
        assert response.status_code == 201, response.text

    first_answers = client.get(
        f"/interview/{first_session}/answers"
    ).json()["answers"]
    second_answers = client.get(
        f"/interview/{second_session}/answers"
    ).json()["answers"]

    assert len(first_answers) == 1
    assert len(second_answers) == 1
    assert first_answers[0]["interview_id"] == first_session
    assert second_answers[0]["interview_id"] == second_session


def test_invalid_session_and_question_are_rejected(client):
    question = _questions(client)[0]
    valid_session = _start_interview(client)

    invalid_session = client.post(
        "/interview/transcribe",
        data={
            "interview_id": "99999999",
            "question_id": str(question["id"]),
        },
        files=_audio_file(),
    )
    assert invalid_session.status_code == 404

    invalid_question = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(valid_session),
            "question_id": "99999999",
        },
        files=_audio_file(),
    )
    assert invalid_question.status_code == 404


def test_audio_validation(client):
    interview_id = _start_interview(client)
    question = _questions(client)[0]
    data = {
        "interview_id": str(interview_id),
        "question_id": str(question["id"]),
    }

    unsupported = client.post(
        "/interview/transcribe",
        data=data,
        files=_audio_file(
            content=b"not audio",
            filename="answer.txt",
            content_type="text/plain",
        ),
    )
    assert unsupported.status_code == 415

    empty = client.post(
        "/interview/transcribe",
        data=data,
        files=_audio_file(content=b""),
    )
    assert empty.status_code == 400

    oversized = client.post(
        "/interview/transcribe",
        data=data,
        files=_audio_file(content=b"a" * (1024 * 1024 + 1)),
    )
    assert oversized.status_code == 413


def test_audio_type_fallbacks():
    assert transcription_service.normalize_audio_type(
        "audio/webm;codecs=opus", "answer.webm"
    ) == ("audio/webm", "webm")
    assert transcription_service.normalize_audio_type(
        "application/octet-stream", "answer.webm"
    ) == ("audio/webm", "webm")
    assert transcription_service.normalize_audio_type(
        "", "answer.m4a"
    ) == ("audio/mp4", "m4a")
    assert transcription_service.normalize_audio_type(
        "audio/mp4", "answer.mp4"
    ) == ("audio/mp4", "m4a")


def test_transcription_without_key_fails_safely(monkeypatch):
    monkeypatch.setattr(transcription_service, "GROQ_API_KEY", "")

    with pytest.raises(transcription_service.TranscriptionError) as error:
        transcription_service.transcribe_audio(
            b"fake audio",
            "answer.webm",
            "audio/webm",
        )

    assert error.value.status_code == 503
    assert "GROQ_API_KEY" in error.value.message


def test_transcription_rejects_placeholder_key(monkeypatch):
    monkeypatch.setattr(
        transcription_service,
        "GROQ_API_KEY",
        "your_groq_api_key_here",
    )

    with pytest.raises(transcription_service.TranscriptionError) as error:
        transcription_service.transcribe_audio(
            b"fake audio",
            "answer.webm",
            "audio/webm",
        )

    assert error.value.status_code == 503
    assert "GROQ_API_KEY" in error.value.message


class _FakeTranscriptions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.create_kwargs: dict | None = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(text=self.text)


class _FakeGroq:
    def __init__(self, text: str) -> None:
        self.client_kwargs: dict | None = None
        self.transcriptions = _FakeTranscriptions(text)
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)
        self.closed = False

    def __call__(self, **kwargs):
        self.client_kwargs = kwargs
        return self

    def close(self) -> None:
        self.closed = True


def test_groq_sdk_call_is_wired_correctly(monkeypatch):
    fake_client = _FakeGroq("  Hello from Groq  ")
    monkeypatch.setattr(transcription_service, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        transcription_service,
        "Groq",
        lambda **kwargs: fake_client.__call__(**kwargs),
    )

    transcript = transcription_service.transcribe_audio(
        b"audio-bytes",
        "answer.webm",
        "audio/webm",
    )

    assert transcript == "Hello from Groq"
    assert fake_client.client_kwargs == {
        "api_key": "test-key",
        "timeout": transcription_service.GROQ_TIMEOUT_SECONDS,
        "max_retries": 1,
    }
    assert fake_client.transcriptions.create_kwargs == {
        "file": ("answer.webm", b"audio-bytes", "audio/webm"),
        "model": transcription_service.GROQ_MODEL,
        "response_format": "json",
        "temperature": 0.0,
    }
    assert fake_client.closed is True


def test_empty_groq_transcript_is_reported(monkeypatch):
    fake_client = _FakeGroq("   ")
    monkeypatch.setattr(transcription_service, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        transcription_service,
        "Groq",
        lambda **kwargs: fake_client.__call__(**kwargs),
    )

    with pytest.raises(transcription_service.TranscriptionError) as error:
        transcription_service.transcribe_audio(
            b"audio-bytes",
            "answer.webm",
            "audio/webm",
        )

    assert error.value.status_code == 422
    assert "No speech" in error.value.message
    assert fake_client.closed is True


class _RaisingTranscriptions:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def create(self, **_kwargs):
        raise self.error


class _FailingGroq:
    def __init__(self, error: BaseException) -> None:
        self.audio = SimpleNamespace(
            transcriptions=_RaisingTranscriptions(error)
        )
        self.closed = False

    def __call__(self, **_kwargs):
        return self

    def close(self) -> None:
        self.closed = True


_GROQ_REQUEST = httpx.Request(
    "POST",
    "https://api.groq.com/openai/v1/audio/transcriptions",
)


def _groq_http_error(error_type, status_code):
    return error_type(
        "provider error",
        response=httpx.Response(status_code, request=_GROQ_REQUEST),
        body=None,
    )


_GROQ_FAILURE_CASES = [
    (
        transcription_service.APITimeoutError(request=_GROQ_REQUEST),
        504,
        "timed out",
    ),
    (
        _groq_http_error(transcription_service.AuthenticationError, 401),
        502,
        "rejected its API credentials",
    ),
    (
        _groq_http_error(transcription_service.RateLimitError, 429),
        503,
        "busy",
    ),
    (
        _groq_http_error(transcription_service.APIStatusError, 500),
        502,
        "could not process",
    ),
    (
        transcription_service.APIConnectionError(request=_GROQ_REQUEST),
        502,
        "Could not reach",
    ),
    (
        RuntimeError("unexpected provider detail including test-key"),
        502,
        "failed unexpectedly",
    ),
]


@pytest.mark.parametrize(
    ("provider_error", "expected_status", "expected_text"),
    _GROQ_FAILURE_CASES,
)
def test_groq_failures_are_mapped_safely(
    monkeypatch,
    caplog,
    provider_error,
    expected_status,
    expected_text,
):
    caplog.set_level(logging.DEBUG)
    fake_client = _FailingGroq(provider_error)
    monkeypatch.setattr(transcription_service, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        transcription_service,
        "Groq",
        lambda **kwargs: fake_client.__call__(**kwargs),
    )

    with pytest.raises(transcription_service.TranscriptionError) as error:
        transcription_service.transcribe_audio(
            b"audio-bytes",
            "answer.webm",
            "audio/webm",
        )

    assert error.value.status_code == expected_status
    assert expected_text in error.value.message
    assert "test-key" not in error.value.message
    assert "unexpected provider detail" not in error.value.message
    assert "test-key" not in caplog.text
    assert "unexpected provider detail" not in caplog.text
    assert fake_client.closed is True


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:5173", "http://127.0.0.1:5173"],
)
def test_cors_allows_local_vite_origin(client, origin):
    response = client.options(
        "/interview/questions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
