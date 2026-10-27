"""Phase 3 tests: audio storage, content analysis, and voice measurements.

Recordings are synthetic WAV files generated with the standard library, so the
suite is deterministic and does not need a real microphone, Groq key, or MySQL.
The semantic model is mocked where Phase 2 reuse is exercised.
"""

from __future__ import annotations

import io
import math
import struct
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

import pytest
from sqlalchemy import func, select

import routes.interview as interview_routes
import services.evaluation as evaluation_service
import services.voice_analysis as voice_service
from config import AUDIO_STORAGE_DIR
from database import SessionLocal
from models import (
    AnswerEvaluation,
    ContentAnalysis,
    InterviewAnswer,
    InterviewQuestion,
    VoiceAnalysis,
)
from services.audio_storage import (
    AudioStorageError,
    resolve_audio_path,
    sha256_bytes,
    store_audio,
)
from services.content_analysis import analyze_content
from services.nlp_preprocessing import tokenize
from services.voice_analysis import (
    VoiceAnalysisError,
    VoiceAnalysisUnavailable,
    analyze_audio,
)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_interview(client) -> int:
    response = client.post("/interview/start")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_question(reference_answer: str | None) -> int:
    with SessionLocal() as db:
        question = InterviewQuestion(
            question_text=f"Phase 3 test question {uuid.uuid4()}",
            category="Phase 3 Test",
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


def _attach_audio(answer_id: int, content: bytes, extension: str = "wav") -> str:
    """Store bytes the same way the transcription endpoint does."""
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        stored = store_audio(
            content,
            answer.interview_id,
            answer.question_id,
            extension,
        )
        answer.audio_path = stored.relative_path
        answer.audio_sha256 = stored.sha256
        db.commit()
        return stored.relative_path


def _count(model) -> int:
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(model))


def _analysis_count(model, answer_id: int) -> int:
    with SessionLocal() as db:
        return db.scalar(
            select(func.count())
            .select_from(model)
            .where(model.answer_id == answer_id)
        )


def _patch_semantic(monkeypatch, score: float = 80.0) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "semantic_similarity",
        lambda _candidate, _reference: score,
    )


def _transcribe(
    client,
    monkeypatch,
    interview_id: int,
    question_id: int,
    audio: bytes,
    transcript: str,
    filename: str = "answer.wav",
    content_type: str = "audio/wav",
) -> dict:
    monkeypatch.setattr(
        interview_routes,
        "transcribe_audio",
        lambda *_args, **_kwargs: transcript,
    )
    response = client.post(
        "/interview/transcribe",
        data={
            "interview_id": str(interview_id),
            "question_id": str(question_id),
        },
        files={"file": (filename, audio, content_type)},
    )
    assert response.status_code == 201, response.text
    return response.json()["answer"]


def _wav_bytes(segments: list[tuple[float, float, float]]) -> bytes:
    """Build mono 16-bit PCM from ``(amplitude, seconds, frequency)`` parts.

    An amplitude of 0 produces silence. Frequencies of 0 are allowed for
    silence. The signal is deterministic, so VAD/pitch behaviour is repeatable.
    """
    frames = bytearray()
    for amplitude, seconds, frequency in segments:
        sample_count = int(SAMPLE_RATE * seconds)
        for index in range(sample_count):
            if amplitude <= 0 or frequency <= 0:
                value = 0.0
            else:
                value = amplitude * math.sin(
                    2.0 * math.pi * frequency * index / SAMPLE_RATE
                )
            frames.extend(
                struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
            )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def _speech_wav() -> bytes:
    # 3.2 seconds of speech, split by two 0.5 second pauses.
    return _wav_bytes(
        [
            (0.25, 0.8, 180),
            (0.0, 0.5, 0),
            (0.25, 0.8, 180),
            (0.0, 0.5, 0),
            (0.25, 0.8, 180),
            (0.0, 0.5, 0),
            (0.25, 0.8, 180),
        ]
    )


def _write_temp_wav(content: bytes, directory: Path) -> Path:
    path = directory / "recording.wav"
    path.write_bytes(content)
    return path


def _to_mp3(wav_content: bytes) -> bytes:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="ai-phase3-") as directory:
        source = Path(directory) / "source.wav"
        target = Path(directory) / "source.mp3"
        source.write_bytes(wav_content)
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                str(target),
            ],
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return target.read_bytes()


# ---------------------------------------------------------------------------
# Audio storage
# ---------------------------------------------------------------------------


def test_audio_storage_is_content_addressed_and_path_safe():
    content = _speech_wav()
    first = store_audio(content, 9001, 1, "wav")
    second = store_audio(content, 9001, 1, "wav")

    assert first.sha256 == sha256_bytes(content)
    assert first.relative_path == second.relative_path
    assert first.size_bytes == len(content)

    stored_path = resolve_audio_path(first.relative_path)
    assert stored_path.is_file()
    assert stored_path.read_bytes() == content
    assert stored_path.is_relative_to(AUDIO_STORAGE_DIR.resolve())

    with pytest.raises(AudioStorageError):
        resolve_audio_path("../../outside.wav")

    with pytest.raises(AudioStorageError):
        store_audio(b"", 9001, 1, "wav")


# ---------------------------------------------------------------------------
# Content analysis service
# ---------------------------------------------------------------------------


def test_content_analysis_measures_fillers_repetition_and_sentences():
    reference = (
        "Inheritance lets a child class reuse and extend the methods of a "
        "parent class."
    )
    transcript = (
        "um well basically inheritance lets a child class reuse methods. "
        "Inheritance inheritance lets a child class extend methods! "
        "A parent class provides the shared behaviour."
    )
    result = analyze_content(transcript, reference)

    assert result.status == "completed"
    assert result.coverage_source == "keyword_coverage"
    assert result.coverage_score is not None
    assert 0.0 <= result.coverage_score <= 100.0
    assert result.filler_word_count >= 3
    assert result.filler_rate_percent is not None
    assert {word for word, _count in result.filler_words} >= {
        "um",
        "well",
        "basically",
    }
    repeated = dict(result.repeated_terms)
    assert repeated["inheritance"] >= 3
    assert result.repetition_rate_percent is not None
    assert result.sentence_count == 3
    assert result.average_sentence_words is not None


def test_content_analysis_without_reference_is_explicitly_not_evaluable():
    result = analyze_content(
        "um well this answer has enough meaningful words to analyse.",
        None,
    )

    assert result.status == "not_evaluable"
    assert result.status_detail
    assert result.coverage_score is None
    assert result.coverage_source == "none"
    assert result.matched_concepts == ()
    assert result.missing_concepts == ()
    # Transcript-only statistics are still valid and must not be hidden.
    assert result.filler_word_count == 2
    assert result.sentence_count == 1


def test_content_analysis_filler_only_transcript_is_unevaluable():
    result = analyze_content("um uh hmm", "Inheritance reuses parent methods.")

    assert result.status == "unevaluable"
    assert result.status_detail
    assert result.coverage_score is None
    assert result.coverage_source == "none"


def test_content_analysis_empty_transcript_reports_zero_not_a_score():
    result = analyze_content("", "Inheritance reuses parent methods.")

    assert result.status == "unevaluable"
    assert result.coverage_score is None
    assert result.filler_word_count == 0
    assert result.filler_rate_percent is None
    assert result.sentence_count == 0
    assert result.average_sentence_words is None


# ---------------------------------------------------------------------------
# Voice analysis service
# ---------------------------------------------------------------------------


def test_voice_analysis_measures_speech_pauses_and_speaking_rate(tmp_path):
    pytest.importorskip("librosa")
    path = _write_temp_wav(_speech_wav(), tmp_path)
    transcript = " ".join(["word"] * 20)

    result = analyze_audio(path, transcript)

    assert result.status == "completed"
    assert result.status_detail is None
    assert result.speech_detected is True
    assert result.duration_seconds is not None
    assert result.analyzed_duration_seconds is not None
    assert result.speech_duration_seconds is not None
    assert result.speech_duration_seconds >= 2.0
    assert result.speech_ratio_percent is not None
    assert result.silence_duration_seconds is not None
    assert result.pause_count is not None
    assert result.pause_count >= 2
    assert result.longest_pause_seconds is not None
    assert 0.3 <= result.longest_pause_seconds <= 0.9
    assert result.pauses_per_minute is not None

    expected_wpm = 20 / (result.speech_duration_seconds / 60.0)
    assert abs(result.speaking_rate_wpm - expected_wpm) < 0.5
    assert result.word_count == 20

    assert result.rms_mean is not None
    assert result.rms_mean > 0
    assert result.rms_mean_dbfs is not None
    assert result.peak_amplitude is not None
    assert result.clipping_percent == 0.0
    assert isinstance(result.pitch_reliable, bool)
    if result.pitch_reliable:
        assert result.pitch_median_hz is not None
        assert 60 <= result.pitch_median_hz <= 400
        assert result.pitch_range_hz is not None
    else:
        assert result.pitch_median_hz is None


def test_voice_analysis_silence_has_no_invented_speech_metrics(tmp_path):
    pytest.importorskip("librosa")
    path = _write_temp_wav(
        _wav_bytes([(0.0, 2.0, 0)]),
        tmp_path,
    )

    result = analyze_audio(path, "word " * 10)

    assert result.status == "insufficient_speech"
    assert result.status_detail
    assert result.speech_detected is False
    assert result.speech_duration_seconds == 0.0
    assert result.pause_count == 0
    assert result.speaking_rate_wpm is None
    assert result.pitch_reliable is False
    assert result.pitch_median_hz is None
    # Energy measurements are still real values for silence.
    assert result.rms_mean == 0.0
    assert result.rms_mean_dbfs == -120.0
    assert any("No speech was detected" in note for note in result.quality_notes)


def test_voice_analysis_short_clip_does_not_report_pitch(tmp_path):
    pytest.importorskip("librosa")
    path = _write_temp_wav(_wav_bytes([(0.25, 0.4, 180)]), tmp_path)

    result = analyze_audio(path, "a few words")

    assert result.pitch_reliable is False
    assert result.pitch_median_hz is None
    if result.speech_detected:
        assert result.speaking_rate_wpm is None


def test_voice_analysis_clipping_is_measured(tmp_path):
    pytest.importorskip("librosa")
    path = _write_temp_wav(_wav_bytes([(1.0, 1.0, 180)]), tmp_path)

    result = analyze_audio(path, "clipped")

    assert result.peak_amplitude == 1.0
    assert result.clipping_percent is not None
    assert result.clipping_percent > 0
    assert any("clipping" in note.lower() for note in result.quality_notes)


def test_voice_analysis_rejects_malformed_audio(tmp_path):
    path = tmp_path / "malformed.wav"
    path.write_bytes(b"this is not an audio file" * 20)

    with pytest.raises(VoiceAnalysisError):
        analyze_audio(path, "a transcript")


def test_voice_analysis_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    path = _write_temp_wav(_speech_wav(), tmp_path)

    def fail_soundfile(_soundfile, _path):
        raise RuntimeError("libsndfile cannot read this container")

    monkeypatch.setattr(voice_service, "_decode_with_soundfile", fail_soundfile)

    result = analyze_audio(path, " ".join(["word"] * 10))

    assert result.status == "completed"
    # FFmpeg output is mono 16 kHz, so source details are unknown, not guessed.
    assert result.sample_rate_hz is None
    assert result.channels is None
    assert any("FFmpeg" in note for note in result.quality_notes)


def test_missing_voice_dependencies_are_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        voice_service,
        "_import_audio_stack",
        lambda: (_ for _ in ()).throw(VoiceAnalysisUnavailable()),
    )

    with pytest.raises(VoiceAnalysisUnavailable):
        analyze_audio(Path("unused.wav"), "transcript")


# ---------------------------------------------------------------------------
# Content API
# ---------------------------------------------------------------------------


def test_content_analysis_reuses_phase2_coverage_and_keeps_scores(client, monkeypatch):
    reference = (
        "Inheritance lets a child class reuse and extend the methods of a "
        "parent class."
    )
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer_id = _create_answer(interview_id, question_id, reference)
    _patch_semantic(monkeypatch, 80.0)

    evaluation_response = client.post(f"/evaluation/{answer_id}")
    assert evaluation_response.status_code == 201, evaluation_response.text
    evaluation = evaluation_response.json()["evaluation"]
    assert evaluation["status"] == "evaluated"

    created = client.post(f"/analysis/{answer_id}/content")
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["reused"] is False
    analysis = body["content_analysis"]

    assert analysis["coverage_source"] == "phase2_evaluation"
    assert analysis["coverage_score"] == evaluation["keyword_score"]
    assert analysis["matched_concepts"] == evaluation["matched_keywords"]
    assert analysis["missing_concepts"] == evaluation["missing_keywords"]
    assert analysis["status"] == "completed"
    assert analysis["is_stale"] is False
    assert analysis["methodology_version"] == "phase3-v1"

    # Phase 2 evaluation is unchanged by Phase 3.
    unchanged = client.get(f"/evaluation/{answer_id}")
    assert unchanged.status_code == 200
    assert unchanged.json() == evaluation

    repeat = client.post(f"/analysis/{answer_id}/content")
    assert repeat.status_code == 200
    assert repeat.json()["reused"] is True
    assert repeat.json()["content_analysis"]["id"] == analysis["id"]
    assert _analysis_count(ContentAnalysis, answer_id) == 1

    fetched = client.get(f"/analysis/{answer_id}/content")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == analysis["id"]


def test_content_analysis_without_phase2_uses_keyword_coverage(client):
    reference = "TCP is connection-oriented and reliable. UDP is connectionless."
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "TCP is reliable and connection-oriented.",
    )

    response = client.post(f"/analysis/{answer_id}/content")
    assert response.status_code == 201, response.text
    analysis = response.json()["content_analysis"]
    assert analysis["coverage_source"] == "keyword_coverage"
    assert analysis["coverage_score"] is not None
    assert analysis["status"] == "completed"


def test_content_analysis_without_reference_persists_explicit_status(client):
    question_id = _create_question(None)
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "um well basically inheritance lets a child class reuse methods.",
    )

    response = client.post(f"/analysis/{answer_id}/content")
    assert response.status_code == 201, response.text
    analysis = response.json()["content_analysis"]
    assert analysis["status"] == "not_evaluable"
    assert analysis["status_detail"]
    assert analysis["coverage_score"] is None
    assert analysis["coverage_source"] == "none"
    assert analysis["filler_word_count"] == 3
    assert analysis["sentence_count"] == 1

    assert client.get(f"/analysis/{answer_id}/content").status_code == 200


def test_content_analysis_is_marked_stale_and_updated_in_place(client):
    reference = (
        "Inheritance lets a child class reuse the methods of a parent class."
    )
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer_id = _create_answer(interview_id, question_id, reference)

    created = client.post(f"/analysis/{answer_id}/content")
    assert created.status_code == 201
    analysis_id = created.json()["content_analysis"]["id"]

    _set_transcript(
        answer_id,
        "TCP is reliable and UDP is connectionless. TCP also preserves order.",
    )
    stale = client.get(f"/analysis/{answer_id}/content")
    assert stale.status_code == 200
    assert stale.json()["is_stale"] is True

    refreshed = client.post(f"/analysis/{answer_id}/content")
    assert refreshed.status_code == 200
    assert refreshed.json()["reused"] is False
    refreshed_analysis = refreshed.json()["content_analysis"]
    assert refreshed_analysis["id"] == analysis_id
    assert refreshed_analysis["is_stale"] is False
    assert refreshed_analysis["coverage_source"] == "keyword_coverage"

    _set_reference(
        question_id,
        "A primary key uniquely identifies each row in a database table.",
    )
    assert client.get(f"/analysis/{answer_id}/content").json()["is_stale"] is True

    again = client.post(f"/analysis/{answer_id}/content")
    assert again.status_code == 200
    assert again.json()["content_analysis"]["id"] == analysis_id
    assert _analysis_count(ContentAnalysis, answer_id) == 1


# ---------------------------------------------------------------------------
# Voice API
# ---------------------------------------------------------------------------


def test_transcription_persists_audio_and_voice_analysis_succeeds(
    client,
    monkeypatch,
):
    pytest.importorskip("librosa")
    reference = "TCP is a reliable transport protocol."
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    transcript = " ".join(["protocol"] * 20)

    answer = _transcribe(
        client,
        monkeypatch,
        interview_id,
        question_id,
        _speech_wav(),
        transcript,
    )
    answer_id = answer["id"]
    assert answer["has_audio"] is True

    with SessionLocal() as db:
        saved = db.get(InterviewAnswer, answer_id)
        assert saved is not None
        assert saved.audio_path
        assert saved.audio_sha256 == sha256_bytes(_speech_wav())
        assert resolve_audio_path(saved.audio_path).is_file()

    created = client.post(f"/analysis/{answer_id}/voice")
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["reused"] is False
    analysis = body["voice_analysis"]

    assert analysis["status"] == "completed"
    assert analysis["speech_detected"] is True
    assert analysis["word_count"] == len(tokenize(transcript))
    assert analysis["speaking_rate_wpm"] is not None
    assert analysis["speaking_rate_wpm"] > 0
    assert analysis["duration_seconds"] is not None
    assert analysis["pause_count"] is not None and analysis["pause_count"] >= 2
    assert analysis["is_stale"] is False
    assert analysis["methodology_version"] == "phase3-v1"

    repeat = client.post(f"/analysis/{answer_id}/voice")
    assert repeat.status_code == 200
    assert repeat.json()["reused"] is True
    assert repeat.json()["voice_analysis"]["id"] == analysis["id"]

    fetched = client.get(f"/analysis/{answer_id}/voice")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == analysis["id"]
    assert _analysis_count(VoiceAnalysis, answer_id) == 1


def test_voice_analysis_accepts_mp3(client, monkeypatch):
    pytest.importorskip("librosa")
    question_id = _create_question("Explain transport protocols.")
    interview_id = _start_interview(client)
    mp3 = _to_mp3(_speech_wav())

    answer = _transcribe(
        client,
        monkeypatch,
        interview_id,
        question_id,
        mp3,
        " ".join(["word"] * 12),
        filename="answer.mp3",
        content_type="audio/mpeg",
    )

    response = client.post(f"/analysis/{answer['id']}/voice")
    assert response.status_code == 201, response.text
    analysis = response.json()["voice_analysis"]
    assert analysis["status"] == "completed"
    assert analysis["speech_detected"] is True
    assert analysis["speaking_rate_wpm"] is not None


def test_voice_analysis_without_stored_audio_is_unavailable(client):
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "Inheritance lets a child class reuse parent methods.",
    )

    response = client.post(f"/analysis/{answer_id}/voice")
    assert response.status_code == 201, response.text
    analysis = response.json()["voice_analysis"]

    assert analysis["status"] == "unavailable"
    assert analysis["status_detail"]
    assert analysis["speech_detected"] is False
    assert analysis["duration_seconds"] is None
    assert analysis["speaking_rate_wpm"] is None
    assert analysis["pitch_median_hz"] is None
    assert analysis["quality_notes"] == []

    repeat = client.post(f"/analysis/{answer_id}/voice")
    assert repeat.status_code == 200
    assert repeat.json()["reused"] is True
    assert repeat.json()["voice_analysis"]["id"] == analysis["id"]


def test_voice_analysis_corrupt_audio_is_failed_not_a_crash(client):
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "Inheritance lets a child class reuse parent methods.",
    )
    _attach_audio(answer_id, b"not really audio" * 30)

    response = client.post(f"/analysis/{answer_id}/voice")
    assert response.status_code == 201, response.text
    analysis = response.json()["voice_analysis"]

    assert analysis["status"] == "failed"
    assert analysis["status_detail"]
    assert analysis["speech_detected"] is False
    assert analysis["speaking_rate_wpm"] is None
    assert analysis["pitch_reliable"] is False


def test_voice_analysis_detects_audio_fingerprint_mismatch(client):
    pytest.importorskip("librosa")
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "Inheritance lets a child class reuse parent methods.",
    )
    _attach_audio(answer_id, _speech_wav())

    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        answer.audio_sha256 = "0" * 64
        db.commit()

    response = client.post(f"/analysis/{answer_id}/voice")
    assert response.status_code == 201, response.text
    analysis = response.json()["voice_analysis"]
    assert analysis["status"] == "failed"
    assert "fingerprint" in analysis["status_detail"].lower()
    assert analysis["speaking_rate_wpm"] is None


def test_voice_analysis_missing_file_is_unavailable(client):
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "Inheritance lets a child class reuse parent methods.",
    )
    relative_path = _attach_audio(answer_id, _speech_wav())
    resolve_audio_path(relative_path).unlink()

    response = client.post(f"/analysis/{answer_id}/voice")
    assert response.status_code == 201, response.text
    analysis = response.json()["voice_analysis"]
    assert analysis["status"] == "unavailable"
    assert analysis["speaking_rate_wpm"] is None


def test_voice_analysis_is_marked_stale_after_transcript_change(client):
    pytest.importorskip("librosa")
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        " ".join(["word"] * 10),
    )
    _attach_audio(answer_id, _speech_wav())

    created = client.post(f"/analysis/{answer_id}/voice")
    assert created.status_code == 201
    analysis_id = created.json()["voice_analysis"]["id"]
    first_wpm = created.json()["voice_analysis"]["speaking_rate_wpm"]

    _set_transcript(answer_id, " ".join(["word"] * 20))
    stale = client.get(f"/analysis/{answer_id}/voice")
    assert stale.status_code == 200
    assert stale.json()["is_stale"] is True

    refreshed = client.post(f"/analysis/{answer_id}/voice")
    assert refreshed.status_code == 200
    assert refreshed.json()["reused"] is False
    updated = refreshed.json()["voice_analysis"]
    assert updated["id"] == analysis_id
    assert updated["is_stale"] is False
    assert updated["word_count"] == 20
    assert updated["speaking_rate_wpm"] > first_wpm
    assert _analysis_count(VoiceAnalysis, answer_id) == 1


def test_voice_missing_dependencies_returns_503_without_fake_metrics(
    client,
    monkeypatch,
):
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)
    answer_id = _create_answer(
        interview_id,
        question_id,
        "Inheritance lets a child class reuse parent methods.",
    )
    _attach_audio(answer_id, _speech_wav())

    def unavailable(_path, _transcript):
        raise VoiceAnalysisUnavailable()

    monkeypatch.setattr(
        "routes.analysis.analyze_audio",
        unavailable,
    )

    response = client.post(f"/analysis/{answer_id}/voice")
    assert response.status_code == 503, response.text
    assert "voice analysis" in response.json()["detail"].lower()
    assert client.get(f"/analysis/{answer_id}/voice").status_code == 404
    assert _analysis_count(VoiceAnalysis, answer_id) == 0


def test_transcription_survives_audio_storage_failure(client, monkeypatch):
    question_id = _create_question("Explain inheritance.")
    interview_id = _start_interview(client)

    def storage_failure(*_args, **_kwargs):
        raise OSError("disk is full")

    monkeypatch.setattr(
        interview_routes,
        "store_audio",
        storage_failure,
    )

    answer = _transcribe(
        client,
        monkeypatch,
        interview_id,
        question_id,
        b"fake-webm-audio" * 20,
        "Inheritance lets a child class reuse parent methods.",
        filename="answer.webm",
        content_type="audio/webm",
    )
    assert answer["has_audio"] is False

    voice = client.post(f"/analysis/{answer['id']}/voice")
    assert voice.status_code == 201
    assert voice.json()["voice_analysis"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Combined endpoint and safety
# ---------------------------------------------------------------------------


def test_interview_analyses_returns_both_types(client, monkeypatch):
    pytest.importorskip("librosa")
    reference = "Inheritance lets a child class reuse parent methods."
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer = _transcribe(
        client,
        monkeypatch,
        interview_id,
        question_id,
        _speech_wav(),
        " ".join(["word"] * 10),
    )
    answer_id = answer["id"]

    assert client.post(f"/analysis/{answer_id}/content").status_code == 201
    assert client.post(f"/analysis/{answer_id}/voice").status_code == 201

    combined = client.get(f"/interview/{interview_id}/analyses")
    assert combined.status_code == 200, combined.text
    body = combined.json()
    assert body["interview_id"] == interview_id
    assert len(body["content_analyses"]) == 1
    assert len(body["voice_analyses"]) == 1
    assert body["content_analyses"][0]["answer_id"] == answer_id
    assert body["voice_analyses"][0]["answer_id"] == answer_id
    assert body["content_analyses"][0]["is_stale"] is False
    assert body["voice_analyses"][0]["is_stale"] is False

    other_interview = _start_interview(client)
    empty = client.get(f"/interview/{other_interview}/analyses")
    assert empty.status_code == 200
    assert empty.json()["content_analyses"] == []
    assert empty.json()["voice_analyses"] == []


def test_missing_ids_and_invalid_ids_are_safe(client):
    assert client.post("/analysis/99999999/content").status_code == 404
    assert client.get("/analysis/99999999/content").status_code == 404
    assert client.post("/analysis/99999999/voice").status_code == 404
    assert client.get("/analysis/99999999/voice").status_code == 404
    assert client.get("/interview/99999999/analyses").status_code == 404

    assert client.post("/analysis/not-an-id/content").status_code == 422
    assert client.get("/analysis/not-an-id/voice").status_code == 422


def test_health_reports_phase3_analysis_settings(client):
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_methodology_version"] == "phase3-v1"
    assert isinstance(body["voice_analysis_available"], bool)


def test_phase2_evaluation_survives_content_and_voice_analysis(client, monkeypatch):
    pytest.importorskip("librosa")
    reference = (
        "Inheritance lets a child class reuse and extend the methods of a "
        "parent class."
    )
    question_id = _create_question(reference)
    interview_id = _start_interview(client)
    answer = _transcribe(
        client,
        monkeypatch,
        interview_id,
        question_id,
        _speech_wav(),
        reference,
    )
    answer_id = answer["id"]
    _patch_semantic(monkeypatch, 75.0)

    evaluation = client.post(f"/evaluation/{answer_id}")
    assert evaluation.status_code == 201
    evaluation_body = evaluation.json()["evaluation"]

    assert client.post(f"/analysis/{answer_id}/content").status_code == 201
    assert client.post(f"/analysis/{answer_id}/voice").status_code == 201

    with SessionLocal() as db:
        rows = db.scalars(
            select(AnswerEvaluation).where(
                AnswerEvaluation.answer_id == answer_id
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].id == evaluation_body["id"]
        assert rows[0].combined_score == evaluation_body["combined_score"]
        assert rows[0].semantic_score == 75.0
        assert rows[0].keyword_score == 100.0

    assert client.get(f"/evaluation/{answer_id}").json() == evaluation_body
