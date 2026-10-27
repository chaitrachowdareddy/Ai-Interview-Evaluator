"""Phase 4 tests: feature schema, model artifacts, baseline, API, training.

These tests use the project's isolated SQLite database and temp model artifacts.
No real labelled dataset, MySQL server, Groq key, or microphone is required.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import uuid
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import func, select

import config
import services.evaluation as evaluation_service
import services.ml_scoring as ml_scoring
from config import (
    ANALYSIS_METHODOLOGY_VERSION,
    EVALUATION_METHODOLOGY_VERSION,
    ML_FEATURE_SCHEMA_VERSION,
    ML_PREDICTION_METHODOLOGY_VERSION,
)
from database import SessionLocal
from models import (
    AnswerEvaluation,
    InterviewAnswer,
    InterviewQuestion,
    ScorePrediction,
    VoiceAnalysis,
)
from services.evaluation import fingerprint
from services.ml_features import (
    FEATURE_NAMES,
    NO_AUDIO_FINGERPRINT,
    FeatureValidationError,
    validate_feature_dict,
    validate_feature_value,
)
from services.ml_scoring import (
    METHOD_BASELINE,
    METHOD_ML,
    STATUS_COMPLETED,
)
from services.ml_training import (
    TrainingDataError,
    TrainingRow,
    load_training_dataset,
    split_rows,
    train_from_file,
)

REFERENCE = (
    "Inheritance lets a derived class reuse and override behavior from a base "
    "class, modelling an is-a relationship."
)
TRANSCRIPT = (
    "Inheritance allows a derived or child class to reuse behavior from a base "
    "class and override inherited methods when needed."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BrokenModel:
    """A bundle estimator whose prediction always fails."""

    def predict(self, _values):
        raise RuntimeError("deliberate inference failure")


@pytest.fixture(autouse=True)
def isolated_model_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config,
        "ML_MODEL_PATH",
        tmp_path / "no-installed-model.joblib",
    )
    ml_scoring.clear_model_cache()
    yield
    ml_scoring.clear_model_cache()


def _start_interview(client) -> int:
    response = client.post("/interview/start")
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_question(reference_answer: str | None = REFERENCE) -> int:
    with SessionLocal() as db:
        question = InterviewQuestion(
            question_text=f"Phase 4 test question {uuid.uuid4()}",
            category="Phase 4 Test",
            reference_answer=reference_answer,
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        return question.id


def _create_answer(
    interview_id: int,
    question_id: int,
    transcript: str = TRANSCRIPT,
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


def _new_answer(client, transcript: str = TRANSCRIPT) -> tuple[int, int, int]:
    interview_id = _start_interview(client)
    question_id = _create_question()
    answer_id = _create_answer(interview_id, question_id, transcript)
    return interview_id, question_id, answer_id


def _set_transcript(answer_id: int, transcript: str) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        answer.transcript = transcript
        db.commit()


def _set_audio_hash(answer_id: int, value: str) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        answer.audio_sha256 = value
        db.commit()


def _patch_semantic(monkeypatch, score: float) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "semantic_similarity",
        lambda _candidate, _reference: score,
    )


def _evaluate(client, monkeypatch, answer_id: int, score: float = 82.0) -> dict:
    _patch_semantic(monkeypatch, score)
    response = client.post(f"/evaluation/{answer_id}")
    assert response.status_code in {200, 201}, response.text
    return response.json()["evaluation"]


def _content(client, answer_id: int) -> dict:
    response = client.post(f"/analysis/{answer_id}/content")
    assert response.status_code in {200, 201}, response.text
    return response.json()["content_analysis"]


def _voice_row(
    answer_id: int,
    *,
    status: str = "completed",
    speech_detected: bool = True,
    audio_sha256: str | None = None,
    speaking_rate_wpm: float | None = 142.0,
    pauses_per_minute: float | None = 3.0,
    speech_ratio_percent: float | None = 78.0,
    rms_variation_percent: float | None = 32.0,
) -> None:
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        assert answer is not None
        existing = db.scalar(
            select(VoiceAnalysis).where(
                VoiceAnalysis.answer_id == answer_id,
                VoiceAnalysis.methodology_version
                == ANALYSIS_METHODOLOGY_VERSION,
            )
        )
        if existing is None:
            existing = VoiceAnalysis(
                answer_id=answer.id,
                interview_id=answer.interview_id,
                question_id=answer.question_id,
                methodology_version=ANALYSIS_METHODOLOGY_VERSION,
            )
            db.add(existing)
        existing.status = status
        existing.status_detail = (
            None if status == "completed" else "Synthetic test status."
        )
        existing.speech_detected = speech_detected
        existing.speaking_rate_wpm = speaking_rate_wpm
        existing.pauses_per_minute = pauses_per_minute
        existing.speech_ratio_percent = speech_ratio_percent
        existing.rms_variation_percent = rms_variation_percent
        existing.audio_sha256 = audio_sha256 or (
            answer.audio_sha256 or NO_AUDIO_FINGERPRINT
        )
        existing.transcript_hash = fingerprint(answer.transcript)
        db.commit()


def _bundle(model, **overrides):
    bundle = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "methodology_version": ML_PREDICTION_METHODOLOGY_VERSION,
        "model_version": "test-rf-1",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "sklearn_version": __import__("sklearn").__version__,
        "validation_status": "validated",
        "metrics": {"test": {"mae": 4.2, "rmse": 5.1, "r2": 0.72}},
        "limitations": ["Synthetic test artifact."],
        "target_min": 0.0,
        "target_max": 100.0,
    }
    bundle.update(overrides)
    return bundle


def _write_bundle(path: Path, model=None, **overrides) -> Path:
    if model is None:
        model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
            np.zeros((4, len(FEATURE_NAMES))),
            [40.0, 50.0, 60.0, 70.0],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_bundle(model, **overrides), path)
    ml_scoring.clear_model_cache()
    return path


def _prediction(client, answer_id: int):
    return client.post(f"/prediction/{answer_id}")


def _count_predictions(answer_id: int) -> int:
    with SessionLocal() as db:
        return int(
            db.scalar(
                select(func.count())
                .select_from(ScorePrediction)
                .where(ScorePrediction.answer_id == answer_id)
            )
            or 0
        )


def _training_rows(
    count: int,
    groups: int,
    *,
    correlated: bool = False,
) -> list[TrainingRow]:
    rng = random.Random(11)
    rows: list[TrainingRow] = []
    for index in range(count):
        semantic = rng.uniform(20.0, 95.0)
        keyword = rng.uniform(10.0, 90.0)
        target = (
            0.55 * semantic + 0.35 * keyword + rng.uniform(-3.0, 3.0)
            if correlated
            else rng.uniform(20.0, 90.0)
        )
        rows.append(
            TrainingRow(
                sample_id=f"sample-{index}",
                group_id=f"group-{index % groups}",
                target=round(max(0.0, min(100.0, target)), 2),
                features={
                    "semantic_score": round(semantic, 2),
                    "keyword_score": round(keyword, 2),
                    "filler_rate_percent": round(rng.uniform(0.0, 12.0), 2),
                    "repetition_rate_percent": round(rng.uniform(0.0, 25.0), 2),
                    "average_sentence_words": round(rng.uniform(6.0, 30.0), 2),
                    "speaking_rate_wpm": round(rng.uniform(70.0, 210.0), 2),
                    "pauses_per_minute": round(rng.uniform(0.0, 18.0), 2),
                    "speech_ratio_percent": round(rng.uniform(20.0, 100.0), 2),
                    "rms_variation_percent": round(rng.uniform(5.0, 70.0), 2),
                    "has_content_features": 1.0,
                    "has_voice_features": 1.0,
                },
            )
        )
    return rows


def _write_csv(path: Path, rows: list[TrainingRow], *, columns=None) -> Path:
    fieldnames = columns or ["group_id", "target_score", *FEATURE_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = {
                "group_id": row.group_id,
                "target_score": row.target,
                **row.features,
            }
            writer.writerow({name: values.get(name, "") for name in fieldnames})
    return path


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


def test_feature_schema_is_versioned_and_validated(client):
    response = client.get("/prediction/schema")
    assert response.status_code == 200, response.text
    body = response.json()

    schema = body["feature_schema"]
    assert schema["feature_schema_version"] == ML_FEATURE_SCHEMA_VERSION
    assert [item["name"] for item in schema["features"]] == list(FEATURE_NAMES)
    assert schema["features"][0]["name"] == "semantic_score"
    assert schema["features"][0]["required"] is True
    assert schema["features"][1]["name"] == "keyword_score"
    assert any(
        item["name"] == "pitch" for item in schema["excluded_by_design"]
    )
    assert body["model_status"]["available"] is False
    assert body["baseline_methodology"]["method"] == METHOD_BASELINE

    complete = {name: None for name in FEATURE_NAMES}
    complete["semantic_score"] = 70.0
    complete["keyword_score"] = 60.0
    validated = validate_feature_dict(complete)
    assert validated["speaking_rate_wpm"] is None

    with pytest.raises(FeatureValidationError):
        validate_feature_dict({**complete, "unknown": 1.0})
    with pytest.raises(FeatureValidationError):
        validate_feature_dict({"semantic_score": 50.0})
    with pytest.raises(FeatureValidationError):
        validate_feature_value("semantic_score", 101.0)
    with pytest.raises(FeatureValidationError):
        validate_feature_value("keyword_score", None)


# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------


def test_model_artifact_states(tmp_path):
    assert ml_scoring.get_model_status().state == "unavailable"

    invalid = tmp_path / "invalid.joblib"
    invalid.write_bytes(b"this is not a joblib artifact")
    config.ML_MODEL_PATH = invalid
    ml_scoring.clear_model_cache()
    assert ml_scoring.get_model_status().state == "invalid"

    incompatible = _write_bundle(
        tmp_path / "order.joblib",
        feature_names=list(reversed(FEATURE_NAMES)),
    )
    config.ML_MODEL_PATH = incompatible
    ml_scoring.clear_model_cache()
    status = ml_scoring.get_model_status()
    assert status.available is False
    assert status.state == "incompatible"

    wrong_schema = _write_bundle(
        tmp_path / "schema.joblib",
        feature_schema_version="other-schema",
    )
    config.ML_MODEL_PATH = wrong_schema
    ml_scoring.clear_model_cache()
    assert ml_scoring.get_model_status().state == "incompatible"

    wrong_version = _write_bundle(
        tmp_path / "version.joblib",
        sklearn_version="0.0.1",
    )
    config.ML_MODEL_PATH = wrong_version
    ml_scoring.clear_model_cache()
    assert ml_scoring.get_model_status().state == "incompatible"

    unvalidated = _write_bundle(
        tmp_path / "unvalidated.joblib",
        validation_status="unvalidated",
    )
    config.ML_MODEL_PATH = unvalidated
    ml_scoring.clear_model_cache()
    status = ml_scoring.get_model_status()
    assert status.available is False
    assert status.state == "unvalidated"

    valid = _write_bundle(tmp_path / "valid.joblib")
    config.ML_MODEL_PATH = valid
    ml_scoring.clear_model_cache()
    status = ml_scoring.get_model_status()
    assert status.available is True
    assert status.model_version == "test-rf-1"
    assert status.validation_status == "validated"
    assert status.artifact_name == "valid.joblib"


# ---------------------------------------------------------------------------
# Baseline and missing data
# ---------------------------------------------------------------------------


def test_baseline_prediction_is_labelled_and_never_zero_fills_voice(
    client,
    monkeypatch,
):
    interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id, score=82.0)
    _content(client, answer_id)

    first = _prediction(client, answer_id)
    assert first.status_code == 201, first.text
    body = first.json()
    prediction = body["prediction"]

    assert prediction["status"] == STATUS_COMPLETED
    assert prediction["method"] == METHOD_BASELINE
    assert prediction["method_label"] == "Rule-based baseline"
    assert prediction["is_ml_prediction"] is False
    assert prediction["score"] is not None
    assert 0.0 <= prediction["score"] <= 100.0
    assert prediction["features"]["speaking_rate_wpm"] is None
    assert prediction["features"]["has_voice_features"] == 0.0
    assert prediction["features"]["has_content_features"] == 1.0
    assert "speaking_rate_wpm" in prediction["missing_features"]
    names = [item["name"] for item in prediction["components"]["components"]]
    assert "answer" in names
    assert "content_style" in names
    assert "delivery" not in names
    assert any("Voice analysis is missing" in item for item in prediction["warnings"])
    assert any(
        "not a machine-learning prediction" in item
        for item in prediction["limitations"]
    )
    assert body["model_status"]["available"] is False

    second = _prediction(client, answer_id)
    assert second.status_code == 200, second.text
    assert second.json()["reused"] is True
    assert _count_predictions(answer_id) == 1

    saved = client.get(f"/prediction/{answer_id}")
    assert saved.status_code == 200
    assert saved.json()["score"] == prediction["score"]
    assert saved.json()["is_stale"] is False

    listed = client.get(f"/interview/{interview_id}/predictions")
    assert listed.status_code == 200
    assert len(listed.json()["predictions"]) == 1

    summary = client.get(f"/interview/{interview_id}/summary")
    assert summary.status_code == 200
    summary_body = summary.json()
    assert summary_body["predictions_completed"] == 1
    assert summary_body["predictions_baseline"] == 1
    assert summary_body["predictions_ml"] == 0
    assert summary_body["predictions_stale"] == 0
    assert summary_body["prediction_method"] == METHOD_BASELINE
    assert summary_body["average_predicted_score"] == prediction["score"]
    assert summary_body["ml_model_available"] is False
    assert summary_body["prediction_methodology_version"] == (
        ML_PREDICTION_METHODOLOGY_VERSION
    )
    question = next(
        item
        for item in summary_body["questions"]
        if item["answer_id"] == answer_id
    )
    assert question["predicted_score"] == prediction["score"]
    assert question["prediction_status"] == STATUS_COMPLETED
    assert question["prediction_is_stale"] is False


def test_unreliable_voice_is_excluded_then_included_and_stale(
    client,
    monkeypatch,
):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id)
    _content(client, answer_id)
    _voice_row(
        answer_id,
        status="insufficient_speech",
        speech_detected=False,
        speaking_rate_wpm=140.0,
    )

    first = _prediction(client, answer_id).json()["prediction"]
    assert first["features"]["has_voice_features"] == 0.0
    assert first["features"]["speaking_rate_wpm"] is None
    assert any("insufficient_speech" in item for item in first["warnings"])

    _voice_row(answer_id, status="completed", speech_detected=True)
    second = _prediction(client, answer_id).json()["prediction"]
    assert second["features"]["has_voice_features"] == 1.0
    assert second["features"]["speaking_rate_wpm"] == 142.0
    components = [item["name"] for item in second["components"]["components"]]
    assert "delivery" in components

    _set_audio_hash(answer_id, hashlib.sha256(b"new-recording").hexdigest())
    third = _prediction(client, answer_id).json()["prediction"]
    assert third["features"]["has_voice_features"] == 0.0
    assert third["features"]["speaking_rate_wpm"] is None
    assert any("Voice analysis is stale" in item for item in third["warnings"])
    assert _count_predictions(answer_id) == 1


def test_missing_and_stale_phase2_sources_are_not_scored(client, monkeypatch):
    _interview_id, _question_id, answer_id = _new_answer(client)

    missing = _prediction(client, answer_id)
    assert missing.status_code == 201
    assert missing.json()["prediction"]["status"] == "unavailable"
    assert missing.json()["prediction"]["method"] == "none"
    assert missing.json()["prediction"]["score"] is None
    assert "Phase 2 evaluation" in missing.json()["prediction"]["status_detail"]

    _evaluate(client, monkeypatch, answer_id)
    completed = _prediction(client, answer_id).json()["prediction"]
    assert completed["status"] == STATUS_COMPLETED

    _set_transcript(answer_id, TRANSCRIPT + " This text changed afterwards.")
    stale = _prediction(client, answer_id).json()["prediction"]
    assert stale["status"] == "unavailable"
    assert stale["score"] is None
    assert "stale" in stale["status_detail"].lower()
    assert _count_predictions(answer_id) == 1


def test_stale_content_is_excluded_without_invalidating_evaluation(
    client,
    monkeypatch,
):
    _interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id, score=75.0)
    _content(client, answer_id)

    _set_transcript(
        answer_id,
        "A derived class can reuse and override base-class behavior through "
        "inheritance in object-oriented programming.",
    )
    # Refresh only the Phase 2 evaluation. Its hash is current; the saved content
    # analysis still describes the old transcript.
    _evaluate(client, monkeypatch, answer_id, score=88.0)

    prediction = _prediction(client, answer_id).json()["prediction"]
    assert prediction["status"] == STATUS_COMPLETED
    assert prediction["features"]["has_content_features"] == 0.0
    assert prediction["features"]["filler_rate_percent"] is None
    assert any("Content analysis is stale" in item for item in prediction["warnings"])
    components = [item["name"] for item in prediction["components"]["components"]]
    assert "content_style" not in components


# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------


def test_valid_ml_artifact_is_used_and_score_is_bounded(
    client,
    monkeypatch,
    tmp_path,
):
    _write_bundle(
        tmp_path / "ml.joblib",
        model=DummyRegressor(strategy="constant", constant=150.0).fit(
            np.zeros((2, len(FEATURE_NAMES))),
            [50.0, 50.0],
        ),
    )
    config.ML_MODEL_PATH = tmp_path / "ml.joblib"
    ml_scoring.clear_model_cache()

    interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id)
    _content(client, answer_id)

    body = _prediction(client, answer_id).json()
    prediction = body["prediction"]
    assert prediction["status"] == STATUS_COMPLETED
    assert prediction["method"] == METHOD_ML
    assert prediction["is_ml_prediction"] is True
    assert prediction["method_label"] == "ML prediction"
    assert prediction["score"] == 100.0
    assert prediction["model_version"] == "test-rf-1"
    assert prediction["components"]["model"]["validation_status"] == "validated"
    assert body["model_status"]["available"] is True

    summary = client.get(f"/interview/{interview_id}/summary").json()
    assert summary["predictions_ml"] == 1
    assert summary["predictions_baseline"] == 0
    assert summary["prediction_method"] == METHOD_ML
    assert summary["ml_model_available"] is True


def test_model_inference_failure_is_reported_not_replaced(
    client,
    monkeypatch,
    tmp_path,
):
    _write_bundle(tmp_path / "broken.joblib", model=BrokenModel())
    config.ML_MODEL_PATH = tmp_path / "broken.joblib"
    ml_scoring.clear_model_cache()

    _interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id)
    _content(client, answer_id)

    prediction = _prediction(client, answer_id).json()["prediction"]
    assert prediction["status"] == "failed"
    assert prediction["method"] == "none"
    assert prediction["score"] is None
    assert "could not score" in prediction["status_detail"]


# ---------------------------------------------------------------------------
# Rescoring, duplicates, stale summary
# ---------------------------------------------------------------------------


def test_rescoring_updates_one_row_and_stale_rows_leave_summary(
    client,
    monkeypatch,
):
    interview_id, _question_id, answer_id = _new_answer(client)
    _evaluate(client, monkeypatch, answer_id, score=60.0)
    _content(client, answer_id)
    _voice_row(answer_id)

    first = _prediction(client, answer_id).json()["prediction"]
    first_id = first["id"]

    _set_transcript(
        answer_id,
        "Inheritance lets a child class reuse and override inherited methods "
        "from a parent class, which is an is-a relationship.",
    )
    _evaluate(client, monkeypatch, answer_id, score=92.0)
    _content(client, answer_id)

    second = _prediction(client, answer_id).json()["prediction"]
    assert second["id"] == first_id
    assert second["score"] != first["score"]
    assert _count_predictions(answer_id) == 1

    _set_transcript(answer_id, "The transcript changed once more.")
    stale = client.get(f"/prediction/{answer_id}").json()
    assert stale["is_stale"] is True

    summary = client.get(f"/interview/{interview_id}/summary").json()
    assert summary["predictions_completed"] == 0
    assert summary["predictions_stale"] == 1
    assert summary["average_predicted_score"] is None

    refreshed = _prediction(client, answer_id).json()["prediction"]
    assert refreshed["id"] == first_id
    assert refreshed["status"] == "unavailable"
    assert _count_predictions(answer_id) == 1


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------


def test_prediction_api_errors_are_actionable(client):
    assert _prediction(client, 999999).status_code == 404
    assert client.get("/prediction/999999").status_code == 404
    assert client.get("/interview/999999/predictions").status_code == 404

    _interview_id, _question_id, answer_id = _new_answer(client)
    missing = client.get(f"/prediction/{answer_id}")
    assert missing.status_code == 404
    assert "No prediction" in missing.json()["detail"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_group_split_has_no_leakage_and_dataset_validation(tmp_path):
    rows = _training_rows(24, 8, correlated=True)
    train, test = split_rows(rows, test_fraction=0.25, random_seed=42)
    assert all(row.group_id not in {item.group_id for item in test} for row in train)
    assert len(train) + len(test) == len(rows)

    with pytest.raises(TrainingDataError):
        split_rows(rows[:1], test_fraction=0.25, random_seed=42)

    good_path = _write_csv(tmp_path / "good.csv", rows)
    dataset = load_training_dataset(good_path)
    assert len(dataset.rows) == 24
    assert all(set(row.features) == set(FEATURE_NAMES) for row in dataset.rows)

    missing_column = _write_csv(
        tmp_path / "missing.csv",
        rows,
        columns=["group_id", "target_score", "semantic_score"],
    )
    with pytest.raises(TrainingDataError, match="missing required"):
        load_training_dataset(missing_column)

    unknown_column = _write_csv(
        tmp_path / "unknown.csv",
        rows,
        columns=["group_id", "target_score", *FEATURE_NAMES, "candidate_name"],
    )
    with pytest.raises(TrainingDataError, match="unrecognized"):
        load_training_dataset(unknown_column)

    duplicate_path = _write_csv(tmp_path / "duplicate.csv", rows + rows[:1])
    with pytest.raises(TrainingDataError, match="duplicate"):
        load_training_dataset(duplicate_path)

    bad_target_path = _write_csv(tmp_path / "bad-target.csv", rows)
    contents = bad_target_path.read_text(encoding="utf-8").splitlines()
    contents[1] = contents[1].replace(str(rows[0].target), "150.0")
    bad_target_path.write_text("\n".join(contents) + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="target_score"):
        load_training_dataset(bad_target_path)


def test_training_refuses_small_data_and_marks_it_unvalidated(tmp_path):
    rows = _training_rows(10, 4, correlated=True)
    data = _write_csv(tmp_path / "small.csv", rows)
    output = tmp_path / "small.joblib"

    with pytest.raises(TrainingDataError, match="at least 60"):
        train_from_file(data, output_path=output)

    report = train_from_file(
        data,
        output_path=output,
        allow_small=True,
        overwrite=True,
    )
    assert report["validation_status"] == "unvalidated"
    assert output.is_file()
    assert Path(report["metadata_path"]).is_file()

    config.ML_MODEL_PATH = output
    ml_scoring.clear_model_cache()
    status = ml_scoring.get_model_status()
    assert status.available is False
    assert status.state == "unvalidated"

    # Inference must use the baseline, never the unvalidated artifact.
    _interview_id, _question_id, answer_id = _new_answer_no_client()
    with SessionLocal() as db:
        answer = db.get(InterviewAnswer, answer_id)
        question = db.get(InterviewQuestion, answer.question_id)
        result = ml_scoring.compute_prediction(db, answer, question)
    assert result.method == METHOD_BASELINE
    assert result.score is not None


def test_training_produces_validated_artifact_with_supported_metrics(tmp_path):
    rows = _training_rows(60, 20, correlated=True)
    data = _write_csv(tmp_path / "labelled.csv", rows)
    output = tmp_path / "validated.joblib"

    report = train_from_file(
        data,
        output_path=output,
        overwrite=True,
        random_seed=7,
    )
    assert report["validation_status"] == "validated", report
    assert report["validation_reasons"] == []
    assert output.is_file()

    metrics = report["metrics"]
    assert metrics["test_samples"] + metrics["train_samples"] == 60
    assert metrics["train_groups"] + metrics["test_groups"] == 20
    for split in ("test", "train", "mean_baseline_test"):
        assert math.isfinite(metrics[split]["mae"])
        assert math.isfinite(metrics[split]["rmse"])
        assert metrics[split]["r2"] is not None
    assert metrics["improves_on_mean_baseline"] is True

    bundle = joblib.load(output)
    assert bundle["feature_names"] == list(FEATURE_NAMES)
    assert bundle["feature_schema_version"] == ML_FEATURE_SCHEMA_VERSION
    assert bundle["validation_status"] == "validated"
    assert bundle["target_min"] == 0.0
    assert bundle["target_max"] == 100.0
    assert bundle["model_version"] == report["model_version"]

    config.ML_MODEL_PATH = output
    ml_scoring.clear_model_cache()
    status = ml_scoring.get_model_status()
    assert status.available is True
    assert status.validation_status == "validated"


# ---------------------------------------------------------------------------
# Direct database helpers used by the training test (no FastAPI client needed)
# ---------------------------------------------------------------------------


def _new_answer_no_client() -> tuple[int, int, int]:
    from database import SessionLocal as _SessionLocal
    from models import InterviewSession

    with _SessionLocal() as db:
        session = InterviewSession(status="in_progress")
        db.add(session)
        db.commit()
        db.refresh(session)
        interview_id = session.id

        question = InterviewQuestion(
            question_text=f"Phase 4 direct question {uuid.uuid4()}",
            category="Phase 4 Test",
            reference_answer=REFERENCE,
        )
        db.add(question)
        db.commit()
        db.refresh(question)

        answer = InterviewAnswer(
            interview_id=interview_id,
            question_id=question.id,
            transcript=TRANSCRIPT,
        )
        db.add(answer)
        db.commit()
        db.refresh(answer)

        evaluation = AnswerEvaluation(
            answer_id=answer.id,
            interview_id=interview_id,
            question_id=question.id,
            methodology_version=EVALUATION_METHODOLOGY_VERSION,
            semantic_score=80.0,
            keyword_score=70.0,
            combined_score=77.0,
            matched_keywords=[],
            missing_keywords=[],
            status="evaluated",
            status_detail=None,
            transcript_hash=fingerprint(answer.transcript),
            reference_hash=fingerprint(question.reference_answer or ""),
        )
        db.add(evaluation)
        db.commit()
        return interview_id, question.id, answer.id
