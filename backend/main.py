"""FastAPI application entry point for the AI Interview Evaluator."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from config import (
    ANALYSIS_METHODOLOGY_VERSION,
    CORS_ORIGINS,
    DATABASE_URL_SOURCE,
    ENV_FILE_DISPLAY_PATH,
    ENV_FILE_EXISTS,
    FEEDBACK_METHODOLOGY_VERSION,
    GROQ_API_KEY,
    GROQ_API_KEY_SOURCE,
    GROQ_MODEL,
    MAX_AUDIO_MB,
    ML_FEATURE_SCHEMA_VERSION,
    ML_PREDICTION_METHODOLOGY_VERSION,
    is_groq_configured,
)
from database import (
    SessionLocal,
    describe_database_error,
    describe_database_target,
    ensure_db_ready,
)
from routes.analysis import router as analysis_router
from routes.evaluation import router as evaluation_router
from routes.feedback import router as feedback_router
from routes.interview import router as interview_router
from routes.prediction import router as prediction_router
from schemas import HealthResponse
from services.ml_scoring import get_model_status
from services.voice_analysis import voice_analysis_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_interview_evaluator")

DATABASE_TARGET = describe_database_target()

TRANSCRIPTION_HINT = (
    "Create a free key at https://console.groq.com/keys, add your key as "
    "GROQ_API_KEY in backend/.env, then restart FastAPI."
)


def _health_payload(
    database: str,
    error: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one consistent, non-secret health payload for success and failure."""
    configured = is_groq_configured(GROQ_API_KEY)
    model_status = get_model_status()
    payload: dict[str, object] = {
        "status": "ok" if database == "connected" else "error",
        "database": database,
        **DATABASE_TARGET,
        "database_error_category": None,
        "database_hint": None,
        "transcription_configured": configured,
        "transcription_model": GROQ_MODEL,
        "transcription_provider": "groq",
        "transcription_hint": None if configured else TRANSCRIPTION_HINT,
        "max_audio_mb": MAX_AUDIO_MB,
        "env_file": ENV_FILE_DISPLAY_PATH,
        "env_file_found": ENV_FILE_EXISTS,
        "database_url_source": DATABASE_URL_SOURCE,
        "groq_api_key_source": GROQ_API_KEY_SOURCE,
        "analysis_methodology_version": ANALYSIS_METHODOLOGY_VERSION,
        "voice_analysis_available": voice_analysis_available(),
        "prediction_methodology_version": ML_PREDICTION_METHODOLOGY_VERSION,
        "feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "ml_model_available": model_status.available,
        "ml_model_state": model_status.state,
        "ml_model_version": model_status.model_version,
        "ml_model_validation_status": model_status.validation_status,
        "ml_model_reason": model_status.reason,
        "feedback_methodology_version": FEEDBACK_METHODOLOGY_VERSION,
    }
    if error:
        payload.update(error)
    return payload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Do not crash the whole API if MySQL is temporarily unavailable. Health
    # checks and request handlers report a clear 503 instead, and the database
    # is initialized again on the next request.
    try:
        await run_in_threadpool(ensure_db_ready)
        logger.info("Database initialized and interview questions are ready.")
    except Exception as exc:  # noqa: BLE001 - startup must stay diagnostic-safe
        category = describe_database_error(exc)["database_error_category"]
        logger.warning(
            "Database initialization failed (%s). Check DATABASE_URL and that "
            "MySQL is running; endpoints will retry on the next request.",
            category,
        )
    yield


app = FastAPI(
    title="AI Interview Evaluator",
    description=(
        "Phase 5: interview questions, audio transcription, saved answers, "
        "explainable evaluation, content plus voice measurements, versioned "
        "ML/baseline score predictions, and deterministic evidence-based "
        "per-question plus interview feedback."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview_router)
app.include_router(evaluation_router)
app.include_router(analysis_router)
app.include_router(prediction_router)
app.include_router(feedback_router)


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(
    request: Request,
    _exc: SQLAlchemyError,
) -> JSONResponse:
    # Never return the raw driver error: it can contain host/user details.
    logger.error(
        "Database error while handling %s %s.",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The database is unavailable. Check DATABASE_URL in "
                "backend/.env and make sure the MySQL server is running."
            )
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "AI Interview Evaluator API is running",
        "phase": "5",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        ensure_db_ready()
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - report a safe, actionable 503
        diagnostics = describe_database_error(exc)
        logger.warning(
            "Health check failed: database unavailable (%s).",
            diagnostics["database_error_category"],
        )
        return JSONResponse(
            status_code=503,
            content=_health_payload("unavailable", diagnostics),
        )

    return HealthResponse(**_health_payload("connected"))
