"""Environment-driven configuration for the AI Interview Evaluator backend.

Values are read from actual environment variables first, then from
``backend/.env`` (without overriding variables that are already set).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ENV_FILE_DISPLAY_PATH = "backend/.env"

# Remember which variables already existed in the real process before the
# ``.env`` file is merged, and which ones the file defines. This lets the health
# endpoint explain where a setting came from without printing its value.
_PROCESS_ENV_KEYS = set(os.environ)
ENV_FILE_EXISTS = ENV_FILE.is_file()
try:
    ENV_FILE_KEYS = set(dotenv_values(ENV_FILE)) if ENV_FILE_EXISTS else set()
except Exception:  # noqa: BLE001 - diagnostics must never block startup
    ENV_FILE_KEYS = set()

# Load local development settings. ``override=False`` keeps real environment
# variables (for example, ones set in the current PowerShell session) intact.
load_dotenv(ENV_FILE, override=False)


def setting_source(name: str) -> str:
    """Return a safe description of where a setting came from."""
    if name in _PROCESS_ENV_KEYS:
        return "environment"
    if name in ENV_FILE_KEYS:
        return ENV_FILE_DISPLAY_PATH
    return "default"


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value >= minimum else default


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value >= 0 else default


# The default matches a common local MySQL installation. Credentials should be
# supplied through DATABASE_URL in backend/.env.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root@127.0.0.1:3306/ai_interview_evaluator",
).strip()
DATABASE_URL_SOURCE = setting_source("DATABASE_URL")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_KEY_SOURCE = setting_source("GROQ_API_KEY")

# Values copied from ``.env.example`` are treated as "not configured" so the UI
# does not report a working Groq setup before a real key is supplied.
_GROQ_PLACEHOLDER_KEYS = {
    "your_groq_api_key_here",
    "your-api-key",
    "your_api_key",
    "changeme",
    "replace_me",
}


def is_groq_configured(api_key: str | None = None) -> bool:
    """Return ``True`` only when a non-placeholder Groq API key is present."""
    value = (GROQ_API_KEY if api_key is None else api_key).strip()
    return bool(value) and value.lower() not in _GROQ_PLACEHOLDER_KEYS

# Groq currently supports whisper-large-v3-turbo and whisper-large-v3.
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo").strip()

GROQ_TIMEOUT_SECONDS = _read_int("GROQ_TIMEOUT_SECONDS", 60, minimum=5)

MAX_AUDIO_MB = _read_int("MAX_AUDIO_MB", 25, minimum=1)
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Phase 3 analysis settings
# ---------------------------------------------------------------------------
ANALYSIS_METHODOLOGY_VERSION = "phase3-v1"

# Original recordings are kept so voice analysis can measure the real audio
# instead of re-deriving it from the transcript. Relative paths resolve against
# the backend directory (the same place backend/.env lives).
_storage_setting = os.getenv("AUDIO_STORAGE_DIR", "uploads").strip() or "uploads"
AUDIO_STORAGE_DIR = (
    Path(_storage_setting).expanduser()
    if Path(_storage_setting).expanduser().is_absolute()
    else BASE_DIR / _storage_setting
)

# Pitch tracking is the expensive part of voice analysis, so it only examines
# the first N seconds of detected speech. Silence and energy metrics use the
# whole (possibly capped) file. This keeps analysis latency predictable.
VOICE_PITCH_MAX_SECONDS = _read_int("VOICE_PITCH_MAX_SECONDS", 30, minimum=5)

# Decoding compressed or very long recordings is capped so a single analysis
# request cannot consume unbounded memory or CPU. Durations reported for a
# capped file refer to the analysed portion; the response says when this
# happened.
VOICE_MAX_ANALYSIS_SECONDS = _read_int(
    "VOICE_MAX_ANALYSIS_SECONDS",
    300,
    minimum=10,
)

# Energy-based voice activity detection. Audio this far below the loudest
# frame is treated as silence. A pause is a silent gap of at least this many
# seconds between two detected speech segments.
VOICE_VAD_TOP_DB = _read_int("VOICE_VAD_TOP_DB", 35, minimum=10)
VOICE_MIN_PAUSE_SECONDS = _read_float("VOICE_MIN_PAUSE_SECONDS", 0.3)
if not 0.1 <= VOICE_MIN_PAUSE_SECONDS <= 5.0:
    VOICE_MIN_PAUSE_SECONDS = 0.3

# Analysis always runs at this sample rate. Pitch is only estimated from
# 60-400 Hz, the usual range for adult speech fundamentals.
VOICE_TARGET_SAMPLE_RATE = 16000
VOICE_PITCH_MIN_HZ = 60.0
VOICE_PITCH_MAX_HZ = 400.0
VOICE_MIN_SPEECH_SECONDS = 0.3
VOICE_MIN_VOICED_SECONDS = 0.25

CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Phase 2 evaluation settings
# ---------------------------------------------------------------------------

# A small, general-purpose English sentence-embedding model. It is loaded lazily
# on the first evaluation, so the API can start without downloading the model.
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()
SEMANTIC_MODEL_DEVICE = os.getenv("SEMANTIC_MODEL_DEVICE", "cpu").strip() or "cpu"

EVALUATION_METHODOLOGY_VERSION = "phase2-v1"

_SEMANTIC_WEIGHT = _read_float("EVALUATION_SEMANTIC_WEIGHT", 0.70)
_KEYWORD_WEIGHT = _read_float("EVALUATION_KEYWORD_WEIGHT", 0.30)

# Keep the weights in one place. If the environment values do not add up to 1,
# normalize them so the final score remains on the documented 0-100 scale.
_weight_total = _SEMANTIC_WEIGHT + _KEYWORD_WEIGHT
if _weight_total <= 0:
    _SEMANTIC_WEIGHT, _KEYWORD_WEIGHT = 0.70, 0.30
elif abs(_weight_total - 1.0) > 1e-9:
    _SEMANTIC_WEIGHT = _SEMANTIC_WEIGHT / _weight_total
    _KEYWORD_WEIGHT = _KEYWORD_WEIGHT / _weight_total

EVALUATION_WEIGHTS: dict[str, float] = {
    "semantic": _SEMANTIC_WEIGHT,
    "keyword": _KEYWORD_WEIGHT,
}

# Answers with fewer meaningful tokens than this are marked "unevaluable"
# instead of allowing a short sentence to receive a misleading score.
EVALUATION_MIN_MEANINGFUL_TOKENS = _read_int(
    "EVALUATION_MIN_MEANINGFUL_TOKENS",
    3,
    minimum=1,
)

# ---------------------------------------------------------------------------
# Phase 4 ML scoring settings
# ---------------------------------------------------------------------------

# These versions are stored with every prediction. A change to the feature
# schema or prediction methodology makes older rows explicitly stale instead of
# silently mixing incompatible results.
ML_PREDICTION_METHODOLOGY_VERSION = "phase4-v1"
ML_FEATURE_SCHEMA_VERSION = "phase4-features-v1"

# Trained artifacts are local files and are deliberately not committed. The
# inference service reports an explicit "unavailable" status when none exists;
# it never invents a model output.
_ml_model_setting = (
    os.getenv("ML_MODEL_PATH", "").strip() or "ml_artifacts/interview_score.joblib"
)
ML_MODEL_PATH = (
    Path(_ml_model_setting).expanduser()
    if Path(_ml_model_setting).expanduser().is_absolute()
    else BASE_DIR / _ml_model_setting
)

# The training script refuses to produce a "validated" artifact below these
# thresholds. Passing --allow-small is possible for experimentation, but such an
# artifact is marked unvalidated and inference will not use it.
ML_MIN_TRAINING_SAMPLES = _read_int("ML_MIN_TRAINING_SAMPLES", 60, minimum=10)
ML_MIN_TRAINING_GROUPS = _read_int("ML_MIN_TRAINING_GROUPS", 15, minimum=3)
ML_TEST_FRACTION = _read_float("ML_TEST_FRACTION", 0.25)
if not 0.1 <= ML_TEST_FRACTION <= 0.5:
    ML_TEST_FRACTION = 0.25
ML_RANDOM_SEED = _read_int("ML_RANDOM_SEED", 42, minimum=0)

# A trained Random Forest bundle records this range so a future target change
# cannot be loaded by mistake.
ML_TARGET_MIN = 0.0
ML_TARGET_MAX = 100.0

# ---------------------------------------------------------------------------
# Phase 5 feedback settings
# ---------------------------------------------------------------------------

# Phase 5 feedback is deterministic and evidence-based. It does not call an
# LLM and it never invents a score, quotation, or measurement. A change to the
# generation logic bumps this version so older rows become explicitly stale.
FEEDBACK_METHODOLOGY_VERSION = "phase5-v1"

# Fixed, documented bands for the Phase 2 combined score. They describe lexical
# and semantic agreement with one reference answer, not overall ability.
FEEDBACK_STRONG_SCORE = 75.0
FEEDBACK_WEAK_SCORE = 50.0

# The baseline's reference ranges are reused when interpreting objective Phase
# 3 delivery measurements. They are not used to infer confidence or personality.
FEEDBACK_SPEAKING_RATE_MIN_WPM = 110.0
FEEDBACK_SPEAKING_RATE_MAX_WPM = 170.0
FEEDBACK_FILLER_RATE_MAX_PERCENT = 3.0
FEEDBACK_REPETITION_RATE_MAX_PERCENT = 10.0
FEEDBACK_SENTENCE_WORDS_MIN = 8.0
FEEDBACK_SENTENCE_WORDS_MAX = 25.0

