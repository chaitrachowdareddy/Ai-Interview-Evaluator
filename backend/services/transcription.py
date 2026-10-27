"""Groq Whisper integration and audio type validation."""

from __future__ import annotations

import logging
from pathlib import Path

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    is_groq_configured,
)

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """A safe, user-facing transcription failure with an HTTP status."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Formats accepted by Groq speech-to-text. Values are the file extensions sent
# to Groq. Browsers commonly produce audio/webm (Chrome, Firefox, Edge) and
# audio/mp4 (Safari), but we accept the other Groq-supported formats as well.
MIME_TO_EXTENSION: dict[str, str] = {
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "video/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpga": "mpga",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}

EXTENSION_TO_MIME: dict[str, str] = {
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "mp3": "audio/mpeg",
    "mpga": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
}


def normalize_audio_type(
    content_type: str | None,
    filename: str | None,
) -> tuple[str, str]:
    """Return a supported ``(mime_type, extension)`` pair.

    Browsers may include codecs in the type (for example
    ``audio/webm;codecs=opus``) or send an empty/octet-stream type. In the
    latter case, the file extension is used as a fallback.
    """
    mime_type = (content_type or "").split(";", 1)[0].strip().lower()

    if mime_type in MIME_TO_EXTENSION:
        return mime_type, MIME_TO_EXTENSION[mime_type]

    extension = Path(filename or "").suffix.lower().lstrip(".")
    if extension in EXTENSION_TO_MIME:
        return EXTENSION_TO_MIME[extension], extension

    raise TranscriptionError(
        "Unsupported audio format. Please record with a browser that supports "
        "WebM, Ogg, MP4/M4A, MP3, WAV, or FLAC audio.",
        status_code=415,
    )


def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
) -> str:
    """Transcribe audio with Groq and return non-empty transcript text."""
    if not is_groq_configured(GROQ_API_KEY):
        raise TranscriptionError(
            "Transcription is not configured. Add GROQ_API_KEY to backend/.env "
            "and restart the backend.",
            status_code=503,
        )

    client = Groq(
        api_key=GROQ_API_KEY,
        timeout=GROQ_TIMEOUT_SECONDS,
        max_retries=1,
    )

    try:
        result = client.audio.transcriptions.create(
            file=(filename, audio_bytes, content_type),
            model=GROQ_MODEL,
            response_format="json",
            temperature=0.0,
        )
    except APITimeoutError as exc:
        logger.warning("Groq transcription timed out.")
        raise TranscriptionError(
            "The transcription service timed out. Please try again.",
            status_code=504,
        ) from exc
    except AuthenticationError as exc:
        logger.error("Groq rejected the configured API key.")
        raise TranscriptionError(
            "The transcription service rejected its API credentials. "
            "Check GROQ_API_KEY in backend/.env.",
            status_code=502,
        ) from exc
    except RateLimitError as exc:
        logger.warning("Groq rate limit reached.")
        raise TranscriptionError(
            "The transcription service is busy. Please wait a moment and try again.",
            status_code=503,
        ) from exc
    except APIStatusError as exc:
        logger.warning(
            "Groq API error (status=%s).",
            getattr(exc, "status_code", "unknown"),
        )
        raise TranscriptionError(
            "The transcription service could not process this recording. "
            "Please try again.",
            status_code=502,
        ) from exc
    except APIConnectionError as exc:
        logger.warning("Could not connect to the Groq API.")
        raise TranscriptionError(
            "Could not reach the transcription service. Check your internet "
            "connection and try again.",
            status_code=502,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak SDK details to clients
        # Do not log the traceback: unexpected SDK errors can include request
        # details. The exception type is enough for local diagnosis.
        logger.error(
            "Unexpected Groq transcription failure (%s).",
            type(exc).__name__,
        )
        raise TranscriptionError(
            "Transcription failed unexpectedly. Please try again.",
            status_code=502,
        ) from exc
    finally:
        client.close()

    transcript = (getattr(result, "text", "") or "").strip()
    if not transcript:
        raise TranscriptionError(
            "No speech was detected in the recording. Please record your answer "
            "again and speak clearly.",
            status_code=422,
        )

    return transcript
