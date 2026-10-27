"""Persist original recordings for Phase 3 voice analysis.

Phase 1 sent the upload straight to Groq and stored only metadata, so voice
analysis had nothing to measure. This module writes the exact uploaded bytes to
``AUDIO_STORAGE_DIR`` using a content-addressed name and stores a portable
relative path plus SHA-256 in the database.

Storage is deliberately independent of analysis: a storage failure never fails
transcription. Callers can continue with ``audio_sha256``/``audio_path`` unset,
and voice analysis then reports ``unavailable`` instead of inventing values.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config import AUDIO_STORAGE_DIR

logger = logging.getLogger(__name__)


class AudioStorageError(Exception):
    """A safe storage failure. Raw OS paths/details stay in the server log."""


@dataclass(frozen=True)
class StoredAudio:
    relative_path: str
    sha256: str
    size_bytes: int


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extension(extension: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", (extension or "").lower())
    return cleaned or "bin"


def _storage_root() -> Path:
    try:
        root = AUDIO_STORAGE_DIR.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        raise AudioStorageError(
            "The recording storage directory could not be created."
        ) from exc


def store_audio(
    audio_bytes: bytes,
    interview_id: int,
    question_id: int,
    extension: str,
) -> StoredAudio:
    """Write audio bytes atomically and return their portable location.

    Names are content-addressed, so resubmitting identical audio reuses the
    existing file. The write uses a temporary file in the same directory and
    ``os.replace`` so a partially written recording is never referenced.
    """

    if not audio_bytes:
        raise AudioStorageError("The recording is empty.")

    root = _storage_root()
    digest = sha256_bytes(audio_bytes)
    folder = root / f"interview-{int(interview_id)}"
    target = (
        folder
        / f"question-{int(question_id)}-{digest[:16]}.{_safe_extension(extension)}"
    )

    try:
        folder.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".recording-",
                dir=folder,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(audio_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, target)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
                raise
    except OSError as exc:
        raise AudioStorageError(
            "The recording could not be saved for later voice analysis."
        ) from exc

    return StoredAudio(
        relative_path=target.relative_to(root).as_posix(),
        sha256=digest,
        size_bytes=len(audio_bytes),
    )


def resolve_audio_path(relative_path: str) -> Path:
    """Resolve a stored relative path, refusing anything outside the root."""

    if not relative_path:
        raise AudioStorageError("No recording path is stored for this answer.")

    root = AUDIO_STORAGE_DIR.expanduser().resolve()
    try:
        candidate = (root / relative_path).resolve()
    except OSError as exc:
        raise AudioStorageError(
            "The stored recording path could not be resolved."
        ) from exc

    if not candidate.is_relative_to(root):
        raise AudioStorageError(
            "The stored recording path is outside the configured storage "
            "directory."
        )

    return candidate
