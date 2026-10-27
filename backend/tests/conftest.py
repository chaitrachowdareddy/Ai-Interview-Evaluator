"""Pytest setup: run the API against an isolated in-memory SQLite database."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Configure the app before any backend module is imported. These settings are
# test-only; the real application uses the MySQL URL from backend/.env.
os.environ["DATABASE_URL"] = "sqlite+pysqlite://"
os.environ["GROQ_API_KEY"] = ""
os.environ["MAX_AUDIO_MB"] = "1"

# Phase 3 stores original recordings on disk. Keep test audio out of the
# repository and remove it when the test session ends.
TEST_AUDIO_DIR = Path(tempfile.mkdtemp(prefix="ai-interview-audio-tests-"))
os.environ["AUDIO_STORAGE_DIR"] = str(TEST_AUDIO_DIR)
atexit.register(shutil.rmtree, TEST_AUDIO_DIR, ignore_errors=True)

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from main import app

    with TestClient(app) as test_client:
        yield test_client
