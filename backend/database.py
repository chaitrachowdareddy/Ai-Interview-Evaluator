"""SQLAlchemy engine, session factory, and safe database initialization."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import (
    ArgumentError,
    NoSuchModuleError,
    OperationalError,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def _create_engine():
    engine_kwargs: dict = {"pool_pre_ping": True}

    if DATABASE_URL.startswith("sqlite"):
        # Allow FastAPI's worker threads to use the connection and keep an
        # in-memory database alive for tests.
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if DATABASE_URL in {"sqlite://", "sqlite+pysqlite://"} or ":memory:" in DATABASE_URL:
            from sqlalchemy.pool import StaticPool

            engine_kwargs["poolclass"] = StaticPool
    else:
        # MySQL closes idle connections after a while. Recycling avoids stale
        # connections during local development.
        engine_kwargs["pool_recycle"] = 280
        engine_kwargs["connect_args"] = {"connect_timeout": 5}

    return create_engine(DATABASE_URL, **engine_kwargs)


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def describe_database_target() -> dict[str, object]:
    """Return non-secret connection details for safe health diagnostics."""
    target: dict[str, object] = {
        "database_driver": None,
        "database_host": None,
        "database_port": None,
        "database_name": None,
        "database_user_configured": False,
        "database_password_configured": False,
    }

    try:
        url = make_url(DATABASE_URL)
    except Exception:  # noqa: BLE001 - never fail the health check
        return target

    target.update(
        database_driver=url.drivername or None,
        database_host=url.host,
        database_port=url.port,
        database_name=url.database,
        database_user_configured=bool(url.username),
        database_password_configured=bool(url.password),
    )
    return target


_DATABASE_HINTS: dict[str, str] = {
    "authentication_failed": (
        "MySQL rejected the configured username/password. Update DATABASE_URL "
        "in backend/.env with valid credentials (URL-encode special characters), "
        "then restart FastAPI."
    ),
    "permission_denied": (
        "The MySQL user cannot access this database. Grant the required "
        "privileges to that user (or use a user that has them), then restart "
        "FastAPI."
    ),
    "unknown_database": (
        "The configured MySQL database does not exist. Run setup.sql (it only "
        "creates the database) or point DATABASE_URL at an existing database, "
        "then restart FastAPI."
    ),
    "connection_failed": (
        "MySQL is not accepting connections at the configured host/port. Start "
        "the MySQL service and verify the host and port in DATABASE_URL."
    ),
    "unknown_host": (
        "The configured MySQL host could not be resolved. Check the host name "
        "in DATABASE_URL."
    ),
    "connection_timeout": (
        "The MySQL connection timed out. Confirm MySQL is running and reachable "
        "at the host and port in DATABASE_URL."
    ),
    "auth_plugin_error": (
        "MySQL rejected the authentication plugin. Make sure the dependencies "
        "in requirements.txt are installed (they include cryptography)."
    ),
    "missing_driver": (
        "The configured database driver is not installed. Run "
        "'pip install -r requirements.txt' in backend."
    ),
    "invalid_configuration": (
        "DATABASE_URL is not a valid SQLAlchemy URL. Use "
        "mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME and URL-encode special "
        "characters."
    ),
    "database_file_unavailable": (
        "The configured SQLite database file could not be opened. Check the "
        "path and folder permissions in DATABASE_URL."
    ),
    "unavailable": (
        "The database could not be reached. Check DATABASE_URL in backend/.env, "
        "confirm MySQL is running, and restart FastAPI."
    ),
}


def describe_database_error(exc: BaseException) -> dict[str, str]:
    """Classify a database failure without exposing credentials or driver text."""
    original = getattr(exc, "orig", None) or exc
    try:
        raw_message = str(original)
    except Exception:  # noqa: BLE001 - some drivers raise odd objects
        raw_message = ""
    lowered = raw_message.lower()

    args = getattr(original, "args", ())
    code = args[0] if args and isinstance(args[0], int) else None

    def result(category: str) -> dict[str, str]:
        return {
            "database_error_category": category,
            "database_hint": _DATABASE_HINTS[category],
        }

    if isinstance(exc, (ArgumentError, NoSuchModuleError)) or (
        "could not parse sqlalchemy url" in lowered
    ):
        return result("invalid_configuration")
    if isinstance(exc, ModuleNotFoundError) or "no module named" in lowered:
        return result("missing_driver")

    if code == 1044:
        return result("permission_denied")
    if code == 1045:
        return result("authentication_failed")
    if code == 1049:
        return result("unknown_database")
    if code in {2002, 2003}:
        return result("connection_failed")
    if code == 2005:
        return result("unknown_host")
    if code == 2059 or "authentication plugin" in lowered:
        return result("auth_plugin_error")

    if "access denied for user" in lowered and " to database " in lowered:
        return result("permission_denied")
    if "access denied" in lowered:
        return result("authentication_failed")
    if "unknown database" in lowered:
        return result("unknown_database")
    if "unknown host" in lowered or "name or service not known" in lowered:
        return result("unknown_host")
    if "connection refused" in lowered or "can't connect" in lowered:
        return result("connection_failed")
    if "timed out" in lowered or "timeout" in lowered:
        return result("connection_timeout")
    if "unable to open database file" in lowered:
        return result("database_file_unavailable")

    return result("unavailable")


_init_lock = threading.Lock()
_db_ready = False


def _ensure_reference_answer_column() -> None:
    """Add ``reference_answer`` to an existing Phase 1 questions table.

    The project intentionally has no migration framework. This is a small,
    additive, rerunnable migration: it never changes or removes existing rows.
    New databases already get the column from ``create_all``.
    """

    columns = {
        column["name"]
        for column in inspect(engine).get_columns("interview_questions")
    }
    if "reference_answer" in columns:
        return

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE interview_questions "
                    "ADD COLUMN reference_answer TEXT NULL"
                )
            )
    except OperationalError:
        # Another process may have added the column between inspection and the
        # ALTER. Re-check before treating this as a real failure.
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("interview_questions")
        }
        if "reference_answer" not in columns:
            raise
        return

    logger.info("Added optional reference_answer column to interview_questions.")


def _ensure_answer_audio_columns() -> None:
    """Add Phase 3 recording columns to an existing Phase 1 answers table.

    Additive and rerunnable, like the reference-answer migration. The audio
    file itself lives on disk (``AUDIO_STORAGE_DIR``); only its portable
    relative path and SHA-256 fingerprint are stored in the database.
    """

    additions = (
        ("audio_path", "VARCHAR(500) NULL"),
        ("audio_sha256", "VARCHAR(64) NULL"),
    )
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("interview_answers")
    }

    for name, definition in additions:
        if name in columns:
            continue

        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE interview_answers "
                        f"ADD COLUMN {name} {definition}"
                    )
                )
        except OperationalError:
            # Re-check: another process may have added the column first.
            columns = {
                column["name"]
                for column in inspect(engine).get_columns(
                    "interview_answers"
                )
            }
            if name not in columns:
                raise
            continue

        columns.add(name)
        logger.info("Added optional %s column to interview_answers.", name)



def init_db() -> None:
    """Create missing tables and insert seed questions without destroying data."""
    # Importing models registers every table on Base.metadata.
    import models  # noqa: F401  (imported for its side effect)
    from seed import seed_questions

    Base.metadata.create_all(bind=engine)
    _ensure_reference_answer_column()
    _ensure_answer_audio_columns()

    with SessionLocal() as db:
        added = seed_questions(db)

    logger.info("Database ready; %s seed question(s) inserted.", added)


def ensure_db_ready() -> None:
    """Initialize the database once, retrying later if the first attempt failed."""
    global _db_ready

    if _db_ready:
        return

    with _init_lock:
        if _db_ready:
            return
        init_db()
        _db_ready = True


def reset_db_ready() -> None:
    """Allow a later request to re-check the database (used by tests)."""
    global _db_ready

    with _init_lock:
        _db_ready = False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a thread-safe-per-request ORM session."""
    ensure_db_ready()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
