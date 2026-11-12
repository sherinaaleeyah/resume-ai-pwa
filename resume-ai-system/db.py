"""
Database layer for TalentMatch.

Uses SQLAlchemy Core so the same code runs on PostgreSQL (Neon, in production) and on
SQLite (a throw-away local file, for quick development when DATABASE_URL is not set).

How the rest of the app uses it:

    from db import get_db

    with get_db() as conn:                      # commits on success, rolls back on error
        row  = conn.execute("SELECT * FROM jobs WHERE id = :id", {"id": 5}).fetchone()
        rows = conn.execute("SELECT * FROM jobs").fetchall()   # list of dicts
        new_id = conn.execute("INSERT ... RETURNING id", {...}).scalar()

Rules:
  * SQL uses named parameters (:name) and a dict of values, never string formatting.
  * Rows come back as plain dicts, so row["title"] works like it did with sqlite3.Row.
  * The schema is created by setup_db.py (run once), not when the web app starts.
"""
import os
from contextlib import contextmanager

from sqlalchemy import (
    Column, Float, ForeignKey, Integer, LargeBinary, MetaData, Table, Text,
    create_engine, event, inspect, text,
)
from sqlalchemy.engine import make_url

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Share of the combined score that comes from keyword matching; the rest is semantic similarity.
DEFAULT_KEYWORD_WEIGHT = 0.6


def running_on_azure():
    """Azure App Service always sets WEBSITE_SITE_NAME."""
    return bool(os.getenv("WEBSITE_SITE_NAME"))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("full_name", Text, nullable=False),
    Column("email", Text, nullable=False, unique=True),     # always stored lower-case
    Column("company_name", Text, nullable=False),
    Column("job_title", Text, nullable=False, server_default=""),
    Column("phone", Text, nullable=False, server_default=""),
    Column("password_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)

jobs = Table(
    "jobs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # The recruiter who owns the job. Every recruiter query filters on this.
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False, index=True),
    Column("title", Text, nullable=False),
    Column("company", Text, nullable=False),
    Column("description", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("active", Integer, nullable=False, server_default=text("1")),
    Column("keyword_weight", Float, nullable=False, server_default=text(str(DEFAULT_KEYWORD_WEIGHT))),
)

candidate_scores = Table(
    "candidate_scores", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Integer, ForeignKey("jobs.id"), nullable=False, index=True),
    Column("candidate_name", Text, nullable=False),
    Column("file_name", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("matched_skills", Text, nullable=False),
    Column("missing_skills", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("summary", Text, nullable=False, server_default=""),
    Column("keyword_score", Float),
    Column("semantic_score", Float),
    Column("resume_text", Text),
)

# The uploaded resume file itself. Kept in its own table so the ranking queries never
# load file bytes. One file per candidate; removed together with the candidate.
resume_files = Table(
    "resume_files", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("candidate_id", Integer, ForeignKey("candidate_scores.id", ondelete="CASCADE"),
           nullable=False, unique=True),
    Column("filename", Text, nullable=False),
    Column("content_type", Text, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("data", LargeBinary, nullable=False),
)

activity = Table(
    "activity", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", Integer, ForeignKey("jobs.id"), index=True),
    Column("message", Text, nullable=False),
    Column("score", Float),
    Column("created_at", Text, nullable=False),
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_engine = None


def get_database_url():
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        # Some providers hand out the old "postgres://" form, which SQLAlchemy rejects.
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        return url
    if running_on_azure():
        raise RuntimeError(
            "DATABASE_URL is not set. Add it under App Service > Environment variables "
            "(use the Neon connection string), then restart the app."
        )
    # Local development fallback: a throw-away SQLite file (ignored by git via *.db).
    return "sqlite:///" + os.path.join(BASE_DIR, "local_dev.db")


def get_engine():
    global _engine
    if _engine is None:
        url = get_database_url()
        is_sqlite = make_url(url).get_backend_name() == "sqlite"
        kwargs = {}
        if not is_sqlite:
            # Neon suspends idle databases and drops their connections: test a connection
            # before using it, and don't keep one for longer than 5 minutes.
            kwargs.update(pool_pre_ping=True, pool_recycle=300)
        _engine = create_engine(url, **kwargs)

        if is_sqlite:
            @event.listens_for(_engine, "connect")
            def _sqlite_foreign_keys(dbapi_conn, _record):
                # SQLite ignores foreign keys unless asked; PostgreSQL always enforces them.
                # Turning this on locally makes development behave like production.
                dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return _engine


def is_sqlite():
    return get_engine().dialect.name == "sqlite"


# ---------------------------------------------------------------------------
# Small wrapper so queries read like sqlite3 but return plain dicts
# ---------------------------------------------------------------------------
class _Result:
    def __init__(self, result):
        self._result = result

    def fetchone(self):
        row = self._result.mappings().fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(row) for row in self._result.mappings().all()]

    def scalar(self):
        return self._result.scalar()

    @property
    def rowcount(self):
        return self._result.rowcount


class _Connection:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        return _Result(self._conn.execute(text(sql), params or {}))


@contextmanager
def get_db():
    """One transaction: commits when the block ends, rolls back if it raises."""
    with get_engine().begin() as conn:
        yield _Connection(conn)


# ---------------------------------------------------------------------------
# Schema management (called by setup_db.py)
# ---------------------------------------------------------------------------
def init_schema():
    """Create any missing tables. Safe to run repeatedly; never alters existing tables."""
    metadata.create_all(get_engine())


def drop_everything():
    """Delete all TalentMatch tables and data. Used only by `setup_db.py --reset`."""
    metadata.drop_all(get_engine())


def schema_ready():
    names = set(inspect(get_engine()).get_table_names())
    return {t.name for t in metadata.sorted_tables} <= names
