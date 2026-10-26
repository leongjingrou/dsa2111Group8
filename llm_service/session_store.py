"""Versioned SQLite snapshots shared by backend replicas on a local volume."""

import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path


class SessionConflict(RuntimeError):
    """Another request saved a newer version of this consultation."""


class SessionCapacity(RuntimeError):
    """The configured limit of active consultations has been reached."""


class SessionMissing(KeyError):
    """The consultation does not exist or its idle timeout has elapsed."""


class SessionStore:
    """Persist snapshots with an idle timeout and optimistic concurrency checks."""

    def __init__(self, path, *, ttl_seconds=3600, maximum=4096):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.maximum = maximum
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated REAL NOT NULL,
                    state TEXT NOT NULL
                )
            """)

    @contextmanager
    def connect(self):
        """Commit on success, roll back on failure, and always close the handle."""
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            with connection:
                yield connection

    def create(self, state):
        session_id = str(uuid.uuid4())
        with self.connect() as connection:
            # Serialize capacity checks and inserts across worker processes.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM sessions WHERE updated < ?",
                (time.time() - self.ttl_seconds,),
            )
            active_count = connection.execute("""
                SELECT COUNT(*) FROM sessions
                WHERE json_extract(state, '$.finished') IS NOT 1
            """).fetchone()[0]
            if active_count >= self.maximum:
                raise SessionCapacity("Consultation capacity reached")
            connection.execute(
                "INSERT INTO sessions (id, version, updated, state) VALUES (?, 0, ?, ?)",
                (session_id, time.time(), json.dumps(state)),
            )
        return session_id

    def load(self, session_id):
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT version, state FROM sessions
                WHERE id = ? AND updated >= ?
                """,
                (session_id, time.time() - self.ttl_seconds),
            ).fetchone()
        if row is None:
            raise SessionMissing(session_id)
        version, serialized_state = row
        return version, json.loads(serialized_state)

    def save(self, session_id, version, state):
        """Replace only the version loaded by this request; reject stale writes."""
        with self.connect() as connection:
            result = connection.execute(
                """
                UPDATE sessions SET state = ?, updated = ?, version = version + 1
                WHERE id = ? AND version = ?
                """,
                (json.dumps(state), time.time(), session_id, version),
            )
            if result.rowcount != 1:
                raise SessionConflict(
                    "Consultation changed during this request; reload before continuing"
                )

    def delete(self, session_id):
        with self.connect() as connection:
            result = connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return result.rowcount == 1
