from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass
class SessionRecord:
    started_at: str
    ended_at: str
    source_name: str
    duration_seconds: float
    total_juggles: int
    average_touch_interval: float | None
    best_streak: int


@dataclass
class SessionSummary:
    sessions_played: int = 0
    total_juggles: int = 0
    personal_best: int = 0
    best_streak: int = 0
    average_duration_seconds: float = 0.0


class SessionStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or Path(__file__).with_name("futbol_flow.db")
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    total_juggles INTEGER NOT NULL,
                    average_touch_interval REAL,
                    best_streak INTEGER NOT NULL
                )
                """
            )

    def save_session(self, session: SessionRecord) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (
                    started_at,
                    ended_at,
                    source_name,
                    duration_seconds,
                    total_juggles,
                    average_touch_interval,
                    best_streak
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.started_at,
                    session.ended_at,
                    session.source_name,
                    session.duration_seconds,
                    session.total_juggles,
                    session.average_touch_interval,
                    session.best_streak,
                ),
            )
            return int(cursor.lastrowid)

    def fetch_summary(self) -> SessionSummary:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS sessions_played,
                    COALESCE(SUM(total_juggles), 0) AS total_juggles,
                    COALESCE(MAX(total_juggles), 0) AS personal_best,
                    COALESCE(MAX(best_streak), 0) AS best_streak,
                    COALESCE(AVG(duration_seconds), 0.0) AS average_duration_seconds
                FROM sessions
                """
            ).fetchone()

        return SessionSummary(
            sessions_played=int(row[0]),
            total_juggles=int(row[1]),
            personal_best=int(row[2]),
            best_streak=int(row[3]),
            average_duration_seconds=float(row[4]),
        )

    def fetch_recent_sessions(self, limit: int = 4) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT started_at, ended_at, source_name, duration_seconds, total_juggles, average_touch_interval, best_streak
                FROM sessions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            SessionRecord(
                started_at=row[0],
                ended_at=row[1],
                source_name=row[2],
                duration_seconds=float(row[3]),
                total_juggles=int(row[4]),
                average_touch_interval=float(row[5]) if row[5] is not None else None,
                best_streak=int(row[6]),
            )
            for row in rows
        ]
