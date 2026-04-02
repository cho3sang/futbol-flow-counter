from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

from session_store import SessionRecord, SessionStore, SessionSummary


@dataclass
class SessionRuntime:
    started_at_monotonic: float | None = None
    started_at_wallclock: datetime | None = None
    pause_started_at: float | None = None
    paused_duration: float = 0.0
    active_session_saved: bool = False
    new_record_session: bool = False
    personal_best_target: int = 0


class SessionController:
    def __init__(self, store: SessionStore | None = None) -> None:
        self.store = store or SessionStore()
        self.summary = self.store.fetch_summary()
        self.runtime = SessionRuntime(personal_best_target=self.summary.personal_best)

    @property
    def personal_best_target(self) -> int:
        return self.runtime.personal_best_target

    @property
    def new_record_session(self) -> bool:
        return self.runtime.new_record_session

    def refresh_summary(self) -> SessionSummary:
        self.summary = self.store.fetch_summary()
        if self.runtime.started_at_monotonic is None or self.runtime.active_session_saved:
            self.runtime.personal_best_target = self.summary.personal_best
        return self.summary

    def begin_session(self) -> None:
        self.runtime = SessionRuntime(
            started_at_monotonic=time.monotonic(),
            started_at_wallclock=datetime.now().replace(microsecond=0),
            personal_best_target=self.summary.personal_best,
        )

    def pause(self) -> None:
        if self.runtime.started_at_monotonic is None or self.runtime.pause_started_at is not None:
            return
        self.runtime.pause_started_at = time.monotonic()

    def resume(self) -> None:
        if self.runtime.pause_started_at is None:
            return

        self.runtime.paused_duration += time.monotonic() - self.runtime.pause_started_at
        self.runtime.pause_started_at = None

    def elapsed_seconds(self) -> float:
        if self.runtime.started_at_monotonic is None:
            return 0.0

        now = time.monotonic()
        paused_now = (now - self.runtime.pause_started_at) if self.runtime.pause_started_at else 0.0
        return max(0.0, now - self.runtime.started_at_monotonic - self.runtime.paused_duration - paused_now)

    def persist_if_needed(
        self,
        *,
        total_juggles: int,
        average_touch_interval: float | None,
        best_streak: int,
        source_name: str,
        reason: str,
    ) -> bool:
        if (
            self.runtime.active_session_saved
            or self.runtime.started_at_monotonic is None
            or self.runtime.started_at_wallclock is None
        ):
            return False

        duration = self.elapsed_seconds()
        if total_juggles <= 0 and duration < 12.0:
            return False

        normalized_source = source_name.replace("Source: ", "") or "Manual session"
        if normalized_source == "No source selected":
            normalized_source = "Manual session"

        self.store.save_session(
            SessionRecord(
                started_at=self.runtime.started_at_wallclock.isoformat(sep=" "),
                ended_at=datetime.now().replace(microsecond=0).isoformat(sep=" "),
                source_name=normalized_source,
                duration_seconds=duration,
                total_juggles=total_juggles,
                average_touch_interval=average_touch_interval,
                best_streak=best_streak,
            )
        )
        self.runtime.active_session_saved = True
        self.refresh_summary()
        return reason != "close"

    def mark_record_if_needed(self, total_juggles: int) -> bool:
        if total_juggles > self.runtime.personal_best_target and not self.runtime.new_record_session:
            self.runtime.new_record_session = True
            return True
        return False

    def live_personal_best(self, total_juggles: int) -> int:
        return max(self.summary.personal_best, total_juggles)

    def live_best_streak(self, best_streak: int) -> int:
        return max(self.summary.best_streak, best_streak)
