from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time
import unittest

from session_controller import SessionController
from session_store import SessionStore


class SessionControllerTests(unittest.TestCase):
    def test_persist_if_needed_saves_meaningful_session_and_refreshes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "session_test.db")
            controller = SessionController(store)
            controller.begin_session()
            controller.runtime.started_at_monotonic = time.monotonic() - 18.0
            controller.runtime.started_at_wallclock = datetime.now().replace(microsecond=0) - timedelta(seconds=18)

            saved = controller.persist_if_needed(
                total_juggles=7,
                average_touch_interval=0.64,
                best_streak=5,
                source_name="Source: Webcam 0",
                reason="reset",
            )

            self.assertTrue(saved)
            self.assertEqual(controller.summary.sessions_played, 1)
            self.assertEqual(controller.summary.personal_best, 7)
            self.assertEqual(controller.summary.best_streak, 5)


if __name__ == "__main__":
    unittest.main()
