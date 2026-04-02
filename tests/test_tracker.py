from __future__ import annotations

from collections import deque
import unittest

from tracker import BallDetection, JuggleTracker, TrackerConfig


class TrackerLogicTests(unittest.TestCase):
    def test_score_candidate_prefers_close_round_match(self) -> None:
        tracker = JuggleTracker()
        tracker.last_detection = BallDetection(center=(120, 160), radius=26, score=0.9)
        tracker.reference_center = (120, 160)

        close_score = tracker._score_candidate((128, 168), radius=26, circularity=0.93, hough_bonus=0.0)
        far_score = tracker._score_candidate((320, 340), radius=26, circularity=0.93, hough_bonus=0.0)
        lopsided_score = tracker._score_candidate((128, 168), radius=26, circularity=0.44, hough_bonus=0.0)

        self.assertGreater(close_score, far_score)
        self.assertGreater(close_score, lopsided_score)

    def test_is_juggle_event_detects_rebound_in_kick_zone(self) -> None:
        tracker = JuggleTracker(TrackerConfig(kick_zone_ratio=0.70, reversal_speed=160.0, min_travel_px=30.0))
        base = 100.0
        tracker.motion_points = deque(
            [
                (base + 0.00, 240.0, 250.0),
                (base + 0.05, 240.0, 272.0),
                (base + 0.10, 240.0, 316.0),
                (base + 0.15, 240.0, 274.0),
                (base + 0.20, 240.0, 232.0),
            ],
            maxlen=7,
        )

        self.assertTrue(tracker._is_juggle_event(base + 0.21, frame_height=400))

    def test_split_motion_windows_avoids_overlapping_midpoint(self) -> None:
        points = [
            (0.00, 240.0, 250.0),
            (0.05, 240.0, 272.0),
            (0.10, 240.0, 316.0),
            (0.15, 240.0, 274.0),
            (0.20, 240.0, 232.0),
        ]

        downward_window, upward_window = JuggleTracker._split_motion_windows(points)

        self.assertEqual(downward_window, points[:3])
        self.assertEqual(upward_window, points[3:])
        self.assertTrue(set(downward_window).isdisjoint(set(upward_window)))

    def test_is_juggle_event_respects_touch_cooldown(self) -> None:
        tracker = JuggleTracker(TrackerConfig(kick_zone_ratio=0.70, reversal_speed=160.0, min_travel_px=30.0))
        base = 100.0
        tracker.motion_points = deque(
            [
                (base + 0.00, 240.0, 250.0),
                (base + 0.05, 240.0, 272.0),
                (base + 0.10, 240.0, 316.0),
                (base + 0.15, 240.0, 274.0),
                (base + 0.20, 240.0, 232.0),
            ],
            maxlen=7,
        )
        tracker.last_touch_time = base + 0.05

        self.assertFalse(tracker._is_juggle_event(base + 0.21, frame_height=400))

    def test_upward_reversal_factor_is_configurable(self) -> None:
        points = deque(
            [
                (0.00, 240.0, 240.0),
                (0.05, 240.0, 260.0),
                (0.10, 240.0, 300.0),
                (0.15, 240.0, 293.0),
                (0.20, 240.0, 286.0),
            ],
            maxlen=7,
        )

        permissive = JuggleTracker(
            TrackerConfig(kick_zone_ratio=0.70, reversal_speed=160.0, upward_reversal_factor=0.40, min_travel_px=30.0)
        )
        strict = JuggleTracker(
            TrackerConfig(kick_zone_ratio=0.70, reversal_speed=160.0, upward_reversal_factor=0.90, min_travel_px=30.0)
        )
        permissive.motion_points = points.copy()
        strict.motion_points = points.copy()

        self.assertTrue(permissive._is_juggle_event(1.21, frame_height=400))
        self.assertFalse(strict._is_juggle_event(1.21, frame_height=400))

    def test_record_touch_updates_streak_and_average_gap(self) -> None:
        tracker = JuggleTracker(TrackerConfig(streak_gap_reset_seconds=2.0))

        tracker._record_touch(10.0)
        tracker._record_touch(10.8)
        tracker._record_touch(11.6)

        self.assertEqual(tracker.total_juggles, 3)
        self.assertEqual(tracker.current_streak, 3)
        self.assertEqual(tracker.best_streak, 3)
        self.assertAlmostEqual(tracker.average_touch_interval or 0.0, 0.8, places=2)


if __name__ == "__main__":
    unittest.main()
