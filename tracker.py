from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque

import cv2
import numpy as np


@dataclass
class TrackerConfig:
    min_radius: int = 8
    max_radius: int = 72
    min_area: int = 140
    kick_zone_ratio: float = 0.72
    reversal_speed: float = 165.0
    upward_reversal_factor: float = 0.58
    min_travel_px: float = 32.0
    tracking_distance_px: float = 180.0
    touch_cooldown: float = 0.30
    trail_length: int = 22
    target_width: int = 960
    max_prediction_frames: int = 8
    motion_prediction_frames: int = 2
    streak_gap_reset_seconds: float = 2.25
    velocity_arrow_scale: float = 0.06


@dataclass
class BallDetection:
    center: tuple[int, int]
    radius: int
    score: float


@dataclass
class TrackerMetrics:
    total_juggles: int
    status_text: str
    confidence_percent: int
    last_touch_seconds: float | None
    detected: bool
    predicted: bool
    current_speed: int
    velocity: tuple[float, float]
    current_streak: int
    best_streak: int
    average_touch_interval: float | None
    lost_frames: int


class JuggleTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.show_trail = True
        self._make_background_subtractor()
        self.reset_session()

    @property
    def average_touch_interval(self) -> float | None:
        if not self.touch_intervals:
            return None
        return sum(self.touch_intervals) / len(self.touch_intervals)

    def _make_background_subtractor(self) -> None:
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=32,
            detectShadows=False,
        )

    def _make_kalman_filter(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kalman.processNoiseCov = np.array(
            [
                [1e-2, 0.0, 0.0, 0.0],
                [0.0, 1e-2, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0, 2.5],
            ],
            dtype=np.float32,
        )
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.6
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 4.0
        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def reset_session(self) -> None:
        self.total_juggles = 0
        self.last_touch_time = 0.0
        self.last_detection: BallDetection | None = None
        self.last_status = "Ready for kickoff"
        self.frames_seen = 0
        self.touch_flash_frames = 0
        self.lost_frames = 0
        self.current_streak = 0
        self.best_streak = 0
        self.touch_intervals: list[float] = []
        self.last_frame_time: float | None = None
        self.kalman_initialized = False
        self.predicted_center: tuple[int, int] | None = None
        self.reference_center: tuple[int, int] | None = None
        self.current_velocity = (0.0, 0.0)
        self.trail: Deque[tuple[int, int, bool]] = deque(maxlen=self.config.trail_length)
        self.motion_points: Deque[tuple[float, float, float]] = deque(maxlen=7)
        self._make_kalman_filter()

    def restart(self) -> None:
        self._make_background_subtractor()
        self.reset_session()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, TrackerMetrics]:
        now = time.monotonic()
        self.frames_seen += 1
        dt = self._frame_delta(now)

        frame = self._resize_for_tracking(frame)
        predicted_point = self._predict_track(dt)
        detection = self._detect_ball(frame)

        status = "Calibrating background" if self.frames_seen < 12 else "Searching for the ball"
        confidence = 0
        predicted = False
        tracking_center = predicted_point

        if self.current_streak and (now - self.last_touch_time) > self.config.streak_gap_reset_seconds:
            self.current_streak = 0

        if detection:
            confidence = int(round(detection.score * 100))
            self.lost_frames = 0
            self.last_detection = detection
            tracking_center = self._register_measurement(detection.center)
            self.reference_center = tracking_center
            self._append_trail(tracking_center, predicted=False)
            self.motion_points.append((now, float(tracking_center[0]), float(tracking_center[1])))
            status = "Calibrating background" if self.frames_seen < 12 else "Tracking locked"

            if self.frames_seen >= 12 and self._is_juggle_event(now, frame.shape[0]):
                self._record_touch(now)
                self.touch_flash_frames = 12
                status = "Juggle counted"
        else:
            self.lost_frames += 1
            if predicted_point and self.lost_frames <= self.config.max_prediction_frames:
                predicted = True
                tracking_center = predicted_point
                self.reference_center = predicted_point
                self._append_trail(predicted_point, predicted=True)
                if self.lost_frames <= self.config.motion_prediction_frames:
                    self.motion_points.append((now, float(predicted_point[0]), float(predicted_point[1])))
                status = "Predicting through occlusion" if self.frames_seen >= 12 else "Calibrating background"

            if self.lost_frames > self.config.max_prediction_frames:
                self.motion_points.clear()
                self.last_detection = None
                self.reference_center = None
                self.predicted_center = None
                self.current_velocity = (0.0, 0.0)
                self.kalman_initialized = False
                tracking_center = None
                self._make_kalman_filter()

        if self.touch_flash_frames > 0:
            self.touch_flash_frames -= 1

        self.last_status = status
        annotated = self._draw_overlay(
            frame.copy(),
            detection=detection,
            tracking_center=tracking_center,
            confidence_percent=confidence,
            predicted=predicted,
        )
        metrics = TrackerMetrics(
            total_juggles=self.total_juggles,
            status_text=status,
            confidence_percent=confidence,
            last_touch_seconds=(now - self.last_touch_time) if self.total_juggles else None,
            detected=detection is not None,
            predicted=predicted,
            current_speed=int(round(math.hypot(*self.current_velocity))),
            velocity=self.current_velocity,
            current_streak=self.current_streak,
            best_streak=self.best_streak,
            average_touch_interval=self.average_touch_interval,
            lost_frames=self.lost_frames,
        )
        return annotated, metrics

    def _frame_delta(self, now: float) -> float:
        if self.last_frame_time is None:
            self.last_frame_time = now
            return 1.0 / 30.0

        dt = max(1.0 / 120.0, min(now - self.last_frame_time, 0.2))
        self.last_frame_time = now
        return dt

    def _predict_track(self, dt: float) -> tuple[int, int] | None:
        if not self.kalman_initialized:
            return None

        self.kalman.transitionMatrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        prediction = self.kalman.predict()
        self.predicted_center = (int(prediction[0, 0]), int(prediction[1, 0]))
        self.current_velocity = (float(prediction[2, 0]), float(prediction[3, 0]))
        return self.predicted_center

    def _register_measurement(self, center: tuple[int, int]) -> tuple[int, int]:
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])

        if not self.kalman_initialized:
            self.kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            self.kalman.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            corrected = self.kalman.statePost
            self.kalman_initialized = True
        else:
            corrected = self.kalman.correct(measurement)

        self.predicted_center = (int(corrected[0, 0]), int(corrected[1, 0]))
        self.current_velocity = (float(corrected[2, 0]), float(corrected[3, 0]))
        return self.predicted_center

    def _append_trail(self, center: tuple[int, int], predicted: bool) -> None:
        if self.trail and self.trail[-1][0] == center[0] and self.trail[-1][1] == center[1]:
            return
        self.trail.append((center[0], center[1], predicted))

    def _record_touch(self, now: float) -> None:
        if self.total_juggles:
            gap = now - self.last_touch_time
            self.touch_intervals.append(gap)
            if gap <= self.config.streak_gap_reset_seconds:
                self.current_streak += 1
            else:
                self.current_streak = 1
        else:
            self.current_streak = 1

        self.total_juggles += 1
        self.last_touch_time = now
        self.best_streak = max(self.best_streak, self.current_streak)

    def _resize_for_tracking(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.target_width:
            return frame

        scale = self.config.target_width / float(width)
        return cv2.resize(frame, (self.config.target_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    def _detect_ball(self, frame: np.ndarray) -> BallDetection | None:
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        fg_mask = self.bg_subtractor.apply(blurred)
        _, fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        candidates = self._contour_candidates(fg_mask)
        best_contour_score = max((candidate.score for candidate in candidates), default=0.0)

        if not candidates or best_contour_score < 0.74:
            candidates.extend(self._hough_candidates(gray, fg_mask))

        if not candidates:
            return None

        return max(candidates, key=lambda candidate: candidate.score)

    def _contour_candidates(self, fg_mask: np.ndarray) -> list[BallDetection]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallDetection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.42:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not self.config.min_radius <= radius <= self.config.max_radius:
                continue

            _, _, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / float(max(height, 1))
            if not 0.6 <= aspect_ratio <= 1.45:
                continue

            score = self._score_candidate((x, y), radius, circularity, hough_bonus=0.0)
            candidates.append(BallDetection(center=(int(x), int(y)), radius=int(radius), score=score))

        return candidates

    def _hough_candidates(self, gray: np.ndarray, fg_mask: np.ndarray) -> list[BallDetection]:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(28, self.config.min_radius * 2),
            param1=120,
            param2=18,
            minRadius=self.config.min_radius,
            maxRadius=self.config.max_radius,
        )

        if circles is None:
            return []

        candidates: list[BallDetection] = []
        for x, y, radius in np.round(circles[0, :8]).astype(int):
            circle_mask = np.zeros_like(gray)
            cv2.circle(circle_mask, (x, y), radius, 255, -1)
            motion_ratio = cv2.mean(fg_mask, mask=circle_mask)[0] / 255.0
            if motion_ratio < 0.12:
                continue

            score = self._score_candidate((x, y), radius, 0.82, hough_bonus=min(0.18, motion_ratio * 0.18))
            candidates.append(BallDetection(center=(x, y), radius=radius, score=score))

        return candidates

    def _score_candidate(
        self,
        center: tuple[float, float],
        radius: float,
        circularity: float,
        hough_bonus: float,
    ) -> float:
        circularity_score = self._clamp((circularity - 0.42) / 0.58)

        anchor_center = self.reference_center or (self.last_detection.center if self.last_detection else None)
        if anchor_center:
            distance = math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1])
            distance_score = 1.0 - min(distance / self.config.tracking_distance_px, 1.0)
        else:
            distance_score = 0.55

        radius_midpoint = (self.config.min_radius + self.config.max_radius) / 2.0
        max_distance = max((self.config.max_radius - self.config.min_radius) / 2.0, 1.0)
        size_score = 1.0 - min(abs(radius - radius_midpoint) / max_distance, 1.0)

        score = (circularity_score * 0.42) + (distance_score * 0.38) + (size_score * 0.12) + hough_bonus
        return self._clamp(score)

    def _is_juggle_event(self, now: float, frame_height: int) -> bool:
        if now - self.last_touch_time < self.config.touch_cooldown:
            return False

        if len(self.motion_points) < 5:
            return False

        recent_points = list(self.motion_points)[-5:]
        downward_window, upward_window = self._split_motion_windows(recent_points)
        y_values = [point[2] for point in recent_points]
        lowest_visible_point = max(y_values)
        kick_zone_y = frame_height * self.config.kick_zone_ratio

        if lowest_visible_point < kick_zone_y:
            return False

        downward_speed = self._average_vertical_speed(downward_window)
        upward_speed = self._average_vertical_speed(upward_window)
        travel = lowest_visible_point - min(y_values)

        if downward_speed < self.config.reversal_speed:
            return False

        if upward_speed > (-self.config.reversal_speed * self.config.upward_reversal_factor):
            return False

        if travel < self.config.min_travel_px:
            return False

        return True

    @staticmethod
    def _split_motion_windows(
        points: list[tuple[float, float, float]],
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        midpoint = len(points) // 2
        downward_window = points[: midpoint + 1]
        upward_window = points[midpoint + 1 :]

        if len(upward_window) < 2:
            upward_window = points[-2:]

        return downward_window, upward_window

    @staticmethod
    def _average_vertical_speed(points: list[tuple[float, float, float]]) -> float:
        if len(points) < 2:
            return 0.0

        speeds: list[float] = []
        for (t1, _, y1), (t2, _, y2) in zip(points, points[1:]):
            delta_t = max(t2 - t1, 1e-6)
            speeds.append((y2 - y1) / delta_t)

        return sum(speeds) / len(speeds)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        detection: BallDetection | None,
        tracking_center: tuple[int, int] | None,
        confidence_percent: int,
        predicted: bool,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        kick_zone_y = int(height * self.config.kick_zone_ratio)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, kick_zone_y), (width, height), (20, 44, 28), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        cv2.line(frame, (0, kick_zone_y), (width, kick_zone_y), (106, 240, 151), 2)
        cv2.putText(
            frame,
            "JUGGLE ZONE",
            (18, max(kick_zone_y - 12, 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (168, 250, 198),
            2,
            cv2.LINE_AA,
        )

        if self.show_trail:
            trail_points = list(self.trail)
            for index in range(1, len(trail_points)):
                start = trail_points[index - 1]
                end = trail_points[index]
                blend = index / max(len(trail_points) - 1, 1)
                if start[2] or end[2]:
                    color = (103, 150, 162)
                    thickness = 2
                else:
                    color = (
                        int(58 + (38 * blend)),
                        int(210 + (25 * blend)),
                        int(170 + (70 * blend)),
                    )
                    thickness = max(2, int(2 + blend * 5))
                cv2.line(frame, (start[0], start[1]), (end[0], end[1]), color, thickness, cv2.LINE_AA)

        hud = frame.copy()
        cv2.rectangle(hud, (16, 16), (332, 140), (7, 16, 20), -1)
        cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)

        cv2.putText(
            frame,
            f"Touches: {self.total_juggles}",
            (32, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            (239, 246, 242),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.last_status,
            (32, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (129, 236, 176) if detection else (152, 211, 235) if predicted else (245, 196, 104),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Confidence {confidence_percent}%",
            (32, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (185, 214, 204),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Streak {self.current_streak} | Speed {int(round(math.hypot(*self.current_velocity)))} px/s",
            (32, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (185, 214, 204),
            1,
            cv2.LINE_AA,
        )

        if tracking_center:
            marker_radius = detection.radius if detection else max(self.config.min_radius + 6, 14)
            if predicted and not detection:
                cv2.circle(frame, tracking_center, marker_radius + 3, (166, 194, 209), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "Kalman",
                    (tracking_center[0] + marker_radius + 8, tracking_center[1] - marker_radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (166, 194, 209),
                    1,
                    cv2.LINE_AA,
                )
            else:
                pulse_color = (96, 252, 157) if self.touch_flash_frames else (54, 227, 199)
                outer_radius = marker_radius + (8 if self.touch_flash_frames else 4)
                cv2.circle(frame, tracking_center, outer_radius, pulse_color, 3, cv2.LINE_AA)
                cv2.circle(frame, tracking_center, 3, (240, 248, 244), -1, cv2.LINE_AA)

            self._draw_velocity_arrow(frame, tracking_center)

        return frame

    def _draw_velocity_arrow(self, frame: np.ndarray, center: tuple[int, int]) -> None:
        speed = math.hypot(*self.current_velocity)
        if speed < 24.0:
            return

        length = max(24.0, min(108.0, speed * self.config.velocity_arrow_scale))
        end_x = int(center[0] + (self.current_velocity[0] / speed) * length)
        end_y = int(center[1] + (self.current_velocity[1] / speed) * length)
        cv2.arrowedLine(frame, center, (end_x, end_y), (248, 153, 88), 3, cv2.LINE_AA, tipLength=0.24)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(value, high))
