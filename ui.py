from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from session_controller import SessionController
from session_store import SessionRecord
from tracker import JuggleTracker, TrackerMetrics


class FutbolFlowApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Futbol Flow Counter")
        self.root.geometry("1480x900")
        self.root.minsize(1260, 800)
        self.root.configure(bg="#071013")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.tracker = JuggleTracker()
        self.session_controller = SessionController()
        self.history_summary = self.session_controller.summary

        self.capture: cv2.VideoCapture | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.current_source_name = "No source selected"
        self.current_source_kind = "none"
        self.is_paused = False
        self.loop_interval_ms = 15
        self.loop_after_id: str | None = None

        self.count_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready for kickoff")
        self.record_var = tk.StringVar(value="First saved session becomes your personal best.")
        self.session_var = tk.StringVar(value="00:00")
        self.confidence_var = tk.StringVar(value="0%")
        self.speed_var = tk.StringVar(value="0 px/s")
        self.streak_var = tk.StringVar(value="0")
        self.last_touch_var = tk.StringVar(value="Awaiting first touch")
        self.avg_gap_var = tk.StringVar(value="--")
        self.personal_best_var = tk.StringVar(value=str(self.history_summary.personal_best))
        self.best_streak_var = tk.StringVar(value=str(self.history_summary.best_streak))
        self.saved_sessions_var = tk.StringVar(value=str(self.history_summary.sessions_played))
        self.total_touches_var = tk.StringVar(value=str(self.history_summary.total_juggles))
        self.avg_session_var = tk.StringVar(value=self._format_duration_short(self.history_summary.average_duration_seconds))
        self.recent_history_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value=self.current_source_name)
        self.default_tip = "Clear backgrounds, a steady camera, and full lower-body framing all improve tracking."
        self.tip_var = tk.StringVar(
            value=self.default_tip
        )

        self.kick_zone_var = tk.DoubleVar(value=self.tracker.config.kick_zone_ratio * 100.0)
        self.reversal_var = tk.DoubleVar(value=self.tracker.config.reversal_speed)
        self.upward_recovery_var = tk.DoubleVar(value=self.tracker.config.upward_reversal_factor * 100.0)
        self.area_var = tk.IntVar(value=self.tracker.config.min_area)
        self.mirror_var = tk.BooleanVar(value=True)
        self.show_trail_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_history_panel()
        self._apply_settings()
        self._show_placeholder()
        self._schedule_next_loop(0)

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg="#071013")
        shell.pack(fill="both", expand=True, padx=26, pady=22)

        header = tk.Frame(shell, bg="#101b20", highlightbackground="#22323a", highlightthickness=1)
        header.pack(fill="x", pady=(0, 18))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        left_header = tk.Frame(header, bg="#101b20")
        left_header.grid(row=0, column=0, sticky="w", padx=24, pady=22)

        tk.Label(
            left_header,
            text="COMPUTER VISION TRAINING SUITE",
            bg="#183128",
            fg="#92f2b4",
            font=("Avenir Next", 10, "bold"),
            padx=12,
            pady=6,
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            left_header,
            text="Futbol Flow",
            bg="#101b20",
            fg="#f2f8f4",
            font=("Avenir Next", 32, "bold"),
        ).pack(anchor="w")

        tk.Label(
            left_header,
            text="Predictive ball tracking, rebound-based juggle counting, and a saved history of every session.",
            bg="#101b20",
            fg="#b4c8c0",
            font=("Avenir Next", 13),
        ).pack(anchor="w", pady=(8, 0))

        right_header = tk.Frame(header, bg="#101b20")
        right_header.grid(row=0, column=1, sticky="e", padx=24, pady=22)

        for label, value in (
            ("Tracking", "Kalman Assisted"),
            ("Storage", "SQLite Session Log"),
            ("Overlay", "Velocity + Trail"),
        ):
            pill = tk.Frame(right_header, bg="#152329", highlightbackground="#253740", highlightthickness=1)
            pill.pack(fill="x", pady=4)
            tk.Label(
                pill,
                text=label,
                bg="#152329",
                fg="#8ea5a1",
                font=("Avenir Next", 9, "bold"),
            ).pack(anchor="w", padx=12, pady=(9, 0))
            tk.Label(
                pill,
                text=value,
                bg="#152329",
                fg="#eef7f2",
                font=("Avenir Next", 11, "bold"),
            ).pack(anchor="w", padx=12, pady=(3, 9))

        body = tk.Frame(shell, bg="#071013")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        self.video_card = tk.Frame(body, bg="#10191d", highlightbackground="#203038", highlightthickness=1)
        self.video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.video_card.grid_columnconfigure(0, weight=1)
        self.video_card.grid_rowconfigure(1, weight=1)

        tk.Label(
            self.video_card,
            text="Live Frame",
            bg="#10191d",
            fg="#f0f6f2",
            font=("Avenir Next", 18, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))

        self.video_label = tk.Label(
            self.video_card,
            bg="#061013",
            fg="#eef6f2",
            font=("Avenir Next", 20, "bold"),
            justify="center",
            bd=0,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=12)

        footer = tk.Frame(self.video_card, bg="#10191d")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=1)

        tk.Label(
            footer,
            textvariable=self.source_var,
            bg="#10191d",
            fg="#b8cbc4",
            font=("Avenir Next", 11),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            footer,
            textvariable=self.tip_var,
            bg="#10191d",
            fg="#7f9690",
            font=("Avenir Next", 11),
            anchor="e",
        ).grid(row=0, column=1, sticky="e")

        sidebar = tk.Frame(body, bg="#071013", width=390)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)

        self._build_score_card(sidebar)
        self._build_history_card(sidebar)
        self._build_control_card(sidebar)
        self._build_settings_card(sidebar)

    def _build_score_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg="#0f1a1f", highlightbackground="#21333b", highlightthickness=1)
        card.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text="Session Count",
            bg="#0f1a1f",
            fg="#a0b7af",
            font=("Avenir Next", 11, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 4))

        tk.Label(
            card,
            textvariable=self.count_var,
            bg="#0f1a1f",
            fg="#91f3af",
            font=("Avenir Next", 52, "bold"),
        ).pack(anchor="w", padx=18)

        self.status_pill = tk.Label(
            card,
            textvariable=self.status_var,
            bg="#20352a",
            fg="#d8ffe4",
            font=("Avenir Next", 11, "bold"),
            padx=12,
            pady=7,
        )
        self.status_pill.pack(anchor="w", padx=18, pady=(4, 10))

        self.record_banner = tk.Label(
            card,
            textvariable=self.record_var,
            bg="#1d2f35",
            fg="#d7eff2",
            font=("Avenir Next", 10, "bold"),
            padx=12,
            pady=8,
        )
        self.record_banner.pack(fill="x", padx=18, pady=(0, 14))

        details = tk.Frame(card, bg="#0f1a1f")
        details.pack(fill="x", padx=18, pady=(0, 10))
        details.grid_columnconfigure(0, weight=1)
        details.grid_columnconfigure(1, weight=1)

        self._metric_tile(details, "Session Clock", self.session_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._metric_tile(details, "Tracking", self.confidence_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._metric_tile(details, "Ball Speed", self.speed_var).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(12, 0))
        self._metric_tile(details, "Current Streak", self.streak_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(12, 0),
        )

        info = tk.Frame(card, bg="#0f1a1f")
        info.pack(fill="x", padx=18, pady=(10, 18))
        info.grid_columnconfigure(0, weight=1)
        info.grid_columnconfigure(1, weight=1)

        self._info_row(info, "Last Touch", self.last_touch_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._info_row(info, "Avg Gap", self.avg_gap_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _build_history_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg="#102228", highlightbackground="#24424b", highlightthickness=1)
        card.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text="History",
            bg="#102228",
            fg="#92f2b4",
            font=("Avenir Next", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 8))

        summary_row = tk.Frame(card, bg="#102228")
        summary_row.pack(fill="x", padx=18)
        summary_row.grid_columnconfigure(0, weight=1)
        summary_row.grid_columnconfigure(1, weight=1)
        summary_row.grid_columnconfigure(2, weight=1)

        self._mini_tile(summary_row, "Personal Best", self.personal_best_var).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        self._mini_tile(summary_row, "Best Streak", self.best_streak_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6,
        )
        self._mini_tile(summary_row, "Sessions", self.saved_sessions_var).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=(6, 0),
        )

        lower_row = tk.Frame(card, bg="#102228")
        lower_row.pack(fill="x", padx=18, pady=(12, 0))
        lower_row.grid_columnconfigure(0, weight=1)
        lower_row.grid_columnconfigure(1, weight=1)

        self._mini_tile(lower_row, "All-Time Touches", self.total_touches_var).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8),
        )
        self._mini_tile(lower_row, "Avg Session", self.avg_session_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )

        tk.Label(
            card,
            text="Recent Sessions",
            bg="#102228",
            fg="#d8ebe4",
            font=("Avenir Next", 11, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 6))

        self.recent_history_label = tk.Label(
            card,
            textvariable=self.recent_history_var,
            bg="#102228",
            fg="#d8ebe4",
            font=("Avenir Next", 10),
            justify="left",
            anchor="w",
            wraplength=338,
        )
        self.recent_history_label.pack(fill="x", padx=18, pady=(0, 18))

    def _build_control_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg="#0f1a1f", highlightbackground="#21333b", highlightthickness=1)
        card.pack(fill="x", pady=(0, 16))

        tk.Label(
            card,
            text="Controls",
            bg="#0f1a1f",
            fg="#eef6f2",
            font=("Avenir Next", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 14))

        button_grid = tk.Frame(card, bg="#0f1a1f")
        button_grid.pack(fill="x", padx=18, pady=(0, 18))
        button_grid.grid_columnconfigure(0, weight=1)
        button_grid.grid_columnconfigure(1, weight=1)

        self._make_button(button_grid, "Start Webcam", self.start_webcam, "#78e39b", "#06110a").grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8),
            pady=(0, 10),
        )
        self._make_button(button_grid, "Open Video", self.open_video, "#1b2c33", "#f0f7f3").grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(0, 10),
        )
        self.pause_button = self._make_button(button_grid, "Pause", self.toggle_pause, "#1b2c33", "#f0f7f3")
        self.pause_button.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self._make_button(button_grid, "Reset Count", self.reset_session, "#1b2c33", "#f0f7f3").grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )

    def _build_settings_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg="#0f1a1f", highlightbackground="#21333b", highlightthickness=1)
        card.pack(fill="x")

        tk.Label(
            card,
            text="Tuning",
            bg="#0f1a1f",
            fg="#eef6f2",
            font=("Avenir Next", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 4))

        tk.Label(
            card,
            text="Adjust the kick zone and sensitivity to match your room, camera angle, and ball size.",
            bg="#0f1a1f",
            fg="#9eb3ad",
            font=("Avenir Next", 11),
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=18, pady=(0, 12))

        self._slider_block(card, "Kick Zone Height", self.kick_zone_var, 55, 85, self._on_slider_change, "%")
        self._slider_block(card, "Rebound Speed", self.reversal_var, 90, 320, self._on_slider_change, " px/s")
        self._slider_block(card, "Upward Recovery", self.upward_recovery_var, 35, 90, self._on_slider_change, "%")
        self._slider_block(card, "Motion Area", self.area_var, 80, 360, self._on_slider_change, " px")

        toggles = tk.Frame(card, bg="#0f1a1f")
        toggles.pack(fill="x", padx=18, pady=(8, 18))

        self._toggle_row(toggles, "Mirror Webcam Feed", self.mirror_var).pack(fill="x", pady=(0, 10))
        self._toggle_row(toggles, "Show Trajectory Trail", self.show_trail_var).pack(fill="x")

    def _metric_tile(self, parent: tk.Frame, title: str, variable: tk.StringVar) -> tk.Frame:
        tile = tk.Frame(parent, bg="#142329", highlightbackground="#22353d", highlightthickness=1)
        tk.Label(
            tile,
            text=title,
            bg="#142329",
            fg="#88a09a",
            font=("Avenir Next", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(
            tile,
            textvariable=variable,
            bg="#142329",
            fg="#eef6f2",
            font=("Avenir Next", 14, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 10))
        return tile

    def _mini_tile(self, parent: tk.Frame, title: str, variable: tk.StringVar) -> tk.Frame:
        tile = tk.Frame(parent, bg="#123038", highlightbackground="#23434b", highlightthickness=1)
        tk.Label(
            tile,
            text=title,
            bg="#123038",
            fg="#8fb7af",
            font=("Avenir Next", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(
            tile,
            textvariable=variable,
            bg="#123038",
            fg="#eef7f2",
            font=("Avenir Next", 14, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 10))
        return tile

    def _info_row(self, parent: tk.Frame, title: str, variable: tk.StringVar) -> tk.Frame:
        row = tk.Frame(parent, bg="#142329", highlightbackground="#22353d", highlightthickness=1)
        tk.Label(
            row,
            text=title,
            bg="#142329",
            fg="#88a09a",
            font=("Avenir Next", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(
            row,
            textvariable=variable,
            bg="#142329",
            fg="#eef6f2",
            font=("Avenir Next", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 8))
        return row

    def _slider_block(
        self,
        parent: tk.Frame,
        title: str,
        variable: tk.Variable,
        minimum: int,
        maximum: int,
        command,
        suffix: str,
    ) -> None:
        block = tk.Frame(parent, bg="#0f1a1f")
        block.pack(fill="x", padx=18, pady=(0, 10))

        value_label = tk.Label(block, text="", bg="#0f1a1f", fg="#d4e6df", font=("Avenir Next", 11, "bold"))

        def refresh_value(_: str | None = None) -> None:
            raw = variable.get()
            value = int(round(float(raw)))
            value_label.config(text=f"{value}{suffix}")
            command()

        tk.Label(
            block,
            text=title,
            bg="#0f1a1f",
            fg="#8ea6a0",
            font=("Avenir Next", 10, "bold"),
        ).pack(anchor="w")

        value_label.pack(anchor="e")

        tk.Scale(
            block,
            from_=minimum,
            to=maximum,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            variable=variable,
            command=refresh_value,
            bg="#0f1a1f",
            fg="#f0f7f3",
            activebackground="#92f2b4",
            highlightthickness=0,
            troughcolor="#1f3138",
            sliderrelief="flat",
            bd=0,
        ).pack(fill="x")
        refresh_value()

    def _toggle_row(self, parent: tk.Frame, title: str, variable: tk.BooleanVar) -> tk.Frame:
        row = tk.Frame(parent, bg="#142329", highlightbackground="#22353d", highlightthickness=1)
        tk.Label(
            row,
            text=title,
            bg="#142329",
            fg="#edf7f1",
            font=("Avenir Next", 11, "bold"),
        ).pack(side="left", padx=12, pady=10)

        tk.Checkbutton(
            row,
            variable=variable,
            command=self._apply_settings,
            bg="#142329",
            activebackground="#142329",
            selectcolor="#142329",
            fg="#91f3af",
            activeforeground="#91f3af",
            highlightthickness=0,
            text="On",
            font=("Avenir Next", 11, "bold"),
        ).pack(side="right", padx=12)
        return row

    def _make_button(self, parent: tk.Frame, text: str, command, bg: str, fg: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Avenir Next", 11, "bold"),
            padx=14,
            pady=12,
        )

    def _apply_settings(self) -> None:
        self.tracker.config.kick_zone_ratio = float(self.kick_zone_var.get()) / 100.0
        self.tracker.config.reversal_speed = float(self.reversal_var.get())
        self.tracker.config.upward_reversal_factor = float(self.upward_recovery_var.get()) / 100.0
        self.tracker.config.min_area = int(self.area_var.get())
        self.tracker.show_trail = bool(self.show_trail_var.get())
        if not self.tracker.show_trail:
            self.tracker.trail.clear()

    def _on_slider_change(self) -> None:
        self._apply_settings()

    def _refresh_history_panel(self) -> None:
        self.history_summary = self.session_controller.refresh_summary()
        self.personal_best_var.set(str(self.history_summary.personal_best))
        self.best_streak_var.set(str(self.history_summary.best_streak))
        self.saved_sessions_var.set(str(self.history_summary.sessions_played))
        self.total_touches_var.set(str(self.history_summary.total_juggles))
        self.avg_session_var.set(self._format_duration_short(self.history_summary.average_duration_seconds))
        self.recent_history_var.set(self._format_recent_sessions(self.session_controller.store.fetch_recent_sessions(4)))

    def start_webcam(self) -> None:
        self._open_capture(0, "Webcam 0", "camera")

    def open_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Open a soccer juggling video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if not file_path:
            return

        self._open_capture(file_path, Path(file_path).name, "video")

    def _open_capture(self, source: int | str, source_name: str, source_kind: str) -> None:
        self._save_session_if_needed("switch-source")
        self._release_capture()

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            messagebox.showerror(
                "Unable to open source",
                "The camera or video file could not be opened. Check permissions or try a different source.",
            )
            return

        if source_kind == "camera":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.capture = capture
        self.current_source_name = f"Source: {source_name}"
        self.current_source_kind = source_kind
        self.source_var.set(self.current_source_name)
        self.tip_var.set("Keep the ball in the lower half of frame to help the rebound detector lock in cleanly.")
        self.tracker.restart()
        self._apply_settings()
        self._reset_session_clock()
        self.is_paused = False
        self.pause_button.config(text="Pause")
        self._set_status_text("Live tracking ready")

    def toggle_pause(self) -> None:
        if not self.capture:
            return

        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")

        if self.is_paused:
            self.session_controller.pause()
            self._set_status_text("Paused")
        else:
            self.session_controller.resume()
            self._set_status_text("Tracking resumed")

    def reset_session(self) -> None:
        self._save_session_if_needed("reset")
        self.tracker.restart()
        self._apply_settings()
        self.count_var.set("0")
        self.confidence_var.set("0%")
        self.speed_var.set("0 px/s")
        self.streak_var.set("0")
        self.last_touch_var.set("Awaiting first touch")
        self.avg_gap_var.set("--")
        self.tip_var.set("Reset complete. Start moving the ball again to build a fresh juggle count.")
        self._reset_session_clock()
        self._set_status_text("Session reset")

    def _reset_session_clock(self) -> None:
        self.session_controller.begin_session()
        self.session_var.set("00:00")
        self._update_record_banner(0)

    def _save_session_if_needed(self, reason: str) -> None:
        saved = self.session_controller.persist_if_needed(
            total_juggles=self.tracker.total_juggles,
            average_touch_interval=self.tracker.average_touch_interval,
            best_streak=self.tracker.best_streak,
            source_name=self.current_source_name,
            reason=reason,
        )
        if saved:
            self._refresh_history_panel()
            self.tip_var.set("Session saved to local history. Open a new source or reset to keep training.")

    def _release_capture(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

    def _show_placeholder(self) -> None:
        self.video_label.config(
            image="",
            text="Start Webcam\nor\nOpen Video\n\nFutbol Flow tracks the ball, predicts through brief occlusion,\nand saves every session locally.",
            bg="#061013",
            fg="#eef6f2",
            width=70,
            height=24,
        )
        self.photo_image = None

    def _render_frame(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((1000, 720))
        self.photo_image = ImageTk.PhotoImage(image)
        self.video_label.config(image=self.photo_image, text="")

    def _set_status_text(self, status: str) -> None:
        self.status_var.set(status)

        if "Juggle counted" in status:
            self.status_pill.config(bg="#2c5e3b", fg="#e8fff0")
        elif "Predicting" in status:
            self.status_pill.config(bg="#244254", fg="#dceffd")
        elif "Tracking" in status:
            self.status_pill.config(bg="#214d40", fg="#ddfff1")
        elif "Paused" in status:
            self.status_pill.config(bg="#4d4021", fg="#fff1cf")
        elif "ready" in status.lower() or "reset" in status.lower():
            self.status_pill.config(bg="#20352a", fg="#d8ffe4")
        else:
            self.status_pill.config(bg="#433827", fg="#ffe2ad")

    def _update_record_banner(self, total_juggles: int) -> None:
        if total_juggles > self.session_controller.personal_best_target:
            self.record_var.set(f"NEW RECORD · {total_juggles} touches")
            self.record_banner.config(bg="#214b2c", fg="#e8fff0")
        elif self.session_controller.personal_best_target > 0:
            remaining = max(self.session_controller.personal_best_target - total_juggles, 0)
            self.record_var.set(f"{remaining} to beat your PB of {self.session_controller.personal_best_target}")
            self.record_banner.config(bg="#1d2f35", fg="#d7eff2")
        else:
            self.record_var.set("First saved session becomes your personal best.")
            self.record_banner.config(bg="#1d2f35", fg="#d7eff2")

    def _update_tip(self, metrics: TrackerMetrics) -> None:
        if metrics.predicted:
            self.tip_var.set("Kalman prediction is carrying the trail through a brief occlusion.")
        elif metrics.status_text == "Juggle counted":
            self.tip_var.set("Touch logged. Keep the ball in the kick zone to build your streak.")
        elif metrics.status_text == "Searching for the ball" and metrics.lost_frames >= 20:
            self.tip_var.set("Tracking lost. Try brighter lighting, step back a little, or keep the full ball visible.")
        elif metrics.status_text == "Searching for the ball" and metrics.lost_frames >= 8:
            self.tip_var.set("Ball slipping out of track. Keep the full ball clear of your legs and near the kick zone.")
        elif metrics.status_text == "Tracking locked":
            self.tip_var.set(self.default_tip)

    def _update_metrics(self, metrics: TrackerMetrics) -> None:
        self.count_var.set(str(metrics.total_juggles))
        self._set_status_text(metrics.status_text)
        self.confidence_var.set(f"{metrics.confidence_percent}%")
        self.speed_var.set(f"{metrics.current_speed} px/s")
        self.streak_var.set(str(metrics.current_streak))

        if metrics.last_touch_seconds is None:
            self.last_touch_var.set("Awaiting first touch")
        else:
            self.last_touch_var.set(f"{metrics.last_touch_seconds:0.1f}s ago")

        if metrics.average_touch_interval is None:
            self.avg_gap_var.set("--")
        else:
            self.avg_gap_var.set(f"{metrics.average_touch_interval:0.2f}s")

        self._update_tip(metrics)

        if self.session_controller.mark_record_if_needed(metrics.total_juggles):
            self.tip_var.set("New personal best. This session will be saved to your history panel.")

        live_personal_best = self.session_controller.live_personal_best(metrics.total_juggles)
        live_best_streak = self.session_controller.live_best_streak(metrics.best_streak)
        self.personal_best_var.set(str(live_personal_best))
        self.best_streak_var.set(str(live_best_streak))
        self._update_record_banner(metrics.total_juggles)

    def _update_session_clock(self) -> None:
        elapsed = self.session_controller.elapsed_seconds()
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.session_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.session_var.set(f"{minutes:02d}:{seconds:02d}")

    def _handle_source_end(self) -> None:
        self.is_paused = True
        self.pause_button.config(text="Resume")
        self._set_status_text("Source ended")
        self.tip_var.set("Session finished. Open another clip or reset to start a new tracked run.")
        self._save_session_if_needed("source-ended")

    def _schedule_next_loop(self, delay_ms: int | None = None) -> None:
        next_delay = self.loop_interval_ms if delay_ms is None else delay_ms
        self.loop_after_id = self.root.after(next_delay, self._update_loop)

    def _update_loop(self) -> None:
        loop_started = time.perf_counter()
        if self.capture and self.capture.isOpened():
            self._update_session_clock()

            if not self.is_paused:
                ok, frame = self.capture.read()

                if ok:
                    if self.current_source_kind == "camera" and self.mirror_var.get():
                        frame = cv2.flip(frame, 1)

                    annotated, metrics = self.tracker.process_frame(frame)
                    self._render_frame(annotated)
                    self._update_metrics(metrics)
                else:
                    self._handle_source_end()

        elapsed_ms = int((time.perf_counter() - loop_started) * 1000)
        self._schedule_next_loop(max(5, self.loop_interval_ms - elapsed_ms))

    def on_close(self) -> None:
        if self.loop_after_id is not None:
            self.root.after_cancel(self.loop_after_id)
            self.loop_after_id = None
        self._save_session_if_needed("close")
        self._release_capture()
        self.root.destroy()

    @staticmethod
    def _format_duration_short(duration_seconds: float) -> str:
        total_seconds = int(round(duration_seconds))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        return f"{minutes:d}m {seconds:02d}s"

    def _format_recent_sessions(self, sessions: list[SessionRecord]) -> str:
        if not sessions:
            return "No saved sessions yet.\nYour first tracked run will appear here."

        lines: list[str] = []
        for session in sessions:
            started_at = datetime.fromisoformat(session.started_at)
            stamp = started_at.strftime("%b %d %I:%M %p")
            gap_text = (
                f" · gap {session.average_touch_interval:0.2f}s"
                if session.average_touch_interval is not None
                else ""
            )
            lines.append(
                f"{stamp} · {session.total_juggles} touches · {self._format_duration_short(session.duration_seconds)}{gap_text}"
            )
        return "\n".join(lines)
