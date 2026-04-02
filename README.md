# Futbol Flow Counter

Futbol Flow is a desktop Python app that uses OpenCV to track a soccer ball and count juggling touches when the ball reverses from falling to rising inside the kick zone.

## Features

- Live webcam mode and local video-file mode
- OpenCV ball detection using motion filtering, contour analysis, and Hough circle fallback
- Kalman-assisted prediction that keeps the track alive through short occlusions
- Rebound-based juggle counting instead of raw motion counting
- Desktop dashboard with live count, timer, velocity, streaks, saved-session history, and tuning sliders
- Local SQLite session logging with personal bests, best streaks, and recent-session summaries
- Adjustable kick zone, rebound sensitivity, and motion area thresholds

## Setup

1. Create a local virtual environment:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

4. Launch the app:

```bash
python3 app.py
```

5. Run the tracker logic tests:

```bash
python3 -m unittest discover -s tests -v
```

## How It Counts

The tracker looks for a moving circular object, stores a short trail of ball positions, and counts a touch when:

- the ball drops into the lower kick zone
- recent motion is clearly downward
- that motion reverses upward quickly enough
- short occlusions are bridged with a Kalman filter so the track can recover smoothly

This works best with:

- one ball in frame
- a steady camera angle
- enough distance to keep your feet and the full ball visible
- a background that contrasts with the ball and your clothing

## Tuning Tips

- If the app misses touches, lower the kick zone or reduce rebound speed slightly.
- If the app counts noise, raise rebound speed or increase motion area.
- Mirror mode only affects webcam input, which usually feels better during training.

## Notes

This is a lightweight OpenCV tracker, not a trained sports model, so the cleanest results come from good lighting and a clear practice space.

Session history is stored locally in `futbol_flow.db`.
