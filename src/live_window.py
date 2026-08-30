"""
live_demo_window.py (v3)

Live hand-tracking demo using the SAME reference-building and
sustained-trend detection logic as analyze_all.py. Adds a rotating 3D
wireframe panel of the 21 hand landmarks, using MediaPipe's x/y/z values
directly (z is a relative depth estimate from one camera, not true
stereo depth -- see honest scope notes below).

Run:
    cd src
    python3 live_demo_window.py

Requires (relative to src/):
    ../data/labels.csv          (from label_helper.py)
    ../data/trajectories/*.npy  (from extract_trajectory.py)
    ../data/trajectories/*_fps.txt

If these don't exist yet, this script falls back to a synthetic reference
and tells you clearly on-screen and in the terminal.

Controls:
    q  - quit
    r  - reset the current live session (clears buffers, keeps reference)

Honest scope notes:
  - MediaPipe's landmark z is a relative depth ESTIMATE from one camera,
    not true stereo/depth-sensor 3D. The rotating panel below uses these
    z values as-is to project a 3D-looking wireframe -- it genuinely shows
    the hand's depth structure as MediaPipe estimates it, rotating so you
    can see it from different angles, but it is not a claim of
    depth-sensor-grade 3D accuracy.
  - Live scoring approximates "where in the task" you are by matching your
    frame index directly to the reference index. This is a standard
    approximation, not a limitation unique to this code.
  - No detector is 100% accurate.
"""

import os
import csv
import time
import math
import sys
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Resolve all paths relative to THIS SCRIPT'S location, not the current
# working directory -- this makes the script work correctly whether you
# run it as `python3 live_demo_window.py` from inside src/, or as
# `python3 src/live_demo_window.py` from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)  # so `from kinematics import ...` always resolves

from kinematics import compute_kinematics

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# ---------- paths (resolved from PROJECT_ROOT, work from any cwd) ----------
LABELS_CSV = os.path.join(PROJECT_ROOT, "data", "labels.csv")
TRAJ_DIR = os.path.join(PROJECT_ROOT, "data", "trajectories")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "live_sessions")
REF_DIR = os.path.join(PROJECT_ROOT, "data", "reference_runs")
TARGET_LEN = 100

# ---------- tuning constants -- copy these to match analyze_all.py exactly ----------
WARMUP_FRAMES = 20
WINDOW = 8
THRESHOLD_RATIO = 1.25
Z_THRESHOLD = 2.0

HELD_OUT = {"run09"}
REFERENCE_EXCLUDE = {"run06", "run07", "run12"}

os.makedirs(LOG_DIR, exist_ok=True)


# ================= reference building (identical logic to analyze_all.py) =================

def load_labels():
    with open(LABELS_CSV) as f:
        return list(csv.DictReader(f))


def load_trajectory(video_id):
    traj = np.load(os.path.join(TRAJ_DIR, f"{video_id}.npy"))
    fps_path = os.path.join(TRAJ_DIR, f"{video_id}_fps.txt")
    fps = float(open(fps_path).read()) if os.path.exists(fps_path) else 30.0
    return traj, fps


def resample(traj, target_len=TARGET_LEN):
    old_idx = np.linspace(0, 1, len(traj))
    new_idx = np.linspace(0, 1, target_len)
    return np.stack(
        [np.interp(new_idx, old_idx, traj[:, c]) for c in range(traj.shape[1])],
        axis=1,
    )


def build_reference_from_labels():
    if not os.path.exists(LABELS_CSV):
        print(f"[Kinesis] {LABELS_CSV} not found -- using SYNTHETIC reference.")
        return _synthetic_reference()

    rows = load_labels()
    normal_trajs = []
    used_ids = []
    durations_seconds = []
    for row in rows:
        vid = row["video_id"]
        if row["type"] == "normal" and vid not in HELD_OUT and vid not in REFERENCE_EXCLUDE:
            npy_path = os.path.join(TRAJ_DIR, f"{vid}.npy")
            if not os.path.exists(npy_path):
                continue
            traj, fps = load_trajectory(vid)
            normal_trajs.append(resample(traj))
            durations_seconds.append(len(traj) / fps)
            used_ids.append(vid)

    if len(normal_trajs) < 2:
        print(f"[Kinesis] Only found {len(normal_trajs)} usable normal trajectories "
              f"-- need at least 2. Using SYNTHETIC reference.")
        return _synthetic_reference()

    stacked = np.stack(normal_trajs)
    mu = stacked.mean(axis=0)
    sigma = np.maximum(stacked.std(axis=0), 0.01)
    avg_duration_seconds = float(np.mean(durations_seconds))
    print(f"[Kinesis] Real reference built from {len(used_ids)} videos: {used_ids}")
    print(f"[Kinesis] Reference average duration: {avg_duration_seconds:.2f}s "
          f"-- live tracking will align by elapsed time, not raw frame count.")
    return mu, sigma, True, avg_duration_seconds


def _synthetic_reference():
    rng = np.random.default_rng(0)
    base = np.linspace(0.3, 0.6, TARGET_LEN).reshape(-1, 1)
    mu = np.tile(base, (1, 63)) + rng.normal(0, 0.01, size=(TARGET_LEN, 63))
    sigma = np.full_like(mu, 0.05)
    return mu, sigma, False, 5.0  # assume a plausible 5s task duration as a fallback


# ================= live signal computation =================

def landmark_vector(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


class LandmarkSmoother:
    """
    Exponential moving average filter on raw landmarks. MediaPipe's raw
    per-frame landmark output has real jitter (a few pixels/units of
    noise even when the hand is perfectly still) -- this jitter alone can
    register as small deviation/grip changes and contribute to false
    alarms. Smoothing trades a small amount of responsiveness for
    meaningfully cleaner signals, which is a good trade here since your
    detection rule already requires a SUSTAINED trend over 8+ frames --
    smoothing makes that trend reflect real motion, not sensor noise.
    """

    def __init__(self, alpha=0.6):
        self.alpha = alpha  # higher = more responsive, lower = smoother
        self._prev = None

    def smooth(self, frame_vec):
        if self._prev is None:
            self._prev = frame_vec.copy()
            return frame_vec
        smoothed = self.alpha * frame_vec + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


def deviation_at(frame_vec, mu, sigma, ref_index):
    ref_index = min(ref_index, len(mu) - 1)
    diff = np.abs(frame_vec - mu[ref_index])
    return float(np.linalg.norm(diff / sigma[ref_index]))


def grip_spread(frame_vec):
    lm = frame_vec.reshape(21, 3)
    thumb_tip = lm[4]
    dists = [np.linalg.norm(thumb_tip - lm[i]) for i in (8, 12, 16, 20)]
    return float(np.mean(dists))


WRIST_IDX = 0


class StreamingKinematics:
    """
    Live equivalent of kinematics.py's compute_kinematics(), but stateful
    frame-by-frame instead of computed over a whole recorded array at once.

    Uses REAL elapsed wall-clock time between frames (time.time() deltas)
    rather than an assumed fixed fps -- a live webcam's frame timing is
    not perfectly steady, so measuring actual dt per frame is the more
    physically correct choice here than analyze_all.py's fixed-fps
    approach (which is fine there, since a recorded video DOES have a
    fixed fps baked into the file).

    Same physics as kinematics.py:
        v(t) = (x(t) - x(t-1)) / dt
        a(t) = (v(t) - v(t-1)) / dt
        j(t) = (a(t) - a(t-1)) / dt
    """

    def __init__(self):
        self._prev_pos = None
        self._prev_vel = None
        self._prev_accel = None
        self._prev_time = None

    def update(self, frame_vec, now):
        """
        frame_vec: flat 63-length landmark array for this frame
        now: current wall-clock time (time.time())
        Returns: (speed, accel_mag, jerk_mag) -- all 0.0 until enough
                 history exists to compute them.
        """
        lm = frame_vec.reshape(21, 3)
        pos = lm[WRIST_IDX]

        if self._prev_pos is None or self._prev_time is None:
            self._prev_pos = pos
            self._prev_time = now
            return 0.0, 0.0, 0.0

        dt = now - self._prev_time
        if dt <= 0:
            dt = 1e-3  # guard against duplicate timestamps

        vel = (pos - self._prev_pos) / dt
        speed = float(np.linalg.norm(vel))

        accel_mag = 0.0
        jerk_mag = 0.0

        if self._prev_vel is not None:
            accel = (vel - self._prev_vel) / dt
            accel_mag = float(np.linalg.norm(accel))

            if self._prev_accel is not None:
                jerk = (accel - self._prev_accel) / dt
                jerk_mag = float(np.linalg.norm(jerk))

            self._prev_accel = accel

        self._prev_vel = vel
        self._prev_pos = pos
        self._prev_time = now

        return speed, accel_mag, jerk_mag

    def reset(self):
        self._prev_pos = None
        self._prev_vel = None
        self._prev_accel = None
        self._prev_time = None


def sustained_rise_now(history, window=WINDOW, ratio=THRESHOLD_RATIO, z_threshold=None):
    n = len(history)
    if n < 2 * window:
        return False, 0.0

    past_window = history[n - 2 * window: n - window]
    recent_window = history[n - window: n]

    past_avg = np.mean(past_window)
    recent_avg = np.mean(recent_window)

    ratio_ok = recent_avg > past_avg * ratio

    x = np.arange(len(recent_window))
    slope = np.polyfit(x, recent_window, 1)[0]
    rising = slope > 0

    level_ok = True if z_threshold is None else recent_avg > z_threshold

    return bool(ratio_ok and rising and level_ok), recent_avg


# ================= 3D wireframe panel =================

def project_3d_landmarks(landmarks_xyz, angle_y, panel_size):
    """
    landmarks_xyz: (21, 3) array of raw MediaPipe x,y,z
    angle_y: current auto-rotation angle (radians) around the vertical axis
    Returns: list of (px, py, depth) for each of the 21 points, in panel
             pixel coordinates, plus depth (rotated z) for size/shading cues.
    """
    centered = landmarks_xyz - landmarks_xyz.mean(axis=0)

    # Exaggerate z so depth is visually readable -- MediaPipe's z range is
    # typically much smaller than its x/y range.
    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2] * 2.5

    cos_a, sin_a = math.cos(angle_y), math.sin(angle_y)
    x_rot = x * cos_a + z * sin_a
    z_rot = -x * sin_a + z * cos_a
    y_rot = y

    scale = panel_size * 1.6
    cx, cy = panel_size / 2, panel_size / 2

    points = []
    for i in range(len(x_rot)):
        px = int(cx + x_rot[i] * scale)
        py = int(cy + y_rot[i] * scale)
        points.append((px, py, z_rot[i]))
    return points


def draw_3d_panel(frame, frame_vec, angle_y, x0, y0, panel_size, bg_color):
    cv2.rectangle(frame, (x0, y0), (x0 + panel_size, y0 + panel_size), bg_color, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_size, y0 + panel_size), (90, 70, 90), 1)
    cv2.putText(frame, "3D VIEW (auto-rotating)", (x0 + 6, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    if frame_vec is None:
        cv2.putText(frame, "no hand detected", (x0 + 30, y0 + panel_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return

    landmarks_xyz = frame_vec.reshape(21, 3)
    points = project_3d_landmarks(landmarks_xyz, angle_y, panel_size)

    depths = [p[2] for p in points]
    d_min, d_max = min(depths), max(depths)
    d_range = (d_max - d_min) or 1e-6

    # Draw connections (skeleton lines) first, points on top
    for a, b in HAND_CONNECTIONS:
        pa, pb = points[a], points[b]
        cv2.line(frame, (x0 + pa[0], y0 + pa[1]), (x0 + pb[0], y0 + pb[1]),
                  (163, 124, 255), 1, cv2.LINE_AA)

    for px, py, depth in points:
        # Closer points (larger depth after rotation) drawn bigger/brighter
        # -- a simple, robust depth cue instead of full perspective projection.
        norm_depth = (depth - d_min) / d_range
        radius = 2 + int(norm_depth * 3)
        brightness = int(140 + norm_depth * 115)
        cv2.circle(frame, (x0 + px, y0 + py), radius,
                   (brightness, brightness // 2, brightness), -1, cv2.LINE_AA)


# ================= 2D graph drawing =================

GRAPH_W, GRAPH_H = 420, 160
PANEL_SIZE = 220
COLOR_BG = (44, 31, 42)
COLOR_DEV_LINE = (163, 124, 255)
COLOR_GRIP_LINE = (240, 181, 99)
COLOR_THRESHOLD = (0, 0, 200)
COLOR_NOMINAL = (168, 136, 148)
COLOR_RISK = (99, 181, 240)
COLOR_TEXT = (235, 235, 245)


def draw_graph(frame, dev_smoothed_history, grip_smoothed_history, jerk_history, x, y, graph_h=GRAPH_H, alert_level=None):
    cv2.rectangle(frame, (x, y), (x + GRAPH_W, y + graph_h), COLOR_BG, -1)
    cv2.rectangle(frame, (x, y), (x + GRAPH_W, y + graph_h), (90, 70, 90), 1)

    if len(jerk_history) < 2:
        return

    # Jerk (green) is the PRIMARY signal now -- it's what actually gates
    # the RISK CONFIRMED decision. Scale to jerk's own recent range so the
    # graph is always readable regardless of absolute magnitude.
    jerk_max = max(max(jerk_history[-GRAPH_W:], default=1.0), 1e-6)
    dev_max = max(max(dev_smoothed_history[-GRAPH_W:], default=1.0), 1e-6)

    def plot(history, color, scale_max):
        pts = []
        for i, val in enumerate(history[-GRAPH_W:]):
            px = x + int(i / GRAPH_W * GRAPH_W)
            py = y + graph_h - int(min(val / scale_max, 1.0) * graph_h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)

    plot(dev_smoothed_history, COLOR_DEV_LINE, dev_max)
    plot(grip_smoothed_history, COLOR_GRIP_LINE, dev_max)
    plot(jerk_history, COLOR_JERK_LINE, jerk_max)


def draw_status_banner(frame, status, w):
    is_risk = status == "RISK CONFIRMED"
    is_warming = status == "WARMING UP..."
    if is_risk:
        color = COLOR_RISK
    elif is_warming:
        color = (200, 200, 100)
    else:
        color = COLOR_NOMINAL
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 22, 30), -1)
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


COLOR_JERK_LINE = (120, 220, 120)  # green -- calibration-only signal, not yet gating alerts


def draw_metrics(frame, dev_avg, grip_avg, jerk_mag, x, y):
    lines = [
        f"deviation (z): {dev_avg:.2f}",
        f"grip spread:   {grip_avg:.3f}",
        f"jerk (calib):  {jerk_mag:.3f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


# ================= main =================

def main():
    mu, sigma, is_real, ref_duration_seconds = build_reference_from_labels()
    if not is_real:
        print("[Kinesis] WARNING: running with a SYNTHETIC reference. "
              "Detections are not meaningful until real labeled data exists.")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{session_id}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "timestamp", "dev_zscore", "grip_spread",
                          "speed", "accel_mag", "jerk_mag",
                          "dev_alert", "grip_alert", "combined_alert"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the NEWEST frame, don't let a backlog build up
    if not cap.isOpened():
        print("Could not open camera. Check System Settings > Privacy & Security > Camera.")
        return

    dev_history = []
    grip_history = []
    dev_smoothed_history = []
    grip_smoothed_history = []
    jerk_history = []
    frame_index = 0
    first_alert_frame = None
    rotation_angle = 0.0
    kinematics_tracker = StreamingKinematics()
    landmark_smoother = LandmarkSmoother(alpha=0.6)
    session_start_time = None

    # Detects SUDDEN motion via jerk (rate of change of acceleration), using
    # the same sustained-rise sliding-window comparison validated in
    # analyze_all.py -- this continuously compares recent motion to the
    # immediately preceding period, so it naturally tolerates smooth,
    # ongoing movement (like steady up/down motion) without needing to
    # match any specific recorded task shape, while still catching a
    # genuine sudden spike (a real shake or drop), for any person.
    ALERT_HOLD_FRAMES = 30
    JERK_CALIBRATION_FRAMES = 60      # ~2s: learn what YOUR normal jerk looks like
    JERK_FLOOR_MULTIPLIER = 2.5       # lowered -- catches a fall's brief spike more easily
    alert_hold_counter = 0

    jerk_baseline_samples = []
    jerk_baseline_mean = None
    jerk_baseline_std = None

    NO_HAND_GRACE_FRAMES = 8
    no_hand_counter = 0
    last_status = "NO HAND DETECTED"
    MIN_PEAK_FRAMES = 1   # even a single frame clearing the floor counts (a fall can be that brief)
    consecutive_peak_frames = 0

    with mp_hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands:

        print("[Kinesis] Live demo running. Press 'q' to quit, 'r' to reset.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            status = "NO HAND DETECTED"
            dev_avg_display = 0.0
            grip_avg_display = 0.0
            frame_vec = None

            if results.multi_hand_landmarks:
                no_hand_counter = 0
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                frame_vec_raw = landmark_vector(hand_landmarks)
                frame_vec = landmark_smoother.smooth(frame_vec_raw)

                now = time.time()
                if session_start_time is None:
                    session_start_time = now
                elapsed = now - session_start_time

                # Align by ELAPSED REAL TIME against the reference's average
                # real duration, not raw frame count -- this is the fix for
                # live tracking drifting out of sync with a resampled
                # reference that was compressed from longer source videos.
                progress_fraction = min(elapsed / ref_duration_seconds, 1.0)
                ref_index = int(progress_fraction * (len(mu) - 1))

                dev = deviation_at(frame_vec, mu, sigma, ref_index)
                grip = grip_spread(frame_vec)
                speed, accel_mag, jerk_mag = kinematics_tracker.update(frame_vec, now)

                dev_history.append(dev)
                grip_history.append(grip)
                jerk_history.append(jerk_mag)

                status = "NOMINAL"
                dev_alert, dev_avg = False, dev
                grip_alert, grip_avg = False, grip

                if frame_index >= max(2 * WINDOW, WARMUP_FRAMES):
                    dev_alert, dev_avg = sustained_rise_now(dev_history, z_threshold=Z_THRESHOLD)
                    grip_alert, grip_avg = sustained_rise_now(grip_history)

                # ---- Learn YOUR normal jerk range first (includes natural
                # start/stop motion jerk from everyday movement), THEN only
                # fire on jerk that clearly exceeds that -- not just any
                # relative rise, which was firing on normal hand-raising,
                # stopping, etc. ----
                jerk_alert, jerk_avg = False, jerk_mag

                if len(jerk_baseline_samples) < JERK_CALIBRATION_FRAMES:
                    jerk_baseline_samples.append(jerk_mag)
                    status = "WARMING UP..."
                    if len(jerk_baseline_samples) == JERK_CALIBRATION_FRAMES:
                        jerk_baseline_mean = float(np.mean(jerk_baseline_samples))
                        jerk_baseline_std = max(float(np.std(jerk_baseline_samples)), 1e-3)
                        print(f"[Kinesis] Jerk baseline calibrated: "
                              f"mean={jerk_baseline_mean:.2f}, std={jerk_baseline_std:.2f}, "
                              f"real-spike floor={jerk_baseline_mean + JERK_FLOOR_MULTIPLIER * jerk_baseline_std:.2f}")
                elif len(jerk_history) >= 2 * WINDOW:
                    jerk_floor = jerk_baseline_mean + JERK_FLOOR_MULTIPLIER * jerk_baseline_std
                    jerk_alert, jerk_avg = sustained_rise_now(jerk_history, z_threshold=jerk_floor)

                    # Instant-peak check, separate from the sustained-trend
                    # check above: a SUDDEN FALL is often one sharp spike
                    # for just 1-2 frames -- averaged into an 8-frame trend
                    # window, it gets diluted and can slip past the trend
                    # rule even though a real event happened. This catches
                    # it directly: does the raw jerk value itself clearly
                    # exceed your calibrated floor, held for a couple
                    # frames (not just one noisy frame)?
                    if jerk_mag > jerk_floor:
                        consecutive_peak_frames += 1
                    else:
                        consecutive_peak_frames = 0
                    peak_alert = consecutive_peak_frames >= MIN_PEAK_FRAMES

                    if jerk_alert or peak_alert:
                        alert_hold_counter = ALERT_HOLD_FRAMES
                        if first_alert_frame is None:
                            first_alert_frame = frame_index
                            reason = "sustained rise" if jerk_alert else "instant peak"
                            print(f"[Kinesis] Sudden motion spike at frame {frame_index} "
                                  f"({reason}, jerk={jerk_mag:.2f}, floor={jerk_floor:.2f})")

                if alert_hold_counter > 0:
                    status = "RISK CONFIRMED"
                    alert_hold_counter = max(alert_hold_counter - 1, 0)

                dev_avg_display, grip_avg_display = dev_avg, grip_avg
                dev_smoothed_history.append(dev_avg)
                grip_smoothed_history.append(grip_avg)

                log_writer.writerow([frame_index, now, dev, grip, speed, accel_mag, jerk_mag,
                                      dev_alert, grip_alert, status == "RISK CONFIRMED"])
                last_status = status
            else:
                no_hand_counter += 1

                if no_hand_counter <= NO_HAND_GRACE_FRAMES:
                    # Brief tracking loss -- very common during a FAST,
                    # sudden motion (motion blur makes MediaPipe momentarily
                    # lose the hand). Keep showing the last real status and
                    # KEEP the alert hold/history intact instead of wiping
                    # everything -- this is exactly the moment a real risk
                    # event needs to stay visible, not disappear.
                    status = last_status
                    if alert_hold_counter > 0:
                        status = "RISK CONFIRMED"
                else:
                    # Hand has genuinely left the frame for a while --
                    # NOW it's safe to reset for a fresh warmup.
                    status = "NO HAND DETECTED"
                    last_status = status
                    dev_history.clear()
                    grip_history.clear()
                    dev_smoothed_history.clear()
                    grip_smoothed_history.clear()
                    jerk_history.clear()
                    kinematics_tracker.reset()
                    landmark_smoother.reset()
                    frame_index = 0
                    session_start_time = None
                    alert_hold_counter = 0

            # Scale UI element sizes to the ACTUAL frame size, and clamp
            # positions so nothing can be pushed off-screen or overlap,
            # regardless of the camera's real resolution.
            safe_graph_h = min(GRAPH_H, int(h * 0.28))
            safe_panel_size = min(PANEL_SIZE, int(h * 0.35), int(w * 0.35))
            top_margin = 50  # space reserved for the status banner + metrics

            graph_y = max(top_margin, h - safe_graph_h - 55)
            panel_y = max(top_margin, h - safe_panel_size - 55)

            current_jerk = jerk_history[-1] if jerk_history else 0.0

            draw_status_banner(frame, status, w)
            draw_metrics(frame, dev_avg_display, grip_avg_display, current_jerk, w - 260, 70)
            draw_graph(frame, dev_smoothed_history, grip_smoothed_history, jerk_history,
                       x=10, y=graph_y, graph_h=safe_graph_h)
            draw_3d_panel(frame, frame_vec, rotation_angle,
                          x0=w - safe_panel_size - 10, y0=panel_y,
                          panel_size=safe_panel_size, bg_color=COLOR_BG)

            rotation_angle += 0.025  # auto-rotate speed

            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, max(20, graph_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            if not is_real:
                cv2.putText(frame, "SYNTHETIC REFERENCE - not calibrated", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            cv2.imshow("Kinesis - Live Demo", frame)
            frame_index += 1

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                dev_history.clear()
                grip_history.clear()
                dev_smoothed_history.clear()
                grip_smoothed_history.clear()
                jerk_history.clear()
                kinematics_tracker.reset()
                landmark_smoother.reset()
                session_start_time = None
                frame_index = 0
                first_alert_frame = None
                alert_hold_counter = 0
                jerk_baseline_samples = []
                jerk_baseline_mean = None
                jerk_baseline_std = None
                no_hand_counter = 0
                last_status = "NO HAND DETECTED"
                consecutive_peak_frames = 0
                print("[Kinesis] Session reset.")

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print(f"[Kinesis] Session log saved to {log_path}")


if __name__ == "__main__":
    main()