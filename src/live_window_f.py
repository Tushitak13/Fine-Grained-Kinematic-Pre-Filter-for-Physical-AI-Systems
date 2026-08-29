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
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# ---------- paths (relative to src/, matching analyze_all.py's layout) ----------
LABELS_CSV = "../data/labels.csv"
TRAJ_DIR = "../data/trajectories"
LOG_DIR = "../data/live_sessions"
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
    for row in rows:
        vid = row["video_id"]
        if row["type"] == "normal" and vid not in HELD_OUT and vid not in REFERENCE_EXCLUDE:
            npy_path = os.path.join(TRAJ_DIR, f"{vid}.npy")
            if not os.path.exists(npy_path):
                continue
            traj, _ = load_trajectory(vid)
            normal_trajs.append(resample(traj))
            used_ids.append(vid)

    if len(normal_trajs) < 2:
        print(f"[Kinesis] Only found {len(normal_trajs)} usable normal trajectories "
              f"-- need at least 2. Using SYNTHETIC reference.")
        return _synthetic_reference()

    stacked = np.stack(normal_trajs)
    mu = stacked.mean(axis=0)
    sigma = np.maximum(stacked.std(axis=0), 0.01)
    print(f"[Kinesis] Real reference built from {len(used_ids)} videos: {used_ids}")
    return mu, sigma, True


def _synthetic_reference():
    rng = np.random.default_rng(0)
    base = np.linspace(0.3, 0.6, TARGET_LEN).reshape(-1, 1)
    mu = np.tile(base, (1, 63)) + rng.normal(0, 0.01, size=(TARGET_LEN, 63))
    sigma = np.full_like(mu, 0.05)
    return mu, sigma, False


# ================= live signal computation =================

def landmark_vector(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def deviation_at(frame_vec, mu, sigma, ref_index):
    ref_index = min(ref_index, len(mu) - 1)
    diff = np.abs(frame_vec - mu[ref_index])
    return float(np.linalg.norm(diff / sigma[ref_index]))


def grip_spread(frame_vec):
    lm = frame_vec.reshape(21, 3)
    thumb_tip = lm[4]
    dists = [np.linalg.norm(thumb_tip - lm[i]) for i in (8, 12, 16, 20)]
    return float(np.mean(dists))


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


def draw_graph(frame, dev_smoothed_history, grip_smoothed_history, x, y, graph_h=GRAPH_H):
    cv2.rectangle(frame, (x, y), (x + GRAPH_W, y + graph_h), COLOR_BG, -1)
    cv2.rectangle(frame, (x, y), (x + GRAPH_W, y + graph_h), (90, 70, 90), 1)

    # Label sits ABOVE the box only if there's room, else inside the top edge,
    # so it can never overlap the "press q" instruction line above it.
    label_y = y - 8 if y - 20 > 0 else y + 14

    if len(dev_smoothed_history) < 2:
        cv2.putText(frame, "pink=deviation  amber=grip  red=threshold", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return

    max_val = max(Z_THRESHOLD * 1.5, 1.0)

    def plot(history, color):
        pts = []
        for i, val in enumerate(history[-GRAPH_W:]):
            px = x + int(i / GRAPH_W * GRAPH_W)
            py = y + graph_h - int(min(val / max_val, 1.0) * graph_h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)

    plot(dev_smoothed_history, COLOR_DEV_LINE)
    plot(grip_smoothed_history, COLOR_GRIP_LINE)

    threshold_y = y + graph_h - int(min(Z_THRESHOLD / max_val, 1.0) * graph_h)
    cv2.line(frame, (x, threshold_y), (x + GRAPH_W, threshold_y), COLOR_THRESHOLD, 1)

    cv2.putText(frame, "pink=deviation  amber=grip  red=threshold", (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def draw_status_banner(frame, status, w):
    is_risk = status == "RISK CONFIRMED"
    color = COLOR_RISK if is_risk else COLOR_NOMINAL
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 22, 30), -1)
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_metrics(frame, dev_avg, grip_avg, x, y):
    lines = [f"deviation (z): {dev_avg:.2f}", f"grip spread:   {grip_avg:.3f}"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


# ================= main =================

def main():
    mu, sigma, is_real = build_reference_from_labels()
    if not is_real:
        print("[Kinesis] WARNING: running with a SYNTHETIC reference. "
              "Detections are not meaningful until real labeled data exists.")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{session_id}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "timestamp", "dev_zscore", "grip_spread",
                          "dev_alert", "grip_alert", "combined_alert"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open camera. Check System Settings > Privacy & Security > Camera.")
        return

    dev_history = []
    grip_history = []
    dev_smoothed_history = []
    grip_smoothed_history = []
    frame_index = 0
    first_alert_frame = None
    rotation_angle = 0.0

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
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                frame_vec = landmark_vector(hand_landmarks)
                ref_index = min(frame_index, len(mu) - 1)

                dev = deviation_at(frame_vec, mu, sigma, ref_index)
                grip = grip_spread(frame_vec)

                dev_history.append(dev)
                grip_history.append(grip)

                status = "NOMINAL"
                dev_alert, dev_avg = False, dev
                grip_alert, grip_avg = False, grip

                if frame_index >= max(2 * WINDOW, WARMUP_FRAMES):
                    dev_alert, dev_avg = sustained_rise_now(dev_history, z_threshold=Z_THRESHOLD)
                    grip_alert, grip_avg = sustained_rise_now(grip_history)

                    if dev_alert or grip_alert:
                        status = "RISK CONFIRMED"
                        if first_alert_frame is None:
                            first_alert_frame = frame_index
                            print(f"[Kinesis] Alert first fired at frame {frame_index}")

                dev_avg_display, grip_avg_display = dev_avg, grip_avg
                dev_smoothed_history.append(dev_avg)
                grip_smoothed_history.append(grip_avg)

                log_writer.writerow([frame_index, time.time(), dev, grip,
                                      dev_alert, grip_alert, status == "RISK CONFIRMED"])
            else:
                dev_history.append(0.0)
                grip_history.append(0.0)
                dev_smoothed_history.append(0.0)
                grip_smoothed_history.append(0.0)

            # Scale UI element sizes to the ACTUAL frame size, and clamp
            # positions so nothing can be pushed off-screen or overlap,
            # regardless of the camera's real resolution.
            safe_graph_h = min(GRAPH_H, int(h * 0.28))
            safe_panel_size = min(PANEL_SIZE, int(h * 0.35), int(w * 0.35))
            top_margin = 50  # space reserved for the status banner + metrics

            graph_y = max(top_margin, h - safe_graph_h - 55)
            panel_y = max(top_margin, h - safe_panel_size - 55)

            draw_status_banner(frame, status, w)
            draw_metrics(frame, dev_avg_display, grip_avg_display, w - 260, 70)
            draw_graph(frame, dev_smoothed_history, grip_smoothed_history,
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
                frame_index = 0
                first_alert_frame = None
                print("[Kinesis] Session reset.")

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print(f"[Kinesis] Session log saved to {log_path}")


if __name__ == "__main__":
    main()