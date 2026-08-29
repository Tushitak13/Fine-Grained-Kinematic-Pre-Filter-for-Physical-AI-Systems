"""
Final analysis pipeline.

Reads: data/labels.csv, data/trajectories/*.npy, data/trajectories/*_fps.txt
       (all produced earlier by label_helper.py + extract_trajectory.py)

Does:
  1. Builds the reference "normal" profile from trajectories labeled `normal`
     (excluding whichever you're holding out for the demo)
  2. Scores every video with two signals:
       - trajectory deviation vs reference (DTW + z-score)
       - grip spread (thumb-to-fingertip distance)
  3. Combines both into one alert stream using the sustained-trend rule
  4. Validates: for failure videos, computes lead time vs the labeled drop_frame.
     For normal videos, counts false alarms.
  5. Prints the summary numbers to use in your pitch.

Run once, after all videos are labeled and extracted:
    python analyze_all.py
"""

import os
import csv
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

LABELS_CSV = "data/labels.csv"
TRAJ_DIR = "data/trajectories"
TARGET_LEN = 100

# Frames to ignore at the very start of every clip — the hand is still
# entering frame / settling into grip, which is naturally noisy and not
# a real anomaly. Without this, that settling period fires false alerts
# on almost every video. Tune this based on how long your hand actually
# takes to settle after the video starts (check a few clips visually).
WARMUP_FRAMES = 20

# Videos you're holding back untouched for the live demo — never used to
# build the reference or to tune thresholds. Fill in your actual held-out ids.
HELD_OUT = {"run09"}   # <-- edit to match whichever you chose

# Normal runs excluded from REFERENCE BUILDING ONLY (still scored/reported
# below like any other video). These have inconsistent task length/phase
# compared to the rest of the normal set (e.g. run06/run07 are much shorter,
# run12 includes an extra "set object down" phase not present in others).
# Mixing them into the reference blurs the averaged path and causes false
# alarms on runs whose pacing/phase doesn't match that blur. Excluding them
# from the reference (but still testing against it) gives a cleaner, more
# consistent "normal" baseline while still validating on this variation.
REFERENCE_EXCLUDE = {"run06", "run07", "run12"}


# ---------- shared helpers ----------

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


# ---------- signal 1: trajectory deviation vs reference ----------

def build_reference(rows):
    normal_trajs = []
    for row in rows:
        if (row["type"] == "normal"
                and row["video_id"] not in HELD_OUT
                and row["video_id"] not in REFERENCE_EXCLUDE):
            traj, _ = load_trajectory(row["video_id"])
            normal_trajs.append(resample(traj))
    stacked = np.stack(normal_trajs)
    mu = stacked.mean(axis=0)
    # Floor sigma at 0.01 (not 1e-6) — coordinates are normalized to roughly
    # [0,1], so a near-zero sigma at low-variance reference points otherwise
    # makes the z-score wildly hypersensitive to tiny, harmless differences.
    sigma = np.maximum(stacked.std(axis=0), 0.01)
    return mu, sigma


def deviation_series(traj, mu, sigma):
    _, path = fastdtw(traj, mu, dist=euclidean)
    deviations = np.zeros(len(traj))
    counts = np.zeros(len(traj))
    for live_idx, ref_idx in path:
        diff = np.abs(traj[live_idx] - mu[ref_idx])
        z = np.linalg.norm(diff / sigma[ref_idx])
        deviations[live_idx] += z
        counts[live_idx] += 1
    counts[counts == 0] = 1
    return deviations / counts


# ---------- signal 2: grip spread ----------

def grip_spread_series(traj):
    scores = []
    for frame in traj:
        lm = frame.reshape(21, 3)
        thumb_tip = lm[4]
        dists = [np.linalg.norm(thumb_tip - lm[i]) for i in (8, 12, 16, 20)]
        scores.append(np.mean(dists))
    return np.array(scores)


# ---------- shared trend/alarm rule ----------

def sustained_rise_alerts(scores, window=8, threshold_ratio=1.25, z_threshold=None):
    """
    Fires when the RECENT window's average has risen by threshold_ratio over
    the PRIOR window's average, AND the recent window has a positive slope
    (sustained rise, not a single noisy frame or short blip).
    If z_threshold is given (for the deviation signal, already in std-dev
    units), also requires the recent window's average level to be above it.

    This compares window AVERAGES rather than single frames, which is what
    makes it robust to one-frame landmark jitter — a single noisy point can
    no longer flip an alert on its own; it has to persist across the window.
    """
    n = len(scores)
    alerts = [False] * n
    start = max(window * 2, WARMUP_FRAMES)

    for i in range(start, n):
        past_window = scores[i - 2 * window: i - window]
        recent_window = scores[i - window: i]

        past_avg = np.mean(past_window)
        recent_avg = np.mean(recent_window)

        ratio_ok = recent_avg > past_avg * threshold_ratio

        x = np.arange(len(recent_window))
        slope = np.polyfit(x, recent_window, 1)[0]
        rising = slope > 0

        level_ok = True if z_threshold is None else recent_avg > z_threshold

        alerts[i] = bool(ratio_ok and rising and level_ok)

    return alerts


def combine_alerts(alerts_a, alerts_b):
    """Fire if either signal's trend rule fires on that frame."""
    return [a or b for a, b in zip(alerts_a, alerts_b)]


# ---------- validation ----------

def first_true(bool_list):
    for i, v in enumerate(bool_list):
        if v:
            return i
    return None


def main():
    rows = load_labels()
    mu, sigma = build_reference(rows)
    ref_count = sum(
        1 for r in rows
        if r["type"] == "normal"
        and r["video_id"] not in HELD_OUT
        and r["video_id"] not in REFERENCE_EXCLUDE
    )
    print(f"Reference built from {ref_count} normal videos "
          f"(excluded from reference: {sorted(REFERENCE_EXCLUDE)}).\n")

    print("--- dev score distributions (for threshold calibration) ---")
    for row in rows:
        video_id = row["video_id"]
        if video_id in HELD_OUT:
            continue
        traj, fps = load_trajectory(video_id)
        dev = deviation_series(traj, mu, sigma)
        print(f"{video_id:8} [{row['type']:8}] dev: mean={dev.mean():.2f} max={dev.max():.2f} p90={np.percentile(dev,90):.2f}")
    print()

    lead_times = []
    false_alarm_count = 0
    normal_count = 0

    for row in rows:
        video_id = row["video_id"]
        if video_id in HELD_OUT:
            continue  # never touch held-out videos during tuning

        traj, fps = load_trajectory(video_id)

        dev = deviation_series(traj, mu, sigma)
        dev_alerts = sustained_rise_alerts(dev, z_threshold=15.0)

        grip = grip_spread_series(traj)
        grip_alerts = sustained_rise_alerts(grip)

        combined = combine_alerts(dev_alerts, grip_alerts)
        alert_frame = first_true(combined)

        if row["type"] == "failure":
            drop_frame = int(row["drop_frame"])
            if alert_frame is None:
                print(f"{video_id} [failure]: MISSED — no alert fired before drop at frame {drop_frame}")
            else:
                lead_seconds = (drop_frame - alert_frame) / fps
                lead_times.append(lead_seconds)
                status = "early" if lead_seconds > 0 else "LATE"
                print(f"{video_id} [failure]: alert at frame {alert_frame}, drop at {drop_frame} -> {lead_seconds:.2f}s {status}")
        else:
            normal_count += 1
            if alert_frame is not None:
                false_alarm_count += 1
                print(f"{video_id} [normal]: false alarm at frame {alert_frame}")
            else:
                print(f"{video_id} [normal]: correctly stayed quiet")

    print("\n--- Summary (use these numbers in your pitch) ---")
    if lead_times:
        print(f"Average lead time: {np.mean(lead_times):.2f}s over {len(lead_times)} failure video(s)")
        print(f"Lead time range: {min(lead_times):.2f}s - {max(lead_times):.2f}s")
    else:
        print("No successful early detections yet — thresholds likely need tuning.")
    print(f"False alarm rate: {false_alarm_count}/{normal_count} normal videos")


if __name__ == "__main__":
    main()
