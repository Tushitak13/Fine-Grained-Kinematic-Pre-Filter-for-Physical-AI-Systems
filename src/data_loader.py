"""
data_loader.py

Reads pipeline output files (labels.csv, trajectories, reference envelope,
precomputed scores). Falls back to synthetic data if a file isn't present
yet, so the app never crashes while teammates are still producing real data.
"""

import os
import csv
import numpy as np

DATA_DIR = "data"
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
TRAJ_DIR = os.path.join(DATA_DIR, "trajectories")
SCORES_DIR = os.path.join(DATA_DIR, "scores")
REF_DIR = os.path.join(DATA_DIR, "reference_runs")
NUM_LANDMARK_VALUES = 63  # 21 landmarks x xyz


def load_labels():
    """
    Returns a list of dicts: {video_id, type, drop_frame, split}
    split defaults to REFERENCE/VALIDATION/HELD-OUT based on type if the
    'split' column doesn't exist in labels.csv yet.
    """
    if not os.path.exists(LABELS_FILE):
        # Fallback: synthetic label set so the UI has something to show
        return [
            {"video_id": "demo_normal_01", "type": "normal", "drop_frame": "", "split": "REFERENCE"},
            {"video_id": "demo_fail_01", "type": "failure", "drop_frame": "90", "split": "VALIDATION"},
            {"video_id": "demo_borderline_01", "type": "borderline", "drop_frame": "", "split": "HELD-OUT"},
        ]

    rows = []
    with open(LABELS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = row.get("type", "").strip().lower()
            split = row.get("split", "").strip().upper() if "split" in row else ""
            if not split:
                # Sensible default split assignment if not explicitly labeled
                split = "REFERENCE" if row_type == "normal" else "VALIDATION"
            rows.append({
                "video_id": row["video_id"].strip(),
                "type": row_type,
                "drop_frame": row.get("drop_frame", "").strip(),
                "split": split,
            })
    return rows


def _synthetic_trajectory(num_frames=120, seed=0):
    """Generates a smooth, plausible-looking fake hand trajectory for fallback."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, num_frames).reshape(-1, 1)
    noise = rng.normal(0, 0.01, size=(num_frames, NUM_LANDMARK_VALUES))
    trajectory = np.tile(base, (1, NUM_LANDMARK_VALUES)) + noise
    return trajectory.astype(np.float32)


def load_trajectory(video_id, num_frames=120):
    path = os.path.join(TRAJ_DIR, f"{video_id}.npy")
    if os.path.exists(path):
        return np.load(path), True  # (data, is_real)
    # Fallback synthetic trajectory, seeded by video_id so it's stable across reruns
    seed = abs(hash(video_id)) % (2**32)
    return _synthetic_trajectory(num_frames, seed=seed), False


def load_fps(video_id, default_fps=30):
    path = os.path.join(TRAJ_DIR, f"{video_id}_fps.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return default_fps
    return default_fps


def load_reference_envelope(num_frames=120):
    mean_path = os.path.join(REF_DIR, "envelope_mean.npy")
    std_path = os.path.join(REF_DIR, "envelope_std.npy")

    if os.path.exists(mean_path):
        mean = np.load(mean_path)
        std = np.load(std_path) if os.path.exists(std_path) else None
        return mean, std, True

    # Fallback: flat synthetic reference envelope
    mean = _synthetic_trajectory(num_frames, seed=42)
    std = np.full_like(mean, 0.05)
    return mean, std, False


def load_precomputed_scores(video_id):
    """
    Returns a list of dicts {frame, D_t, dD_dt, P_t, R_t} if a precomputed
    scores CSV exists, otherwise None (caller should compute on the fly
    using filter_logic.DeviationFilter instead).
    """
    path = os.path.join(SCORES_DIR, f"{video_id}_scores.csv")
    if not os.path.exists(path):
        return None

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame"]),
                "D_t": float(row["D_t"]),
                "dD_dt": float(row["dD_dt"]),
                "P_t": float(row["P_t"]),
                "R_t": float(row["R_t"]),
            })
    return rows