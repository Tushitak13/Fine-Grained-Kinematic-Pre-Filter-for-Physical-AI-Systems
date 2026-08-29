"""
filter_logic.py

Core scoring logic for TrajGuard / Fine-Grained Kinematic Pre-Filter.

DeviationFilter takes a reference "normal" trajectory (mean + std envelope)
and, frame by frame, computes:
    D_t     - deviation of current frame from the reference envelope
    dD_dt   - rate of change of that deviation
    P_t     - a physical signal (grip aperture: thumb-index tip distance)
    R_t     - combined risk score = w1*D_t + w2*max(dD_dt, 0) + w3*P_t

This same class is used by both live_engine.py (live webcam) and
data_loader.py / replay_view.py (pre-recorded validation clips), so the
scoring logic is identical in both modes -- what differs is only where
the frames come from.
"""

import numpy as np

# MediaPipe hand landmark indices
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8

# Default risk formula weights: R_t = W1*D_t + W2*max(dD_dt, 0) + W3*P_t
DEFAULT_WEIGHTS = {"w1": 0.4, "w2": 0.5, "w3": 0.1}

# Default alert threshold -- tune this once real data is available
DEFAULT_THRESHOLD = 0.5

# How many consecutive frames R_t must stay above threshold before we
# treat it as a CONFIRMED risk (this is the "sustained and rising" rule
# that suppresses single-frame noise / false alarms)
DEFAULT_SUSTAIN_FRAMES = 3


class DeviationFilter:
    def __init__(self, reference_mean, reference_std=None,
                 weights=None, threshold=DEFAULT_THRESHOLD,
                 sustain_frames=DEFAULT_SUSTAIN_FRAMES):
        """
        reference_mean: np.array shape (num_frames, 63) - the averaged
                         "normal" trajectory (21 landmarks x xyz)
        reference_std:   np.array shape (num_frames, 63) or None -
                         per-frame standard deviation across normal runs,
                         used to normalize D_t so it's scale-independent.
                         If None, defaults to 1.0 everywhere (no normalization).
        weights:         dict with keys w1, w2, w3
        threshold:       R_t value above which a frame counts as "risky"
        sustain_frames:  number of consecutive risky frames needed before
                         status flips from NOMINAL to RISK CONFIRMED
        """
        self.reference_mean = reference_mean
        self.reference_std = reference_std if reference_std is not None \
            else np.ones_like(reference_mean)
        self.reference_std[self.reference_std == 0] = 1.0  # avoid divide-by-zero

        self.weights = weights or DEFAULT_WEIGHTS
        self.threshold = threshold
        self.sustain_frames = sustain_frames

        # Running state (used in live mode, frame by frame)
        self._prev_D = None
        self._consecutive_risky_frames = 0
        self._status = "NOMINAL"

    def grip_aperture(self, frame_vec):
        """frame_vec: flat 63-length array (21 landmarks x xyz)."""
        landmarks = frame_vec.reshape(21, 3)
        thumb = landmarks[THUMB_TIP_IDX]
        index = landmarks[INDEX_TIP_IDX]
        return float(np.linalg.norm(thumb - index))

    def score_frame(self, frame_vec, ref_index):
        """
        Score a single frame against the reference at position ref_index.
        Returns a dict: {D_t, dD_dt, P_t, R_t, status}

        This is the function live_engine.py calls once per webcam frame,
        and data_loader.py / replay logic calls once per row when
        replaying a pre-recorded clip.
        """
        ref_index = min(ref_index, len(self.reference_mean) - 1)
        mean_t = self.reference_mean[ref_index]
        std_t = self.reference_std[ref_index]

        # Normalized deviation (z-score style distance from the envelope)
        D_t = float(np.linalg.norm((frame_vec - mean_t) / std_t))
        P_t = self.grip_aperture(frame_vec)

        dD_dt = 0.0 if self._prev_D is None else (D_t - self._prev_D)
        self._prev_D = D_t

        w1, w2, w3 = self.weights["w1"], self.weights["w2"], self.weights["w3"]
        R_t = w1 * D_t + w2 * max(dD_dt, 0.0) + w3 * P_t

        # Sustained-and-rising gate: only count consecutive frames above
        # threshold, this is what suppresses single-frame noise
        if R_t >= self.threshold:
            self._consecutive_risky_frames += 1
        else:
            self._consecutive_risky_frames = 0

        if self._consecutive_risky_frames >= self.sustain_frames:
            self._status = "RISK CONFIRMED"
        else:
            self._status = "NOMINAL"

        return {
            "D_t": D_t,
            "dD_dt": dD_dt,
            "P_t": P_t,
            "R_t": R_t,
            "status": self._status,
        }

    def reset(self):
        """Call this between clips/sessions so state doesn't leak across runs."""
        self._prev_D = None
        self._consecutive_risky_frames = 0
        self._status = "NOMINAL"

    def score_full_trajectory(self, trajectory):
        """
        Score an entire pre-aligned trajectory at once (used by replay_view.py
        for validation clips, where the whole video is already available).

        trajectory: np.array shape (num_frames, 63), already DTW-aligned to
                    self.reference_mean's frame count.

        Returns a list of per-frame score dicts (same shape as score_frame's output).
        """
        self.reset()
        results = []
        for t in range(len(trajectory)):
            result = self.score_frame(trajectory[t], ref_index=t)
            result["frame"] = t
            results.append(result)
        return results


def naive_score_full_trajectory(trajectory, reference_mean, reference_std=None, threshold=0.5):
    """
    A deliberately "dumb" baseline scorer for the naive-vs-ours comparison:
    flags a frame purely on raw deviation, with NO rate-of-change gating and
    NO sustained-frames requirement. Used to demonstrate why the
    "significant and rising" rule in DeviationFilter reduces false alarms.

    Returns a list of dicts: {frame, D_t, flagged (bool)}
    """
    reference_std = reference_std if reference_std is not None \
        else np.ones_like(reference_mean)
    reference_std[reference_std == 0] = 1.0

    results = []
    for t in range(len(trajectory)):
        ref_index = min(t, len(reference_mean) - 1)
        D_t = float(np.linalg.norm(
            (trajectory[t] - reference_mean[ref_index]) / reference_std[ref_index]
        ))
        results.append({
            "frame": t,
            "D_t": D_t,
            "flagged": D_t >= threshold,
        })
    return results