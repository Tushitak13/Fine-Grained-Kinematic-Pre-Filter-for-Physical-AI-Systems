"""
kinematics.py

Layer 1's physics core. Computes true kinematic quantities (velocity,
acceleration, jerk) from a hand trajectory, using REAL time deltas from
the video's actual fps -- not frame-index approximations.

These are genuine physical derivatives, computed via finite differences:
    v(t) = (x(t) - x(t-1)) / dt
    a(t) = (v(t) - v(t-1)) / dt
    j(t) = (a(t) - a(t-1)) / dt
where dt = 1/fps (seconds per frame).

Why jerk matters for failure detection specifically: jerk is the
standard physics term for "how suddenly is motion changing." A smooth,
controlled lift has low jerk throughout. A slip, jolt, or loss-of-grip
event produces a jerk SPIKE -- often even when raw position hasn't
deviated far yet. This is a genuinely different, complementary signal
to deviation-from-reference: deviation asks "are you somewhere wrong,"
jerk asks "is your motion suddenly out of control," and a failure can
trigger one before the other.

(Empirical note from this project's own validated dataset: for the
lift/shake/drop task tested, jerk did NOT show a clean separation
between failure and normal clips -- failures were gradual drift
patterns rather than sudden jerks. Jerk remains a legitimate signal
worth computing and calibrating per-task, but should not be assumed
useful without checking real data first -- see analyze_all.py's
calibration output.)

Usage:
    from kinematics import compute_kinematics

    traj, fps = load_trajectory(video_id)   # shape (num_frames, 63)
    kin = compute_kinematics(traj, fps)
    # kin["speed"]          -> (num_frames,) wrist speed, in position-units/sec
    # kin["accel_mag"]      -> (num_frames,) wrist acceleration magnitude
    # kin["jerk_mag"]       -> (num_frames,) wrist jerk magnitude
    # kin["grip_velocity"]  -> (num_frames,) rate of change of grip spread
"""

import numpy as np

WRIST_IDX = 0  # landmark 0 is the wrist -- the most stable reference point
               # for whole-hand motion (fingertips move relative to it too,
               # so using the wrist avoids conflating "hand moved" with
               # "fingers moved relative to the hand")


def landmark_xyz(traj, idx):
    """traj: (num_frames, 63). Returns (num_frames, 3) for one landmark."""
    return traj[:, idx * 3: idx * 3 + 3]


def finite_difference(series, dt):
    """
    First-order finite difference: derivative[t] = (series[t] - series[t-1]) / dt
    First frame has no prior sample, so its derivative is defined as 0
    (no motion assumed at the very first frame) rather than left undefined.
    series: (num_frames, D) for any dimensionality D (3 for xyz, 1 for scalar)
    """
    deriv = np.zeros_like(series, dtype=np.float64)
    deriv[1:] = (series[1:] - series[:-1]) / dt
    return deriv


def compute_kinematics(traj, fps, grip_series=None):
    """
    traj: (num_frames, 63) raw landmark trajectory
    fps: real frames-per-second of the source video
    grip_series: optional (num_frames,) precomputed grip-spread values;
                 if not given, grip-based kinematics are skipped.

    Returns a dict of (num_frames,) arrays:
        position    : wrist position (3D, for reference/plotting)
        velocity    : wrist velocity vector (3D)
        speed       : |velocity| -- scalar speed magnitude
        acceleration: wrist acceleration vector (3D)
        accel_mag   : |acceleration| -- scalar
        jerk        : wrist jerk vector (3D)
        jerk_mag    : |jerk| -- scalar, THE key "suddenness" signal
        grip_velocity: rate of change of grip spread (if grip_series given)
    """
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}. Cannot compute real-time derivatives without it.")

    dt = 1.0 / fps

    position = landmark_xyz(traj, WRIST_IDX)          # (N, 3)
    velocity = finite_difference(position, dt)         # (N, 3)
    acceleration = finite_difference(velocity, dt)      # (N, 3)
    jerk = finite_difference(acceleration, dt)          # (N, 3)

    speed = np.linalg.norm(velocity, axis=1)
    accel_mag = np.linalg.norm(acceleration, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    result = {
        "position": position,
        "velocity": velocity,
        "speed": speed,
        "acceleration": acceleration,
        "accel_mag": accel_mag,
        "jerk": jerk,
        "jerk_mag": jerk_mag,
    }

    if grip_series is not None:
        grip_series = np.asarray(grip_series, dtype=np.float64).reshape(-1, 1)
        grip_velocity = finite_difference(grip_series, dt).reshape(-1)
        result["grip_velocity"] = grip_velocity

    return result


def normalize_series(series, reference_series):
    """
    Z-score normalize a live/test series against a reference series'
    mean and std -- puts jerk/speed/accel on a comparable, unitless scale,
    same principle as the existing deviation z-score, applied to real
    kinematic quantities instead of raw position.
    """
    mu = np.mean(reference_series)
    sigma = max(np.std(reference_series), 1e-6)
    return (series - mu) / sigma
