import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import joblib

# ---------------------------
# Physics-based event detection
# ---------------------------
def detect_hits_and_bounces(df_test, thresholds, min_frames=10, window=2):
    """
    Detect hits and bounces using physics.
    Returns dict: {frame_index: "hit"/"bounce"}
    """
    df = df_test.copy()
    vert_acc = df["ay_raw"].values
    vert_acc_abs = df["ay_abs_raw"].values
    horiz_speed = df["vx_raw"].values
    vert_speed = df["vy_raw"].values
    jerk_vals = df["jerk"].values

    peaks, _ = find_peaks(
        vert_acc_abs,
        height=thresholds["VERT_ACC_MIN"],
        prominence=thresholds["VERT_ACC_PROM"],
        distance=3
    )

    candidate_events = []

    for idx in peaks:
        if idx < window or idx + window >= len(df):
            continue

        # Velocity states around event
        vx_before, vx_after = horiz_speed[idx - window], horiz_speed[idx + window]
        vy_before, vy_after = vert_speed[idx - window], vert_speed[idx + window]
        ay_val = vert_acc[idx]
        jerk_val = jerk_vals[idx]

        # Derived physics metrics
        horiz_flip = vx_before * vx_after < 0
        vert_flip = vy_before * vy_after < 0
        horiz_delta = abs(vx_after) - abs(vx_before)
        vert_ratio = abs(vy_after) / (abs(vy_before) + 1e-6)
        angle_before = np.arctan2(vy_before, vx_before)
        angle_after = np.arctan2(vy_after, vx_after)
        angle_change = abs(angle_after - angle_before)

        # Event scoring
        hit_points = 0.0
        bounce_points = 0.0

        # Hit scoring
        hit_points += 2.0 if horiz_flip else 0.0 #  vx changes direction
        hit_points += 1.5 if horiz_delta > thresholds["HORIZ_SPEED_DELTA"] else 0.0 # Magnitude of vx increases sharply in additional speed (not in ratio)
        hit_points += 1.0 if vert_ratio > 1.1 else 0.0 # vy after the event increases by more than 10% (in magnitude)
        hit_points += 1.0 if jerk_val > thresholds["JERK_THRESHOLD"] else 0.0 # If the rate of change of acceleration is high
        hit_points += 1.0 if angle_change > np.pi / 4 else 0.0 # If trajectory angle changes by more than 45°
        
        # Bounce scoring
        bounce_points += 2.0 if ay_val < thresholds["VERT_ACC_BOUNCE"] else 0.0 # ay is very negative
        bounce_points += 1.5 if vert_flip else 0.0 # vy changes direction
        bounce_points += 1.0 if vert_ratio < 0.8 else 0.0 # vy after the event decrease by more than 20% (energy loss)

        # Decide event type
        event_type = None
        if hit_points >= bounce_points and hit_points >= 2.5:
            event_type = "hit"
        elif bounce_points > hit_points and bounce_points >= 2.0:
            event_type = "bounce"

        # Computing strength score to help chosing between close events
        strength_score = abs(ay_val) + jerk_val + abs(horiz_delta) + angle_change

        if event_type:
            candidate_events.append((df.index[idx], event_type, strength_score))

    # Strongest Event Selection (to prevent from selecting event too close)
    candidate_events.sort(key=lambda x: x[0])
    final_events = {}
    for frame_id, label, score in candidate_events:
        if not final_events:
            final_events[frame_id] = (label, score)
            continue
        last_frame = max(final_events.keys())
        if frame_id - last_frame >= min_frames:
            final_events[frame_id] = (label, score)
        else:
            # Keep the event with higher strength
            if score > final_events[last_frame][1]:
                final_events[last_frame] = (label, score)

    # Return events by frame
    return {frame: label for frame, (label, _) in final_events.items()}


def load_threshold(threshold_path="thresholds_physics.joblib"):
    """Load physics-based thresholds from disk."""
    with open(threshold_path, "rb") as f:
        threshold = joblib.load(f)
    return threshold
