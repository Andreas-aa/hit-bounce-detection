import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.signal import find_peaks


FEATURE_COLS = [
    "delta_angle",
    "vx_sign_change", 
    "vy_sign_change",
    "v", "a", "jerk",
    "vx", 'vy', 'ax', 'ay', 'jx', 'jy',
    "v_mean", "v_std",
    "a_mean", "a_std",
    "j_mean", "j_std",
    "log_v", "log_a", "log_j",
    "vx_abs_raw", "vy_abs_raw",
    "ax_abs_raw", "ay_abs_raw",
    "jx_abs_raw", "jy_abs_raw",
]

FEATURE_COLS_DEEP = FEATURE_COLS + ["x_i", "y_i"]

SMOOTH_WINDOW = 7


# Load trained model
def load_model(model_path="model/rf_model.joblib"):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

# Load trained model
def load_threshold(threshold_path="thresholds_physics.joblib"):
    with open(threshold_path, "rb") as f:
        threshold = joblib.load(f)
    return threshold

if __name__ == "__main__":
    model = load_model()


def build_features(
    df: pd.DataFrame,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """
    Feature builder for ball hit / bounce detection.

    """

    # ------------------------------------------------------------------
    # Numeric positions and index
    # ------------------------------------------------------------------
    new_df = df.copy()
    new_df.index = pd.to_numeric(new_df.index, errors="coerce")
    new_df = new_df.sort_index()
    new_df["x_i"] = pd.to_numeric(new_df["x"], errors="coerce")
    new_df["y_i"] = pd.to_numeric(new_df["y"], errors="coerce")
    new_df = new_df.dropna(new_df=["x_i", "y_i"])
    

    # ------------------------------------------------------------------
    # Raw positions
    # ------------------------------------------------------------------
    new_df["x_raw"] = new_df["x_i"]
    new_df["y_raw"] = new_df["y_i"]

    # ------------------------------------------------------------------
    # Centered smoothing on positions
    # ------------------------------------------------------------------

    # Centered rolling mean reduces high-frequency measurement noise
    # without eliminating physical discontinuities (hits / bounces).
    new_df["x_smooth"] = (
        new_df["x_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )
    new_df["y_smooth"] = (
        new_df["y_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------------------
    # Time step (central)
    # ------------------------------------------------------------------
    t = new_df.index.to_series()

    # ------------------------------------------------------------------
    # Smoothed derivatives (stable kinematics)
    # ------------------------------------------------------------------
    x_smooth = new_df["x_smooth"].to_numpy()
    y_smooth = new_df["y_smooth"].to_numpy()

    vx = np.gradient(x_smooth, t)
    vy = np.gradient(y_smooth, t)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    jx = np.gradient(ax, t)
    jy = np.gradient(ay, t)

    new_df["vx"] = vx
    new_df["vy"] = vy
    new_df["ax"] = ax
    new_df["ay"] = ay
    new_df["jx"] = jx
    new_df["jy"] = jy

    # ------------------------------------------------------------------
    # Raw derivatives (impulse-sensitive)
    # ------------------------------------------------------------------
    x_raw = new_df["x_raw"].to_numpy()
    y_raw = new_df["y_raw"].to_numpy()

    vx_raw = np.gradient(x_raw, t)
    vy_raw = np.gradient(y_raw, t)

    ax_raw = np.gradient(vx_raw, t)
    ay_raw = np.gradient(vy_raw, t)

    jx_raw = np.gradient(ax_raw, t)
    jy_raw = np.gradient(ay_raw, t)

    new_df["vx_raw"] = vx_raw
    new_df["vy_raw"] = vy_raw
    new_df["ax_raw"] = ax_raw
    new_df["ay_raw"] = ay_raw
    new_df["jx_raw"] = jx_raw
    new_df["jy_raw"] = jy_raw

    # ------------------------------------------------------------------
    # Raw derivatubes in absolute
    # ------------------------------------------------------------------

    new_df["vx_abs_raw"] = np.abs(new_df["vx_raw"])
    new_df["vy_abs_raw"] = np.abs(new_df["vy_raw"])
    new_df["ax_abs_raw"] = np.abs(new_df["ax_raw"])
    new_df["ay_abs_raw"] = np.abs(new_df["ay_raw"])
    new_df["jx_abs_raw"] = np.abs(new_df["jx_raw"])
    new_df["jy_abs_raw"] = np.abs(new_df["jy_raw"])

    # ------------------------------------------------------------------
    # Magnitudes (smoothed)
    # ------------------------------------------------------------------
    new_df["v"] = np.sqrt(new_df["vx"]**2 + new_df["vy"]**2)
    new_df["a"] = np.sqrt(new_df["ax"]**2 + new_df["ay"]**2)
    new_df["jerk"] = np.sqrt(new_df["jx"]**2 + new_df["jy"]**2)

    # ------------------------------------------------------------------
    # Log magnitudes : preserves order and compresses large values
    # ------------------------------------------------------------------
    new_df["log_v"] = np.log1p(new_df["v"])    
    new_df["log_a"] = np.log1p(new_df["a"])
    new_df["log_j"] = np.log1p(new_df["jerk"])

    # ------------------------------------------------------------------
    # Directional features
    # ------------------------------------------------------------------
    new_df["angle"] = np.arctan2(new_df["vy"], new_df["vx"])
    new_df["delta_angle"] = np.gradient(new_df["angle"])

    # ------------------------------------------------------------------
    # Centered rolling statistics (smoothed)
    # ------------------------------------------------------------------
    new_df["v_mean"] = new_df["v"].rolling(smooth_window, center=True, min_periods=1).mean()
    new_df["v_std"]  = new_df["v"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    new_df["a_mean"] = new_df["a"].rolling(smooth_window, center=True, min_periods=1).mean()
    new_df["a_std"]  = new_df["a"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    new_df["j_mean"] = new_df["jerk"].rolling(smooth_window, center=True, min_periods=1).mean()
    new_df["j_std"]  = new_df["jerk"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    # ------------------------------------------------------------------
    # Motion sign changes
    # ------------------------------------------------------------------
    new_df["vx_sign"] = np.sign(new_df["vx"]).fillna(0.0)
    new_df["vx_sign_change"] = (
        new_df["vx_sign"].diff().abs() > 0
    ).astype(int)
    
    new_df["vy_sign"] = np.sign(new_df["vy"]).fillna(0.0)
    new_df["vy_sign_change"] = (
        new_df["vy_sign"].diff().abs() > 0
    ).astype(int)

    return new_df


def transform_for_model(
    df: pd.DataFrame,
    preprocessors: dict,
):
    X = df[FEATURE_COLS]
    X_deep = df[FEATURE_COLS_DEEP]

    X_scaled = preprocessors["scaler"].transform(X)
    X_deep_scaled = preprocessors["scaler_deep"].transform(X_deep)

    return X, X_scaled, X_deep_scaled


# ==============================
# Physics-based event detection
# ==============================
def detect_hits_and_bounces(df_test, thresholds, min_frames=10, window=2):
    """
    Detect events ("hit" or "bounce") from motion trajectory data.
    """
    df = df_test.copy()

    # Raw signals
    vert_acc = df["ay_raw"].values
    vert_acc_abs = df["ay_abs_raw"].values
    horiz_speed = df["vx_raw"].values
    vert_speed = df["vy_raw"].values
    jerk_vals = df["jerk"].values

    # Candidate event peaks
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
        hit_points += 2.0 if horiz_flip else 0.0 # vx changes direction
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

    # ==============================
    # Strongest Event Selection (to prevent from selecting event too close)
    # ==============================
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


def supervized_hit_bounce_detection(json_path: Path):
    # Load JSON into a DataFrame
    df = pd.DataFrame(columns=["x", "y", "visible"])
    df.index.name = "image_frame"

    with json_path.open("r", encoding="utf-8") as f:
        ball_data = json.load(f)

    file_df = pd.DataFrame(ball_data).T
    file_df.index.name = "image_frame"

    # Ensure correct columns
    file_df = file_df.reindex(columns=["x", "y", "visible"])

    # Build features and preprocess
    new_df = build_features(file_df, smooth_window=SMOOTH_WINDOW)
    preprocessors = joblib.load("preprocessors.joblib")
    # Predicting using Random Forest, no need for Deep Learning Features
    _, X_new, _ = transform_for_model(new_df, preprocessors)

    # Load model and predict
    model = load_model()
    y_pred = model.predict(X_new)

    # Add predictions as a new column
    new_df["action"] = y_pred

    # Update original JSON
    # Convert DataFrame back to dictionary with same structure
    updated_json = new_df[["x", "y", "visible", "action"]].T.to_dict()

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(updated_json, f, indent=4)

    print(f"Predictions added to '{json_path}' successfully!")

    return updated_json


def unsupervized_hit_bounce_detection(json_path: Path):
    # Load JSON into a DataFrame
    df = pd.DataFrame(columns=["x", "y", "visible"])
    df.index.name = "image_frame"

    with json_path.open("r", encoding="utf-8") as f:
        ball_data = json.load(f)

    file_df = pd.DataFrame(ball_data).T
    file_df.index.name = "image_frame"

    # Ensure correct columns
    file_df = file_df.reindex(columns=["x", "y", "visible"])

    # Build features and preprocess
    new_df = build_features(file_df, smooth_window=SMOOTH_WINDOW)

    thresh = load_threshold()

    # Predicting using the function
    y_pred = detect_hits_and_bounces(new_df, thresholds=thresh)
    y_pred_array = np.array(["air"] * len(new_df), dtype=object)
    for frame, label in y_pred.items():
        if frame in df.index:
            idx = df.index.get_loc(frame)
            y_pred_array[idx] = label

    new_df["action"] = y_pred_array


    # Update original JSON
    # Convert DataFrame back to dictionary with same structure
    updated_json = new_df[["x", "y", "visible", "action"]].T.to_dict()

    # with json_path.open("w", encoding="utf-8") as f:
    #     json.dump(updated_json, f, indent=4)

    print(f"Predictions added to '{json_path}' successfully!")

    return updated_json
