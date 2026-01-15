import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    model = load_model()


def build_features(
    subset_df: pd.DataFrame,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """
    Feature builder for ball hit / bounce detection.

    """

    # ------------------------------------------------------------------
    # Numeric positions and index
    # ------------------------------------------------------------------
    subset = subset_df.copy()
    subset.index = pd.to_numeric(subset.index, errors="coerce")
    subset = subset.sort_index()
    subset["x_i"] = pd.to_numeric(subset["x"], errors="coerce")
    subset["y_i"] = pd.to_numeric(subset["y"], errors="coerce")
    subset = subset.dropna(subset=["x_i", "y_i"])
    

    # ------------------------------------------------------------------
    # Raw positions
    # ------------------------------------------------------------------
    subset["x_raw"] = subset["x_i"]
    subset["y_raw"] = subset["y_i"]

    # ------------------------------------------------------------------
    # Centered smoothing on positions
    # ------------------------------------------------------------------

    # Centered rolling mean reduces high-frequency measurement noise
    # without eliminating physical discontinuities (hits / bounces).
    subset["x_smooth"] = (
        subset["x_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )
    subset["y_smooth"] = (
        subset["y_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------------------
    # Time step (central)
    # ------------------------------------------------------------------
    t = subset.index.to_series()

    # ------------------------------------------------------------------
    # Smoothed derivatives (stable kinematics)
    # ------------------------------------------------------------------
    x_smooth = subset["x_smooth"].to_numpy()
    y_smooth = subset["y_smooth"].to_numpy()

    vx = np.gradient(x_smooth, t)
    vy = np.gradient(y_smooth, t)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    jx = np.gradient(ax, t)
    jy = np.gradient(ay, t)

    subset["vx"] = vx
    subset["vy"] = vy
    subset["ax"] = ax
    subset["ay"] = ay
    subset["jx"] = jx
    subset["jy"] = jy

    # ------------------------------------------------------------------
    # Raw derivatives (impulse-sensitive)
    # ------------------------------------------------------------------
    x_raw = subset["x_raw"].to_numpy()
    y_raw = subset["y_raw"].to_numpy()

    vx_raw = np.gradient(x_raw, t)
    vy_raw = np.gradient(y_raw, t)

    ax_raw = np.gradient(vx_raw, t)
    ay_raw = np.gradient(vy_raw, t)

    jx_raw = np.gradient(ax_raw, t)
    jy_raw = np.gradient(ay_raw, t)

    subset["vx_raw"] = vx_raw
    subset["vy_raw"] = vy_raw
    subset["ax_raw"] = ax_raw
    subset["ay_raw"] = ay_raw
    subset["jx_raw"] = jx_raw
    subset["jy_raw"] = jy_raw

    # ------------------------------------------------------------------
    # Raw derivatubes in absolute
    # ------------------------------------------------------------------

    subset["vx_abs_raw"] = np.abs(subset["vx_raw"])
    subset["vy_abs_raw"] = np.abs(subset["vy_raw"])
    subset["ax_abs_raw"] = np.abs(subset["ax_raw"])
    subset["ay_abs_raw"] = np.abs(subset["ay_raw"])
    subset["jx_abs_raw"] = np.abs(subset["jx_raw"])
    subset["jy_abs_raw"] = np.abs(subset["jy_raw"])

    # ------------------------------------------------------------------
    # Magnitudes (smoothed)
    # ------------------------------------------------------------------
    subset["v"] = np.sqrt(subset["vx"]**2 + subset["vy"]**2)
    subset["a"] = np.sqrt(subset["ax"]**2 + subset["ay"]**2)
    subset["jerk"] = np.sqrt(subset["jx"]**2 + subset["jy"]**2)

    # ------------------------------------------------------------------
    # Log magnitudes : preserves order and compresses large values
    # ------------------------------------------------------------------
    subset["log_v"] = np.log1p(subset["v"])    
    subset["log_a"] = np.log1p(subset["a"])
    subset["log_j"] = np.log1p(subset["jerk"])

    # ------------------------------------------------------------------
    # Directional features
    # ------------------------------------------------------------------
    subset["angle"] = np.arctan2(subset["vy"], subset["vx"])
    subset["delta_angle"] = np.gradient(subset["angle"])

    # ------------------------------------------------------------------
    # Centered rolling statistics (smoothed)
    # ------------------------------------------------------------------
    subset["v_mean"] = subset["v"].rolling(smooth_window, center=True, min_periods=1).mean()
    subset["v_std"]  = subset["v"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    subset["a_mean"] = subset["a"].rolling(smooth_window, center=True, min_periods=1).mean()
    subset["a_std"]  = subset["a"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    subset["j_mean"] = subset["jerk"].rolling(smooth_window, center=True, min_periods=1).mean()
    subset["j_std"]  = subset["jerk"].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    # ------------------------------------------------------------------
    # Motion sign changes
    # ------------------------------------------------------------------
    subset["vx_sign"] = np.sign(subset["vx"]).fillna(0.0)
    subset["vx_sign_change"] = (
        subset["vx_sign"].diff().abs() > 0
    ).astype(int)
    
    subset["vy_sign"] = np.sign(subset["vy"]).fillna(0.0)
    subset["vy_sign_change"] = (
        subset["vy_sign"].diff().abs() > 0
    ).astype(int)

    return subset


def transform_for_model(
    df: pd.DataFrame,
    preprocessors: dict,
):
    X = df[FEATURE_COLS]
    X_deep = df[FEATURE_COLS_DEEP]

    X_scaled = preprocessors["scaler"].transform(X)
    X_deep_scaled = preprocessors["scaler_deep"].transform(X_deep)

    return X_scaled, X_deep_scaled


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
    X_new, X_new_deep = transform_for_model(new_df, preprocessors)

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
