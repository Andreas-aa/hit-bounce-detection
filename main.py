import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import os
import joblib
import Path

FEATURE_COLS = [
    "delta_angle",
    "vx_sign_change", 
    "vy_sign_change",
    "v_mean", "v_std",
    "a_mean", "a_std",
    "j_mean", "j_std",
    "vx_abs_raw", "vy_abs_raw",
    "ax_abs_raw", "ay_abs_raw",
    "jx_abs_raw", "jy_abs_raw",
]

FEATURE_COLS_DEEP = FEATURE_COLS + ["x_i", "y_i"]

# Load trained model
def load_model(model_path="model/lstm_model.joblib"):
 with open(model_path, "rb") as f:
     model = joblib.load(f)
 return model




if __name__ == "__main__":
 model = load_model()
 



def build_features(
    subset_df: pd.DataFrame,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """
    Feature builder for ball hit / bounce detection.

    """

    def central_diff(series):
        """Central diff with forward/backward at boundaries."""
        diff = series.shift(-1) - series.shift(1)

        # forward diff at start
        diff.iloc[0] = series.iloc[1] - series.iloc[0]

        # backward diff at end
        diff.iloc[-1] = series.iloc[-1] - series.iloc[-2]

        return diff

    def second_diff(series):
        """Second derivative with asymmetric boundaries."""
        diff2 = series.shift(-1) - 2 * series + series.shift(1)

        diff2.iloc[0] = series.iloc[2] - 2 * series.iloc[1] + series.iloc[0]
        diff2.iloc[-1] = series.iloc[-1] - 2 * series.iloc[-2] + series.iloc[-3]

        return diff2

    def angular_diff(series):
        """Central difference with boundary handling and pi-wrapping"""
        diff = series.shift(-1) - series.shift(1)
        
        # wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # forward difference at start
        diff.iloc[0] = series.iloc[1] - series.iloc[0]
        diff.iloc[0] = (diff.iloc[0] + np.pi) % (2 * np.pi) - np.pi
        
        # backward difference at end
        diff.iloc[-1] = series.iloc[-1] - series.iloc[-2]
        diff.iloc[-1] = (diff.iloc[-1] + np.pi) % (2 * np.pi) - np.pi
        
        return diff / 2  # divide by 2 for central difference

    subset = subset_df.copy()
    subset = subset.sort_index()

    # --- numeric positions ---
    subset["x_i"] = pd.to_numeric(subset["x"], errors="coerce")
    subset["y_i"] = pd.to_numeric(subset["y"], errors="coerce")
    subset = subset.dropna(subset=["x_i", "y_i"])

    # ------------------------------------------------------------------
    # Raw positions (always preserved)
    # ------------------------------------------------------------------
    subset["x_raw"] = subset["x_i"]
    subset["y_raw"] = subset["y_i"]

    # ------------------------------------------------------------------
    # Centered smoothing on positions
    # ------------------------------------------------------------------

    # Centered rolling mean reduces high-frequency measurement noise
    # without eliminating physical discontinuities (hits / bounces).
    # Impulsive events are preserved as extrema and sign changes in
    # derived kinematic quantities.
    subset["x_s"] = (
        subset["x_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )
    subset["y_s"] = (
        subset["y_raw"]
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
    )


    # ------------------------------------------------------------------
    # Time step (central)
    # ------------------------------------------------------------------
    subset.index = pd.to_numeric(subset.index, errors="coerce")
    t = subset.index.to_series()
    dt = t.shift(-1) - t.shift(1)
    dt.iloc[0] = t.iloc[1] - t.iloc[0]
    dt.iloc[-1] = t.iloc[-1] - t.iloc[-2]

    # ------------------------------------------------------------------
    # Smoothed derivatives (stable kinematics)
    # ------------------------------------------------------------------
    dx_s = central_diff(subset["x_s"])
    dy_s = central_diff(subset["y_s"])

    subset["vx"] = dx_s / dt
    subset["vy"] = dy_s / dt
    
    subset["ax"] = second_diff(subset["x_s"]) / dt
    subset["ay"] = second_diff(subset["y_s"]) / dt

    subset["jx"] = second_diff(subset["vx"]) / dt
    subset["jy"] = second_diff(subset["vy"]) / dt

    # ------------------------------------------------------------------
    # Raw derivatives (impulse-sensitive)
    # ------------------------------------------------------------------
    dx_r = central_diff(subset["x_raw"])
    dy_r = central_diff(subset["y_raw"])

    subset["vx_raw"] = dx_r / dt
    subset["vy_raw"] = dy_r / dt

    subset["ax_raw"] = second_diff(subset["x_raw"]) / dt
    subset["ay_raw"] = second_diff(subset["y_raw"]) / dt

    subset["jx_raw"] = second_diff(subset["vx_raw"]) / dt
    subset["jy_raw"] = second_diff(subset["vy_raw"]) / dt

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
    # Directional features
    # ------------------------------------------------------------------
    subset["angle"] = np.arctan2(subset["vy"], subset["vx"])

    subset["delta_angle"] = angular_diff(subset["angle"])

    # ------------------------------------------------------------------
    # Centered rolling statistics (smoothed features)
    # ------------------------------------------------------------------
    w = smooth_window

    subset["v_mean"] = subset["v"].rolling(w, center=True, min_periods=1).mean()
    subset["v_std"]  = subset["v"].rolling(w, center=True, min_periods=1).std().fillna(0)

    subset["a_mean"] = subset["a"].rolling(w, center=True, min_periods=1).mean()
    subset["a_std"]  = subset["a"].rolling(w, center=True, min_periods=1).std().fillna(0)

    subset["j_mean"] = subset["jerk"].rolling(w, center=True, min_periods=1).mean()
    subset["j_std"]  = subset["jerk"].rolling(w, center=True, min_periods=1).std().fillna(0)

    # ------------------------------------------------------------------
    # Motion sign changes (robust bounce indicator)
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
    new_df = build_features(file_df, smooth_window=7)
    preprocessors = joblib.load("model/preprocessors.joblib")
    X_new, X_new_deep = transform_for_model(new_df, preprocessors)

    # Load model and predict
    model = load_model()
    y_pred = model.predict(X_new_deep)
    y_pred_labels = preprocessors["label_encoder"].inverse_transform(y_pred)

    # 4️⃣ Add predictions as a new column
    new_df["action"] = y_pred_labels

    # 5️⃣ Update original JSON
    # Convert DataFrame back to dictionary with same structure
    updated_json = new_df[["x", "y", "visible", "action"]].T.to_dict()
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(updated_json, f, indent=4)

    print(f"Predictions added to '{json_path}' successfully!")