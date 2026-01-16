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


def build_features(df: pd.DataFrame, smooth_window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """
    Build features from raw ball positions for hit/bounce detection.

    Returns a DataFrame with raw, smoothed, derivatives, magnitudes,
    directional features, rolling statistics, and sign-change indicators.
    """
    new_df = df.copy()
    new_df.index = pd.to_numeric(new_df.index, errors="coerce")
    new_df = new_df.sort_index()
    new_df["x_i"] = pd.to_numeric(new_df["x"], errors="coerce")
    new_df["y_i"] = pd.to_numeric(new_df["y"], errors="coerce")
    new_df = new_df.dropna(subset=["x_i", "y_i"])

    # Raw and smoothed positions
    new_df["x_raw"] = new_df["x_i"]
    new_df["y_raw"] = new_df["y_i"]
    new_df["x_smooth"] = new_df["x_raw"].rolling(smooth_window, center=True, min_periods=1).mean()
    new_df["y_smooth"] = new_df["y_raw"].rolling(smooth_window, center=True, min_periods=1).mean()

    # Time Frame vector
    t = new_df.index.to_series()

    # Smoothed derivatives
    x_smooth = new_df["x_smooth"].to_numpy()
    y_smooth = new_df["y_smooth"].to_numpy()
    new_df["vx"] = np.gradient(x_smooth, t)
    new_df["vy"] = np.gradient(y_smooth, t)
    new_df["ax"] = np.gradient(new_df["vx"], t)
    new_df["ay"] = np.gradient(new_df["vy"], t)
    new_df["jx"] = np.gradient(new_df["ax"], t)
    new_df["jy"] = np.gradient(new_df["ay"], t)

    # Raw derivatives
    x_raw = new_df["x_raw"].to_numpy()
    y_raw = new_df["y_raw"].to_numpy()
    new_df["vx_raw"] = np.gradient(x_raw, t)
    new_df["vy_raw"] = np.gradient(y_raw, t)
    new_df["ax_raw"] = np.gradient(new_df["vx_raw"], t)
    new_df["ay_raw"] = np.gradient(new_df["vy_raw"], t)
    new_df["jx_raw"] = np.gradient(new_df["ax_raw"], t)
    new_df["jy_raw"] = np.gradient(new_df["ay_raw"], t)

    # Absolute raw derivatives
    for col in ["vx", "vy", "ax", "ay", "jx", "jy"]:
        new_df[f"{col}_abs_raw"] = np.abs(new_df[f"{col}_raw"])

    # Magnitudes
    new_df["v"] = np.sqrt(new_df["vx"]**2 + new_df["vy"]**2)
    new_df["a"] = np.sqrt(new_df["ax"]**2 + new_df["ay"]**2)
    new_df["jerk"] = np.sqrt(new_df["jx"]**2 + new_df["jy"]**2)

    # Log magnitudes
    new_df["log_v"] = np.log1p(new_df["v"])
    new_df["log_a"] = np.log1p(new_df["a"])
    new_df["log_j"] = np.log1p(new_df["jerk"])

    # Directional features
    new_df["angle"] = np.arctan2(new_df["vy"], new_df["vx"])
    new_df["delta_angle"] = np.gradient(new_df["angle"])

    # Centered rolling statistics
    for feat, col in zip(["v", "a", "jerk"], ["v", "a", "j"]):
        new_df[f"{col}_mean"] = new_df[feat].rolling(smooth_window, center=True, min_periods=1).mean()
        new_df[f"{col}_std"] = new_df[feat].rolling(smooth_window, center=True, min_periods=1).std().fillna(0)

    # Motion sign changes
    for axis in ["vx", "vy"]:
        new_df[f"{axis}_sign"] = np.sign(new_df[axis]).fillna(0.0)
        new_df[f"{axis}_sign_change"] = (new_df[f"{axis}_sign"].diff().abs() > 0).astype(int)

    return new_df


def transform_for_model(df: pd.DataFrame, preprocessors: dict):
    """
    Apply scaling for supervised model.
    Returns: raw features, scaled features, deep scaled features
    """
    X = df[FEATURE_COLS]
    X_deep = df[FEATURE_COLS_DEEP]

    X_scaled = preprocessors["scaler"].transform(X)
    X_deep_scaled = preprocessors["scaler_deep"].transform(X_deep)

    return X, X_scaled, X_deep_scaled
