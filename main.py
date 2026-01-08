import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import os
import pickle


# Load trained model
def load_model(model_path="model/trained_model.pkl"):
 with open(model_path, "rb") as f:
     model = pickle.load(f)
 return model


def detect_hit(input_features, model):
 """
 Detect Hit event.
 
 Parameters:
     input_features (array-like): Features for prediction
     model: Trained supervised model
     
 Returns:
     bool: True if Hit detected, False otherwise
 """
 prediction = model.predict([input_features])
 return prediction[0] == "hit"


def detect_bounce(input_features, model):
 """
 Detect Bounce event.
 
 Parameters:
     input_features (array-like): Features for prediction
     model: Trained supervised model
     
 Returns:
     bool: True if Bounce detected, False otherwise
 """
 prediction = model.predict([input_features])
 return prediction[0] == "bounce"


if __name__ == "__main__":
 model = load_model()
 
 # Example input (replace with real data)
 sample_input = [0.5, 1.2, 0.8]
 
 print("Hit detected:", detect_hit(sample_input, model))
 print("Bounce detected:", detect_bounce(sample_input, model))







def unsupervised_hit_bounce_detection(json_path: str,
                                      fps: float = 50.0,
                                      smooth_window: int = 7,
                                      min_gap_interp: int = 3,
                                      bounce_min_sep_frames: int = 20,
                                      hit_min_sep_frames: int = 15) -> Dict[str, Any]:
    """
    Unsupervised detection of 'hit' & 'bounce' events from ball-tracking JSON.

    Parameters
    ----------
    json_path : str
        Path to the ball-tracking JSON file (keys = frame numbers as strings).
    fps : float
        Frames per second. Adjust to match the video (e.g., 25/50/60).
    smooth_window : int
        Rolling window (centered) used to smooth positions.
    min_gap_interp : int
        Maximum consecutive invisible frames to linearly interpolate.
    bounce_min_sep_frames : int
        Minimal frame distance between consecutive bounces (non-maximum suppression).
    hit_min_sep_frames : int
        Minimal frame distance between consecutive hits (non-maximum suppression).

    Returns
    -------
    Dict[str, Any]
        The same JSON structure, enriched with "pred_action" in each frame dict.
    """

    if not isinstance(json_path, str):
        raise TypeError("json_path must be a string path to a JSON file.")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # ---- Load JSON ----
    with open(json_path, "r", encoding="utf-8") as f:
        ball_data = json.load(f)

    # ---- Dict -> sorted DataFrame without pred_action for unsupervised method ----
    records = []
    for k, v in ball_data.items():
        frm = int(k)
        records.append({
            "frame": frm,
            "x": float(v.get("x", np.nan)),
            "y": float(v.get("y", np.nan)),
            "visible": bool(v.get("visible", True))
        })
    df = pd.DataFrame(records).sort_values("frame").reset_index(drop=True)

    # ---- Handle visibility & small gaps ----
    df["x_i"] = df["x"]
    df["y_i"] = df["y"]
    df.loc[~df["visible"], ["x_i", "y_i"]] = np.nan

    # Interpolate only short gaps (limit = min_gap_interp)
    df["x_i"] = df["x_i"].interpolate(limit=min_gap_interp)
    df["y_i"] = df["y_i"].interpolate(limit=min_gap_interp)

    # ---- Smoothing ----
    df["x_s"] = df["x_i"].rolling(window=smooth_window, center=True, min_periods=1).mean()
    df["y_s"] = df["y_i"].rolling(window=smooth_window, center=True, min_periods=1).mean()

    # ---- Kinematics (central differences) ----
    dt = 1.0 / fps
    df["vx"] = (df["x_s"].shift(-1) - df["x_s"].shift(1)) / (2 * dt)
    df["vy"] = (df["y_s"].shift(-1) - df["y_s"].shift(1)) / (2 * dt)
    df["ax"] = (df["vx"].shift(-1) - df["vx"].shift(1)) / (2 * dt)
    df["ay"] = (df["vy"].shift(-1) - df["vy"].shift(1)) / (2 * dt)
    df["jy"] = (df["ay"].shift(-1) - df["ay"].shift(1)) / (2 * dt)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)

    # ---- Candidate events ----
    # Bounce: local min in y_s + vy flips from negative to positive + ay positive spike
    df["is_local_min_y"] = (df["y_s"] < df["y_s"].shift(1)) & (df["y_s"] < df["y_s"].shift(-1))
    df["vy_flip_dn_up"] = (df["vy"].shift(1) < 0) & (df["vy"] > 0)
    ay_pos_thr = np.nanpercentile(df["ay"], 90)
    df["ay_spike_pos"] = df["ay"] >= ay_pos_thr

    bounce_candidates = df.index[
        df["is_local_min_y"] & df["vy_flip_dn_up"] & df["ay_spike_pos"]
    ].tolist()

    # Hit: speed & jerk spikes away from ground minima
    speed_thr = np.nanpercentile(df["speed"], 90)
    jy_thr = np.nanpercentile(np.abs(df["jy"]), 90)
    y_bottom_thr = np.nanpercentile(df["y_s"], 10)

    hit_candidates = df.index[
        (df["speed"] >= speed_thr) &
        (np.abs(df["jy"]) >= jy_thr) &
        (~df["is_local_min_y"]) &
        (df["y_s"] > y_bottom_thr)
    ].tolist()

    # ---- Non-maximum suppression (spacing) ----
    def nms(indices, min_sep):
        if not indices:
            return []
        selected = [indices[0]]
        for i in indices[1:]:
            if (i - selected[-1]) >= min_sep:
                selected.append(i)
        return selected

    bounce_idx = nms(sorted(bounce_candidates), bounce_min_sep_frames)
    hit_idx = nms(sorted(hit_candidates), hit_min_sep_frames)

    # ---- Resolve conflicts near minima ----
    w = 5  # neighborhood size (frames)
    hit_set = set(hit_idx)
    bounce_set = set(bounce_idx)
    for b in list(bounce_set):
        neighborhood = set(range(b - w, b + w + 1))
        overlap_hits = list(hit_set.intersection(neighborhood))
        if overlap_hits:
            # Prefer bounce if strict local min; otherwise drop bounce
            if not df.loc[b, "is_local_min_y"]:
                bounce_set.discard(b)
            else:
                for h in overlap_hits:
                    hit_set.discard(h)

    # ---- Build predictions ----
    pred = np.array(["air"] * len(df))
    for i in bounce_set:
        pred[i] = "bounce"
    for i in hit_set:
        pred[i] = "hit"
        

    # ---- Merge back into original JSON dict ----
    enriched = {}
    for idx, row in df.iterrows():
        k = str(int(row["frame"]))
        obj = ball_data.get(k, {}).copy()
        obj["pred_action"] = pred[idx]
        enriched[k] = obj

    return enriched
