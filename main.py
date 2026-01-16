import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from build_features import build_features, transform_for_model
from physics import detect_hits_and_bounces, load_threshold

SMOOTH_WINDOW = 7

def suppress_close_events(candidates, min_frames=3):
    """
    Keep only the highest-probability event within a window.
    Suppression is class-aware (hit does not suppress bounce).
    """
    candidates.sort(key=lambda x: x[0])
    final_events = {}
    for frame, label, proba in candidates:
        if not final_events:
            final_events[frame] = (label, proba)
            continue

        last_frame = max(final_events.keys())
        last_label, last_proba = final_events[last_frame]

        if frame - last_frame >= min_frames:
            final_events[frame] = (label, proba)
        else:
            # suppress only if same class
            if label == last_label and proba > last_proba:
                final_events[last_frame] = (label, proba)

    return final_events

def load_model(model_path="model/rf_model.joblib"):
    """Load trained RandomForest model."""
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def supervized_hit_bounce_detection(json_path: Path):
    """ML-based prediction with temporal suppression (3 frames)."""
    # Load JSON into DataFrame
    with json_path.open("r", encoding="utf-8") as f:
        ball_data = json.load(f)
    file_df = pd.DataFrame(ball_data).T
    file_df.index.name = "image_frame"
    file_df = file_df.reindex(columns=["x", "y", "visible"])

    # Feature engineering
    new_df = build_features(file_df, smooth_window=SMOOTH_WINDOW)
    preprocessors = joblib.load("preprocessors.joblib")
    X_new, _, _ = transform_for_model(new_df, preprocessors)

    # Model inference
    model = load_model()
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)
    class_index = {c: i for i, c in enumerate(model.classes_)}
    predicted_proba = np.array([y_proba[i, class_index[y_pred[i]]] for i in range(len(y_pred))])

    # Candidates for suppression
    candidates = [(frame, label, proba) for frame, label, proba in zip(new_df.index, y_pred, predicted_proba) if label != "air"]
    final_events = suppress_close_events(candidates, min_frames=3)

    # Assign final labels
    new_df["action"] = "air"
    for frame, (label, _) in final_events.items():
        new_df.loc[frame, "action"] = label

    # Export JSON
    enriched_json = new_df[["x", "y", "visible", "action"]].T.to_dict()
    # with json_path.open("w", encoding="utf-8") as f:
    #     json.dump(enriched_json, f, indent=4)

    print(f"Predictions added to '{json_path}' successfully!")
    return enriched_json


def unsupervized_hit_bounce_detection(json_path: Path):
    """Physics-based hit/bounce detection."""
    df = pd.DataFrame(columns=["x", "y", "visible"])
    df.index.name = "image_frame"

    # Load JSON into DataFrame
    with json_path.open("r", encoding="utf-8") as f:
        ball_data = json.load(f)

    file_df = pd.DataFrame(ball_data).T
    file_df.index.name = "image_frame"
    file_df = file_df.reindex(columns=["x", "y", "visible"])

    new_df = build_features(file_df, smooth_window=SMOOTH_WINDOW)
    thresh = load_threshold()
    y_pred = detect_hits_and_bounces(new_df, thresholds=thresh)

    # Initialize all frames as "air"
    y_pred_array = np.array(["air"] * len(new_df), dtype=object)
    for frame, label in y_pred.items():
        try:
            idx = new_df.index.get_loc(frame)
            y_pred_array[idx] = label
        except KeyError:
            pass
    new_df["action"] = y_pred_array

    # Export JSON
    enriched_json = new_df[["x", "y", "visible", "action"]].T.to_dict()
    # with json_path.open("w", encoding="utf-8") as f:
    #     json.dump(enriched_json, f, indent=4)

    print(f"Predictions added to '{json_path}' successfully!")
    return enriched_json