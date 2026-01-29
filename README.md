🎾 # Hit & Bounce Detection from Ball Trajectories


This project detects **ball hits** and **bounces** from 2D trajectory data using:

- **Supervised ML (Random Forest)**
- **Physics-based rules**

It includes preprocessing, feature extraction, and **temporal suppression** to improve real-life prediction accuracy.

📍 Data source:
The model is developed and evaluated on trajectory data extracted from a Roland-Garros match (Alcaraz vs Sinner), capturing realistic ball dynamics from professional-level tennis.

---

## Project Structure
```text
HIT-BOUNCE-DETECTION/
├── model/                    # Saved Random Forest models
├── per_point_v2/             # JSON files for training and testing
├── tool/                     # Viz tool
├── unused_models/            # LSTM, MLP, XGB and other models but not kept for final evaluation 
├── build_features.py         # Feature extraction for ML and physics
├── physics.py                # Physics-based event detection
├── main.py                   # Main script for supervised and unsupervised detection
├── preprocessors.joblib      # Saved preprocessing scalers and label encoder
├── thresholds_physics.joblib # Physics thresholds for bounce/hit detection
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Features

- Computes **velocity, acceleration, jerk** (raw and smoothed)
- Computes **log magnitudes**, **directional changes**, and **rolling statistics**
- Detects **motion sign changes** in x and y axes
- Physics-based rules consider: Thresholds values for detection based on a train set and percentile
  - Vertical acceleration (ay)
  - Horizontal speed changes in magnitude and sign change
  - Vertical speed ratio 
  - Jerk (rate of change of acceleration)
  - Trajectory angle changes

## Assumptions

### Frame precision
Due to natural measurement noise, evaluating **frame-level accuracy** can be misleading. Using **temporal suppression** and **temporal tolerance** provides a more realistic assessment of detection performance, as reflected in the results.
- **Suppression of events predicted too close to each other** :
  - ±3 frames for ML predictions
  - ±10 frames for physics rules
- Evaluation is reported both **at frame level** and with **temporal tolerance** (±2 frames) to account for slight timing variations.

### ML (Random Forest)

- **Class weights are balanced** to further mitigate class imbalance, ensuring rare events like hits and bounces are properly considered during training.
- The hyperparamaters are selected in gridsearch using **F1-macro**, which balances precision and recall across all classes. This is important because the dataset is heavily dominated by the “air” class.

### Physics-based rules
- Rules are applied only to **candidate frames with high absolute vertical acceleration.** This reduces false positives, since hits or bounces produce sudden, significant changes in vertical motion.
- **Bounce rules are intentionally softer than hit rules** to prioritize correct bounce detection, since hits are easier to detect from physics features and can sometimes overshadow bounces.




## Installation

```bash
git clone <repo_url>
cd your_repo
pip install -r requirements.txt
```

## Usage
### Supervised detection (Random Forest)
```bash
from pathlib import Path
from main import supervized_hit_bounce_detection

json_path = Path("path/to/trajectory.json")
results = supervized_hit_bounce_detection(json_path)
```
- Returns enriched JSON with "action" for each frame while saving the results in the input file
- Temporal suppression applied with a ±3 frame window to avoid over-counting nearby events

### Unsupervised physics-based detection
```bash
from pathlib import Path
from main import unsupervized_hit_bounce_detection

json_path = Path("path/to/trajectory.json")
results = unsupervized_hit_bounce_detection(json_path)
```
- Returns enriched JSON with "action" for each frame while saving the results in the input file
- Uses physics thresholds from thresholds_physics.joblib
- Computes scores for hit and bounce
- Temporal suppression applied with a ±10 frame window to avoid over-counting nearby events


## Evaluation (with last 20% frames)

### Supervised Random Forest Model

**Standard evaluation (frame-level)**

| Event  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| air    | 0.99      | 0.99   | 0.99     |
| bounce | 0.68      | 0.56   | 0.61     |
| hit    | 0.68      | 0.50   | 0.58     |

**Temporal tolerance evaluation (±2 frames)**

| Event  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Bounce | 0.944     | 0.776  | 0.852    |
| Hit    | 0.912     | 0.666  | 0.770    |


### Physics-based Rule Detection

**Standard evaluation (frame-level)**

| Event  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| air    | 0.98      | 0.98   | 0.98     |
| bounce | 0.24      | 0.32   | 0.27     |
| hit    | 0.41      | 0.35   | 0.38     |


**Temporal tolerance evaluation (±2 frames)**

| Event  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Bounce | 0.507     | 0.692  | 0.585    |
| Hit    | 0.635     | 0.539  | 0.583    |


 ## **Observation:**  
 - **Random Forest** outperforms physics-based rules when considering temporal tolerance, achieving higher F1-scores for both hits and bounces.  
 - **Physics-based detection** remains interpretable and useful for high-confidence events, but has low scoring metrics, even more for frame-level detection.  
