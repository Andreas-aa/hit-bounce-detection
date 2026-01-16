# Hit & Bounce Detection

This project detects **ball hits** and **bounces** from 2D trajectory data using:

- **Supervised ML (Random Forest)**
- **Physics-based rules**

It includes preprocessing, feature extraction, and **temporal suppression** to improve real-life prediction accuracy.

---

## Project Structure

HIT-BOUNCE-DETECTION/
├── model/ # Saved Random Forest models
├── tool/ # Utility scripts
├── build_features.py # Feature extraction for ML and physics
├── physics.py # Physics-based event detection
├── main.py # Main script for supervised and unsupervised detection
├── preprocessors.joblib # Saved preprocessing scalers and label encoder
├── thresholds_physics.joblib # Physics thresholds for bounce/hit detection
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Features

- Computes **velocity, acceleration, jerk** (raw and smoothed)
- Computes **log magnitudes**, **directional changes**, and **rolling statistics**
- Detects **motion sign changes** in x and y axes
- Physics-based rules consider:
  - Vertical acceleration (ay)
  - Horizontal speed changes in magnitude and sign change
  - Vertical speed ratio 
  - Jerk (rate of change of acceleration)
  - Trajectory angle changes
- Temporal suppression:
  - ±3 frames for ML predictions
  - ±10 frames for physics rules
- Evaluation at **frame level** and **temporal tolerance** (±2 frames)

---

## Installation

```bash
git clone <repo_url>
cd HIT-BOUNCE-DETECTION
pip install -r requirements.txt


Usage
Supervised detection (Random Forest)
from pathlib import Path
from main import supervized_hit_bounce_detection

json_path = Path("path/to/trajectory.json")
results = supervized_hit_bounce_detection(json_path)


Uses pre-trained rf_model.joblib and preprocessors.joblib

Adds "action" key to each frame: "air", "hit", "bounce"

Temporal suppression applied (3-frame window)

Unsupervised physics-based detection
from main import unsupervized_hit_bounce_detection

json_path = Path("path/to/trajectory.json")
results = unsupervized_hit_bounce_detection(json_path)


Uses physics thresholds from thresholds_physics.joblib

Detects candidate peaks in vertical acceleration

Computes scores for hit and bounce

Temporal suppression applied (10-frame window)

Returns enriched JSON with "action" for each frame

Evaluation
Supervised Random Forest Model

Standard evaluation (frame-level)

Event	Precision	Recall	F1-Score
air	0.98	0.98	0.98
bounce	0.24	0.32	0.27
hit	0.41	0.35	0.38

Confusion Matrix

[[22928 304 153]
 [199   100 9  ]
 [195   16  112]]


Temporal tolerance evaluation (±2 frames)

Event	Precision	Recall	F1-Score
Bounce	0.507	0.692	0.585
Hit	0.635	0.539	0.583

Note: Frame-level precision is low for hits/bounces due to natural jitter. Temporal tolerance is more realistic for real-life applications.

Physics-based Rule Detection

Standard evaluation (frame-level)

Event	Precision	Recall	F1-Score
air	0.99	0.99	0.99
bounce	0.68	0.56	0.61
hit	0.68	0.50	0.58

Temporal tolerance evaluation (±2 frames)

Event	Precision	Recall	F1-Score
Bounce	0.944	0.776	0.852
Hit	0.912	0.666	0.770

Suppression of neighbor frames is key: only the strongest nearby event is kept, improving real-life predictions.

Notes

ML model performs better in temporal tolerance evaluation, even if frame-level precision seems low.

Physics-based detection is more interpretable and performs well in high-confidence events.

Temporal suppression prevents over-counting of hits/bounces that occur in quick succession.

build_features.py contains all feature engineering logic.

physics.py contains detect_hits_and_bounces and temporal suppression logic for physics detection.
