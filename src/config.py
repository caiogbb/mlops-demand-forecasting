from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "demand.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "features.csv"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "demand_forecasting",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

PROMOTION_METRIC = os.getenv(
    "PROMOTION_METRIC",
    "rmse",
)

MIN_IMPROVEMENT = float(
    os.getenv(
        "MIN_IMPROVEMENT",
        "0.0",
    )
)

EXPERIMENT_NAME = "demand-forecasting"
RANDOM_STATE = 42
