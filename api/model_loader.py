from __future__ import annotations

from functools import lru_cache

import mlflow
import mlflow.pyfunc

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    return mlflow.pyfunc.load_model(model_uri)
