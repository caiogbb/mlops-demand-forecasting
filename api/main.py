from __future__ import annotations

import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
)
from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app,
)

from api.model_loader import (
    load_model,
)
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from src.config import (
    MODEL_ALIAS,
    MODEL_NAME,
)

app = FastAPI(
    title="Demand Forecasting API",
    version="1.0.0",
)

app.mount(
    "/metrics",
    make_asgi_app(),
)

REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
) -> PredictionResponse:
    REQUESTS.inc()

    try:
        with LATENCY.time():
            model = load_model()

            dataframe = pd.DataFrame([payload.model_dump()])

            prediction = float(model.predict(dataframe)[0])

            prediction = max(
                0.0,
                prediction,
            )

        return PredictionResponse(
            prediction=round(
                prediction,
                2,
            ),
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    except Exception as exception:
        raise HTTPException(
            status_code=503,
            detail=(f"Model is unavailable: {exception}"),
        ) from exception
