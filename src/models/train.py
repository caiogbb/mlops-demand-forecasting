from __future__ import annotations

import json
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
)

from src.config import (
    EXPERIMENT_NAME,
    METRICS_PATH,
    MIN_IMPROVEMENT,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    PROCESSED_DATA_PATH,
    PROMOTION_METRIC,
    RANDOM_STATE,
)

FEATURES = [
    "price",
    "promotion",
    "day_of_week",
    "month",
    "day_of_year",
    "time_index",
    "demand_lag_1",
    "demand_lag_7",
    "rolling_mean_7",
]

TARGET = "demand"


@dataclass
class Candidate:
    name: str
    model: object
    params: dict


def candidates() -> list[Candidate]:
    return [
        Candidate(
            name="random_forest",
            model=RandomForestRegressor(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            params={
                "n_estimators": 250,
                "max_depth": 12,
                "min_samples_leaf": 2,
            },
        ),
        Candidate(
            name="hist_gradient_boosting",
            model=HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
            params={
                "max_iter": 250,
                "learning_rate": 0.06,
                "max_leaf_nodes": 31,
            },
        ),
    ]


def evaluate_model(
    model: object,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)

    return {
        "mae": float(
            mean_absolute_error(
                target,
                predictions,
            )
        ),
        "rmse": float(
            mean_squared_error(
                target,
                predictions,
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(
                target,
                predictions,
            )
        ),
    }


def current_champion_metric(
    client: MlflowClient,
) -> float | None:
    try:
        version = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )

        run = client.get_run(version.run_id)

        metric_value = run.data.metrics.get(PROMOTION_METRIC)

        if metric_value is None:
            return None

        return float(metric_value)

    except Exception:
        return None


def should_promote(
    candidate_metric: float,
    champion_metric: float | None,
) -> bool:
    if champion_metric is None:
        return True

    threshold = champion_metric - MIN_IMPROVEMENT

    return candidate_metric < threshold


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    dataframe = pd.read_csv(PROCESSED_DATA_PATH)

    features = dataframe[FEATURES]

    target = dataframe[TARGET]

    split_index = int(len(dataframe) * 0.8)

    x_train = features.iloc[:split_index]

    x_test = features.iloc[split_index:]

    y_train = target.iloc[:split_index]

    y_test = target.iloc[split_index:]

    cross_validation = TimeSeriesSplit(n_splits=4)

    results: list[dict] = []

    for candidate in candidates():
        with mlflow.start_run(run_name=candidate.name) as run:
            cv_rmse_values = []

            for train_index, validation_index in cross_validation.split(x_train):
                candidate.model.fit(
                    x_train.iloc[train_index],
                    y_train.iloc[train_index],
                )

                fold_metrics = evaluate_model(
                    candidate.model,
                    x_train.iloc[validation_index],
                    y_train.iloc[validation_index],
                )

                cv_rmse_values.append(fold_metrics["rmse"])

            candidate.model.fit(
                x_train,
                y_train,
            )

            holdout_metrics = evaluate_model(
                candidate.model,
                x_test,
                y_test,
            )

            holdout_metrics["cv_rmse_mean"] = float(sum(cv_rmse_values) / len(cv_rmse_values))

            mlflow.log_params(
                {
                    "algorithm": candidate.name,
                    **candidate.params,
                }
            )

            mlflow.log_metrics(holdout_metrics)

            mlflow.set_tags(
                {
                    "pipeline": "training",
                    "dataset_rows": str(len(dataframe)),
                    "selection_metric": (PROMOTION_METRIC),
                }
            )

            signature = infer_signature(
                x_train,
                candidate.model.predict(x_train),
            )

            mlflow.sklearn.log_model(
                sk_model=candidate.model,
                name="model",
                signature=signature,
                input_example=x_train.head(3),
            )

            results.append(
                {
                    "name": candidate.name,
                    "run_id": run.info.run_id,
                    "metrics": holdout_metrics,
                }
            )

    best_result = min(
        results,
        key=lambda result: result["metrics"][PROMOTION_METRIC],
    )

    model_uri = f"runs:/{best_result['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client = MlflowClient()

    champion_metric = current_champion_metric(client)

    candidate_metric = float(best_result["metrics"][PROMOTION_METRIC])

    promoted = should_promote(
        candidate_metric,
        champion_metric,
    )

    if promoted:
        client.set_registered_model_alias(
            MODEL_NAME,
            MODEL_ALIAS,
            registered_model.version,
        )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "best_model": best_result,
        "registered_version": (registered_model.version),
        "previous_champion_metric": (champion_metric),
        "promoted_to_champion": promoted,
    }

    METRICS_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
