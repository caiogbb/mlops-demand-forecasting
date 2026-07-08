from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.data.validation import validate_dataframe


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(dataframe)

    features = dataframe.copy()

    features["date"] = pd.to_datetime(features["date"])

    features["day_of_week"] = features["date"].dt.dayofweek
    features["month"] = features["date"].dt.month
    features["day_of_year"] = features["date"].dt.dayofyear
    features["time_index"] = range(len(features))

    features["demand_lag_1"] = features["demand"].shift(1)
    features["demand_lag_7"] = features["demand"].shift(7)

    features["rolling_mean_7"] = features["demand"].shift(1).rolling(window=7).mean()

    return features.dropna().reset_index(drop=True)


def main() -> None:
    dataframe = pd.read_csv(RAW_DATA_PATH)

    features = build_features(dataframe)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(f"Created {len(features)} feature rows at {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
