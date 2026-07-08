from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RAW_DATA_PATH


def generate_data(days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    dates = pd.date_range(
        start="2024-01-01",
        periods=days,
        freq="D",
    )

    day_index = np.arange(days)

    weekly_seasonality = 18 * np.sin(2 * np.pi * day_index / 7)

    yearly_seasonality = 12 * np.sin(2 * np.pi * day_index / 365.25)

    trend = 0.04 * day_index

    promotion = rng.binomial(
        n=1,
        p=0.18,
        size=days,
    )

    price = np.clip(
        25 + rng.normal(0, 2.2, days) - promotion * 2,
        15,
        None,
    )

    demand = 120 + trend + weekly_seasonality + yearly_seasonality + promotion * 28 - 2.1 * price

    demand += rng.normal(
        0,
        7,
        days,
    )

    demand = np.clip(
        demand,
        0,
        None,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "price": price.round(2),
            "promotion": promotion,
            "demand": demand.round(2),
        }
    )


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_data()

    dataframe.to_csv(
        RAW_DATA_PATH,
        index=False,
    )

    print(f"Generated {len(dataframe)} rows at {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
