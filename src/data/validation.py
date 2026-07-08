from __future__ import annotations

import pandas as pd

from src.config import RAW_DATA_PATH

REQUIRED_COLUMNS = {
    "date",
    "price",
    "promotion",
    "demand",
}


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    if dataframe[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if (dataframe["price"] <= 0).any():
        raise ValueError("Price must be positive.")

    if (dataframe["demand"] < 0).any():
        raise ValueError("Demand must be non-negative.")

    valid_promotions = set(dataframe["promotion"].unique()).issubset({0, 1})

    if not valid_promotions:
        raise ValueError("Promotion must contain only 0 or 1.")

    pd.to_datetime(
        dataframe["date"],
        errors="raise",
    )


def main() -> None:
    dataframe = pd.read_csv(RAW_DATA_PATH)

    validate_dataframe(dataframe)

    print(f"Dataset validation passed: {len(dataframe)} rows.")


if __name__ == "__main__":
    main()
