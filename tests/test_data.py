import pytest

from src.data.generate_data import (
    generate_data,
)
from src.data.validation import (
    validate_dataframe,
)
from src.features.build_features import (
    build_features,
)


def test_generated_data_is_valid():
    dataframe = generate_data(days=60)

    validate_dataframe(dataframe)

    assert len(dataframe) == 60

    assert (dataframe["demand"] >= 0).all()


def test_feature_generation():
    dataframe = generate_data(days=60)

    features = build_features(dataframe)

    assert not features.empty

    required_features = {
        "demand_lag_1",
        "demand_lag_7",
        "rolling_mean_7",
    }

    assert required_features.issubset(features.columns)


def test_negative_demand_is_rejected():
    dataframe = generate_data(days=20)

    dataframe.loc[
        0,
        "demand",
    ] = -1

    with pytest.raises(ValueError):
        validate_dataframe(dataframe)
