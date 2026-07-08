from pydantic import (
    BaseModel,
    Field,
)


class PredictionRequest(BaseModel):
    price: float = Field(gt=0)

    promotion: int = Field(
        ge=0,
        le=1,
    )

    day_of_week: int = Field(
        ge=0,
        le=6,
    )

    month: int = Field(
        ge=1,
        le=12,
    )

    day_of_year: int = Field(
        ge=1,
        le=366,
    )

    time_index: int = Field(ge=0)

    demand_lag_1: float = Field(ge=0)

    demand_lag_7: float = Field(ge=0)

    rolling_mean_7: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_alias: str
