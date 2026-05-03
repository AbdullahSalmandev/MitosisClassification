from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    payload_type: str | None = None


class PredictionItem(BaseModel):
    index: int
    label: str
    score: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
