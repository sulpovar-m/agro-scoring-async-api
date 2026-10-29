from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import MAX_BATCH_SIZE

RiskLevel = Literal["low", "medium", "high"]


class FarmRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "farm_id": "FARM-001",
                "region": "Krasnodar",
                "crop_type": "wheat",
                "area_ha": 1200,
                "temperature_avg": 24,
                "precipitation_mm": 80,
                "payment_delay_days": 45,
                "previous_defaults": 0,
                "debt": 3000000,
            }
        },
    )
    farm_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    crop_type: str = Field(min_length=1)
    area_ha: float = Field(gt=0)
    temperature_avg: float = Field(ge=-60, le=60)
    precipitation_mm: float = Field(ge=0)
    payment_delay_days: int = Field(ge=0)
    previous_defaults: int = Field(ge=0)
    debt: float = Field(ge=0)


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: RiskLevel
    recommendation: str
    model_version: str


class StoredPrediction(PredictionResponse):
    id: int
    source: str
    created_at: str


class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[PredictionResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictions(BaseModel):
    data: list[StoredPrediction]
    meta: PaginationMeta


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class QueueStatus(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class ErrorResponse(BaseModel):
    detail: str | list[dict]
    request_id: str
