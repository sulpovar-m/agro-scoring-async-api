import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request

from app.config import MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS, QUEUE_MAX_SIZE
from app.errors import ServiceError
from app.models import risk
from app.schemas.llm import LLMChatRequest, LLMChatResponse, PredictWithLLMResponse
from app.schemas.prediction import (
    BatchRequest,
    BatchResponse,
    FarmRequest,
    HealthResponse,
    ModelInfoResponse,
    PaginatedPredictions,
    PredictionResponse,
    QueueStatus,
    RiskLevel,
    StoredPrediction,
)


# Заголовок объявлен как параметр, чтобы его можно было передать из Swagger.
async def request_header(x_request_id: Annotated[str | None, Header()] = None):
    return x_request_id


router = APIRouter(dependencies=[Depends(request_header)])


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {"status": "ok"}


@router.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    return {
        "model_name": risk.MODEL_NAME,
        "model_version": risk.MODEL_VERSION,
        "model_type": risk.MODEL_TYPE,
        "status": "ready" if risk.MODEL_READY else "unavailable",
    }


@router.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict(data: FarmRequest, request: Request):
    return (
        await request.app.state.scoring.predict_batch(
            [data], [request.state.request_id], "single"
        )
    )[0]


@router.post("/batch-predict", response_model=BatchResponse, tags=["Scoring"])
async def batch_predict(data: BatchRequest, request: Request):
    started = time.perf_counter()
    results = await request.app.state.scoring.predict_batch(
        data.items, [request.state.request_id] * len(data.items), "batch"
    )
    return {
        "request_id": request.state.request_id,
        "batch_size": len(results),
        "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "items": results,
    }


@router.post("/predict-queued", response_model=PredictionResponse, tags=["Scoring"])
async def predict_queued(data: FarmRequest, request: Request):
    return await request.app.state.batcher.submit(data, request.state.request_id)


@router.get("/queue-status", response_model=QueueStatus, tags=["System"])
async def queue_status(request: Request):
    return {
        "queue_size": request.app.state.batcher.queue.qsize(),
        "queue_max_size": QUEUE_MAX_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_batch_wait_ms": MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": request.app.state.settings.inference_timeout_seconds,
    }


@router.get("/predictions", response_model=list[StoredPrediction], tags=["History"])
async def predictions(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    risk_level: RiskLevel | None = None,
):
    return await request.app.state.repository.recent(limit, risk_level)


# Статический путь должен предшествовать /predictions/{request_id}.
@router.get(
    "/predictions/by-weekday", response_model=PaginatedPredictions, tags=["History"]
)
async def by_weekday(
    request: Request,
    weekday: str = "monday",
    page: int = Query(1, ge=1),
    per_page: int = Query(5, ge=1, le=100),
):
    weekdays = {
        "sunday": "0",
        "monday": "1",
        "tuesday": "2",
        "wednesday": "3",
        "thursday": "4",
        "friday": "5",
        "saturday": "6",
    }
    weekday = weekday.lower()
    if weekday not in weekdays:
        raise ServiceError(400, "Unknown weekday")
    data, total = await request.app.state.repository.by_weekday(
        weekdays[weekday], page, per_page
    )
    pages = (total + per_page - 1) // per_page

    def link(number):
        return str(
            request.url.include_query_params(
                weekday=weekday, page=number, per_page=per_page
            )
        )

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": pages,
            "total_items": total,
            "next_page": link(page + 1) if page < pages else None,
            "prev_page": link(page - 1) if page > 1 else None,
        },
    }


@router.get(
    "/predictions/{request_id}", response_model=StoredPrediction, tags=["History"]
)
async def prediction_by_id(request_id: str, request: Request):
    rows = await request.app.state.repository.by_request(request_id)
    if not rows:
        raise ServiceError(404, "Prediction not found")
    if len(rows) > 1:
        raise ServiceError(
            409, "Multiple predictions; use /requests/{request_id}/predictions"
        )
    return rows[0]


@router.get(
    "/requests/{request_id}/predictions",
    response_model=list[StoredPrediction],
    tags=["History"],
)
async def request_predictions(request_id: str, request: Request):
    rows = await request.app.state.repository.by_request(request_id)
    if not rows:
        raise ServiceError(404, "Predictions not found")
    return rows


@router.post("/llm/chat", response_model=LLMChatResponse, tags=["LLM"])
async def llm_chat(data: LLMChatRequest, request: Request):
    return await request.app.state.llm.chat(data, request.state.request_id)


@router.post("/predict-with-llm", response_model=PredictWithLLMResponse, tags=["LLM"])
async def predict_with_llm(data: FarmRequest, request: Request):
    prediction = (
        await request.app.state.scoring.predict_batch(
            [data], [request.state.request_id], "llm"
        )
    )[0]
    prompt = (
        "Объясни кратко по-русски результат учебного агроскоринга. "
        "Не пересчитывай и не изменяй оценку и категорию риска. "
        f"Учебная оценка: risk_score={prediction.risk_score}, "
        f"risk_level={prediction.risk_level}."
    )
    explanation = await request.app.state.llm.chat(
        LLMChatRequest(prompt=prompt, model=request.app.state.settings.llm_model),
        request.state.request_id,
    )
    return {"prediction": prediction, "llm": explanation}
