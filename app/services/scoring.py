import asyncio
import logging

from app.errors import ServiceError
from app.models import risk
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)
ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}


class ScoringService:
    def __init__(self, repository, settings):
        self.repository = repository
        self.settings = settings

    def validate(self, items):
        if not risk.MODEL_READY:
            raise ServiceError(503, "Model is unavailable")
        for item in items:
            if item.region not in ALLOWED_REGIONS:
                raise ServiceError(400, f"Unknown region: {item.region}")

    async def predict_batch(self, items, request_ids, source):
        self.validate(items)
        try:
            scores = await asyncio.wait_for(
                risk.model_predict_batch(items),
                timeout=self.settings.inference_timeout_seconds,
            )
        except TimeoutError as error:
            raise ServiceError(504, "Inference timeout") from error
        predictions = []
        for data, score, request_id in zip(items, scores, request_ids, strict=True):
            level = "low" if score < 0.3 else "medium" if score < 0.7 else "high"
            recommendation = {
                "low": "Стандартное рассмотрение",
                "medium": "Требуется дополнительная проверка",
                "high": "Высокий риск. Требуется ручное рассмотрение",
            }[level]
            predictions.append(
                PredictionResponse(
                    request_id=request_id,
                    farm_id=data.farm_id,
                    risk_score=score,
                    risk_level=level,
                    recommendation=recommendation,
                    model_version=risk.MODEL_VERSION,
                )
            )
        await self.repository.save_many(predictions, source)
        for prediction in predictions:
            logger.info(
                "Prediction saved request_id=%s farm_id=%s source=%s",
                prediction.request_id,
                prediction.farm_id,
                source,
            )
        return predictions
