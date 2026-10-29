"""Учебная модель на правилах из лабораторной работы №1."""

import asyncio

from app.schemas.prediction import FarmRequest

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.1"
MODEL_TYPE = "risk-scoring"
MODEL_READY = True


def calculate_risk(data: FarmRequest) -> float:
    score = 0.1
    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    if data.temperature_avg < 5 or data.temperature_avg > 35:
        score += 0.05
    return round(min(score, 1.0), 2)


async def model_predict_batch(items: list[FarmRequest]) -> list[float]:
    # Имитация накладных расходов одного вызова модели и расчёта пакета.
    await asyncio.sleep(0.020 + 0.003 * len(items))
    return [calculate_risk(item) for item in items]


async def model_predict(data: FarmRequest) -> float:
    return (await model_predict_batch([data]))[0]
