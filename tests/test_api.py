import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from app.config import Settings
from app.main import create_app
from app.models import risk
from app.schemas.prediction import FarmRequest
from app.services.batching import QueueItem

FARM = {
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


def completion(request):
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "choices": [
                {
                    "message": {"content": "Учебная оценка риска: требуется проверка."},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest_asyncio.fixture
async def api(tmp_path):
    app = create_app(
        Settings(database_path=tmp_path / "agro_scoring.db"),
        llm_transport=httpx.MockTransport(completion),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield app, client


async def test_health_and_middleware(api):
    _, c = api
    r = await c.get("/health", headers={"X-Request-ID": "health-check"})
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["X-Request-ID"] == "health-check"
    assert float(r.headers["X-Process-Time"]) >= 0


async def test_predict_persists_and_restarts(api):
    app, c = api
    r = await c.post("/predict", json=FARM, headers={"X-Request-ID": "saved-one"})
    assert r.status_code == 200
    assert r.json()["request_id"] == r.headers["X-Request-ID"] == "saved-one"
    assert r.json()["risk_score"] == 0.5
    async with aiosqlite.connect(app.state.settings.database_path) as db:
        row = await (
            await db.execute("SELECT request_id,source FROM predictions")
        ).fetchone()
        assert row == ("saved-one", "single")
    restarted = create_app(
        replace(app.state.settings), llm_transport=httpx.MockTransport(completion)
    )
    async with restarted.router.lifespan_context(restarted):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted), base_url="http://test"
        ) as other:
            assert (await other.get("/predictions/saved-one")).status_code == 200


@pytest.mark.parametrize(
    "path,payload,code",
    [
        ("/predict", dict(FARM, area_ha=-100), 422),
        ("/predict", dict(FARM, region="Unknown"), 400),
        ("/batch-predict", {"items": [FARM] * 9}, 422),
        ("/batch-predict", {"items": []}, 422),
        ("/llm/chat", {"prompt": ""}, 422),
    ],
)
async def test_invalid_inputs(api, path, payload, code):
    _, c = api
    r = await c.post(path, json=payload)
    assert r.status_code == code
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


async def test_history_and_not_found(api):
    _, c = api
    r = await c.post("/predict", json=FARM)
    rid = r.json()["request_id"]
    assert (await c.get("/predictions/" + rid)).json()["farm_id"] == FARM["farm_id"]
    assert len((await c.get("/predictions?risk_level=medium")).json()) == 1
    assert (await c.get("/predictions/missing")).status_code == 404
    assert (await c.get("/requests/missing/predictions")).status_code == 404


async def test_explicit_batch_single_call_and_atomic_validation(api, monkeypatch):
    _, c = api
    spy = AsyncMock(wraps=risk.model_predict_batch)
    monkeypatch.setattr(risk, "model_predict_batch", spy)
    items = [FARM, dict(FARM, farm_id="FARM-002", previous_defaults=1)]
    r = await c.post("/batch-predict", json={"items": items})
    assert r.status_code == 200 and r.json()["batch_size"] == 2
    assert spy.await_count == 1
    rid = r.json()["request_id"]
    assert (await c.get("/predictions/" + rid)).status_code == 409
    assert len((await c.get("/requests/" + rid + "/predictions")).json()) == 2
    invalid = await c.post(
        "/batch-predict", json={"items": [FARM, dict(FARM, region="unknown")]}
    )
    assert invalid.status_code == 400
    assert len((await c.get("/predictions")).json()) == 2


async def test_pagination(api):
    _, c = api
    await c.post("/batch-predict", json={"items": [FARM] * 8})
    weekday = datetime.now(UTC).strftime("%A").lower()
    r = await c.get(f"/predictions/by-weekday?weekday={weekday}&page=1&per_page=5")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 5
    assert r.json()["meta"]["total_items"] == 8
    assert r.json()["meta"]["total_pages"] == 2
    next_page = await c.get(r.json()["meta"]["next_page"])
    assert len(next_page.json()["data"]) == 3
    assert next_page.json()["meta"]["prev_page"]
    assert (await c.get("/predictions/by-weekday?page=-1")).status_code == 422
    assert (await c.get("/predictions/by-weekday?weekday=never")).status_code == 400
    assert (await c.get(f"/predictions/by-weekday?weekday={weekday}&page=20")).json()[
        "data"
    ] == []


async def test_dynamic_batch_matches_futures(api, monkeypatch):
    _, c = api
    spy = AsyncMock(wraps=risk.model_predict_batch)
    monkeypatch.setattr(risk, "model_predict_batch", spy)
    responses = await asyncio.gather(
        *[
            c.post(
                "/predict-queued",
                json=dict(FARM, farm_id=f"Q-{i}"),
                headers={"X-Request-ID": f"q-{i}"},
            )
            for i in range(8)
        ]
    )
    assert all(r.status_code == 200 for r in responses)
    assert [r.json()["farm_id"] for r in responses] == [f"Q-{i}" for i in range(8)]
    assert [r.json()["request_id"] for r in responses] == [f"q-{i}" for i in range(8)]
    assert spy.await_count == 1
    assert len((await c.get("/predictions")).json()) == 8
    status = (await c.get("/queue-status")).json()
    assert status == dict(
        queue_size=0,
        queue_max_size=100,
        max_batch_size=8,
        max_batch_wait_ms=50,
        inference_timeout_seconds=2.0,
    )


async def test_full_queue_returns_429(api):
    app, c = api
    batcher = app.state.batcher
    await batcher.stop()
    # Приостановленный worker позволяет воспроизводимо заполнить очередь.
    batcher.task = asyncio.create_task(asyncio.Event().wait())
    for i in range(100):
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        batcher.queue.put_nowait(QueueItem(FarmRequest(**FARM), str(i), future))
    r = await c.post("/predict-queued", json=FARM)
    assert r.status_code == 429
    assert (await c.get("/queue-status")).json()["queue_size"] == 100


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/predict", FARM),
        ("/batch-predict", {"items": [FARM]}),
        ("/predict-queued", FARM),
    ],
)
async def test_inference_timeout(api, path, payload):
    app, c = api
    app.state.settings.inference_timeout_seconds = 0.001
    assert (await c.post(path, json=payload)).status_code == 504
    assert (await c.get("/predictions")).json() == []
    app.state.settings.inference_timeout_seconds = 2.0
    assert (await c.post("/predict-queued", json=FARM)).status_code == 200


async def test_llm_and_score_separation(api):
    _, c = api
    r = await c.post("/llm/chat", json={"prompt": "Объясни агроскоринг"})
    assert r.status_code == 200 and r.json()["returned_model"] == "test-model"
    plain = (await c.post("/predict", json=FARM)).json()
    r = await c.post("/predict-with-llm", json=FARM)
    assert r.status_code == 200
    assert r.json()["prediction"]["risk_score"] == plain["risk_score"]
    assert r.json()["prediction"]["risk_level"] == plain["risk_level"]
    assert r.json()["llm"]["request_id"] == r.json()["prediction"]["request_id"]


@pytest.mark.parametrize(
    "failure,code",
    [
        ("http", 502),
        ("unavailable", 503),
        ("connect", 503),
        ("timeout", 504),
        ("invalid", 502),
    ],
)
async def test_provider_errors(api, failure, code):
    app, c = api

    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("offline", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if failure == "invalid":
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(503 if failure == "unavailable" else 500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider:
        app.state.llm.client = provider
        r = await c.post("/llm/chat", json={"prompt": "test"})
        assert r.status_code == code
        assert r.json()["request_id"] == r.headers["X-Request-ID"]


async def test_model_storage_and_internal_failure(api, monkeypatch, tmp_path):
    app, c = api
    monkeypatch.setattr(risk, "MODEL_READY", False)
    assert (await c.post("/predict", json=FARM)).status_code == 503
    monkeypatch.setattr(risk, "MODEL_READY", True)
    original = app.state.repository.path
    app.state.repository.path = tmp_path / "missing" / "db.sqlite"
    assert (await c.post("/predict", json=FARM)).status_code == 503
    app.state.repository.path = original
    monkeypatch.setattr(
        risk, "model_predict_batch", AsyncMock(side_effect=RuntimeError("test fault"))
    )
    r = await c.post("/predict", json=FARM)
    assert r.status_code == 500
    assert r.json()["request_id"] == r.headers["X-Request-ID"]
    assert (await c.get("/predictions")).json() == []
