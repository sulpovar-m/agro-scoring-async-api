import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.routes import router
from app.config import Settings
from app.errors import ServiceError
from app.repositories.predictions import PredictionRepository
from app.schemas.prediction import ErrorResponse
from app.services.batching import DynamicBatcher
from app.services.llm import LLMService
from app.services.scoring import ScoringService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings=None, llm_transport=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        repository = PredictionRepository(settings.database_path)
        # API остаётся доступным при отказе БД; операции хранения вернут 503.
        try:
            await repository.initialize()
        except ServiceError:
            logger.exception("Database initialization failed")
        async with httpx.AsyncClient(transport=llm_transport) as client:
            app.state.settings = settings
            app.state.repository = repository
            app.state.scoring = ScoringService(repository, settings)
            app.state.llm = LLMService(client, settings)
            app.state.batcher = DynamicBatcher(app.state.scoring)
            app.state.batcher.start()
            try:
                yield
            finally:
                await app.state.batcher.stop()

    app = FastAPI(
        title="Agro Scoring API",
        version="2.0",
        lifespan=lifespan,
        description="Лабораторная работа №2. Сульповар М. Л., МПИ-25-1-1.",
        responses={
            code: {"model": ErrorResponse}
            for code in [400, 404, 409, 422, 429, 500, 502, 503, 504]
        },
    )

    @app.middleware("http")
    async def identify_request(request: Request, call_next):
        started = time.perf_counter()
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error request_id=%s", request.state.request_id)
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request.state.request_id,
                },
            )
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time"] = f"{elapsed:.6f}"
        logger.info(
            "%s %s status=%s request_id=%s time=%.6f",
            request.method,
            request.url.path,
            response.status_code,
            request.state.request_id,
            elapsed,
        )
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.detail, "request_id": request.state.request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(
            status_code=error.status_code,
            headers=error.headers,
            content={"detail": error.detail, "request_id": request.state.request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {"detail": error.errors(), "request_id": request.state.request_id},
                custom_encoder={ValueError: str},
            ),
        )

    app.include_router(router)
    return app


app = create_app()
