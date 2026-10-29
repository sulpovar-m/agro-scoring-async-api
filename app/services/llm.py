import time

import httpx

from app.errors import ServiceError
from app.schemas.llm import LLMChatResponse


class LLMService:
    def __init__(self, client, settings):
        self.client = client
        self.settings = settings

    async def chat(self, data, request_id):
        started = time.perf_counter()
        try:
            response = await self.client.post(
                self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers={"X-Request-ID": request_id},
                json={
                    "model": data.model,
                    "messages": [{"role": "user", "content": data.prompt}],
                    "temperature": data.temperature,
                    "max_tokens": data.max_tokens,
                },
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise ServiceError(504, "LLM timeout") from error
        except httpx.RequestError as error:
            raise ServiceError(503, "LLM service is unavailable") from error
        except httpx.HTTPStatusError as error:
            code = 503 if error.response.status_code == 503 else 502
            raise ServiceError(
                code, f"LLM provider returned HTTP {error.response.status_code}"
            ) from error
        try:
            body = response.json()
            choice = body["choices"][0]
            answer = choice["message"]["content"]
            returned_model = body["model"]
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or not isinstance(returned_model, str)
            ):
                raise ValueError("Empty or invalid completion")
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ServiceError(502, "Invalid LLM provider response") from error
        return LLMChatResponse(
            request_id=request_id,
            requested_model=data.model,
            returned_model=returned_model,
            answer=answer,
            finish_reason=choice.get("finish_reason"),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
