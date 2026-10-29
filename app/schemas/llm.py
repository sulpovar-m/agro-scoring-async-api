from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction import PredictionResponse


class LLMChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=5000)
    model: str = Field(default="auto", min_length=1)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=300, ge=1, le=2000)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str
    answer: str
    finish_reason: str | None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMChatResponse
