"""Validated wire contracts for the OpenAI-compatible sampling bridge."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator


class WireModel(BaseModel):
    # Compatible clients send optional fields the bridge does not consume.
    model_config = ConfigDict(extra="ignore", strict=True)


class ImageURL(WireModel):
    url: StrictStr
    detail: Literal["auto", "low", "high"] = "auto"


class TextPart(WireModel):
    type: Literal["text"]
    text: StrictStr


class ImagePart(WireModel):
    type: Literal["image_url"]
    image_url: ImageURL | StrictStr


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class FunctionCall(WireModel):
    name: StrictStr = Field(min_length=1)
    arguments: StrictStr | dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def object_arguments(cls, value):
        import json
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except ValueError:
                raise ValueError("Tool arguments must contain a JSON object") from None
            if not isinstance(decoded, dict):
                raise ValueError("Tool arguments must contain a JSON object")
        return value


class ToolCall(WireModel):
    id: StrictStr = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(WireModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: StrictStr | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: StrictStr | None = None
    name: StrictStr | None = None
    reasoning_content: StrictStr | None = None
    reasoning: StrictStr | None = None


class FunctionSpec(WireModel):
    name: StrictStr = Field(min_length=1)
    description: StrictStr = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolSpec(WireModel):
    type: Literal["function"] = "function"
    function: FunctionSpec


class CompletionRequest(WireModel):
    model: StrictStr = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolSpec] | None = None
    stream: StrictBool = False
    temperature: float = 0.0
    max_tokens: StrictInt | None = Field(default=None, ge=1)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=1)

    @field_validator("temperature", mode="before")
    @classmethod
    def finite_temperature(cls, value):
        import math
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ValueError("temperature must be a finite number from 0 to 2")
        return value
