"""Provider-neutral model contract, scripted fake, and OpenAI-compatible adapter."""

import json
import os
from collections import deque
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ModelTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content_or_tool_call(self) -> "ModelTurn":
        if not (self.content and self.content.strip()) and not self.tool_calls:
            raise ValueError("Model response must contain text or a tool call.")
        return self


class ModelClientError(RuntimeError):
    """Raised for unavailable or invalid model responses."""


class ModelClient(Protocol):
    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        model: str,
    ) -> ModelTurn:
        """Return one assistant response, either text or tool calls."""


class ScriptedModelClient:
    """A deterministic response queue for repeatable agent/evaluation tests."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = deque(turns)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        model: str,
    ) -> ModelTurn:
        self.requests.append(
            {
                "messages": [message.model_dump(mode="json", exclude_none=True) for message in messages],
                "tools": tools,
                "model": model,
            }
        )
        if not self._turns:
            raise ModelClientError("Scripted model has no response remaining for this turn.")
        return self._turns.popleft()


class OpenAIModelClient:
    """Minimal adapter for APIs implementing OpenAI chat completions and tools."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 45,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._timeout_seconds = timeout_seconds
        if not self._api_key:
            raise ModelClientError("Set OPENAI_API_KEY to use the live model adapter.")

    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        model: str,
    ) -> ModelTurn:
        payload = {
            "model": model,
            "messages": [self._serialize_message(message) for message in messages],
            "tools": tools,
            "tool_choice": "auto",
        }
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read())
        except HTTPError as error:
            raise ModelClientError(f"Model API returned HTTP {error.code}.") from None
        except (URLError, TimeoutError):
            raise ModelClientError("Could not reach the configured model API.") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ModelClientError("Model API returned invalid JSON.") from None

        try:
            message = body["choices"][0]["message"]
            calls = [
                ModelToolCall(
                    id=item["id"],
                    name=item["function"]["name"],
                    arguments=json.loads(item["function"]["arguments"]),
                )
                for item in message.get("tool_calls", [])
            ]
            return ModelTurn(content=message.get("content"), tool_calls=calls)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
            raise ModelClientError("Model API response did not match the expected chat format.") from None

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        serialized = message.model_dump(mode="json", exclude_none=True)
        if message.tool_calls:
            serialized["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return serialized

