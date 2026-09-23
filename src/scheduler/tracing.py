"""Append-only execution traces used for evaluation and debugging."""

from copy import deepcopy
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TraceEventType(StrEnum):
    PATIENT_MESSAGE = "patient_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_EXECUTION = "tool_execution"
    TOOL_CALL = "tool_call"
    STATE_UPDATE = "state_update"
    AGENT_ERROR = "agent_error"


class TraceEvent(BaseModel):
    """One externally meaningful event in an agent run."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1)
    event_type: TraceEventType
    occurred_at: datetime
    payload: dict[str, Any]


class RunTrace:
    """Records ordered events for one run without exposing mutable storage."""

    def __init__(self, policy_version: str, agent_version: str = "0.1.0") -> None:
        self.run_id = str(uuid4())
        self.policy_version = policy_version
        self.agent_version = agent_version
        self._events: list[TraceEvent] = []

    def append(self, event_type: TraceEventType, payload: dict[str, Any]) -> TraceEvent:
        event = TraceEvent(
            sequence=len(self._events) + 1,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            payload=deepcopy(payload),
        )
        self._events.append(event)
        return event

    def events(self) -> tuple[TraceEvent, ...]:
        """Return an ordered, read-only view of the trace."""
        return tuple(event.model_copy(deep=True) for event in self._events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "policy_version": self.policy_version,
            "agent_version": self.agent_version,
            "events": [event.model_dump(mode="json") for event in self._events],
        }
