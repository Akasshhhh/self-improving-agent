"""Working conversation state and the explicit operations that record it."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .tools import ToolResult
from .tracing import RunTrace, TraceEventType


class MessageRole(StrEnum):
    PATIENT = "patient"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationMessage(BaseModel):
    """A single message supplied to the model on later turns."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = ""
    created_at: datetime
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class SchedulingContext(BaseModel):
    """Structured facts gathered during the current scheduling conversation."""

    patient_id: str = Field(min_length=1)
    specialty: str | None = None
    offered_slot_ids: list[str] = Field(default_factory=list)
    pending_booking_slot_id: str | None = None


class ConversationState(BaseModel):
    """Mutable working context, distinct from durable appointment records."""

    policy_version: str = Field(min_length=1)
    context: SchedulingContext
    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationRun:
    """Keeps state and trace consistent through one explicit recording API."""

    def __init__(self, patient_id: str, policy_version: str) -> None:
        self.state = ConversationState(
            policy_version=policy_version,
            context=SchedulingContext(patient_id=patient_id),
        )
        self.trace = RunTrace(policy_version=policy_version)

    def record_patient_message(self, content: str) -> ConversationMessage:
        return self._record_message(MessageRole.PATIENT, content)

    def record_assistant_message(self, content: str) -> ConversationMessage:
        return self._record_message(MessageRole.ASSISTANT, content)

    def record_assistant_tool_calls(
        self, calls: list[dict[str, Any]], content: str = ""
    ) -> ConversationMessage:
        message = self._record_message(
            MessageRole.ASSISTANT,
            content,
            tool_calls=calls,
        )
        for call in calls:
            self.trace.append(TraceEventType.TOOL_CALL, call)
        return message

    def record_tool_result(
        self,
        result: ToolResult,
        *,
        tool_call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        serialized = result.model_dump_json()
        message = self._record_message(
            MessageRole.TOOL,
            serialized,
            tool_name=result.tool_name,
            tool_call_id=tool_call_id,
            tool_arguments=arguments,
        )
        self.trace.append(
            TraceEventType.TOOL_EXECUTION,
            {
                "tool_name": result.tool_name,
                "tool_call_id": tool_call_id,
                "origin": "model",
                "arguments": arguments or {},
                "result": result.model_dump(mode="json"),
            },
        )
        return message

    def record_dispatcher_tool_result(
        self,
        result: ToolResult,
        *,
        arguments: dict[str, Any],
    ) -> None:
        """Trace an application action without inventing a model tool-call message."""
        self.trace.append(
            TraceEventType.TOOL_EXECUTION,
            {
                "tool_name": result.tool_name,
                "tool_call_id": None,
                "origin": "dispatcher",
                "arguments": arguments,
                "result": result.model_dump(mode="json"),
            },
        )

    def update_context(
        self,
        *,
        specialty: str | None = None,
        offered_slot_ids: list[str] | None = None,
    ) -> None:
        update: dict[str, Any] = {}
        if specialty is not None:
            self.state.context.specialty = specialty
            update["specialty"] = specialty
        if offered_slot_ids is not None:
            self.state.context.offered_slot_ids = offered_slot_ids
            update["offered_slot_ids"] = offered_slot_ids
        if update:
            self.trace.append(TraceEventType.STATE_UPDATE, update)

    def set_pending_booking_slot(self, slot_id: str | None) -> None:
        """Record the only slot a later patient confirmation may authorize."""
        self.state.context.pending_booking_slot_id = slot_id
        self.trace.append(
            TraceEventType.STATE_UPDATE, {"pending_booking_slot_id": slot_id}
        )

    def record_agent_error(self, message: str) -> None:
        self.trace.append(TraceEventType.AGENT_ERROR, {"message": message})

    def trace_document(
        self,
        final_appointment_state: list[dict[str, Any]],
        starting_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document = self.trace.as_dict()
        document["messages"] = [message.model_dump(mode="json") for message in self.state.messages]
        document["final_appointment_state"] = final_appointment_state
        if starting_state is not None:
            document["starting_state"] = starting_state
        return document

    def _record_message(
        self,
        role: MessageRole,
        content: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_arguments: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_arguments=tool_arguments,
            tool_calls=tool_calls or [],
        )
        self.state.messages.append(message)
        event_type = (
            TraceEventType.PATIENT_MESSAGE
            if role is MessageRole.PATIENT
            else TraceEventType.ASSISTANT_MESSAGE
        )
        if role is not MessageRole.TOOL:
            self.trace.append(event_type, {"content": content})
        return message
