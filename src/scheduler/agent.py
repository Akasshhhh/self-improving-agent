"""Bounded orchestration loop connecting model turns to deterministic tools."""

from typing import Any

from .llm import ChatMessage, ModelClient, ModelClientError, ModelToolCall
from .policy import AgentPolicy
from .state import ConversationMessage, ConversationRun, MessageRole
from .tools import SchedulingTools


class AgentTurnLimitError(RuntimeError):
    """Raised when the model keeps requesting tools without answering."""


class SchedulingAgent:
    def __init__(
        self,
        *,
        model_client: ModelClient,
        tools: SchedulingTools,
        policy: AgentPolicy,
        model_name: str,
        max_model_turns: int = 6,
    ) -> None:
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be at least one")
        self._model_client = model_client
        self._tools = tools
        self._policy = policy
        self._model_name = model_name
        self._max_model_turns = max_model_turns

    def respond(self, run: ConversationRun, patient_message: str) -> str:
        run.record_patient_message(patient_message)
        tool_schemas = SchedulingTools.schemas()
        for _ in range(self._max_model_turns):
            turn = self._model_client.complete(
                messages=self._messages_for_model(run),
                tools=tool_schemas,
                model=self._model_name,
            )
            if turn.tool_calls:
                self._execute_tool_calls(run, turn.content, turn.tool_calls)
                continue
            if turn.content is None:
                raise ModelClientError("Model response contained neither text nor a tool call.")
            run.record_assistant_message(turn.content)
            return turn.content
        raise AgentTurnLimitError(
            f"Agent exceeded its limit of {self._max_model_turns} model turns for one patient message."
        )

    def _execute_tool_calls(
        self,
        run: ConversationRun,
        assistant_text: str | None,
        calls: list[ModelToolCall],
    ) -> None:
        serialized_calls = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        ]
        run.record_assistant_tool_calls(serialized_calls, content=assistant_text or "")
        for call in calls:
            result = self._tools.execute(call.name, call.arguments)
            run.record_tool_result(
                result,
                tool_call_id=call.id,
                arguments=call.arguments,
            )
            if result.succeeded and call.name == "search_available_slots":
                slots = (result.data or {}).get("slots", [])
                run.update_context(offered_slot_ids=[slot["id"] for slot in slots])

    def _messages_for_model(self, run: ConversationRun) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=self._policy.system_prompt)]
        for item in run.state.messages:
            if item.role is MessageRole.PATIENT:
                messages.append(ChatMessage(role="user", content=item.content))
            elif item.tool_calls:
                calls = [
                    ModelToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
                    for call in item.tool_calls
                ]
                messages.append(
                    ChatMessage(role="assistant", content=item.content or None, tool_calls=calls)
                )
            elif item.role is MessageRole.TOOL:
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=item.content,
                        tool_call_id=item.tool_call_id,
                        name=item.tool_name,
                    )
                )
            else:
                messages.append(ChatMessage(role="assistant", content=item.content))
        return messages

