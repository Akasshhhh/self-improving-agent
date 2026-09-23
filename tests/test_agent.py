from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scheduler.agent import AgentTurnLimitError, SchedulingAgent
from scheduler.llm import ModelToolCall, ModelTurn, ScriptedModelClient
from scheduler.models import Slot
from scheduler.policy import load_policy
from scheduler.repository import SchedulingRepository
from scheduler.state import ConversationRun
from scheduler.tools import SchedulingTools
from scheduler.tracing import TraceEventType


class SchedulingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.repository = SchedulingRepository(
            Path(self.temporary_directory.name) / "scheduling.db"
        )
        self.repository.create_slot(
            Slot(
                id="cardiology-1",
                specialty="cardiology",
                clinician_name="Dr. Rivera",
                starts_at=datetime(2026, 9, 28, 10, tzinfo=timezone.utc),
            )
        )
        self.run = ConversationRun(patient_id="patient-123", policy_version="v1")

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary_directory.cleanup()

    def test_agent_executes_tool_then_returns_follow_up_model_response(self) -> None:
        client = ScriptedModelClient(
            [
                ModelTurn(
                    tool_calls=[
                        ModelToolCall(
                            id="call-search",
                            name="search_available_slots",
                            arguments={"specialty": "cardiology"},
                        )
                    ]
                ),
                ModelTurn(content="I found Monday at 10am. Would you like that slot?"),
            ]
        )
        agent = SchedulingAgent(
            model_client=client,
            tools=SchedulingTools(self.repository, patient_id="patient-123"),
            policy=load_policy(Path(__file__).parents[1] / "policies" / "v1.json"),
            model_name="fake-model",
        )

        answer = agent.respond(self.run, "I need a cardiology appointment.")

        self.assertIn("Monday", answer)
        self.assertEqual(len(client.requests), 2)
        self.assertNotIn("patient_id", str(client.requests))
        tool_call_event = next(
            event
            for event in self.run.trace.events()
            if event.event_type is TraceEventType.TOOL_CALL
        )
        self.assertEqual(tool_call_event.payload["arguments"]["specialty"], "cardiology")
        second_request_roles = [message["role"] for message in client.requests[1]["messages"]]
        self.assertEqual(second_request_roles, ["system", "user", "assistant", "tool"])

    def test_agent_stops_after_configured_model_turn_limit(self) -> None:
        client = ScriptedModelClient(
            [
                ModelTurn(
                    tool_calls=[
                        ModelToolCall(
                            id=f"call-{number}",
                            name="search_available_slots",
                            arguments={"specialty": "cardiology"},
                        )
                    ]
                )
                for number in range(2)
            ]
        )
        agent = SchedulingAgent(
            model_client=client,
            tools=SchedulingTools(self.repository, patient_id="patient-123"),
            policy=load_policy(Path(__file__).parents[1] / "policies" / "v1.json"),
            model_name="fake-model",
            max_model_turns=1,
        )

        with self.assertRaises(AgentTurnLimitError):
            agent.respond(self.run, "Find a cardiology appointment.")


if __name__ == "__main__":
    unittest.main()
