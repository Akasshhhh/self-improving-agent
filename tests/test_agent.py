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

    def test_booking_requires_later_confirmation_of_pending_slot(self) -> None:
        client = ScriptedModelClient([
            ModelTurn(tool_calls=[ModelToolCall(
                id="search", name="search_available_slots", arguments={"specialty": "cardiology"}
            )]),
            ModelTurn(content="Dr. Rivera has an available slot."),
            ModelTurn(tool_calls=[ModelToolCall(
                id="stage", name="book_appointment", arguments={"slot_id": "cardiology-1"}
            )]),
        ])
        agent = SchedulingAgent(
            model_client=client,
            tools=SchedulingTools(self.repository, patient_id="patient-123"),
            policy=load_policy(Path(__file__).parents[1] / "policies" / "v1.json"),
            model_name="fake-model",
        )

        agent.respond(self.run, "Show cardiology slots.")
        prompt = agent.respond(self.run, "I choose Dr. Rivera's slot.")

        self.assertIn("Reply 'confirm'", prompt)
        self.assertEqual(self.repository.list_patient_appointments("patient-123"), [])
        self.assertEqual(self.run.state.context.pending_booking_slot_id, "cardiology-1")
        self.assertIn("book_appointment", [
            event.payload["name"] for event in self.run.trace.events()
            if event.event_type is TraceEventType.TOOL_CALL
        ])

        answer = agent.respond(self.run, "confirm")

        self.assertIn("booked", answer)
        self.assertEqual(
            [item.slot_id for item in self.repository.list_patient_appointments("patient-123")],
            ["cardiology-1"],
        )
        self.assertIsNone(self.run.state.context.pending_booking_slot_id)
        booking_results = [
            event.payload["result"] for event in self.run.trace.events()
            if event.event_type is TraceEventType.TOOL_EXECUTION
            and event.payload["tool_name"] == "book_appointment"
        ]
        self.assertEqual(booking_results[0]["error_code"], "confirmation_required")
        self.assertFalse(booking_results[0]["succeeded"])
        self.assertTrue(booking_results[1]["succeeded"])
        self.assertEqual(len(client.requests), 3)

    def test_confirmation_books_only_the_staged_slot_without_another_model_call(self) -> None:
        self.repository.create_slot(Slot(
            id="cardiology-2", specialty="cardiology", clinician_name="Dr. Shah",
            starts_at=datetime(2026, 9, 29, 14, tzinfo=timezone.utc),
        ))
        client = ScriptedModelClient([
            ModelTurn(tool_calls=[ModelToolCall(
                id="search", name="search_available_slots", arguments={"specialty": "cardiology"}
            )]),
            ModelTurn(content="I found two cardiology slots."),
            ModelTurn(tool_calls=[ModelToolCall(
                id="stage", name="book_appointment", arguments={"slot_id": "cardiology-1"}
            )]),
            ModelTurn(tool_calls=[ModelToolCall(
                id="wrong", name="book_appointment", arguments={"slot_id": "cardiology-2"}
            )]),
        ])
        agent = SchedulingAgent(
            model_client=client,
            tools=SchedulingTools(self.repository, patient_id="patient-123"),
            policy=load_policy(Path(__file__).parents[1] / "policies" / "v1.json"),
            model_name="fake-model",
        )

        agent.respond(self.run, "Show cardiology slots.")
        agent.respond(self.run, "I choose Dr. Rivera's slot.")
        requests_before_confirmation = len(client.requests)
        answer = agent.respond(self.run, "confirm")

        self.assertIn("booked", answer)
        self.assertEqual(
            [item.slot_id for item in self.repository.list_patient_appointments("patient-123")],
            ["cardiology-1"],
        )
        self.assertIsNone(self.run.state.context.pending_booking_slot_id)
        self.assertEqual(len(client.requests), requests_before_confirmation)

    def test_patient_can_decline_and_stale_slot_does_not_book(self) -> None:
        client = ScriptedModelClient([
            ModelTurn(tool_calls=[ModelToolCall(
                id="search", name="search_available_slots", arguments={"specialty": "cardiology"}
            )]),
            ModelTurn(content="I found Dr. Rivera's slot."),
            ModelTurn(tool_calls=[ModelToolCall(
                id="stage", name="book_appointment", arguments={"slot_id": "cardiology-1"}
            )]),
            ModelTurn(tool_calls=[ModelToolCall(
                id="retry-stage", name="book_appointment", arguments={"slot_id": "cardiology-1"}
            )]),
        ])
        agent = SchedulingAgent(
            model_client=client,
            tools=SchedulingTools(self.repository, patient_id="patient-123"),
            policy=load_policy(Path(__file__).parents[1] / "policies" / "v1.json"),
            model_name="fake-model",
        )

        agent.respond(self.run, "Show cardiology slots.")
        agent.respond(self.run, "I choose Dr. Rivera's slot.")
        decline = agent.respond(self.run, "cancel")
        self.assertIn("haven't booked", decline)
        self.assertIsNone(self.run.state.context.pending_booking_slot_id)

        agent.respond(self.run, "I choose Dr. Rivera's slot again.")
        self.repository.book_slot("another-patient", "cardiology-1")
        answer = agent.respond(self.run, "confirm")

        self.assertIn("no longer available", answer)
        self.assertEqual(self.repository.list_patient_appointments("patient-123"), [])
        self.assertIsNone(self.run.state.context.pending_booking_slot_id)


if __name__ == "__main__":
    unittest.main()
