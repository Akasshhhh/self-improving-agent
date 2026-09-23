import json
from tempfile import TemporaryDirectory
import unittest

from scheduler.state import ConversationRun, MessageRole
from scheduler.tools import ToolName, ToolResult
from scheduler.tracing import TraceEventType
from scheduler.trace_store import TraceStore


class ConversationRunTests(unittest.TestCase):
    def test_records_messages_and_tool_results_in_order(self) -> None:
        run = ConversationRun(patient_id="patient-123", policy_version="v1")

        run.record_patient_message("I need a cardiology appointment.")
        run.update_context(specialty="cardiology", offered_slot_ids=["cardiology-1"])
        run.record_tool_result(
            ToolResult(
                tool_name=ToolName.SEARCH_AVAILABLE_SLOTS,
                succeeded=True,
                data={"slots": [{"id": "cardiology-1"}]},
            )
        )
        run.record_assistant_message("I have an appointment at 10am.")

        self.assertEqual(
            [event.event_type for event in run.trace.events()],
            [
                TraceEventType.PATIENT_MESSAGE,
                TraceEventType.STATE_UPDATE,
                TraceEventType.TOOL_EXECUTION,
                TraceEventType.ASSISTANT_MESSAGE,
            ],
        )
        self.assertEqual([event.sequence for event in run.trace.events()], [1, 2, 3, 4])
        self.assertEqual(run.state.context.specialty, "cardiology")
        self.assertEqual(run.state.context.offered_slot_ids, ["cardiology-1"])
        self.assertEqual(run.state.messages[1].role, MessageRole.TOOL)

    def test_trace_is_json_serializable_and_attributed_to_a_policy(self) -> None:
        run = ConversationRun(patient_id="patient-123", policy_version="v2")
        run.record_patient_message("I need an appointment.")

        trace = run.trace.as_dict()

        self.assertEqual(trace["policy_version"], "v2")
        self.assertEqual(json.loads(json.dumps(trace))["events"][0]["sequence"], 1)

    def test_trace_keeps_a_snapshot_when_context_input_changes_later(self) -> None:
        run = ConversationRun(patient_id="patient-123", policy_version="v1")
        offered_slots = ["cardiology-1"]

        run.update_context(offered_slot_ids=offered_slots)
        offered_slots.append("cardiology-2")

        self.assertEqual(
            run.trace.events()[0].payload["offered_slot_ids"], ["cardiology-1"]
        )

    def test_trace_store_writes_messages_and_final_appointment_state(self) -> None:
        run = ConversationRun(patient_id="patient-123", policy_version="v1")
        run.record_patient_message("Book cardiology next week.")
        document = run.trace_document([{"id": "appointment-1", "status": "booked"}])

        with TemporaryDirectory() as directory:
            path = TraceStore(directory).save(document)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["agent_version"], "0.1.0")
        self.assertEqual(saved["policy_version"], "v1")
        self.assertEqual(saved["messages"][0]["content"], "Book cardiology next week.")
        self.assertEqual(saved["final_appointment_state"][0]["status"], "booked")


if __name__ == "__main__":
    unittest.main()
