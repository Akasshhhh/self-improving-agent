import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from scheduler.admin import accept_proposal, main as admin_main, reject_proposal
from scheduler.live_proposals import (
    GeneratedToolProposal,
    ModelTraceReviewer,
    ProposalStore,
    TraceReview,
    deterministic_trace_review,
)
from scheduler.llm import ModelToolCall, ModelTurn, ScriptedModelClient


ROOT = Path(__file__).parents[1]


class LiveProposalTests(unittest.TestCase):
    def test_admin_cli_requires_the_configured_password(self) -> None:
        with TemporaryDirectory() as directory:
            stderr = StringIO()
            with patch.dict(os.environ, {"SCHEDULER_ADMIN_PASSWORD": "correct"}), \
                    patch("scheduler.admin.getpass", return_value="wrong"), \
                    patch("sys.argv", ["scheduler-admin", "--proposal-dir", directory, "list"]):
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    admin_main()

        self.assertEqual(error.exception.code, 2)
        self.assertIn("authentication failed", stderr.getvalue())

    def test_admin_cli_accepts_configured_password(self) -> None:
        with TemporaryDirectory() as directory:
            stdout = StringIO()
            with patch.dict(os.environ, {"SCHEDULER_ADMIN_PASSWORD": "correct"}), \
                    patch("scheduler.admin.getpass", return_value="correct"), \
                    patch("sys.argv", ["scheduler-admin", "--proposal-dir", directory, "list"]):
                with redirect_stdout(stdout):
                    admin_main()

        self.assertEqual(stdout.getvalue(), "")

    def test_reschedule_booking_mismatch_is_detected_from_trace(self) -> None:
        trace = {
            "messages": [
                {"role": "patient", "content": "Please move my existing appointment."},
                {"role": "assistant", "content": "I found a new time."},
            ],
            "events": [
                {"event_type": "tool_call", "payload": {"name": "book_appointment", "arguments": {"slot_id": "new"}}}
            ],
        }

        review = deterministic_trace_review(trace)

        self.assertIsNotNone(review)
        self.assertEqual(review.category, "unsupported_reschedule")
        self.assertEqual(review.evidence_message_indexes, [0])
        self.assertEqual(review.evidence_event_indexes, [0])

    def test_standalone_reschedule_request_is_detected(self) -> None:
        trace = {
            "messages": [{"role": "patient", "content": "reschedule"}],
            "events": [{"event_type": "tool_call", "payload": {"name": "book_appointment"}}],
        }

        review = deterministic_trace_review(trace)

        self.assertIsNotNone(review)
        self.assertEqual(review.category, "unsupported_reschedule")

    def test_trace_judge_must_return_one_structured_tool_call(self) -> None:
        client = ScriptedModelClient([
            ModelTurn(tool_calls=[ModelToolCall(
                id="review-1",
                name="submit_trace_review",
                arguments={
                    "actionable_failure": False,
                    "category": "none",
                    "confidence": 0.9,
                    "patient_goal": "Find available times.",
                    "observed_behavior": "The agent showed available times.",
                    "expected_behavior": "Show relevant availability.",
                    "evidence_message_indexes": [],
                    "evidence_event_indexes": [],
                },
            )])
        ])

        review = ModelTraceReviewer(client, "fake").review({"messages": [], "events": []})

        self.assertFalse(review.actionable_failure)
        system_message = client.requests[0]["messages"][0]["content"]
        self.assertIn("untrusted data", system_message)

    def test_trace_judge_bounds_excess_evidence_indexes(self) -> None:
        client = ScriptedModelClient([ModelTurn(tool_calls=[ModelToolCall(
            id="review-many",
            name="submit_trace_review",
            arguments={
                "actionable_failure": True,
                "category": "confirmation_missing",
                "confidence": 0.9,
                "patient_goal": "Book an appointment.",
                "observed_behavior": "The assistant booked without confirmation.",
                "expected_behavior": "Ask for confirmation first.",
                "evidence_message_indexes": list(range(11)),
                "evidence_event_indexes": list(range(12)),
            },
        )])])

        review = ModelTraceReviewer(client, "fake").review({"messages": [], "events": []})

        self.assertEqual(review.evidence_message_indexes, list(range(10)))
        self.assertEqual(review.evidence_event_indexes, list(range(10)))

    def test_proposal_schema_rejects_extra_executable_content(self) -> None:
        with self.assertRaises(ValidationError):
            GeneratedToolProposal.model_validate({
                "rationale": "A reschedule operation is missing and needs careful implementation.",
                "acceptance_criteria": ["Preserve the original appointment on failure."],
                "tool_contract": {
                    "name": "reschedule_appointment",
                    "purpose": "Move an existing appointment to an available slot.",
                    "parameters": [
                        {"name": "appointment_id", "value_type": "string", "required": True, "description": "Existing appointment."},
                        {"name": "new_slot_id", "value_type": "string", "required": True, "description": "Replacement slot."},
                    ],
                    "result_description": "Updated appointment.",
                    "side_effects": "Updates appointment slot.",
                    "safeguards": ["Require same patient."],
                    "python_code": "exec('danger')",
                },
            })

    def test_admin_rejects_schema_less_legacy_proposals(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProposalStore(root / "proposals")
            store.directory.mkdir(parents=True)
            proposal_path = store.directory / "legacy-proposal.json"
            proposal_path.write_text(json.dumps({
                "proposal_id": proposal_path.stem,
                "status": "pending_review",
                "proposal": {"proposal_type": "policy"},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only schema version 2"):
                accept_proposal(
                    proposal_id=proposal_path.stem,
                    store=store,
                    policy_directory=root / "policies",
                    trace_directory=root / "gate-traces",
                )

    def test_admin_can_reject_a_pending_proposal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProposalStore(root / "proposals")
            store.directory.mkdir(parents=True)
            proposal_path = store.directory / "pending-proposal.json"
            proposal_path.write_text(json.dumps({
                "schema_version": 2,
                "proposal_id": proposal_path.stem,
                "status": "pending_review",
            }), encoding="utf-8")

            result = reject_proposal(proposal_path.stem, store, "Out of scope for this take-home.")

            self.assertEqual(result["status"], "rejected")
            self.assertEqual(
                store.load(proposal_path.stem)[1]["admin_decision"]["reason"],
                "Out of scope for this take-home.",
            )


if __name__ == "__main__":
    unittest.main()
