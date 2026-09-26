import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scheduler.evaluation import EvaluationHarness, Scenario, load_scenarios
from scheduler.improvement import ImprovementLoop, ScriptedImprovementProposer
from scheduler.improvement import (
    FailureAnalysis,
    ImprovementProposal,
    ModelImprovementProposer,
)
from scheduler.llm import ModelToolCall, ModelTurn, ScriptedModelClient
from scheduler.policy import activate_policy, load_active_policy, load_policy, save_policy
from scheduler.trace_store import TraceStore
from scheduler.state import ConversationRun
from scheduler.tracing import TraceEventType


ROOT = Path(__file__).parents[1]


class EvaluationAndImprovementTests(unittest.TestCase):
    def test_proposal_validation_accepts_a_paraphrased_policy_rule(self) -> None:
        scenarios = load_scenarios(ROOT / "scenarios" / "scheduling.json")
        target = next(scenario for scenario in scenarios if scenario.id == "ambiguous_that_one")
        analysis = FailureAnalysis(
            scenario_id=target.id,
            failure_category="ambiguous_reference",
            observed_behavior="The agent selected a slot without clarifying.",
            expected_behavior="Ask which slot the patient means.",
            root_cause="The baseline policy does not cover unclear references.",
            evidence=["The trace contains a booking tool call."],
        )
        proposal = ImprovementProposal(
            failure_category=analysis.failure_category,
            observed_behavior=analysis.observed_behavior,
            expected_behavior=analysis.expected_behavior,
            root_cause=analysis.root_cause,
            policy_rule=(
                "If the patient refers to one of several appointment options without specifying which, "
                "ask them to clarify before creating a booking."
            ),
            regression_assertion={
                "scenario_id": target.id,
                "required_response_terms": target.required_response_terms,
                "forbidden_tool_names": target.forbidden_tool_names,
            },
        )

        ImprovementLoop._validate_proposal(proposal, analysis, target)

    def test_model_proposer_must_return_a_structured_proposal_tool_call(self) -> None:
        analysis = FailureAnalysis(
            scenario_id="ambiguous_that_one",
            failure_category="ambiguous_reference",
            observed_behavior="The agent booked the first of two offered slots.",
            expected_behavior="Ask which slot the patient means.",
            root_cause="The policy does not address ambiguous references.",
            evidence=["book_appointment(slot_id=ambiguous-1)"],
        )
        client = ScriptedModelClient(
            [
                ModelTurn(
                    tool_calls=[
                        ModelToolCall(
                            id="proposal-1",
                            name="submit_improvement_proposal",
                            arguments={
                                "failure_category": "ambiguous_reference",
                                "observed_behavior": analysis.observed_behavior,
                                "expected_behavior": analysis.expected_behavior,
                                "root_cause": analysis.root_cause,
                                "policy_rule": "When a choice is ambiguous, ask which slot and do not guess.",
                                "regression_assertion": {
                                    "scenario_id": "ambiguous_that_one",
                                    "required_response_terms": ["which appointment"],
                                    "forbidden_tool_names": ["book_appointment"],
                                },
                            },
                        )
                    ]
                )
            ]
        )
        policy = load_policy(ROOT / "policies" / "v1.json")

        proposal = ModelImprovementProposer(client, "fake-model").propose(analysis, policy)

        self.assertEqual(proposal.regression_assertion.scenario_id, analysis.scenario_id)
        self.assertIn("ask which slot", proposal.policy_rule)

    def test_scripted_suite_demonstrates_target_improvement_without_regressions(self) -> None:
        scenarios = load_scenarios(ROOT / "scenarios" / "scheduling.json")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_policy(load_policy(ROOT / "policies" / "v1.json"), output / "v1.json")
            harness = EvaluationHarness(scenarios, TraceStore(output / "traces"))
            loop = ImprovementLoop(
                harness=harness,
                baseline_policy=load_policy(ROOT / "policies" / "v1.json"),
                candidate_policy_path=output / "v2.json",
                proposer=ScriptedImprovementProposer(),
                proposal_source="scripted fake model",
            )

            report = loop.run("ambiguous_that_one", output / "report.json")

            self.assertTrue(report["accepted"])
            self.assertEqual(report["baseline"]["score"], 81.8)
            self.assertEqual(report["candidate"]["score"], 90.9)
            self.assertEqual(report["protected_regressions"], [])
            self.assertEqual(report["scenario_suite"], [scenario.id for scenario in scenarios])
            self.assertTrue((output / "v2.json").exists())
            self.assertEqual(load_active_policy(output).version, "v2")
            proposal_record_path = Path(report["proposal_record_path"])
            proposal_record = json.loads(proposal_record_path.read_text(encoding="utf-8"))
            self.assertEqual(proposal_record["status"], "accepted")
            self.assertEqual(proposal_record["decision_reason"], report["decision_reason"])
            self.assertEqual(proposal_record["candidate_score"], 90.9)
            saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["proposal_source"], "scripted fake model")

    def test_booking_rubric_distinguishes_attempt_from_successful_side_effect(self) -> None:
        scenarios = load_scenarios(ROOT / "scenarios" / "scheduling.json")
        happy_path = next(scenario for scenario in scenarios if scenario.id == "happy_path")
        with TemporaryDirectory() as directory:
            result = EvaluationHarness(
                [happy_path], TraceStore(Path(directory) / "traces")
            ).run(load_policy(ROOT / "policies" / "v1.json")).results[0]
            trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))

        self.assertTrue(result.passed)
        self.assertTrue(result.checks["booking_after_confirmation"])
        self.assertTrue(result.checks["booking_attempt_after_selection"])
        patient_turn = 0
        booking_events = []
        for event in trace["events"]:
            if event["event_type"] == "patient_message":
                patient_turn += 1
            if event["event_type"] == "tool_execution" and event["payload"]["tool_name"] == "book_appointment":
                booking_events.append((patient_turn, event["payload"]["result"]))
        self.assertEqual(booking_events[0][0], 2)
        self.assertEqual(booking_events[0][1]["error_code"], "confirmation_required")
        self.assertFalse(booking_events[0][1]["succeeded"])
        self.assertEqual(booking_events[1][0], 3)
        self.assertTrue(booking_events[1][1]["succeeded"])
        self.assertEqual(
            [event["payload"].get("origin") for event in trace["events"]
             if event["event_type"] == "tool_execution"
             and event["payload"]["tool_name"] == "book_appointment"],
            ["model", "dispatcher"],
        )

    def test_booking_rubric_rejects_early_attempt_and_unconfirmed_write(self) -> None:
        scenario = Scenario(
            id="unsafe-booking", description="An unsafe booking sequence.",
            patient_id="patient-1", patient_messages=["Show slots", "I choose one", "confirm"],
            book_attempt_after_patient_message=2,
            book_after_patient_message=3,
        )
        run = ConversationRun("patient-1", "v1")
        run.record_patient_message("Show slots")
        run.trace.append(TraceEventType.TOOL_CALL, {
            "name": "book_appointment", "arguments": {"slot_id": "slot-1"},
        })
        run.trace.append(TraceEventType.TOOL_EXECUTION, {
            "tool_name": "book_appointment", "origin": "dispatcher",
            "arguments": {"slot_id": "slot-1"},
            "result": {"succeeded": True},
        })

        result = EvaluationHarness._score_scenario(scenario, run, [], Path("unused"))

        self.assertFalse(result.checks["booking_attempt_after_selection"])
        self.assertFalse(result.checks["booking_after_confirmation"])

    def test_offline_demo_preserves_a_newer_active_policy(self) -> None:
        scenarios = load_scenarios(ROOT / "scenarios" / "scheduling.json")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_policy(load_policy(ROOT / "policies" / "v1.json"), output / "v1.json")
            activate_policy(load_policy(ROOT / "policies" / "v3.json"), output)
            loop = ImprovementLoop(
                harness=EvaluationHarness(scenarios, TraceStore(output / "traces")),
                baseline_policy=load_policy(output / "v1.json"),
                candidate_policy_path=output / "v2.json",
                proposer=ScriptedImprovementProposer(),
                proposal_source="scripted fake model",
            )

            report = loop.run("ambiguous_that_one", output / "report.json")

            self.assertTrue(report["accepted"])
            self.assertFalse(report["activated"])
            self.assertEqual(load_active_policy(output).version, "v3")
            baseline_target = next(
                result
                for result in report["baseline"]["results"]
                if result["scenario_id"] == "ambiguous_that_one"
            )
            target_trace = json.loads(Path(baseline_target["trace_path"]).read_text(encoding="utf-8"))
            call_event = next(
                event
                for event in target_trace["events"]
                if event["event_type"] == "tool_call"
                and event["payload"]["name"] == "book_appointment"
            )
            self.assertEqual(call_event["payload"]["arguments"]["slot_id"], "ambiguous-1")
            self.assertIn("final_appointment_state", target_trace)

    def test_candidate_is_rejected_if_it_regresses_a_previously_passing_case(self) -> None:
        scenarios = load_scenarios(ROOT / "scenarios" / "scheduling.json")
        happy_path = next(scenario for scenario in scenarios if scenario.id == "happy_path")
        happy_path.model_turns["v2"] = [
            happy_path.model_turns["default"][0],
            happy_path.model_turns["default"][1],
            # No valid booking request or confirmation in the candidate behavior.
        ]

        with TemporaryDirectory() as directory:
            output = Path(directory)
            harness = EvaluationHarness(scenarios, TraceStore(output / "traces"))
            loop = ImprovementLoop(
                harness=harness,
                baseline_policy=load_policy(ROOT / "policies" / "v1.json"),
                candidate_policy_path=output / "v2.json",
                proposer=ScriptedImprovementProposer(),
                proposal_source="scripted fake model",
            )

            report = loop.run("ambiguous_that_one", output / "report.json")

            self.assertFalse(report["accepted"])
            self.assertIn("happy_path", report["protected_regressions"])
            self.assertFalse((output / "v2.json").exists())
            proposal_record = json.loads(
                Path(report["proposal_record_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(proposal_record["status"], "rejected")
            self.assertEqual(
                proposal_record["protected_regressions"], ["happy_path"]
            )


if __name__ == "__main__":
    unittest.main()
