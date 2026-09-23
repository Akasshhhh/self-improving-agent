import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scheduler.evaluation import EvaluationHarness, load_scenarios
from scheduler.improvement import ImprovementLoop, ScriptedImprovementProposer
from scheduler.improvement import FailureAnalysis, ModelImprovementProposer
from scheduler.llm import ModelToolCall, ModelTurn, ScriptedModelClient
from scheduler.policy import load_policy
from scheduler.trace_store import TraceStore


ROOT = Path(__file__).parents[1]


class EvaluationAndImprovementTests(unittest.TestCase):
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
            self.assertEqual(report["baseline"]["score"], 90.0)
            self.assertEqual(report["candidate"]["score"], 100.0)
            self.assertEqual(report["protected_regressions"], [])
            self.assertEqual(report["scenario_suite"], [scenario.id for scenario in scenarios])
            self.assertTrue((output / "v2.json").exists())
            saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["proposal_source"], "scripted fake model")
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


if __name__ == "__main__":
    unittest.main()
