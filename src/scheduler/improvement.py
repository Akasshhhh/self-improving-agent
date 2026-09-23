"""Structured failure analysis and deterministic policy promotion gate."""

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .evaluation import EvaluationHarness, EvaluationReport, Scenario, ScenarioResult
from .llm import ChatMessage, ModelClient
from .policy import AgentPolicy, save_policy


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    failure_category: str
    observed_behavior: str
    expected_behavior: str
    root_cause: str
    evidence: list[str]


class RegressionAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    required_response_terms: list[str] = Field(min_length=1)
    forbidden_tool_names: list[str] = Field(default_factory=list)


class ImprovementProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_category: str
    observed_behavior: str
    expected_behavior: str
    root_cause: str
    policy_rule: str = Field(min_length=10, max_length=500)
    regression_assertion: RegressionAssertion


class ImprovementProposer(Protocol):
    def propose(self, analysis: FailureAnalysis, policy: AgentPolicy) -> ImprovementProposal:
        """Suggest one constrained policy rule based on observed failure evidence."""


class ScriptedImprovementProposer:
    """Offline deterministic stand-in for the proposal model."""

    def propose(self, analysis: FailureAnalysis, policy: AgentPolicy) -> ImprovementProposal:
        del policy
        return ImprovementProposal(
            failure_category=analysis.failure_category,
            observed_behavior=analysis.observed_behavior,
            expected_behavior=analysis.expected_behavior,
            root_cause=analysis.root_cause,
            policy_rule=(
                "When multiple slots have been offered and the patient's reference is ambiguous, "
                "ask which slot they mean. Do not guess or book until they select a specific slot."
            ),
            regression_assertion=RegressionAssertion(
                scenario_id=analysis.scenario_id,
                required_response_terms=["which appointment"],
                forbidden_tool_names=["book_appointment"],
            ),
        )


class ModelImprovementProposer:
    """Use the configured model client while constraining output to one policy rule."""

    def __init__(self, model_client: ModelClient, model_name: str) -> None:
        self._model_client = model_client
        self._model_name = model_name

    def propose(self, analysis: FailureAnalysis, policy: AgentPolicy) -> ImprovementProposal:
        schema = ImprovementProposal.model_json_schema()
        turn = self._model_client.complete(
            model=self._model_name,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Analyze the observed agent failure and propose one concise policy rule. "
                        "Do not propose code changes or changes to scheduling data. "
                        "For an ambiguous slot reference, the rule must tell the agent to ask which slot and not guess. "
                        "Return the proposal only through submit_improvement_proposal."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {"failure": analysis.model_dump(mode="json"), "current_policy": policy.model_dump(mode="json")}
                    ),
                ),
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_improvement_proposal",
                        "description": "Submit one schema-validated improvement proposal.",
                        "parameters": schema,
                    },
                }
            ],
        )
        matching = [call for call in turn.tool_calls if call.name == "submit_improvement_proposal"]
        if len(matching) != 1:
            raise ValueError("Improvement model must return exactly one proposal tool call.")
        return ImprovementProposal.model_validate(matching[0].arguments)


class ImprovementLoop:
    def __init__(
        self,
        *,
        harness: EvaluationHarness,
        baseline_policy: AgentPolicy,
        candidate_policy_path: str | Path,
        proposer: ImprovementProposer,
        proposal_source: str,
    ) -> None:
        self._harness = harness
        self._baseline_policy = baseline_policy
        self._candidate_policy_path = Path(candidate_policy_path)
        self._proposer = proposer
        self._proposal_source = proposal_source

    def run(self, target_scenario_id: str, report_path: str | Path) -> dict[str, Any]:
        scenarios = self._harness.scenarios
        target = next((item for item in scenarios if item.id == target_scenario_id), None)
        if target is None:
            raise ValueError(f"Target scenario '{target_scenario_id}' is not in the frozen suite.")

        baseline = self._harness.run(self._baseline_policy)
        target_result = self._find_result(baseline, target_scenario_id)
        if target_result.passed:
            raise ValueError("The demo target must fail under the baseline policy.")
        analysis = self._analyze(target, target_result)
        proposal = self._proposer.propose(analysis, self._baseline_policy)
        self._validate_proposal(proposal, analysis, target)
        candidate_policy = self._candidate_policy(proposal)
        candidate = self._harness.run(candidate_policy)

        candidate_target = self._find_result(candidate, target_scenario_id)
        baseline_passed = {result.scenario_id for result in baseline.results if result.passed}
        candidate_passed = {result.scenario_id for result in candidate.results if result.passed}
        regressions = sorted(baseline_passed - candidate_passed)
        accepted = candidate_target.passed and not regressions
        if accepted:
            save_policy(candidate_policy, self._candidate_policy_path)

        report = {
            "proposal_source": self._proposal_source,
            "scenario_suite": [scenario.id for scenario in scenarios],
            "analysis": analysis.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "baseline": baseline.as_dict(),
            "candidate": candidate.as_dict(),
            "target_improved": candidate_target.passed,
            "protected_regressions": regressions,
            "accepted": accepted,
            "candidate_policy_path": str(self._candidate_policy_path) if accepted else None,
        }
        target_path = Path(report_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    @staticmethod
    def _find_result(report: EvaluationReport, scenario_id: str) -> ScenarioResult:
        return next(result for result in report.results if result.scenario_id == scenario_id)

    @staticmethod
    def _analyze(scenario: Scenario, result: ScenarioResult) -> FailureAnalysis:
        if scenario.id == "ambiguous_that_one":
            trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
            booking_calls = [
                event["payload"]
                for event in trace["events"]
                if event["event_type"] == "tool_call"
                and event["payload"].get("name") == "book_appointment"
            ]
            booking_detail = (
                f"book_appointment was called with {booking_calls[0].get('arguments', {})}."
                if booking_calls
                else "No booking tool call was recorded."
            )
            assistant_messages = [
                message["content"]
                for message in trace["messages"]
                if message["role"] == "assistant" and message["content"]
            ]
            return FailureAnalysis(
                scenario_id=scenario.id,
                failure_category="ambiguous_reference",
                observed_behavior=(
                    f"After the patient said 'That one', {booking_detail} "
                    f"Assistant messages: {' | '.join(assistant_messages[-2:])}"
                ),
                expected_behavior="The agent asks the patient to identify one specific offered slot before booking.",
                root_cause="The baseline policy does not state how to resolve an ambiguous reference to multiple slots.",
                evidence=[
                    f"Failed rubric checks: {', '.join(result.failures)}",
                    f"Tool-call evidence: {booking_detail}",
                    f"Trace: {result.trace_path}",
                ],
            )
        return FailureAnalysis(
            scenario_id=scenario.id,
            failure_category="agent_behavior",
            observed_behavior=f"The scenario failed rubric checks: {', '.join(result.failures)}.",
            expected_behavior=scenario.description,
            root_cause="The current policy may not describe the expected behavior precisely enough.",
            evidence=[f"Trace: {result.trace_path}"],
        )

    @staticmethod
    def _validate_proposal(
        proposal: ImprovementProposal,
        analysis: FailureAnalysis,
        target: Scenario,
    ) -> None:
        if proposal.failure_category != analysis.failure_category:
            raise ValueError("Proposal failure category must match the observed failure.")
        rule = proposal.policy_rule.lower()
        if not all(marker in rule for marker in ("ambiguous", "ask which", "do not guess")):
            raise ValueError("Proposal rule must explicitly address ambiguity, clarification, and guessing.")
        if proposal.regression_assertion.scenario_id != analysis.scenario_id:
            raise ValueError("Proposal regression assertion must reference the failed scenario.")
        if not set(target.required_response_terms).issubset(
            proposal.regression_assertion.required_response_terms
        ):
            raise ValueError("Proposal must preserve the target scenario's required response checks.")
        if not set(target.forbidden_tool_names).issubset(
            proposal.regression_assertion.forbidden_tool_names
        ):
            raise ValueError("Proposal must preserve the target scenario's forbidden tool checks.")

    def _candidate_policy(self, proposal: ImprovementProposal) -> AgentPolicy:
        version_number = int(self._baseline_policy.version.lstrip("v")) + 1
        rule = proposal.policy_rule.strip()
        prompt = f"{self._baseline_policy.system_prompt.rstrip()}\n\nAdditional policy rule: {rule}"
        return AgentPolicy(version=f"v{version_number}", system_prompt=prompt)
