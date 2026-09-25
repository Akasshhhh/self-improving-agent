"""Deterministic scenario runner and rubric for agent outcomes and traces."""

import json
import argparse
import os
from collections.abc import Callable
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentTurnLimitError, SchedulingAgent
from .llm import ModelClientError, ModelTurn, ScriptedModelClient
from .models import AppointmentStatus, Slot
from .policy import AgentPolicy
from .policy import load_policy
from .repository import SchedulingRepository
from .state import ConversationRun, MessageRole
from .tools import SchedulingTools
from .trace_store import TraceStore
from .tracing import TraceEventType


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str
    patient_id: str
    slots: list[Slot] = Field(default_factory=list)
    patient_messages: list[str] = Field(min_length=1)
    model_turns: dict[str, list[ModelTurn]]
    seed_appointment_slot_id: str | None = None
    seed_appointment_patient_id: str | None = None
    external_booking_before_message: int | None = None
    expected_active_slot_ids: list[str] = Field(default_factory=list)
    expected_cancelled_count: int = 0
    required_tool_names: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    expected_tool_error_codes: list[str] = Field(default_factory=list)
    required_response_terms: list[str] = Field(default_factory=list)

    def turns_for(self, policy: AgentPolicy) -> list[ModelTurn]:
        if policy.version == "v2" and self.id == "ambiguous_that_one":
            prompt = policy.system_prompt.lower()
            required_markers = ("ambiguous", "ask which", "do not guess")
            if not all(marker in prompt for marker in required_markers):
                return self.model_turns.get("default", [])
        return self.model_turns.get(policy.version, self.model_turns.get("default", []))


class ScenarioResult(BaseModel):
    scenario_id: str
    passed: bool
    checks: dict[str, bool]
    failures: list[str]
    trace_path: str


class EvaluationReport(BaseModel):
    policy_version: str
    total_scenarios: int
    passed_scenarios: int
    score: float
    results: list[ScenarioResult]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_scenarios(path: str | Path) -> list[Scenario]:
    with Path(path).open(encoding="utf-8") as scenario_file:
        raw = json.load(scenario_file)
    scenarios = [Scenario.model_validate(item) for item in raw]
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenario IDs must be unique.")
    return scenarios


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the scheduler evaluation and improvement loop.")
    parser.add_argument("--scenarios", default="scenarios/scheduling.json")
    parser.add_argument("--trace-dir", default="artifacts/evaluation/traces")
    parser.add_argument("--report", default="artifacts/evaluation/improvement-report.json")
    parser.add_argument("--policy-dir", default="policies")
    parser.add_argument("--live-proposer", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    from .improvement import ImprovementLoop, ModelImprovementProposer, ScriptedImprovementProposer
    from .llm import OpenAIModelClient

    policies = Path(args.policy_dir)
    trace_store = TraceStore(args.trace_dir)
    suite = load_scenarios(args.scenarios)
    harness = EvaluationHarness(suite, trace_store)
    if args.live_proposer:
        model_name = args.model or os.environ.get("OPENAI_MODEL")
        if not model_name:
            parser.error("provide --model or set OPENAI_MODEL when using --live-proposer")
        proposer = ModelImprovementProposer(OpenAIModelClient(), model_name)
        source = "live model"
    else:
        proposer = ScriptedImprovementProposer()
        source = "scripted fake model"
    loop = ImprovementLoop(
        harness=harness,
        baseline_policy=load_policy(policies / "v1.json"),
        candidate_policy_path=policies / "v2.json",
        proposer=proposer,
        proposal_source=source,
    )
    report = loop.run(target_scenario_id="ambiguous_that_one", report_path=args.report)
    print(
        f"Policy {report['baseline']['policy_version']}: "
        f"{report['baseline']['passed_scenarios']}/{report['baseline']['total_scenarios']} "
        f"({report['baseline']['score']}%)"
    )
    print(
        f"Policy {report['candidate']['policy_version']}: "
        f"{report['candidate']['passed_scenarios']}/{report['candidate']['total_scenarios']} "
        f"({report['candidate']['score']}%)"
    )
    print(f"Improvement accepted: {report['accepted']}; report: {args.report}")


class EvaluationHarness:
    """Runs the same frozen fixture suite against a selected policy version."""

    def __init__(
        self,
        scenarios: list[Scenario],
        trace_store: TraceStore,
        *,
        model_name: str = "scripted-model",
    ) -> None:
        self._scenarios = tuple(scenarios)
        self._trace_store = trace_store
        self._model_name = model_name

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        return self._scenarios

    def run(self, policy: AgentPolicy) -> EvaluationReport:
        results = [self._run_scenario(scenario, policy) for scenario in self._scenarios]
        passed = sum(result.passed for result in results)
        return EvaluationReport(
            policy_version=policy.version,
            total_scenarios=len(results),
            passed_scenarios=passed,
            score=round(passed / len(results) * 100, 1) if results else 0.0,
            results=results,
        )

    def _run_scenario(self, scenario: Scenario, policy: AgentPolicy) -> ScenarioResult:
        with TemporaryDirectory(prefix=f"scheduler-{scenario.id}-") as directory:
            identifiers = count(1)

            def next_appointment_id() -> str:
                return f"appointment-{next(identifiers):04d}"

            repository = SchedulingRepository(
                Path(directory) / "scenario.db",
                appointment_id_factory=next_appointment_id,
            )
            try:
                for slot in scenario.slots:
                    repository.create_slot(slot)
                if scenario.seed_appointment_slot_id:
                    repository.book_slot(
                        scenario.seed_appointment_patient_id or scenario.patient_id,
                        scenario.seed_appointment_slot_id,
                    )

                client = ScriptedModelClient(scenario.turns_for(policy))
                run = ConversationRun(scenario.patient_id, policy.version)
                agent = SchedulingAgent(
                    model_client=client,
                    tools=SchedulingTools(repository, patient_id=scenario.patient_id),
                    policy=policy,
                    model_name=self._model_name,
                )
                for index, message in enumerate(scenario.patient_messages, start=1):
                    if scenario.external_booking_before_message == index:
                        available_slots = repository.list_available_slots(
                            scenario.slots[0].specialty if scenario.slots else ""
                        )
                        slot_id = next(
                            (slot.id for slot in available_slots),
                            scenario.slots[0].id if scenario.slots else "",
                        )
                        if slot_id:
                            repository.book_slot("external-patient", slot_id)
                    try:
                        agent.respond(run, message)
                    except (AgentTurnLimitError, ModelClientError) as error:
                        run.record_agent_error(str(error))

                appointments = repository.list_patient_appointments(scenario.patient_id)
                trace_document = run.trace_document(
                    [appointment.model_dump(mode="json") for appointment in appointments]
                )
                trace_path = self._trace_store.save(trace_document)
                return self._score_scenario(scenario, run, appointments, trace_path)
            finally:
                repository.close()

    @staticmethod
    def _score_scenario(
        scenario: Scenario,
        run: ConversationRun,
        appointments: list[Any],
        trace_path: Path,
    ) -> ScenarioResult:
        events = run.trace.events()
        calls = [event.payload.get("name", "") for event in events if event.event_type is TraceEventType.TOOL_CALL]
        error_codes = [
            event.payload.get("result", {}).get("error_code")
            for event in events
            if event.event_type is TraceEventType.TOOL_EXECUTION
        ]
        active_slots = sorted(
            appointment.slot_id
            for appointment in appointments
            if appointment.status is AppointmentStatus.BOOKED
        )
        cancelled_count = sum(
            appointment.status is AppointmentStatus.CANCELLED for appointment in appointments
        )
        assistant_text = " ".join(
            message.content.lower()
            for message in run.state.messages
            if message.role is MessageRole.ASSISTANT
        )
        checks = {
            "active_appointments": active_slots == sorted(scenario.expected_active_slot_ids),
            "cancelled_appointments": cancelled_count == scenario.expected_cancelled_count,
            "required_tools": set(scenario.required_tool_names).issubset(calls),
            "forbidden_tools": not set(scenario.forbidden_tool_names).intersection(calls),
            "expected_tool_errors": set(scenario.expected_tool_error_codes).issubset(error_codes),
            "required_response_terms": all(term.lower() in assistant_text for term in scenario.required_response_terms),
        }
        failures = [name for name, passed in checks.items() if not passed]
        return ScenarioResult(
            scenario_id=scenario.id,
            passed=not failures,
            checks=checks,
            failures=failures,
            trace_path=str(trace_path),
        )


if __name__ == "__main__":
    main()
