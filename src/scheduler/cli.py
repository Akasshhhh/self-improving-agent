"""Interactive command-line patient scheduling conversation."""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent import AgentTurnLimitError, SchedulingAgent
from .llm import ModelClientError, OpenAIModelClient
from .models import Slot
from .policy import load_policy
from .repository import SchedulingRepository
from .state import ConversationRun
from .tools import SchedulingTools
from .trace_store import TraceStore


def _seed_demo_slots(repository: SchedulingRepository) -> None:
    if repository.has_slots():
        return
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for index, (days, hour, specialty, clinician) in enumerate(
        [
            (2, 10, "cardiology", "Dr. Rivera"),
            (2, 14, "cardiology", "Dr. Shah"),
            (4, 11, "cardiology", "Dr. Rivera"),
            (3, 9, "dermatology", "Dr. Chen"),
            (6, 15, "dermatology", "Dr. Chen"),
        ],
        start=1,
    ):
        starts_at = (now + timedelta(days=days)).replace(hour=hour)
        repository.create_slot(
            Slot(
                id=f"demo-slot-{index}",
                specialty=specialty,
                clinician_name=clinician,
                starts_at=starts_at,
            )
        )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Chat with the clinic scheduling agent.")
    parser.add_argument("--patient-id", default="demo-patient", help="Trusted demo session identity")
    parser.add_argument("--database", default="data/scheduler.db")
    parser.add_argument("--trace-dir", default="artifacts/traces")
    parser.add_argument("--policy", default="policies/v1.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    args = parser.parse_args()
    if not args.model:
        parser.error("provide --model or set OPENAI_MODEL for your OpenAI-compatible endpoint")

    repository = SchedulingRepository(args.database)
    try:
        _seed_demo_slots(repository)
        policy = load_policy(args.policy)
        model_client = OpenAIModelClient()
        tools = SchedulingTools(repository, patient_id=args.patient_id)
        agent = SchedulingAgent(
            model_client=model_client,
            tools=tools,
            policy=policy,
            model_name=args.model,
        )
        run = ConversationRun(patient_id=args.patient_id, policy_version=policy.version)
        print("Clinic scheduling assistant. Type 'exit' to end the conversation.")
        while True:
            try:
                patient_message = input("Patient: ").strip()
            except EOFError:
                break
            if not patient_message:
                continue
            if patient_message.lower() in {"exit", "quit"}:
                break
            try:
                print(f"Assistant: {agent.respond(run, patient_message)}")
            except (AgentTurnLimitError, ModelClientError) as error:
                run.record_agent_error(str(error))
                print("Assistant: I’m having trouble completing that request. Please try again.")

        appointment_state = [
            appointment.model_dump(mode="json")
            for appointment in repository.list_patient_appointments(args.patient_id)
        ]
        trace_path = TraceStore(args.trace_dir).save(run.trace_document(appointment_state))
        print(f"Run trace saved to {trace_path}")
    finally:
        repository.close()


if __name__ == "__main__":
    run_cli()
