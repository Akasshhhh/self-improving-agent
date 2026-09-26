"""Interactive command-line patient scheduling conversation."""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from .agent import AgentTurnLimitError, SchedulingAgent
from .llm import ModelClientError, OpenAIModelClient
from .live_proposals import (
    ModelTraceProposer,
    ModelTraceReviewer,
    ProposalStore,
    review_completed_trace_generated,
)
from .models import Slot
from .policy import load_active_policy, load_policy
from .repository import SchedulingRepository
from .state import ConversationRun
from .tools import SchedulingTools
from .trace_store import TraceStore


def _seed_demo_slots(repository: SchedulingRepository) -> None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    demo_slots = [
        ("demo-slot-1", 2, 10, "cardiology", "Dr. Rivera"),
        ("demo-slot-2", 2, 14, "cardiology", "Dr. Shah"),
        ("demo-slot-3", 4, 11, "cardiology", "Dr. Rivera"),
        ("demo-slot-4", 3, 9, "dermatology", "Dr. Chen"),
        ("demo-slot-5", 6, 15, "dermatology", "Dr. Chen"),
        ("demo-urgent-care-1", 1, 9, "urgent care", "Urgent Care Team"),
        ("demo-urgent-care-2", 2, 13, "urgent care", "Urgent Care Team"),
        ("demo-urgent-care-3", 3, 10, "urgent care", "Urgent Care Team"),
        ("demo-primary-care-1", 1, 11, "primary care", "Dr. Patel"),
        ("demo-primary-care-2", 3, 15, "primary care", "Dr. Patel"),
        ("demo-primary-care-3", 5, 10, "primary care", "Dr. Morgan"),
        ("demo-pediatrics-1", 2, 9, "pediatrics", "Dr. Brooks"),
        ("demo-pediatrics-2", 5, 14, "pediatrics", "Dr. Brooks"),
        ("demo-orthopedics-1", 4, 10, "orthopedics", "Dr. Kim"),
        ("demo-orthopedics-2", 8, 13, "orthopedics", "Dr. Kim"),
        ("demo-gastroenterology-1", 3, 11, "gastroenterology", "Dr. Singh"),
        ("demo-gastroenterology-2", 7, 14, "gastroenterology", "Dr. Singh"),
    ]
    for slot_id, days, hour, specialty, clinician in demo_slots:
        if repository.has_slot(slot_id):
            continue
        starts_at = (now + timedelta(days=days)).replace(hour=hour)
        repository.create_slot(
            Slot(
                id=slot_id,
                specialty=specialty,
                clinician_name=clinician,
                starts_at=starts_at,
            )
        )


def run_cli() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Chat with the clinic scheduling agent.")
    parser.add_argument("--patient-id", default="demo-patient", help="Trusted demo session identity")
    parser.add_argument("--database", default="data/scheduler.db")
    parser.add_argument("--trace-dir", default="artifacts/traces")
    parser.add_argument("--policy", help="Use a specific policy file instead of the active policy.")
    parser.add_argument("--proposal-dir", default="artifacts/proposals")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--workspace", help="Use an isolated demo workspace created by scheduler-demo.")
    parser.add_argument(
        "--timezone",
        default=os.environ.get("SCHEDULER_TIMEZONE"),
        help="IANA timezone for displayed appointment times (defaults to this computer's local timezone)",
    )
    args = parser.parse_args()
    policy_directory = Path("policies")
    if args.workspace:
        workspace = Path(args.workspace)
        policy_directory = workspace / "policies"
        args.database = workspace / "scheduler.db"
        args.trace_dir = workspace / "traces"
        args.proposal_dir = workspace / "proposals"
    if not args.model:
        parser.error("provide --model or set OPENAI_MODEL for your OpenAI-compatible endpoint")
    try:
        display_timezone = (
            ZoneInfo(args.timezone)
            if args.timezone
            else datetime.now().astimezone().tzinfo or timezone.utc
        )
    except ZoneInfoNotFoundError:
        parser.error(f"unknown timezone '{args.timezone}'; use an IANA timezone such as Asia/Kolkata")

    repository = SchedulingRepository(args.database)
    try:
        _seed_demo_slots(repository)
        starting_state = repository.evaluation_snapshot(args.patient_id)
        starting_state["timezone"] = getattr(display_timezone, "key", None) or "UTC"
        model_client = OpenAIModelClient()
        policy = load_policy(args.policy) if args.policy else load_active_policy(policy_directory)
        tools = SchedulingTools(
            repository,
            patient_id=args.patient_id,
            display_timezone=display_timezone,
        )
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
        trace_path = TraceStore(args.trace_dir).save(
            run.trace_document(appointment_state, starting_state)
        )
        print(f"Run trace saved to {trace_path}")
        try:
            review, saved_proposals = review_completed_trace_generated(
                trace_path=trace_path,
                policy=policy,
                reviewer=ModelTraceReviewer(model_client, args.model),
                proposer=ModelTraceProposer(model_client, args.model),
                proposal_store=ProposalStore(args.proposal_dir),
            )
            if not review.actionable_failure:
                print("Trace review: no actionable failure found.")
            elif not saved_proposals:
                print("Trace review: failure was below threshold or has no supported proposal path.")
            for proposal_path, created in saved_proposals:
                record = json.loads(proposal_path.read_text(encoding="utf-8"))
                detail = "created" if created else "matched an existing pending proposal"
                print(
                    f"Trace review: {review.category}; {record['proposal']['proposal_type']} proposal "
                    f"{detail} with status {record['status']}: {proposal_path}"
                )
        except (ModelClientError, ValueError, OSError) as error:
            # Keep the completed conversation and trace even when review is unavailable.
            print(f"Trace review could not complete: {error}")
    finally:
        repository.close()


if __name__ == "__main__":
    run_cli()
