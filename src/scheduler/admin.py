"""Password-gated review CLI for trace-derived improvement proposals."""

import argparse
import json
import os
import secrets
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from .live_proposals import (
    GeneratedPolicyProposal,
    GeneratedToolProposal,
    ProposalStore,
    TraceReview,
    candidate_policy_for_generated_rule,
    revalidate_rejected_generated,
    validate_generated_proposal,
)
from .live_gate import run_live_gate
from .live_scenarios import GeneratedScenario, load_generated_scenarios, load_protected_scenarios, save_generated_scenario
from .llm import ModelClient, ModelClientError, OpenAIModelClient
from .policy import activate_policy, load_active_policy


def accept_proposal(
    *,
    proposal_id: str,
    store: ProposalStore,
    policy_directory: str | Path,
    trace_directory: str | Path,
    generated_scenario_directory: str | Path = "scenarios/generated",
    live_protected_path: str | Path = "scenarios/live_protected.json",
    model_name: str | None = None,
    model_client_factory: Callable[[], ModelClient] = OpenAIModelClient,
) -> dict[str, Any]:
    path, record = store.load(proposal_id)
    if record.get("schema_version") != 2:
        raise ValueError("Unsupported proposal format; only schema version 2 proposals can be accepted.")
    return _accept_generated_proposal(
        path=path,
        record=record,
        store=store,
        policy_directory=policy_directory,
        trace_directory=trace_directory,
        generated_scenario_directory=generated_scenario_directory,
        live_protected_path=live_protected_path,
        model_name=model_name,
        model_client_factory=model_client_factory,
    )


def _accept_generated_proposal(
    *,
    path: Path,
    record: dict[str, Any],
    store: ProposalStore,
    policy_directory: str | Path,
    trace_directory: str | Path,
    generated_scenario_directory: str | Path,
    live_protected_path: str | Path,
    model_name: str | None,
    model_client_factory: Callable[[], ModelClient],
) -> dict[str, Any]:
    if record.get("status") not in {"pending_review", "evaluation_error", "approved_for_test"}:
        raise ValueError("Only pending or retryable generated proposals can be accepted.")
    review = TraceReview.model_validate(record["failure_review"])
    source_trace = json.loads(Path(record["source_trace"]).read_text(encoding="utf-8"))
    proposal = record["proposal"]
    proposal_type = proposal["proposal_type"]
    now = datetime.now(timezone.utc).isoformat()
    if proposal_type == "tool":
        payload = GeneratedToolProposal.model_validate({
            key: value for key, value in proposal.items() if key != "proposal_type"
        })
        validate_generated_proposal("tool", payload, review, source_trace)
        record["status"] = "approved_for_implementation"
        record["admin_decision"] = {
            "action": "accept", "decided_at": now,
            "result": "Developer implementation required; no code or tool registry changed.",
        }
        store.update(path, record)
        return {"status": record["status"], "proposal_id": record["proposal_id"]}
    if proposal_type != "policy":
        raise ValueError("Invalid proposal type cannot be approved.")
    payload = GeneratedPolicyProposal.model_validate({
        key: value for key, value in proposal.items() if key != "proposal_type"
    })
    validate_generated_proposal("policy", payload, review, source_trace)
    baseline_policy = load_active_policy(policy_directory)
    if record.get("source_policy_version") != baseline_policy.version:
        record["status"] = "rejected_by_gate"
        record["admin_decision"] = {
            "action": "accept", "decided_at": now,
            "result": "The source trace used a policy that is no longer active.",
        }
        store.update(path, record)
        return {"status": record["status"], "reason": record["admin_decision"]["result"]}

    version = _next_policy_version(policy_directory)
    candidate = candidate_policy_for_generated_rule(
        baseline_policy, payload.candidate_rule, version
    )
    record["status"] = "approved_for_test"
    record["admin_decision"] = {"action": "accept", "decided_at": now}
    store.update(path, record)
    try:
        if not model_name:
            raise ValueError("Set OPENAI_MODEL or provide --model for real-model evaluation.")
        protected = [
            *load_protected_scenarios(live_protected_path),
            *load_generated_scenarios(generated_scenario_directory),
        ]
        result = run_live_gate(
            target=payload.scenario,
            protected=protected,
            baseline_policy=baseline_policy,
            candidate_policy=candidate,
            trace_directory=trace_directory,
            model_name=model_name,
            model_client_factory=model_client_factory,
            baseline_required_tool=(
                "book_appointment"
                if review.category in {"unsupported_reschedule", "ambiguous_reference"}
                and any(
                    event.get("event_type") == "tool_call"
                    and event.get("payload", {}).get("name") == "book_appointment"
                    for event in source_trace.get("events", [])
                )
                else None
            ),
            target_category=review.category,
        )
        if result["status"] == "promote":
            scenario_path = save_generated_scenario(
                payload.scenario, generated_scenario_directory
            )
            policy_path = activate_policy(candidate, policy_directory)
            record["status"] = "promoted"
            record["activated_policy"] = {
                "version": candidate.version,
                "path": str(policy_path),
                "protected_scenario": str(scenario_path),
            }
        else:
            record["status"] = result["status"]
        record["admin_decision"]["result"] = result["reason"]
        record["admin_decision"]["evaluation"] = result
        store.update(path, record)
        return {
            "status": record["status"],
            "reason": result["reason"],
            "baseline_score": result.get("baseline_score"),
            "candidate_score": result.get("candidate_score"),
            "target_improved": result.get("target_improved"),
            "protected_regressions": result.get("protected_regressions"),
            "active_policy": candidate.version if record["status"] == "promoted" else baseline_policy.version,
        }
    except (ModelClientError, OSError, ValueError) as error:
        record["status"] = "evaluation_error"
        record["admin_decision"]["result"] = str(error)
        store.update(path, record)
        return {"status": "evaluation_error", "reason": str(error)}


def reject_proposal(
    proposal_id: str,
    store: ProposalStore,
    reason: str = "Rejected by admin.",
) -> dict[str, Any]:
    path, record = store.load(proposal_id)
    if record.get("status") not in {"pending_review", "evaluation_error", "approved_for_test"}:
        raise ValueError("Only pending or retryable proposals can be rejected.")
    record["status"] = "rejected"
    record["admin_decision"] = {
        "action": "reject",
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason[:500],
    }
    store.update(path, record)
    return {"status": record["status"], "proposal_id": record["proposal_id"]}


def revise_policy_rule(
    proposal_id: str,
    store: ProposalStore,
    rule: str,
    reason: str,
) -> dict[str, Any]:
    """Keep the model draft and record a human-reviewed rule before the gate."""
    path, record = store.load(proposal_id)
    if record.get("schema_version") != 2 or record.get("status") != "pending_review":
        raise ValueError("Only pending generated policy proposals can be revised.")
    proposal = record.get("proposal", {})
    if proposal.get("proposal_type") != "policy":
        raise ValueError("A tool contract cannot be revised as policy text.")
    if not reason.strip():
        raise ValueError("Explain why the proposed rule needs revision.")
    original = proposal["candidate_rule"]
    revised = GeneratedPolicyProposal.model_validate({
        "rationale": proposal["rationale"],
        "candidate_rule": rule.strip(),
        "scenario": proposal["scenario"],
    })
    trace = json.loads(Path(record["source_trace"]).read_text(encoding="utf-8"))
    review = TraceReview.model_validate(record["failure_review"])
    validate_generated_proposal("policy", revised, review, trace)
    record.setdefault("admin_revisions", []).append({
        "revised_at": datetime.now(timezone.utc).isoformat(),
        "previous_rule": original,
        "new_rule": revised.candidate_rule,
        "reason": reason.strip()[:500],
    })
    proposal["candidate_rule"] = revised.candidate_rule
    store.update(path, record)
    return {"status": record["status"], "proposal_id": proposal_id, "revision_count": len(record["admin_revisions"])}


def revise_policy_scenario_messages(
    proposal_id: str,
    store: ProposalStore,
    messages: list[str],
    reason: str,
) -> dict[str, Any]:
    """Revise replay dialogue after a failed gate without changing its rubric or setup."""
    path, record = store.load(proposal_id)
    if record.get("schema_version") != 2 or record.get("status") not in {
        "pending_review", "rejected_by_gate"
    }:
        raise ValueError("Only pending or gate-rejected generated policy proposals can be revised.")
    proposal = record.get("proposal", {})
    if proposal.get("proposal_type") != "policy":
        raise ValueError("Only policy regression scenarios have patient replay messages.")
    if not reason.strip():
        raise ValueError("Explain why the replay messages need revision.")
    if not isinstance(messages, list) or not all(isinstance(item, str) for item in messages):
        raise ValueError("Messages file must contain a JSON list of patient message strings.")

    previous_scenario = GeneratedScenario.model_validate(proposal["scenario"])
    revised_scenario = GeneratedScenario.model_validate({
        **previous_scenario.model_dump(mode="json"),
        "patient_messages": messages,
    })
    revised = GeneratedPolicyProposal.model_validate({
        "rationale": proposal["rationale"],
        "candidate_rule": proposal["candidate_rule"],
        "scenario": revised_scenario.model_dump(mode="json"),
    })
    trace = json.loads(Path(record["source_trace"]).read_text(encoding="utf-8"))
    review = TraceReview.model_validate(record["failure_review"])
    validate_generated_proposal("policy", revised, review, trace)

    if "admin_decision" in record:
        record.setdefault("decision_history", []).append(record.pop("admin_decision"))
    record.setdefault("admin_revisions", []).append({
        "revised_at": datetime.now(timezone.utc).isoformat(),
        "field": "scenario.patient_messages",
        "previous_messages": previous_scenario.patient_messages,
        "new_messages": revised_scenario.patient_messages,
        "reason": reason.strip()[:500],
    })
    proposal["scenario"] = revised_scenario.model_dump(mode="json")
    record["status"] = "pending_review"
    store.update(path, record)
    return {
        "status": record["status"],
        "proposal_id": proposal_id,
        "revision_count": len(record["admin_revisions"]),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Review clinic-agent improvement proposals.")
    parser.add_argument("--proposal-dir", default="artifacts/proposals")
    parser.add_argument("--policy-dir", default="policies")
    parser.add_argument("--trace-dir", default="artifacts/admin-evaluation/traces")
    parser.add_argument("--generated-scenario-dir", default="scenarios/generated")
    parser.add_argument("--live-protected", default="scenarios/live_protected.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--workspace", help="Use an isolated demo workspace created by scheduler-demo.")
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="List saved proposals.")
    list_parser.add_argument("--status")
    show_parser = commands.add_parser("show", help="Print a proposal for review.")
    show_parser.add_argument("proposal_id")
    accept_parser = commands.add_parser("accept", help="Approve a proposal and run its gate.")
    accept_parser.add_argument("proposal_id")
    reject_parser = commands.add_parser("reject", help="Reject a pending proposal.")
    reject_parser.add_argument("proposal_id")
    reject_parser.add_argument("--reason", default="Rejected by admin.")
    revalidate_parser = commands.add_parser(
        "revalidate", help="Recheck saved malformed generated output with current validators."
    )
    revalidate_parser.add_argument("proposal_id")
    revise_parser = commands.add_parser("revise-rule", help="Revise a pending policy rule before evaluation.")
    revise_parser.add_argument("proposal_id")
    revise_parser.add_argument("--rule-file", required=True, help="UTF-8 file containing the reviewed rule.")
    revise_parser.add_argument("--reason", required=True)
    revise_scenario_parser = commands.add_parser(
        "revise-scenario", help="Revise patient replay messages in a generated policy scenario."
    )
    revise_scenario_parser.add_argument("proposal_id")
    revise_scenario_parser.add_argument(
        "--messages-file", required=True, help="JSON list of 1-4 patient messages."
    )
    revise_scenario_parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.workspace:
        workspace = Path(args.workspace)
        args.policy_dir = workspace / "policies"
        args.proposal_dir = workspace / "proposals"
        args.trace_dir = workspace / "evaluation-traces"
        args.generated_scenario_dir = workspace / "regressions"

    configured_password = os.environ.get("SCHEDULER_ADMIN_PASSWORD")
    if not configured_password:
        parser.error("set SCHEDULER_ADMIN_PASSWORD in .env before using the admin CLI")
    supplied_password = getpass("Admin password: ")
    if not secrets.compare_digest(supplied_password, configured_password):
        parser.error("admin authentication failed")

    store = ProposalStore(args.proposal_dir)
    try:
        if args.command == "list":
            for path, record in store.list(args.status):
                print(
                    f"{record['proposal_id']}  {record['status']}  "
                    f"{record['proposal']['proposal_type']}  "
                    f"{record['failure_review']['category']}  {path}"
                )
        elif args.command == "show":
            _, record = store.load(args.proposal_id)
            print(json.dumps(record, indent=2))
        elif args.command == "accept":
            result = accept_proposal(
                proposal_id=args.proposal_id,
                store=store,
                policy_directory=args.policy_dir,
                trace_directory=args.trace_dir,
                generated_scenario_directory=args.generated_scenario_dir,
                live_protected_path=args.live_protected,
                model_name=args.model,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "revalidate":
            paths = revalidate_rejected_generated(args.proposal_id, store)
            print(json.dumps({"pending_proposals": [str(path) for path in paths]}, indent=2))
        elif args.command == "revise-rule":
            rule = Path(args.rule_file).read_text(encoding="utf-8")
            print(json.dumps(revise_policy_rule(args.proposal_id, store, rule, args.reason), indent=2))
        elif args.command == "revise-scenario":
            messages = json.loads(Path(args.messages_file).read_text(encoding="utf-8"))
            print(json.dumps(revise_policy_scenario_messages(
                args.proposal_id, store, messages, args.reason
            ), indent=2))
        else:
            print(json.dumps(reject_proposal(args.proposal_id, store, args.reason), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


def _next_policy_version(policy_directory: str | Path) -> str:
    versions = []
    for path in Path(policy_directory).glob("v*.json"):
        suffix = path.stem[1:]
        if suffix.isdigit():
            versions.append(int(suffix))
    return f"v{max(versions, default=0) + 1}"


if __name__ == "__main__":
    main()
