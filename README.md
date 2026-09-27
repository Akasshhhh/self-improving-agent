# Clinic Scheduling Agent

This repository contains a patient-appointment scheduling agent for a clinic. A language model holds the conversation and requests scoped tools; deterministic Python code validates those requests and owns SQLite appointment changes. Completed runs produce traces that can be evaluated, turned into bounded improvement proposals, and promoted only after regression checks pass.

## TL;DR

- Live conversation: `.venv/bin/scheduler-agent --timezone Asia/Kolkata`.
- Repeatable offline improvement loop: `.venv/bin/scheduler-eval`.
- A supported trace failure becomes a validated policy proposal plus a new scenario; admin approval runs baseline/candidate replay before activation.
- Results: **scripted** v1 **9/11 (81.8%) → 10/11 (90.9%)**; a separate **real-model rehearsal** **4/5 (80%) → 5/5 (100%)** in both attempts, with no protected regressions ([offline](artifacts/evaluation/improvement-report.json), [live](reports/live-gate-example.json)).

## What this demonstrates

- **Multi-turn agent:** a live CLI uses model tool calls with in-memory conversation state.
- **Safe tools:** session-bound patient identity, validated arguments, and deterministic SQLite side effects.
- **Evaluation:** 11 offline scenarios and four live protected cases cover normal paths and failure modes; database state and ordered traces decide objective outcomes.
- **Self-improvement:** a trace failure becomes a structured policy proposal and regression scenario, then an admin gate compares the old and candidate policies without weakening protected behavior.

## Architecture

```text
Patient ↔ agent/model → tool dispatcher → SQLite
                          └→ trace → evaluation → proposal → admin gate → active policy
```

The model chooses actions and writes patient-facing text. The dispatcher and repository validate identity, permissions, availability, and booking side effects. SQLite is authoritative for appointments; the trace is evidence for debugging and evaluation.

## Setup and commands

Python 3.12+; run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

For live chat and admin review, set `OPENAI_API_KEY`, `OPENAI_MODEL`, and `SCHEDULER_ADMIN_PASSWORD` in `.env`. Offline evaluation needs no API key.

Run the repeatable offline improvement demo:

```bash
.venv/bin/scheduler-eval
```

Create a fresh isolated workspace for a live demo. The directory must not already exist:

```bash
.venv/bin/scheduler-demo artifacts/demo-001
.venv/bin/scheduler-agent \
  --workspace artifacts/demo-001 \
  --timezone Asia/Kolkata
```

After the chat ends with `exit`, review its proposals with the admin CLI:

```bash
.venv/bin/scheduler-admin \
  --workspace artifacts/demo-001 \
  list

.venv/bin/scheduler-admin \
  --workspace artifacts/demo-001 \
  show PROPOSAL_ID

.venv/bin/scheduler-admin \
  --workspace artifacts/demo-001 \
  accept PROPOSAL_ID
```

Replace `PROPOSAL_ID` with the ID printed by `list`. The admin commands ask for `SCHEDULER_ADMIN_PASSWORD`. If the policy gate promotes a proposal, start the agent again with the same workspace; it loads the new active policy.

## Tools exposed

The model gets four tools. Their Pydantic inputs reject extra fields; `patient_id` comes from the CLI session, never model arguments.

| Tool | Model arguments | Scope |
| --- | --- | --- |
| `search_available_slots` | `specialty`, optional `starts_after`, `starts_before` | Read available slots; return times in the display timezone. |
| `list_my_appointments` | `{}` | Read only this patient's appointments. |
| `book_appointment` | `slot_id` | Stage an offered, available slot; a later patient `confirm` lets the dispatcher book that same slot. |
| `cancel_appointment` | `appointment_id` | Repository enforces patient ownership. |

There is no live `reschedule_appointment` tool; rescheduling is therefore an explicitly unsupported capability in the current registry.

## Evaluation rubric

Each scenario gets a fresh SQLite database. A case passes only if all checks pass; the score is the percentage of cases passing.

| Evidence | Actual checks | Grader |
| --- | --- | --- |
| Database | Expected active slot IDs and cancelled count | Deterministic |
| Ordered trace | Required/forbidden tool calls, expected errors, booking request after selection, successful booking after confirmation of the staged slot | Deterministic |
| Response | Required terms; optional live conversational goal | Deterministic term check; narrow model judge for some live goals |

The unsupported-reschedule goal uses a deterministic response check. A transcript-only judge cannot verify a SQLite write, tool success, or ownership. The model judge cannot override failed database or trace checks.

## Scenario coverage

- [Offline, 11 cases](scenarios/scheduling.json): booking, missing specialty, ambiguous time and “that one,” no availability, stale slot, double booking, cancellation, medical question outside scope, injected `patient_id`, and unsupported rescheduling.
- [Live protected, four cases](scenarios/live_protected.json): booking confirmation, ambiguous reference, no availability, cancellation.
- [Generated regression](scenarios/generated/unsupported-reschedule-move-001.json): a move request must preserve the original booking and must not book or cancel as a substitute.

## Improvement loop

On `exit`, the trace saves the conversation, tool arguments/results, starting and final state, and policy version. A deterministic reviewer finds clear failures; a narrow model reviewer handles failures that need intent understanding. A supported failure produces a policy rule and a generated regression scenario for admin review.

The gate freezes that generated target together with the protected scenarios. Baseline and candidate run against the **exact same frozen set**, with the same resolved dates, twice per policy in fresh databases. The baseline must reproduce the failure, the candidate must pass, and protected cases must not regress. Only then is the new policy activated. The lower-level proposal and workspace behavior is documented in [DESIGN.md](DESIGN.md).

The [offline report](artifacts/evaluation/improvement-report.json) shows scripted mechanics: **9/11 → 10/11** for ambiguous reference. The separate [live rehearsal](reports/live-rehearsal.md) shows an admin-reviewed real-model reschedule fix: **4/5 → 5/5** twice. Neither regressed protected passes. The live workspace's v2 differs from tracked offline `policies/v2.json`.

## One concrete example

For `ambiguous_that_one`, v1 attempts `book_appointment` after the patient says “that one” about multiple slots. [Policy v2](policies/v2.json) adds:

> When multiple slots have been offered and the patient's reference is ambiguous, ask which slot they mean. Do not guess or book until they select a specific slot.

The [offline report](artifacts/evaluation/improvement-report.json) records failure under v1 and success under v2; model turns are scripted.

## Guardrails

Identity, tool validation, SQLite ownership/availability checks, and pending-slot confirmation live in code. Medical scope is in the non-editable core prompt. Model-authored policy can add scheduling rules, but cannot alter these checks. Admin replay decides promotion.

## Limits

The live report is one curated rehearsal; raw attempts remain local. Two live attempts do not eliminate model variability, and the current code needs a fresh score after later changes. Scenarios cannot prove every free-text claim. Generated tests may miss the failure; baseline replay must reproduce it. A booking made earlier in the same chat is absent from its run-start snapshot, limiting reschedule replay. Local `--patient-id` and `.env` password are demo controls, not production authentication.

## Links

- [One-page design note](DESIGN.md): tradeoffs and where AI helped versus where engineering judgment overrode it.
- Loom walkthrough: **link pending recording**.
