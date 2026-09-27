# Design note

## TL;DR

The model handles conversation and suggests tool calls. Trusted application code validates identity and owns appointment changes in SQLite. Traces support evaluation and improvement, but a policy becomes active only after the same target and protected scenarios pass without regressions.

## Key design decisions

- **Trusted identity:** the patient ID comes from the CLI session, never from model-controlled tool arguments.
- **Deterministic side effects:** the dispatcher and repository validate requests and return the real result. The model cannot turn a failed booking into a claimed success.
- **Safe booking flow:** `book_appointment` stages an offered slot; the application books only that exact pending slot after explicit patient confirmation. SQLite rechecks availability in a transaction and uses a unique index to prevent double booking.
- **Separate model boundary:** `ModelClient` supports both a scripted client for repeatable offline evaluation and an OpenAI-compatible client for live conversations. A model-turn limit prevents an infinite tool loop.
- **Evidence-based evaluation:** traces record ordered calls, arguments, results, errors, policy version, and appointment state. Deterministic checks decide objective outcomes; a narrow model judge handles only an optional conversational goal.
- **Controlled promotion:** immutable policy files are selected through `active.json`. A generated rule is promoted only after baseline reproduction, candidate improvement, and protected regression checks on the same frozen scenarios.

## State and traces

Conversation state stays in memory for one CLI process. Appointment state is durable in SQLite. The trace is evidence for debugging and evaluation, not a second appointment store. Its bounded starting snapshot contains this patient’s appointments plus slot and occupancy information without exposing other patient identities.

Redis was not added because this take-home uses one process. A multi-worker service would need shared session storage while keeping appointments in a durable database.

## Evaluation and improvement

Each scenario uses a fresh SQLite database and must pass its database, tool-event, and response checks. The offline suite uses scripted responses so the improvement result is repeatable. The live CLI reviews a saved trace and can generate a bounded policy rule and regression scenario.

Before promotion, the generated target is frozen with the protected scenarios. Both policies run against that exact same set with fresh databases and repeated attempts. The baseline must reproduce the failure, the candidate must pass, and protected cases must not regress. Only then is the candidate saved and selected by `active.json`.

A proposed tool contract is reviewable data for a developer; it does not generate code or change the tool registry.

## Proposal and workspace mechanics

`scheduler-demo` refuses to overwrite an existing directory, copies the base policy, and creates isolated database, trace, proposal, evaluation-trace, and regression locations. `--workspace` makes the agent and admin CLIs use those paths.

`ProposalStore` writes schema-versioned JSON atomically, preserves rejected model output with its validation reason, and fingerprints pending proposals so repeated failures do not create duplicates. The admin CLI can display old records, but only schema version 2 proposals in a reviewable state can be accepted. Revalidation and policy revisions happen before the gate; neither changes the active policy by itself.

## Known limitations

- **Rescheduling:** no atomic reschedule tool exists yet. A tool proposal describes the missing contract, but a developer must implement and evaluate it.
- **Scale:** conversation state and SQLite are local to one process. A deployed multi-worker service would need shared state, a production database, and operational controls.
- **Model variance:** live results depend on model behavior, credentials, network, and provider responses. Two attempts reduce noise but do not prove consistency.
- **Coverage:** the 11 offline and four protected live cases cover selected failure modes, not every patient phrasing or clinical situation.
- **Conversational quality:** the model judge checks only a narrow response goal; it cannot fully measure empathy, clarity, or long conversations.
- **Replay evidence:** generated rescheduling tests require a source trace that starts with this patient’s booking. Older traces may need a manually prepared scenario.
- **Administration:** the `.env` password and `--patient-id` are demo controls, not production authentication, authorization, encryption, or audit logging.
- **Timing:** review runs after `exit`; a promoted policy affects the next conversation, not the one that produced the trace.

## AI use and assumptions

AI tools helped scaffold and implement the repository. Engineering judgment kept SQLite authoritative, identity session-bound, and promotion deterministic. The demo uses synthetic data and a trusted local CLI identity; it is not a production privacy or authentication design.
