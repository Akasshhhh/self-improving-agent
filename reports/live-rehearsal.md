# Sanitized live improvement rehearsal

This is a compact, tracked summary of one real-model rehearsal. It contains synthetic clinic data only. Raw traces, proposal records, databases, and API configuration remain local and are ignored by Git.

## Source failure

Under policy `v1`, a patient asked to move an existing cardiology appointment. The agent called `book_appointment` for a replacement slot and treated the request as completed. The original appointment was not safely moved as one operation. This produced an unintended replacement booking and was detected from the trace and final SQLite state.

## Review and proposal

The trace reviewer classified the failure as `unsupported_reschedule`. The model proposed:

- a policy rule that keeps the original appointment unchanged when no rescheduling tool exists;
- a generated regression scenario stored at [unsupported-reschedule-move-001.json](../scenarios/generated/unsupported-reschedule-move-001.json);
- a separate `reschedule_appointment` tool contract for future developer implementation.

An admin revised the policy wording after the first candidate still booked a replacement after confirmation. The accepted rule says that every turn of the move request, including confirmation, must avoid `book_appointment` and `cancel_appointment`, explain that direct rescheduling is unavailable, and preserve the original appointment.

## Gate result

The real model was replayed twice under each policy using fresh SQLite databases and the same frozen scenario inputs. Four protected live scenarios ran in the same comparison.

| Policy | Score | Target | Protected regressions |
| --- | ---: | --- | --- |
| v1 baseline | 4/5 (80%) | failed twice | none |
| reviewed candidate | 5/5 (100%) | passed twice | none |

The gate promoted the candidate policy and recorded it as the active policy in the isolated demo workspace. A subsequent live chat recorded the promoted version in its trace.

The candidate rule is intentionally a policy change only. The proposed rescheduling tool was approved as a developer implementation item; it did not generate code or modify the tool registry.

The detailed per-case decision is in [live-gate-example.json](live-gate-example.json).
