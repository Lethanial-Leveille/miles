# Tasks corrections and local integration

The verified bundles were imported by fast forward into the two canonical
local worktrees. Both feature worktrees are clean.

| Repository | Canonical branch | Exact head | Verified ancestry |
| --- | --- | --- | --- |
| Backend | `feature/tasks-followups` | `62f1848384f60bf058617f2988bd2d196bb1955e` | `fb4a786 → aa3e009 → 62f1848` |
| Nova | `feature/tasks-day-guard` | `f2fd799f10b1482e40711b7ceafa272602d7e13e` | `43dde91 → f2fd799` |

The backend review document matches its blob at `aa3e009`. The backend
`main` is still `51676b9` and Nova `main` is still `d9e8c9f`. The
nightly backup checkpoint remains in the backend continuation handoff.
This integration made no application edits. The scheduling resolver remains
offline, awaiting the existing independent verification gate.

## Migration 047 and request receipts

After 045 and 046, migration 047 adds the SQLite table
`chat_request_receipts` in the same transaction as the version update. It
adds no task columns and does not transform existing tasks, completions,
Evallo rows or conversation history.

Each main chat request with an identity reserves one row before dispatch.
The composite key is the authenticated actor and the client generated
request UUID. The table also stores the SHA 256 fingerprint of the
canonical request envelope and a nullable copy of the final assistant reply.
The fingerprint covers the conversation, request, message, channel and any
confirmation receipt supplied for that turn. The table stores neither the
raw request envelope nor the pending proposal callback or approval receipt
as separate values. The proposal receipt and callback remain in process
memory, and the receipt is issued with the final response for that exact
proposal.

An exact replay with a completed receipt returns the recorded reply without
dispatch or fresh confirmation authority. A changed envelope using the same
key or a reservation without a recorded result receives HTTP 409 on
`/chat`, or a terminal error event on an open `/chat/stream`. The pending
row also covers interrupted or uncertain outcomes. Retrying the same ID
does not run the request again.

Both pending and completed rows survive restart. There is no expiry,
pruning or automatic deletion. This preserves replay protection and means
the table can grow and retain assistant text indefinitely. The approved
proposal's existing confirmation expiry is separate from database
retention. A database backup includes these rows, and restoring an older
snapshot can discard replay history after that snapshot.

An isolated compatibility check initialized a schema 046 database with the
`fb4a786` code, seeded synthetic task, completion and Evallo rows, then
applied 047. Reopening it with the older initializer preserved schema 047,
the seeded rows and both receipt states. This checks storage compatibility.
An older backend binary does not enforce the new request or confirmation
behavior even when the table is present.

## Client compatibility and release sequence

| Pairing | Behavior |
| --- | --- |
| Legacy Nova with corrected backend | Main Chat and Talk can stream ordinary replies and reads. Requests without conversation and request IDs refuse chat tool mutations and app confirmations. Direct Tasks HTTP calls keep their existing authorization and behavior. A Nova build with the rendered day guard retains it. Older bodyless complete and reopen requests still act on the server's current task day. |
| Corrected Nova with corrected backend | Main chat submits conversation and request IDs and sends a proposal receipt for the next approval. Task row actions retain the rendered day guard. A repeated request cannot run the mutation twice. |
| Corrected Nova with prior backend | The prior backend can ignore the extra identity fields and does not issue or check proposal receipts. This pairing lacks the new chat replay and proposal guarantees. If the backend also predates the Tasks day guard, it ignores the rendered day. |

After separate rollout authorization and backup phase acceptance, verify
045 is applied, take the required fresh backup, and prepare rollback.
Release and verify the corrected backend with 046 and 047 before enabling
mutating main chat on the paired Nova build. Coordinate the app update
with the backend window. During the version gap, legacy main chat reads
remain available and chat tool mutations are refused. Check the exact
backend and app versions, stale task taps, replay behavior, and proposal
delivery on the matched pair before accepting the rollout.

For rollback, coordinate the app and backend and pause affected writes
before switching to a backend that ignores the receipt and day guards.
Preserve the 047 ledger and reconcile uncertain requests and future daily
rows before restoring writes. Verify the exact rollback target and merge
parent if applicable. No rollout was performed.

## Verification on the canonical revisions

- Both bundle prerequisites, exact imported heads, worktree cleanliness,
  branch ancestry and the review blob were checked.
- The backend passed 288 targeted tests on Python 3.13 with a disposable
  home and synthetic SQLite data. The passing cases cover legacy and
  identified main chat, cross route replay, lost proposals, task day
  compatibility, migration 046 and the offline resolver.
- The synthetic 046 to 047 upgrade, older initializer and persistent
  receipt check passed.
- The full unsigned generic iOS `Nova` build passed on Xcode 26.4.1.
  The resulting app was verified as an unsigned `iphoneos` arm64 build.
  The standalone Swift checks also passed with synthetic transport,
  including the rendered day and proposal receipt round trip. Physical
  device and rollout validation remain open.

The original [backend bundle](backend.bundle) and [Nova bundle](nova.bundle)
remain available for offline transfer.
