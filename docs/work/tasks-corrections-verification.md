# Focused Tasks correction verification

**Decision. Keep the scheduling resolver offline.** Backend `62f1848` and
Nova `f2fd799` pass the four original High reproduction cases in the
[review at `aa3e009`](tasks-independent-review.md). Nearby cases still permit
an ordinary write for unsupported timing, unrelated writes before a task
refusal, and a spoken success without an archive. A dropped app stream can
also leave a provisional success visible.

## Original reproduction cases

| Review item | Result on the pinned pair |
| --- | --- |
| High 1, H1 and H2 | “Add water plants Saturdays” is refused by the live brain even when the model echoes the entire title. The offline resolver refuses “Add call the bank at half past five” and “Add water plants on the first of the month.” The live guard also refuses those two. No task is created in these live cases. |
| High 2, H3 | “Add reading daily” with only a `dismiss` call produces a code refusal in the audio and final text. “I need to take my vitamins every day” with no call produces “I could not confirm that action.” Neither produces the model's success sentence or creates a task. Verified successful add speech, the specific first day refusal for a prestart completion, and uncertain callback outcomes labelled by action. |
| High 3, H4 | “Add reading” blocks the first round timer with a task read before any effect. The same batch archive sibling is refused in either order. Identical app deliveries across both HTTP routes produce one SQLite task row. The restriction still depends on recognizing the request before any write. |
| High 4, app proposal | A same account conversation cannot confirm a proposal after the first response is lost on either route. The exact receipt works on the next turn with the original conversation, actor and channel. Wrong receipts and receipts moved across scopes are refused. Approval is consumed once. The completed response carries the receipt. A recorded proposal replay returns its question without a receipt. |

## Remaining findings

### High 1. Unsupported timing can still produce an ordinary task

For “Add water plants in four days,” a model that echoes the full phrase as
the title causes the live path to write a one time task for today and speak
“Added the task: Water plants in four days.” The offline resolver returns
`supported` with null recurrence and start. It also returns the same default
plan for “Add call the bank at dawn,” which the live path refuses. These
cases exercise unconsumed relative timing and clock timing in the original
reproduction shape. Both paths need a refusal with no write before the
resolver can be connected.
[Live timing guard](../../src/voice_action_safety.py#L16)
· [Default write](../../src/lists_tasks.py#L232)
· [Offline timing guard](../../src/task_scheduling.py#L28)
· [Offline default plan](../../src/task_scheduling.py#L174)

### High 2. Task intent can arrive too late to prevent unrelated writes or false speech

“Cancel reading” is accepted when `archive_task` appears in the first
batch. The committed intent classifier omits `cancel`, even though the
archive matcher accepts it. With an existing Reading task, a first batch
containing a timer, reminder and task read dispatched both mutation
callbacks. Each wrote a synthetic SQLite marker. The next model round's
archive was refused with “A new request is needed for another change,” and
Reading remained active. Later round reservations therefore cannot enforce the
single mutation limit for the entire request when the first round is
misclassified. For the same utterance with no tool call, the brain spoke
and returned “Done. I archived Reading.” The archive timestamp stayed null.
Recognition for preflight and speech needs to cover the accepted archive
wording before the first round can claim authority.
[Intent and archive matcher](../../src/voice_action_safety.py#L34)
· [Batch preflight](../../src/voice_action_safety.py#L183)
· [No tool speech](../../src/brain.py#L1663)

### Medium. A dropped stream can preserve provisional success on screen

The backend emits model deltas before preflight. A synthetic ASGI stream for
the refused Saturday task emitted “Done. I saved the task.” before the
connection closed, with zero task rows. A Nova transport fixture then
delivered the same delta and ended before the final frame. Nova returned
no reply for Talk Mode and set a transport error, while the success bubble
stayed visible. The completed canonical response and spoken guarded outcome
use the code result. The interrupted screen still needs to discard or clearly
invalidate the provisional reply.
[Backend delta](../../src/brain.py#L1346)
· Nova `Nova/Services/ChatService.swift:132`

## Replay, migration and client compatibility

- **Same request replay passed.** The backend atomically reserves the
  authenticated actor and request ID before dispatch. A matching completed
  input returns its recorded text across transports. A changed input or an
  unfinished reservation is refused. Concurrent same ID delivery and
  interruption after a write each left one task. Fresh Python processes
  opening the same database returned the recorded reply for the known case
  and a conflict for the unfinished case without invoking the brain. A tool
  fixture that committed and then raised produced an uncertainty sentence
  that also replayed without a second write. The blocking route returns
  HTTP 409 for conflicts. The stream sends a terminal error event. The
  standalone Nova fixture reused the full turn identity on 404 fallback and
  did not fall back on a truncated stream. A manual new send creates a new
  ID, including for identical text, so its effects are outside this replay
  guarantee. Pending approvals survive only in memory. A restart also
  ends their confirmation authority.
  [Reservation and response](../../src/server.py#L205)
- **Migration 047 is additive.** It follows 045 and 046 and creates a separate
  receipt table. The synthetic upgrade from 046 preserved daily start,
  task and completion history, and both Evallo tables. A forced error
  after the DDL rolled back the table and version, and concurrent retry
  completed with schema version 47, SQLite integrity OK and no foreign key
  violations. The exact prior database module opened the upgraded file and
  performed a synthetic task write. That establishes storage compatibility.
  Main chat on the earlier backend ignores the new identity fields and lacks
  the replay and receipt guards. Restoring a snapshot discards receipts
  created after it. Any rollback that resumes affected writes needs to
  account for these gaps.
  [Migration and receipt storage](../../src/database.py#L1202)
- **Receipt privacy needs an explicit disposition.** The table writes the
  actor, request UUID, deterministic input digest and complete reply for
  every identified main chat request, including reads. It does not store raw
  request bodies, proposal receipt values or callbacks. Full reply text is
  an additional durable copy of potentially private content in the database
  and its backups. The table has no age column, retention limit or cleanup,
  so it also grows for ordinary chat. Preserve the replay disposition and
  document the intended retention and deletion behavior of reply content.
  [Persisted fields](../../src/database.py#L1202)
  · [Chat persistence boundary](../../src/server.py#L205)
- **The legacy notes are incomplete.** The older HTTP envelope can still
  read, and the structured Tasks day guard keeps its separate documented
  legacy behavior. Main chat without both IDs blocks all noncontrol tool
  writes, including timers, reminders and confirmation, beyond Tasks.
  WebSocket chat always uses that legacy envelope and ignores the new
  identity fields. A paired phone on an earlier backend can send both IDs,
  but inherits the earlier server's behavior. The endpoint notes should
  cover this matrix, and Nova's stream comments should reflect that a
  disconnect cancels unused authority while a running mutation may finish.
  [Current candidate notes](tasks-followups.md#independent-review-corrections)
  · [WebSocket route](../../src/server.py#L1243)

## Verification scope

Python 3.13 ran 389 existing focused tests and 15 scratch probes using
disposable SQLite files, fake model streams, audio collectors and in process
HTTP. The probe assertions checked the actual behavior, including the
reproduced defects. Nova's standalone Swift check exercised the real chat
service, API service and task state with synthetic URL responses. A second
fixture covered the 404 fallback and incomplete response. Both pinned diffs
passed Git whitespace checks. This run verified neither full iOS build nor
physical playback and rendering. It made no application changes and used
no production data, providers or device connections.
