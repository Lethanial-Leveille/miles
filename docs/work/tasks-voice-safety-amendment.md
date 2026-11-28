# Tasks voice dispatch and speech safety amendment

**Approved at `c29d42a`.** This coordinates revision three findings H3
and H4 and its related result handling finding M3. It amends section 3.8 of
[revision three](tasks-revision-three.md) on backend `41d1fb0`. The
[approved scheduling rules](tasks-scheduling-amendment.md) and existing
permission floors, clearance, confirmations and storage rules continue to
apply. This is a Large contract change because it governs side effects.

## Existing boundaries

- `brain.ask_nova_async` feeds first round text to the speech router before
  the final tool list is known. It drains that audio while the tools run.
  Follow up rounds also feed speech before dispatching any additional calls.
  A stream reset can replace draft text on screen. It cannot retract audio.
- `brain._run_tools` dispatches sequentially within one executor invocation
  and checks `tools.permits` for each call. This is a per call gate. An earlier
  write can already have run when a later argument, permission or schedule
  check fails. Separate turn workers and a cancelled await of an executor
  future can also leave work running. Add a pre dispatch decision for the
  complete covered batch and keep the individual clearance check at dispatch.
- `speculative_turn.SpeculativeTurn` buffers local events before the endpoint.
  When hosted search is enabled, its request receives that schema through
  `brain.first_call_kwargs`. A hosted search can run before the local executor
  sees any tool. `pending_action` has an expiry and single consumption rule,
  but its current turn and proposal state are shared across workers. A model
  boolean alone does not establish that the current user approved that proposal.

## Dispatch contract

1. A speculative request may reason and buffer only. It must never expose
   executable hosted tools, start local dispatch, advance or consume
   confirmation state, or emit audio. A tool free speculative answer cannot
   stand in for a committed response that needs tools. For the first slice
   close this gap by bypassing speculation on covered tool capable turns,
   unless tests prove an equivalent side effect free path. A model round with
   covered tools available can emit a write before its intent is known, so
   apply this boundary based on available capabilities. After the existing
   endpoint and identity gates, authorize from the committed utterance and
   its speaker clearance.
2. Treat the full set of mutation or control proposals as one preflight
   before any side effect, in both first and follow up rounds. Resolve tool
   identity and registered schema, validate arguments, call the existing
   `permits` gate, verify explicit intent and any existing confirmation, and
   resolve the schedule from the committed text. The model cannot supply its
   own authority or substitute a normalized date. Use only side effect free
   validation. Check state dependent constraints again at the write boundary
   using the existing transaction or conditional update. No write dispatches
   when a required check is missing, refused or ambiguous.
3. The covered entry points are the shared brain calls for task and list
   mutation, including `task_add`, `task_complete`, `archive_task`,
   `list_add` and `list_check`. A sibling state change, even one outside
   that set, cannot run around a covered refusal. For this slice allow one
   distinct guarded mutation per committed request. If the request or the
   plan contains multiple distinct changes, ask for one at a time before
   executing any of them. Preserve the existing explicit confirmation
   mechanism for approved external batches. A confirmation turn cannot
   append a fresh unapproved mutation to the batch it confirms.
4. Keep independent, vetted read operations eligible for bounded concurrent
   execution after commitment and the existing clearance checks. A read can
   finish even when a sibling write is refused. Reads that supply a write
   prerequisite must finish before preflight is complete. A read that depends
   on a write must wait for its result. Keep tool results associated with
   the original tool use IDs, and never run a control or state change as a
   parallel read. Preserve the existing immediate user interruption path. Leave
   the separate Sessions executor unchanged in this
   slice. Retain its existing storage eligibility guard.
5. Reserve the authorized mutation once before executor submission. A repeated
   call in the same batch or a follow up cannot dispatch it again, including
   after a refusal, timeout or failure. An unrecognized write in a later
   round requires fresh committed user input. A repeated transport or model
   event for the same turn receives its known disposition. A genuinely new
   committed request is evaluated independently. Cancellation before
   dispatch invalidates unused authority and queued audio. Once a synchronous
   worker has begun, cancellation cannot promise rollback. Stop later
   dispatch, suppress stale speech and keep the true or unknown outcome for
   reconciliation. Do not retry an uncertain write automatically.
6. Bind pending confirmation to the proposal actually presented, its exact
   actions, the authorized speaker and the originating conversation. Keep
   the existing next turn and expiry requirements. Approval must be a clear
   committed user response to that proposal, checked in code before dispatch.
   Validate each action and the current clearance again before consumption.
   Denial, ambiguity, expiry, interruption before presentation, changed
   arguments or a replaced proposal cannot authorize execution. Consume valid
   approval once. Follow the existing external action order and report any
   partial success per action. Preserve current calendar and approval guards.

## Speech and results contract

Hold model generated speech during any covered tool capable round until its
tool disposition is settled. A neutral, code written acknowledgement can
bridge a wait. Inspect the final tool list before releasing model text, and
discard provisional text alongside mutations that could imply completion.
Use the existing router, speech queue and canonical response path to deliver
code written mutation outcomes. Apply the hold to follow up rounds as well.

A successful result must describe the effect actually committed. A pending
confirmation speaks the exact question. A refusal or mutation failure says
what did not happen. If an approved batch partly succeeds, name each known
success and failure, and acknowledge uncertain outcomes without claiming
success or rollback. Keep user facing text separate from model instructions
and from sanitized machine outcomes. Retain the existing
`ToolNotice` behavior for unaffected tools. A model paraphrase cannot
override the code's account of the mutation, and a tool error or missing
result never authorizes success playback.

The Nova early speech prototype at `cf6a257` remains separate. Ordinary
stream deltas are provisional. A future committed sentence can be released
only for the current turn and response generation, after every dependency
that could invalidate the sentence has settled. Preserve final reply fallback
and cancel stale queued output on interruption, reset or disconnect. This
proposal adds no Nova protocol or code.

## Implementation sequence and approval

1. Add synthetic contract regressions to the existing brain, tool,
   speculative turn and pending action suites. Reproduce premature “Done”
   before a refused add, and an archive in either position around that add.
   Assert affected rows and audio or transport events. Include wrong
   clearance, malformed arguments, denial, expiry, cross conversation
   confirmation, duplicate calls, cancellation before and after submission,
   mixed reads, failed tools and partial results.
2. Close speculative hosted execution, establish committed preflight and
   reservation in the existing dispatch paths, then replace provisional
   speech with confirmed code outcomes for guarded mutations. Keep existing
   independent read paths and test that reads can overlap safely.
3. Connect the completed resolver only after the voice hold, whole batch
   preflight, duplicate handling and confirmation regressions pass and the
   applicable independent gate is complete. Pass the original committed
   utterance and one captured task day into the resolver, and write only a
   supported normalized plan. Preserve the current limitations on voice
   deadlines and full restatement after clarification.

The owner approved the temporary limit of one guarded mutation per committed
request and the latency cost of holding speech and bypassing tool capable
speculation when safety cannot be proven. This does not set a permanent
product limit. The existing clearance floors,
confirmation policy and production rollout gates carry forward.

## Note from the request intent contract

"This proposal adds no Nova protocol or code" is amended for this pass by
[the intent contract](tasks-intent-contract.md): Nova discards an unfinished
reply when its stream drops. The backend side is that a turn that could change
something sends no model text until its actions finish, on the blocking route,
the streaming route and the voice path. The one change per request limit
stays, and "skip today" counts as that one change. Whole batch preflight now
also checks every mutating call against the validated intent, in every round.
