# Incidents: what broke, and what the failure taught

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Each entry is dated. Every one was reconstructed from logs, the database, or
> captured audio at the time, and the evidence is kept inline.
>
> **Live config values are declared in [CLAUDE.md](../CLAUDE.md), not here.**
> This file carries the reasoning and the measurements behind them. A number
> quoted inside a measurement is what was recorded on the date given and is
> left exactly as measured, so it may differ from what is live now. The config
> table in CLAUDE.md is the one that tracks the present.

Up: [CLAUDE.md](../CLAUDE.md) · [SESSION_START.md](SESSION_START.md) ·
[BACKEND_TODO.md](BACKEND_TODO.md) · [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) ·
[LATENCY.md](LATENCY.md) · [VOICE_OUTPUT.md](VOICE_OUTPUT.md) ·
[BRAIN.md](BRAIN.md) · [INFRASTRUCTURE.md](INFRASTRUCTURE.md) ·
[INCIDENTS.md](INCIDENTS.md)

---

Production failures and their root causes, newest first. This file is for
things that **broke**. Deliberate design choices go in the decision log in
[SESSION_START.md](SESSION_START.md) instead, and the two should not duplicate
each other: if an incident forced a decision, the incident is written here and
the decision is recorded there with a link back.

An entry earns its place by carrying a lesson that would otherwise be relearned.
Several below exist specifically because a stale or absent record cost a later
session real work.

| Date | What happened | Where |
|---|---|---|
| Sep 14 2026 | One deauth took the Pi off the network for 21 hours, and the recovery exposed it to the internet for 9 | [below](#one-deauth-cost-21-hours-offline-and-9-hours-exposed-sep-14-2026) |
| Sep 14 2026 | Fixing one name on three lessons took five read backs and three yeses | [below](#fixing-one-name-took-five-read-backs-sep-14-2026) |
| Sep 14 2026 | His yes created a career fair on a date Nova never said | [below](#his-yes-created-a-date-he-did-not-hear-sep-14-2026) |
| Sep 13 2026 | A busy room held the microphone for up to a minute while he said the wake word | [below](#a-busy-room-held-the-microphone-sep-13-2026) |
| Sep 13 2026 | Confirming one tutoring lesson created a different one, and a failed cancel was reported as done | [below](#confirming-one-lesson-created-another-sep-13-2026) |
| Sep 13 2026 | Nova read the calendar like a printout: past events, "all day", club events as plans | [below](#nova-read-the-calendar-like-a-printout-sep-13-2026) |
| Sep 13 2026 | New calendar and Oura tools answered confidently and wrong | [below](#calendar-and-sleep-tools-answered-confidently-and-wrong-sep-13-2026) |
| Sep 13 2026 | Near misses and false wakes turned out to be one population: the neighbours, sorted by level | [below](#the-near-misses-are-the-neighbours-sep-13-2026) |
| Sep 8 2026 | Nova spoke "Done." on turns where staying silent was correct | [below](#nova-said-done-out-loud-when-she-correctly-stayed-silent-sep-8-2026) |
| Sep 8 2026 | No threshold separates true from false wakes in the new room | [below](#no-threshold-separates-true-from-false-wakes-any-more-sep-8-2026) |
| Sep 6 2026 | Reminders were dropped silently by every restart | [INFRASTRUCTURE.md](INFRASTRUCTURE.md#reminders-are-fired-by-a-poller-not-by-a-thread) |
| Sep 6 2026 | The mic gain guard was reading the speakers | [below](#the-mixer-card-guard-measured-the-speakers-sep-6-2026) |
| Sep 6 2026 | One capsule recorded itself as three microphones; enrollment threw away its audio | [below](#two-guards-that-could-not-do-their-jobs-sep-6-2026) |
| Aug 13 2026 | Nova confabulated a security tool call that never happened | [BRAIN.md](BRAIN.md#nova-knows-the-transcript-is-not-his-words) |
| Aug 10 2026 | An empty room drove a runaway conversation loop | [below](#an-empty-room-drove-a-runaway-conversation-loop-aug-10-2026) |

---

## Cancelling a timer set a new one (Sep 16 2026)

He set a one minute timer by voice, the first timer stored as a row, then tried
to cancel it three times in forty seconds. From the journal and
`conversation_history` 1145 to 1154:

| He said | What happened |
|---|---|
| (a fragment) "Ain't no one." | correctly ignored |
| "cancel the timer." | ignored as not addressed to Nova |
| "cancel the timer." | "I only caught 'ain't no one.'" |
| "Cancel the timer." | `set_timer(0 seconds)`, "Done.", and a timer that rang at once |

His real timer went off at the end, because nothing cancelled it.

Four causes, none of them in the new timer rows, which worked:

- **Local intent could not cancel a timer.** Its gate required the word
  "reminder", so the turn went to Claude.
- **Claude was never told timers could be cancelled.** `cancel_reminder` said
  reminders. With no tool that fit, she reached for `set_timer` and passed zero,
  which the tool accepted.
- **Ignored speech stayed in history.** The fragment and the first ignored
  request sat there as three consecutive user messages, which are sent as one
  turn, and she answered the fragment. That merged turn is also the likeliest
  reason the first request was ignored at all.

Fixed: local intent cancels the one outstanding timer, counted by kind so a
reminder is never cancelled by "cancel the timer"; `cancel_reminder` says it
cancels timers and never to set one to cancel; `set_timer` refuses anything
under one; and an ignored turn's message is deleted from history.

## Moving two lessons took eight turns (Sep 15 2026)

From the app, 23:27 to 23:34: move Charley's Wednesday and Friday lessons half
an hour earlier, and move Andrew's lesson. Nine tool calls, four of them failed,
and the Andrew lesson was never moved. Reconstructed from `tool_call_log` rows
135 to 144 and `conversation_history` 1129 to 1144.

Three separate defects, each found by running his exact phrases through the
parser as it stood:

| He said | Parsed as | What he saw |
|---|---|---|
| day `"wednesday at 4"` | **April 15 2027** | "no event matching 'Charley lesson' on Thursday April 15" |
| new time `"3:30"`, for a 4 PM lesson | 3:30 AM | "Move Charley lesson on Friday from 4 PM to 3:30 AM?" |
| day `"Wednesday"`, for a lesson on Thursday | the right day, no match | "Andrew's lessons aren't on your MILES calendar yet" |

**dateparser reads a bare number as a month.** `"3"` parsed to March 2027 and
`"5"` to May. The lookup failed before the move could run, which is the only
reason no one heard "move it to Sunday May 16 2027". A day with a time written
as `"wednesday at 4pm"` had always worked, which is why this survived: the model
usually adds the pm, and last night it did not.

**A bare clock time was taken at face value.** A 4 PM lesson moved to "3:30" is
obviously 3:30 PM. The code had no notion of which half of the day was meant.

**The lookup only ever searched the named day.** Its error was accurate ("no
Andrew lesson on Wednesday") and Nova generalized it into "not on your calendar
at all", which sent the next turn looking for another calendar.

The staged question read back the wrong time, so nothing wrong was written. The
confirmation held; the cost was his patience.

**A fourth defect, found the next morning when he said there were two Andrew
lessons.** There were: Tuesday at 5:30 PM, the one he had missed, and Thursday
at 4 PM. Nova's listing at 11:33 PM showed only Thursday, for two reasons.
The listing starts at now, so a lesson that ended at 7 PM was dropped exactly
when he was asking to move it. And it named at most ten of his events without
saying it had stopped, so the week ended Friday morning, after bills and
birthdays took the slots. **A fifth, found while moving them:** a Thursday event
moved to "wednesday" resolved to the Wednesday after, because the new day was
read forward from the event's own day. That would have put Andrew's lesson
after the test it was for.

Both lessons were moved by event id on Sep 16: Tuesday's to Wednesday at 4:30
PM, Thursday's to Friday at 5 PM.

Fixed in `calendar_tools.py`: bare hours are written out as clock times before
dateparser sees them, a bare time is read in the half of the day nearest the
event's current time (and 1 to 6 as afternoon on a new event), a lookup with a
bare time matches either reading, and a change that finds nothing on the named
day uses the one event with that title in the week around it. The mechanism is
in [BRAIN.md](BRAIN.md#calendar-times-are-resolved-in-code).

## His news was answered "Done." (Sep 15 2026)

"I actually put in my two weeks recently so my last day is September 25", typed
in the app at 23:22. Nova called `remember`, stored memory 311 correctly, and
the reply was `Done.` (`conversation_history` 1126).

`remember` was fire and forget, on the reasoning that she would already be
answering alongside the call. She was not. When the model calls a tool it tends
to stop and wait for the result, and `remember` sent none back, so the turn had
no words and the fallback supplied "Done.". The prompt made it likelier: the
memory instructions and the tool description both said saving "happens quietly"
and "returns nothing to say".

Measured on that message and "remember that my DSA exam got moved to October 8",
four samples each, history cut at the turn before:

| Condition | Replied in words |
|---|---|
| As shipped | 4/8, and 0/4 when he asked outright |
| A prompt paragraph telling her to answer | 1/8 |
| Result sent back, tools offered, result "stored" | 3/7 |
| Result sent back, no tools, result that says to answer | 6/6 |
| That, through the real `ask_nova_async` | 7/8 |

The lesson is the one this repo keeps relearning in new places: **what Nova is
handed last outweighs what she is told at the top.** The fix went into the tool
result, not the prompt.

## One deauth cost 21 hours offline and 9 hours exposed (Sep 14 2026)

At 15:37 the Alsander AP deauthenticated wlan0 in the middle of the handshake.
That part is unremarkable. Access points deauth clients constantly, and the
supplicant is built to retry.

What followed was not a wifi failure. It was a recovery path that gave up
permanently after one wrong guess, and then a recovery that was more dangerous
than the outage it fixed.

### NetworkManager diagnosed it as a wrong password

A deauth arriving mid handshake is indistinguishable, from where NetworkManager
sits, from an AP rejecting the pre shared key. It made the reasonable guess and
asked for the secret again, entering `no-secrets`.

On a desktop that is a dialog box. There is no desktop. Headless, there is no
secret agent registered to answer, so the request had nowhere to go, and
NetworkManager did the only thing left: it marked the connection failed and
stopped.

**It never retried.** Not once in 21 hours. The connection was not down because
the radio was broken or the AP was gone; it was down because something had
concluded the password was wrong, and a wrong password is not a condition that
retrying fixes. The logic was sound. The premise was false.

This is a failure class that only exists headless, which is exactly why it went
unnoticed for months of development on a Pi that always had a screen or a
session attached to notice. The bug needs nobody watching in order to happen.

### The recovery was worse than the fault

Recovery was a cable. Plugging in ethernet worked immediately, and that is where
the real incident starts.

The wlan0 lease is CGNAT, `100.70.16.218/25`, which is not routable from the
internet. It had been acting as a firewall that nobody had chosen, configured or
checked. The ethernet port handed out a **routable public address with no NAT in
front of it**.

Nothing about any service changed. No config was edited, no port was opened, no
firewall rule was removed. The network moved underneath three daemons that were
all binding the way they always had:

| Service | Bound to | Became |
|---|---|---|
| sshd | `0.0.0.0:22` | open to the internet |
| n8n (docker) | `0.0.0.0:5678` | open to the internet |
| uvicorn | `0.0.0.0:8000` | open to the internet |

That lasted roughly nine hours.

### What found it

Roughly **13,000 SSH authentication attempts from 35 distinct addresses**, which
is simply what the internet does to a fresh port 22 and says nothing about
anyone targeting this Pi. Every username was generic botnet stock: `root`,
`admin`, and a long tail of unknown users. The heaviest three were
`109.160.32.31`, `77.239.124.214` and `43.226.62.221`.

**Every attempt failed. There was no successful login, and no evidence of
compromise.** Two things happened to be standing in the way. `PermitRootLogin
without-password` meant the thousands of root password guesses could not
succeed no matter how long they ran. And no attacker guessed `theycallmelee`,
which was the account that could in principle have been reached, because
`PasswordAuthentication` was `yes` with no rate limiting and no user allowlist.

That second one is luck, not defence. The only reason this entry is not about a
compromise is that a botnet spraying stock usernames never tried the one that
would have worked.

### The lesson

The wifi drop is the least interesting part.

**A service's exposure was a property of the network, not of the service.**
Every daemon here was written to bind `0.0.0.0` on the reasonable assumption
that the LAN was a boundary. It was, right up until the moment a cable changed
what the LAN meant. Binding to loopback is not defence in depth against an
attacker so much as refusing to let the network's shape be load bearing.

The corollary is that an incident's blast radius is decided long before the
incident. Nothing was done wrong during those nine hours. The exposure was
already sitting there in three `ExecStart` lines and a compose file, waiting for
the day the address changed.

### Fixes applied

| Fix | Why | Where |
|---|---|---|
| `miles-wifi` watchdog every two minutes | The gap was never retrying, so the fix is something that retries on a schedule and does not care why the link went | [INFRASTRUCTURE.md](INFRASTRUCTURE.md#systemd-services) |
| wifi powersave disabled | Reduces the deauths that start this | `powersave=2` in `Alsander.nmconnection`, so it reapplies on every activation |
| n8n bound to `127.0.0.1:5678` | Exposure should not depend on the network's shape | `docker-compose.yaml` port mapping |
| uvicorn bound to `localhost` | Same, and `localhost` rather than the literal address so both loopback families are covered | [INFRASTRUCTURE.md](INFRASTRUCTURE.md#systemd-services) |
| `PasswordAuthentication no` | Removes the entire surface those 13,000 attempts were aimed at | `00-hardening.conf`, which sorts ahead of the cloud-init file. See [INFRASTRUCTURE.md](INFRASTRUCTURE.md#ssh-configuration) |
| `AllowUsers theycallmelee` | Ends the unknown user probes at the door | `/etc/ssh/sshd_config` line 59 |
| Tailscale | Recovery no longer requires a publicly addressable anything | `miles` at `100.99.248.127` on the tailnet |

**Tailscale was reported as applied before it was, and is now real.** On
Sep 15 2026 it was listed among these fixes while `command -v tailscale` found
nothing and no `tailscaled` unit existed. The install had been done on the
MacBook, and the tailnet showed one machine. It was installed on the Pi later
the same day: tailscale 1.102.4, `tailscaled` enabled and active, joined as
`miles` at `100.99.248.127`, verified by a round trip to the MacBook.

Worth keeping rather than quietly deleting, because the gap between "I applied
that fix" and "that fix is running on the machine that needed it" is exactly
the drift this file exists to catch, and it happened to the fix list for an
incident about drift.

---

## Fixing one name took five read backs (Sep 14 2026)

He asked for Charlie to be spelled Charley on the three lessons Nova had just
created. From 00:33 to 00:37:

- Nova said she could not change events on other calendars, without calling a
  tool. They were on MILES.
- The one event tool renamed only Monday. His reply asking for the whole week and
  "from now on" was lost to a deploy restart from the assistant's session, not a
  fault in Nova. The system log shows `systemctl restart miles-voice miles-server`
  with no terminal attached at 00:34:16, 00:35:26 and 00:36:05, each landing
  inside his conversation. The first dropped that reply and the proposal waiting
  for it, so the next confirm found nothing; the other two each swallowed a turn
  he then had to repeat.
- Nova then said the other lessons were "already spelled correctly". They were not.
- She asked "Rename Charley lesson on Wednesday at 4 PM to Charley lesson?" herself,
  with nothing staged. He said yes, and the tool failed looking for a Charley that
  did not exist yet.
- Every read back said "Charlie to Charley", which sounds like no change.

Nothing was remembered about the spelling despite "from now on".

**The lesson, for whoever deploys:** a restart ends the turn in progress and wipes
any proposal held in memory. Check the voice journal for recent speech before
restarting, and do not restart while he is talking to her.

Fixed by `rename_calendar_events` (every event with the name, one question), by
spelling a rename whose words sound alike, by prompt lines against asking change
questions herself or refusing without checking, and by a log line when she still
asks one. See [BRAIN.md](BRAIN.md#renaming-a-name-everywhere-and-spelling-what-cannot-be-heard).

---

## His yes created a date he did not hear (Sep 14 2026)

Just after midnight he asked Nova to add a career fair today, 1 to 6. It took
four exchanges and ended on the wrong week.

```
00:11:54  He:   Can you add career fair today from one to six?
00:12:05  Nova: asks whether he meant today, since it is just after midnight
00:12:13  create_calendar_event "monday at 1pm"  -> staged Monday September 21
00:12:34  Nova: Add career fair on Monday September 14 at 1 PM for five hours?
                (no tool call; her own words over the staged 21st)
00:12:39  confirm -> expired, a turn had passed since the proposal
00:15:30  create_calendar_event "Monday at 1 PM" -> staged Monday September 21
00:15:36  Nova: Add career fair on Monday September 14 at 1 PM for 300 minutes?
00:15:51  He:   yes -> "Added career fair."  on September 21
```

Three separate faults:

- **Today's weekday meant next week.** Day names prefer the future, and at 12:10 AM
  on a Monday that made "monday at 1pm" the 21st. It had been fixed for the
  planner's blocked times hours earlier and not for creating an event.
- **The spoken question was not the staged one.** The tool returned "Monday
  September 21" and the instruction to ask exactly that. Nova said "September 14"
  instead. The confirmation guarantee covered what ran, not what he heard, and the
  gap between them was a model turn.
- **Friction on top.** She questioned "today" after midnight, said "300 minutes"
  for a five hour event, and asked a question once with no proposal behind it.

**The lesson:** a guarantee is only as strong as its weakest link to him. Staging
the change in code was not enough while the words he answered were still the
model's. The question is now spoken by code.

The event itself was moved to the 14th by hand the same night.

---

## A busy room held the microphone (Sep 13 2026)

Guests came over that evening and it became very hard to get Nova to respond to
"hey nova". The journal shows she was not failing to hear it. She was stuck
recording the room.

```
21:32:57  Wake word detected! (0.54)        a false wake from the conversation
21:33:52  Recorded 53.3s                    their conversation, to the cap's edge
21:34:07  Wake word detected! (0.41)
21:34:18  You: No way. Hey Nova. Hey Nova.  judged not addressed to her, ignored
21:34:21  Wake word detected! (0.71)
21:34:40  sudo systemctl restart miles-voice   (him, mid recording)
21:36:09  NOTE: hit the 60s recording cap
21:37:07  sudo systemctl restart miles-voice   (him, again)
```

Three causes, the first introduced hours earlier:

- **`MAX_RECORD` had just been raised to 60.** webrtcvad counts anyone talking as
  speech, so in a busy room the silence that ends a recording never comes and the
  cap is the only end. At 18 the room held the mic for 18 seconds; at 60, a minute.
- **Nothing listened for the wake word during a recording.** His "hey nova" landed
  inside the clip being recorded.
- **The transcript check for the wake phrase only looked at the start.** A
  transcript that opened on the room and ended on "Hey Nova. Hey Nova." went to
  Claude, which correctly decided it was not addressed to her.

Nothing crashed: both restarts were his, and `NRestarts` stayed 0. Fixed by a 30
second cap, a second wake model listening during recordings, and the wake phrase
being found anywhere in a transcript; see
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#the-cap-came-down-to-30-and-the-wake-word-interrupts-a-recording).

**The lesson:** the long recording fix was tested against his own long request
and not against the room. Every change to how a recording ends needs checking in
the noisiest place it will be used, not only the quietest.

---

## Confirming one lesson created another (Sep 13 2026)

Between 20:21 and 20:28 he tried to schedule seven tutoring lessons by voice. It
went wrong in four separate ways, and the first was a defect in the confirmation
guarantee itself.

### His yes ran a change he had not been asked about

`pending_action` held one proposal, and a new one replaced it. In a single turn
Nova called `create_calendar_event` seven times (`tool_call_log` rows 87 to 93),
so each call replaced the one before and only the seventh survived. She asked the
question from the first:

```
20:26:58  Nova: Add Isaiah lesson 1 on Tuesday at 3:30 PM for 90 minutes?
20:27:17  He:   Yes, schedule it.
row 94    confirm_pending_action -> "Added Charlie lesson 3."
```

He then found a Charlie lesson on Wednesday he had never agreed to and deleted it.
This is precisely what confirmation exists to prevent. **Several proposals on one
turn now join one batch**, asked as one question and confirmed or cancelled
together; a proposal on a later turn still replaces the old one. It also removes
the one at a time asking he found unusable.

### "Done." over a cancel that did nothing

"Cancel all of the tutoring sessions" was sent to `cancel_reminder` three times
(rows 95 to 97), because lessons are calendar events and there was no way to
delete several at once. Every call returned "No active reminders found". Nova
said "Done.", because the fallback for tools that do not return to the model
speaks "Done." whenever it has nothing else, without looking at the result.

### Two recordings hit the cap

His first and third turns logged `NOTE: hit the 18s recording cap`, and the third
transcript ends mid thought at "also look at the club event". The cap sits below
Whisper's 20 second audio context, so audio past it would not be transcribed even
if it were recorded.

### The scheduling was done in the model's head

Every constraint was spoken and none was enforced. He said lessons had to be
"later than three" and was offered 2 PM; Andrew was put in the Thursday class he
had just described; two lessons overlapped each other on Tuesday; three ninety
minute sessions became "two hours forty five minutes"; Saturday September 20 was
read as Sunday. Nova also cited "your long lifting day" on Tuesday from a planned
split that memory itself said had not started; that memory was deleted at his
request.

**The lesson:** the arithmetic and the constraint checking were all left to the
model, which is the division of labour this project exists to get right. Code
should place the lessons; the model should turn what he said into constraints.

---

## Nova read the calendar like a printout (Sep 13 2026)

At 19:34 he asked "Do you know what my week is looking like?" `tool_call_log` row 69
shows Nova asked for `time_min: "Sunday, September 13, 2026"`. She answered:

> "You've got test prep sessions today at three and four thirty, then Monday gets
> busy. You have LeetCode at ten a.m., PBP at noon, three IEEE events across the
> afternoon and evening, and David's birthday is all day."

Three separate defects, all in what the tool handed her:

- **Past events as upcoming.** A bare date resolves to midnight, so a listing
  asked for at 7:34 PM included the 3 PM and 4:30 PM sessions he had already
  attended. The listing now starts from now, never earlier.
- **"All day" read out as if it were a time.** The tool wrote
  `Monday September 14, all day: David Farina's birthday`, and the label was
  spoken. All day events are now written as a day and a title.
- **Club events read as plans.** Three UF IEEE events sat alongside his own
  sessions, from calendars he keeps so he has options, not commitments. Google
  already marks the difference: his calendars are `owner`, followed ones are
  `reader`. Followed events now come in their own section with an instruction to
  mention them only when he asks what is going on.

The same answer shows her reading his own email address as a calendar name was
headed the same way; that label is dropped for his own calendars.

---

## Calendar and sleep tools answered confidently and wrong (Sep 13 2026)

The first day of the calendar and Oura tools produced three wrong answers that
each looked like a right one. None raised an error. The first was caught by ear;
the other two were found reading the code, then confirmed in `tool_call_log` and
by running the parser against real phrases.

### An hour and forty minutes of sleep, on a night of ten forty three

Asked how he slept, Nova said about an hour and forty minutes. The ring showed
10h 43m. `tool_call_log` row 58, 15:45:20, holds what she was given:

```
get_oura_sleep -> {'sleep_score': 89, 'total_sleep': 100, 'deep_sleep': 98,
                   'rem_sleep': 82, 'efficiency': 93}
```

Those are Oura's **contributor scores** from `daily_sleep`, each out of a
hundred, under names that read as durations. `total_sleep: 100` became a hundred
minutes. The tool now reads the `sleep` periods, and every field carries its
unit: `sleep_score_out_of_100`, and durations as words. A regression test feeds
it the exact contributor block.

### "Tomorrow" started at the current time, tomorrow

`tool_call_log` row 55, 14:46, asked for `time_min: "tomorrow"`. dateparser fills
a missing time of day with the current one, so the listing began at 14:46 on
Sep 14. Row 54, eleven minutes earlier and unwindowed, shows a **12:00 PM event
on Sep 14** that row 55 does not contain. It was dropped, and the day was
reported as if complete.

Found beside it by running phrases on a Sunday afternoon: `"monday"` resolved to
the Monday **before** in listings and freebusy, and to the Monday after in event
creation, the only path that set future preference. "Am I free Monday" checked
last week while "book Monday" wrote to next week.

### Every failure would have read as an empty calendar

Each per calendar fetch sat inside `except Exception: pass`, and the calendar
list fell back to primary on any error. An expired token would have failed every
fetch silently and produced "No upcoming events found on any calendar". Not
observed in production; closed before it could be. Failures now raise, a single
unreadable calendar is named in the answer, and only all of them failing is an
error.

### The fixed tool was never called (18:09)

After the sleep fix and a restart, "How did I sleep?" still got "one hour and
forty minutes". `tool_call_log` has no `get_oura_sleep` row for that turn. Nova
answered from `conversation_history` rows 949 to 953, where she had given the
wrong figure, he had repeated it back, and she had said "That's correct." To the
model that read as an established fact about him. **Fixing a tool does not fix
what the model already said about it.** The prompt now requires live data to be
fetched again every time it is asked for.

### A confirmation that said everything twice (18:10)

History row 959, as spoken: "LeetCode session is Monday at ten a.m. Moving it to
four p.m. on Monday.LeetCode session is currently Monday at ten a.m. to eleven
thirty a.m. I'll move it to four p.m. to five thirty p.m. that same day. Does that
work?" Nova announced the move before the tool ran, then followed the tool's
instruction to read back both versions in full. The missing space came from
`brain.py` joining tool rounds without one. The question is now built in code and
asked as given, and the rounds are joined with a space.

**The shared lesson:** each was a tool handing the model something shaped like
an answer. Every fix moves the same direction, toward code that states exactly
what a value is, or refuses.

## The wake word stopped separating in the new room (Sep 8 2026)

Sep 8 2026. "It responds even though I haven't said hey nova" turned out to be
two independent defects.

### Nova said "Done." out loud when she correctly stayed silent (Sep 8 2026)

`ignore` and `dismiss` carry `returns_to_model=False`, the same as timers and
reminders, so they landed in the branch holding the fallback that exists to stop
a timer being set in silence. `NOT_ADDRESSED_TO_YOU` tells Nova to say nothing
at all when speech was not aimed at her, so **correct behaviour produced an
empty `spoken_parts`, and the fallback then filled it.** The better she behaved,
the more certain she was to speak.

Caught by reading three sources for the same second:

```
01:27:38  Wake word detected! (0.50)     <- not "hey nova"
01:27:50  You: over here
01:27:51  tool_call_log: ignore          <- correct
01:27:52  journal: "Not addressed to Nova, staying quiet."
01:27:52  conversation_history: assistant "Done."   <- spoken anyway
```

`Permission.CONTROL` is the distinction, and its docstring already stated it:
a control tool changes Nova's own conversational state rather than doing work.
Work needs confirming, a state transition does not. The check is `all()` over
the called tools, not `any()`, so a turn that both sets a timer and says goodbye
still confirms the timer.

`save_message` is also skipped on an empty response now. A blank assistant row
is read back as context by the next twenty turns.

### No threshold separates true from false wakes any more (Sep 8 2026)

CLAUDE.md recorded the old room as separating cleanly, successful wakes from
0.520 up, failures topping out at 0.365, **an empty band between**. That was
measured Aug 12 in the old room and it does not hold here:

| score | what followed | real? |
|---|---|---|
| 0.75 | "Oh shoot, this is weird." | false |
| 0.64 | "(beep)" | false |
| 0.50 | "over here" | false |
| 0.46 | "We'll see you later." | false |
| 0.43 | a genuine conversation | true |

**It now overlaps in both directions.** Raising the threshold would have blocked
the real wake at 0.43 and kept the false one at 0.75. This is a model problem,
not a threshold problem, and `hey_nova.onnx` is dated Apr 8 2026, predating both
the gain tuning and this apartment. Five events is not a distribution; the
direction is unambiguous.

The capture that this failure made necessary, and the floor change that
followed, are documented as live mechanism in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#wake-capture).

### The near misses are the neighbours (Sep 13 2026)

Labelled Sep 13 2026. **All 29 wake_misses above 0.20 are `not_wake_phrase`**,
judged by ear: other people talking through the apartment wall. Not one is an
attempt Nova failed to hear.

That alone rules out the reading this capture was built to test. The misses are
not him being ignored, they are the building.

The session then produced a better result by accident. Playing the clips back
through the speakers, while `miles-voice` was still listening, made Nova wake:

| | score | peak |
|---|---|---|
| original, through the wall | 0.337 to 0.392 | -40 to -48 dBFS |
| same content replayed | **0.474, 0.586, 0.687** | -31 to -35 dBFS |

**Identical content, roughly 10 to 13 dB louder, and it crosses the threshold.**
Every one of those clips had just been labelled by ear as containing no wake
phrase at all.

So the near misses and the false wakes are **one population, not two**:
neighbour speech, sorted by how loud it happens to arrive. That reframes the
journal sample of false wakes at 0.43, 0.46, 0.50, 0.64 and 0.75, which were
read as a stale model and are better read as the wall.

Consequences worth keeping:

- **Raising `WAKE_THRESHOLD` cannot fix this.** The same content spans 0.34 to
  0.69 on level alone, and a real wake was observed at 0.43.
- **This is an argument for the array, and a strong one.** The failure is a
  second population of talkers at a fixed bearing that Nova has no other way to
  reject. A steerable beam or a null aimed at the wall addresses the cause;
  every software threshold addresses the symptom.
- The direct versus reverberant null test in `docs/BACKEND_TODO.md` stopped
  being optional. It measures exactly this path.
- **Speaker verification does not catch it, and `ignore` is not evidence that
  it does.** This bullet claimed the opposite until Sep 19 2026. See the
  correction below.

> **Correction, Sep 19 2026.** Over the 30 days to that date, `verification_log`
> holds 148 attempts and **one** rejection. Every one of the 13 turns that ended
> in `ignore` had passed verification: 12 scored 0.61 to 0.82 against a
> threshold of 0.5, on 2 to 20 seconds of embedded audio, and the 13th was
> accepted on session trust. Two of them: "We can go to the office, no, no, no,
> no, but AI is just a challenge fo...", "between what caused a discolored
> crater...".
>
> Mostly his own voice, in the room, not addressed to Nova, and the voiceprint
> accepts them because it is built to. **`ignore` is doing all of the catching,
> alone, one Claude call per false wake.** Not all of them are false wakes
> either: "cancel the timer." at 0.79 was a real command that `ignore` swallowed,
> which is the Sep 16 2026 entry above.
>
> So the wall is not the only false wake population. There is a second one that
> no microphone can separate, because the sound really is the registered
> speaker. The array answers the neighbours; it does not answer this. The two
> need different fixes and the counts above are the only numbers that currently
> separate them.

Artifacts from the playback are quarantined in `data/wake_playback/` with a
README. They are the measurement, not samples, and must never reach a retrain.

---

## The mixer card guard measured the speakers (Sep 6 2026)

**The mixer card is resolved by name, not hardcoded.** `MIC_MIXER_CARD` was
`"0"` until Sep 6 2026, and card 0 on this Pi is the AB13X speaker adapter, not
the Razer. So the startup check read the speakers' capture input, found 255
against an expected 23, and printed a drift warning at every single service
start, while the mic itself sat correctly at 23. The remedy it printed,
`amixer -c 0 sset Mic 23`, is nine percent on that device's 0 to 255 scale; the
microphone's own scale is 0 to 31, where 23 is 74 percent and 7.00dB. The two
devices do not share a scale, so the number was not read off the wrong card so
much as it did not mean the same thing there.

The damage was not the wrong reading. It was that a guard built to be
impossible to ignore fired on every boot and so became impossible to notice.
`config._resolve_mixer_card` now finds it by `MIC_NAME_HINT`, the same way the
mic is found by name in PyAudio and the speaker in `tts.py`, and returns `None`
on a miss rather than falling back, because a fallback silently measures
different hardware and reports it as the microphone.

---

## Two guards that could not do their jobs (Sep 6 2026)

Both fixed Sep 6 2026, both the same shape: a device reference that silently
changed meaning between boots, sitting inside the thing that was supposed to
catch exactly that.

### `MIC_NAME` carried the ALSA card number

`voiceprint_samples.mic` exists so samples gathered on one capsule are never
folded into a centroid built on another, which is the same class of mistake that
poisoned the April voiceprint. PyAudio's device name ends in the hardware
address, and the card number inside it shifts on reboot, so ten samples from one
physical Razer had recorded themselves as **three different microphones**:
`hw:0,0`, `hw:1,0`, and `hw:3,0`.

`get_voiceprint_samples` filters on an exact string, so a recompute scoped to
"this microphone" would have used a third of the samples and reported nothing
unusual. `config.capsule_name` now strips the address, and **migration 023**
collapsed the rows already written.

It lives in `config.py` rather than `audio.py` because importing `audio` opens
PyAudio and takes the exclusive mic lock, so nothing outside the voice process
can touch it, tests included.

### `enroll.py` did not keep the audio

It wrote every sample to one temp file and overwrote it, saving only embeddings
to `enrollment.npz`. **An embedding is locked to the encoder that made it.**
Resemblyzer is 256 dimensions, ECAPA is 192, and there is no conversion, so
swapping encoders meant re recording all twelve samples, and would again for the
encoder after that.

This is the April lesson one step further on. That failure was that only the
mean was saved, so a bad sample could not be identified afterwards; keeping the
individual embeddings fixed it for analysis and left the same hole for re
embedding. **Audio is the only artifact every future encoder can read.**

Recordings now go to `ENROLLMENT_AUDIO_DIR` (`models/enrollment_audio/`,
gitignored), and the npz gains `audio_files` and `encoder` so the mapping is
explicit in the file rather than implied by sort order.

**They are written at the end, in one step, from the same list the embeddings
came from.** Writing them per sample would leave a previous run's `sample_07`
beside a fresh `sample_00` through `sample_05`, which is silently wrong and
looks completely normal. Correspondence is structural, not remembered.

**This must land before the ECAPA swap**, or enrollment gets recorded twice.

---

## An empty room drove a runaway conversation loop (Aug 10 2026)

whisper.cpp does not reliably emit `[BLANK_AUDIO]` for non speech input. Fed room
noise it hallucinates a short plausible token instead. The runaway conversation
loop on Aug 10 2026 was driven by an empty room transcribing as "over." three
times. `is_noise_transcript` in parsing.py matches a frozenset of known
hallucinations against the whole normalized transcript, never as a substring.

The defence is deliberately narrow. `is_noise_transcript` matches whole
normalized transcripts against a frozenset, never substrings, and it is not
the right tool for overheard speech that happens to transcribe as plausible
English. **Do not widen the frozenset to chase those**; `verify_voice` and the
`ignore` tool are what handle them. The reasoning is in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#whisper).
