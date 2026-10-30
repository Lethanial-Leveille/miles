# Testing a feature: examples first, the room last

Written Oct 30 2026, after the resume feature. It came with 550 lines of passing
tests and still said "Anyway, but most granola is heavy on sugar and oil" in the
room. The tests checked that the mechanism did what its author meant. Nobody had
looked at a real sentence she would say and asked whether a person says it.

**Every new feature goes up this ladder, in order.** Rungs 1 to 3 are
deterministic: the same input gives the same output on every run. Rung 4 cannot
be, because his timing, the transcript, the voice and the echo all vary; it is
scripted instead, with the inputs fixed and the expected result written down
before the run.

## 1. Examples, approved before code

Before any code, the behavior is written as concrete cases in plain text, each
an input and the exact output:

> Reply "It's got oats and nuts, which are solid, but most granola is heavy on
> sugar and oil.", cut at "but", 3 seconds later "go on": she says "It's got
> oats and nuts, which are solid, but most granola is heavy on sugar and oil."

He reads them and approves or corrects them. Cover the plain case, the edges
(nothing held, too late, a guest), and anything a person would find odd to
hear. This is the cheapest rung and the one that would have caught "Anyway, but".

## 2. The approved examples become tests

Each example is a test that asserts the exact output, not that something
happened. "A remainder was resumed" passes on "Anyway, but"; the exact sentence
does not.

## 3. Replay on his real data, offline

The Pi already keeps the evidence: the audio behind every stop in
`data/barge_in_stops`, every reply's text in the spoken archive, every turn in
`timing_log`. A replay script runs the new code over the last few weeks of real
cases and prints what she would have said, or would have decided, for each. He
reads the list. No mic, no room, same result every run. `scripts/wake_rule.py`
is the pattern.

Its limit: it only covers what has happened. A case the archive never caught
still needs rung 4.

## 4. The room, scripted and graded

The only rung that needs him. Everything that can be fixed is fixed:

- **A fixed reply** where the feature depends on what she says, so every run
  starts from the same words instead of whatever Claude says that day.
- **A numbered script.** Each step says what to say, word for word; when to say
  it, tied to a **word she says**, never a number of seconds, because a word can
  be heard and seconds cannot be counted while she talks; and what he should
  hear. A step looks like this:

  > **Step 3.** Say exactly: "Hey Nova, read the test paragraph."
  > **When** you hear her say "solid", say "Hey Nova."
  > **Then** stay silent until the chime, and say "go on."
  > **You should hear:** "It's got oats and nuts, which are solid, but most
  > granola is heavy on sugar and oil."
  > **Anything else:** note the step number and keep going.

- **Claude grades it, not him.** He notes the time he starts and says "done" at
  the end. Claude reads the journal from that time and reports every step as
  pass or fail, with the line that shows it. He judges how it felt; the log
  judges what happened.
- **Short.** Five steps at most, about five minutes, and only what rungs 1 to 3
  cannot reach: how it sounds, and the real timing of his voice over hers. A rule
  the tests already pin exactly (a two minute window, a guest) stays out of the
  room. The first script, for resume on Oct 30 2026, had eleven steps times
  three runs; he ran five, and called it too long.
- **Repeat only what failed.** One clean run of the script, then a rerun of any
  step that failed or that he was unsure of.
- **The result is written down,** dated, in the feature's entry in its topic doc:
  which steps passed, which failed, and what was changed.

## 5. Then main, then deploy

New features are built on a branch, `feature/<name>`, never pushed straight to
main. That includes cloud sessions. The room test runs on the branch:

```bash
cd ~/miles && git checkout feature/<name> && sudo systemctl restart miles-voice miles-server
```

`miles-deploy.timer` skips any branch but main, so it stays out of the way. When
rung 4 passes, the branch is merged to main, and the timer deploys it.

`scripts/replay_resume.py` is the rung 3 replay for resume. "Hey Nova, read the
test paragraph" (`TEST_PARAGRAPH` in `local_intent.py`) is the fixed reply for
any room test of barge in or resume; only the exact phrases match, so "the test
paragraph" goes to Claude. So main
means **tested in the room**, and the timer only ever ships that.

To go back to what is live: `git checkout main` and restart.

Small fixes (a typo, a doc, a threshold that was measured offline) can go
straight to main; the ladder is for anything he will hear or that changes what
she does.
