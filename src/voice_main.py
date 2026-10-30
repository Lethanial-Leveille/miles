import random
import re
import threading
from collections import deque
import time
import anthropic
import numpy as np

# audio import triggers mic/wake word hardware init and ALSA silencing
import alerts
import audio
import checkin
import netcheck
import phrasebank
import presence
import resume
import timing
import tts
import actions
import local_intent
import speculative_turn
from tools import registry, permits
from database import latest_message_id
import wake_gate
from brain import ask_nova, speak_what_was_left, TurnResult
from local_intent import TEST_PARAGRAPH
from database import init_db, log_wake_near_miss, save_message
from parsing import is_noise_transcript, split_after_wake_phrase
from config import (CHUNK, WAKE_THRESHOLD, WAKE_LOG_FLOOR, MAX_FOLLOWUP_TURNS,
                    FOLLOWUP_TIMEOUT, ACK_SPOKEN_CHANCE, RATE,
                    WAKE_MISS_FLOOR, WAKE_MISS_PREROLL_MS,
                    WAKE_LEVEL_FLOOR, WAKE_RULE_SCORE_BELOW, WAKE_RULE_PEAK_ABOVE,
                    PRESENCE_SENSOR, PRESENCE_PORT,
                    PRESENCE_BAUD, PRESENCE_SILENT_S, PRESENCE_ENERGY_LOG,
                    PRESENCE_ENERGY_WINDOW_S)

# Wire the speak callback so timer/reminder alerts play audio

_last_wake_log = [0.0]  # mutable so the loop can update it without global

# Rolling window of raw wake frames, so a near miss can keep the audio that
# produced it rather than only the number. Sized to hold the phrase itself,
# since the score arrives at the end of it.
_wake_window = deque(maxlen=max(1, int(WAKE_MISS_PREROLL_MS / (CHUNK / RATE * 1000))))
_last_wake_capture = [0.0]
_last_wake_gated = [0.0]

print("Starting M.I.L.E.S. v0.7...", flush=True)
audio.log_mic_gain()
print("Initializing database...", flush=True)
init_db()

# Reminders are fired by a poller reading the reminders table, not by a thread
# held from the moment they were set. Started here and nowhere else: the server
# process can create reminders but must never deliver them, because its alert
# queue has nothing draining it and a second poller would only race this one for
# the same rows.
#
# This is also the boot rearm. There is nothing to rearm, which is the point:
# anything that came due while the service was down is simply due when the first
# pass runs.
fired = actions.poll_reminders()
if fired:
    print(f"{fired} reminder(s) came due while Nova was offline.", flush=True)
actions.start_reminder_poller()

# Here rather than in the server for the reason the poller is: the alerts that
# presence will decide about are spoken from this process. Logging only for now.
if PRESENCE_SENSOR:
    presence.start_reader(PRESENCE_PORT, PRESENCE_BAUD, PRESENCE_SILENT_S,
                          PRESENCE_ENERGY_WINDOW_S if PRESENCE_ENERGY_LOG else None)

# Load the embedding model off the critical path.
#
# It takes about seven seconds to load and 34ms to query. Left lazy, that seven
# seconds lands on whichever turn first triggers retrieval, which is a turn the
# user is waiting through and which would otherwise have taken four. Boot is not
# latency sensitive and this is, so it moves here.
#
# A daemon thread rather than a blocking call: the wake word loop should be
# listening immediately, and a retrieval in the first few seconds is only as
# slow as it would have been anyway. If the model fails to load, hybrid search
# degrades to keyword and nothing raises.
def _warm_embeddings():
    try:
        import embeddings
        embeddings.get_model()
        # Encoding the intent examples is a separate cost from loading the
        # model, and it is lazy. Left to the first real command it lands on a
        # turn he is waiting through: measured at 10.8s against 100ms warm.
        local_intent.warm()
        print("Embedding model ready.", flush=True)
    except Exception as exc:
        print(f"Embedding model unavailable, retrieval is keyword only: {exc}",
              flush=True)


threading.Thread(target=_warm_embeddings, daemon=True).start()

print("\n=== M.I.L.E.S. v0.7 — Nova is online ===", flush=True)
print("Listening for 'hey nova'... (Ctrl+C to stop)\n", flush=True)

def _say_cached(key):
    """Cached audio first, live synthesis only if nothing is rendered yet.

    The order is not an optimization. In the offline case tts.speak needs the
    same network that just failed and would return silently, so the phrase bank
    is the only thing that can make a sound at all. The fallback exists for a
    bank that has not been rendered, not because it is expected to work."""
    if phrasebank.play(key) is None:
        tts.speak(phrasebank.PHRASES[key][0])


# Belt and braces for the ignore tool.
#
# The tool is the intended path and its description is explicit, but the model
# does not always take it: observed twice in a row saying "I'm not part of that
# conversation." as plain text with no tool call, which then left the follow up
# window open to catch the next sentence of the same overheard exchange.
#
# Matching on her own output is crude and is deliberately narrow. It only has to
# catch the phrasing she actually produces when she has recognised the situation
# and reached for words instead of the tool.
_NOT_ADDRESSED = re.compile(
    r"\b(not (a )?part of (that|this) conversation|"
    r"(wasn't|weren't|not) (talking|speaking) to me|"
    r"that wasn't (meant )?for me|not addressed to me)\b", re.I)


def reads_as_not_addressed(text):
    return bool(text) and bool(_NOT_ADDRESSED.search(text))


def _run_local(match, user_text):
    """Answer without Claude, leaving the same trace behind that Claude would.

    The two save_message calls are not optional bookkeeping. brain.py writes
    both sides of every turn, and a local turn that skipped them would leave a
    hole in history: "set a timer for ten minutes" followed by "make it fifteen
    instead" would reach Claude as a follow up whose subject it never saw."""
    print(f"(local intent: {match.name} {match.slots} @ {match.score:.2f})", flush=True)
    fallback, key, dismissed = local_intent.execute(match)

    save_message("user", user_text)
    # What history records is what she actually said, which is why play returns
    # the variant text rather than a bool. The fallback is only reached when
    # nothing is rendered for the key.
    #
    # A null key means the intent composed its answer rather than choosing one,
    # which weather does because temperature times condition is not a space that
    # can be enumerated and rendered. Checked explicitly instead of passing None
    # through to play and relying on it finding no files.
    spoken = (phrasebank.play(key, thanked=match.slots.get('thanked', False))
              if key else None)
    if spoken is None:
        spoken = fallback
        tts.speak(spoken)
    save_message("assistant", spoken)
    return TurnResult(text=spoken, dismissed=dismissed)


# A voice that does not verify as his is a guest, not an intruder: it gets
# answers from general knowledge, and nothing above genin. See GUEST_BLOCK.
GUEST_TIER = "genin"


def _local_allowed(match, tier):
    """Whether a local intent may answer for this tier.

    Local intents skip Claude, and so skip the permission gate in brain.py. A
    guest setting a timer or cancelling his reminder would be a WRITE at genin,
    so a guest's match runs locally only if its tool is allowed at that tier, or
    it is not a tool at all, like the time of day. Otherwise Claude answers, and
    the gate says it is his to grant."""
    if match.name == "go_on":
        # Not a tool, so the check below would let it through. What she had left
        # may be his: she was reading his calendar when he was cut off, and a
        # guest saying "go on" must not make her finish it.
        return tier is None
    if match.name == "test_paragraph":
        # His test tool, and it writes to his history; a guest gets Claude.
        return tier is None
    if tier is None:
        return True
    try:
        spec = registry.get(match.name)
    except Exception:
        return True
    return permits(spec, tier)


def _run_go_on(user_text):
    """He asked her to go on and something is held: finish it. None if nothing
    played, so Claude answers instead of leaving him in silence.

    Saves both sides for the reason _run_local does, and the assistant side is
    what she actually got through, so a second cut in is marked as one."""
    said = speak_what_was_left()
    if not said:
        return None
    print("(local intent: go_on)", flush=True)
    save_message("user", user_text)
    # Cut again, what is still unsaid belongs to this reply, so a later stop to
    # nothing finishes this row and not the one before it.
    resume.attach(save_message("assistant", said))
    return TurnResult(text=said, dismissed=False)


def _run_test_paragraph(user_text):
    """Read the fixed test paragraph as if it were a reply (docs/TESTING.md).

    Held as a remainder with nothing said and then finished, so it goes through
    the same consumer as a reply: the first sentence at once, barge in live, and
    what he cuts off held for a go on. Saved like _run_go_on, for the same reason."""
    resume.keep(TEST_PARAGRAPH)
    said = speak_what_was_left()
    print("(local intent: test_paragraph)", flush=True)
    save_message("user", user_text)
    resume.attach(save_message("assistant", said))
    return TurnResult(text=said, dismissed=False)


def resume_after_empty_stop():
    """She was stopped and what followed was not speech: finish her sentence.

    Returns the interruption that cut the resumed speech, or None. It is claimed
    here, as after any reply, because he may cut in on this too and the next
    recording must start from what he said."""
    if not speak_what_was_left(amend_history=True):
        return None
    interruption = audio.claim_interruption()
    if interruption is None:
        audio.flush_input()
    return interruption


def run_turn(text, named=False, tier=None, conversation_start=None, followup=False):
    """Run one turn, returning None if it could not be completed.

    The boundary is here rather than in brain.py because brain serves two
    callers whose failure needs are opposite: the server has to surface a
    failure as a 503 so the app can retry, and this loop has to stay alive and
    keep listening. A handler inside brain would have to choose one of those,
    and choosing this loop's answer would hand the server a fabricated response
    to store as a real assistant turn.

    Broad on purpose. The specific exception matters for the log, but no
    exception should be able to end the process: an unhandled one escapes to
    systemd, which restarts under Restart=always, and the room sees a chime
    followed by silence with no way to tell that anything went wrong."""
    # The Claude call a speculation may already have made during the endpoint
    # wait. ask_nova uses it only on an exact match. Cancelled in finally on
    # every way out, claimed or not: after a claim the stream has finished and
    # the cancel does nothing, and on any other exit it stops the generation.
    speculation = speculative_turn.take()
    try:
        # Local first, and inside the try rather than before it. A bad regex or
        # an unparseable duration in here is exactly the kind of exception the
        # boundary exists to absorb, and code sitting outside it would reopen
        # the hole this function was written to close.
        match = local_intent.classify(text)
        if match is not None and _local_allowed(match, tier):
            if match.name == "test_paragraph":
                return _run_test_paragraph(text)
            if match.name != "go_on":
                return _run_local(match, text)
            result = _run_go_on(text)
            if result is not None:
                return result
        return ask_nova(text, speculation=speculation, named=named, tier=tier,
                        followup=followup,
                        conversation_start=conversation_start)
    except anthropic.APIConnectionError as exc:
        # The exception says a connection failed, not which one. Ask locally so
        # she names the cause she has actually ruled everything else out for.
        cause = netcheck.diagnose()
        print(f"Turn failed, {cause}: {exc}", flush=True)
        timing.abandon_turn()
        _say_cached(cause)
        return None
    except Exception as exc:
        print(f"Turn failed ({type(exc).__name__}): {exc}", flush=True)
        timing.abandon_turn()
        _say_cached('error')
        return None
    finally:
        if speculation is not None:
            speculation.cancel()


def speak_pending_alerts():
    """Announce anything still queued, on its own.

    Only ever called from this loop, and only at points where the mic is not
    open for capture. That is the entire deferral mechanism: a background
    thread cannot tell an open mic from an idle room, so it does not get to
    decide when to talk. Fresh alerts are usually gone before this runs,
    having been folded into a response by brain.py instead."""
    for alert in alerts.take_for_speech():
        tts.speak(alert.text)
        audio.flush_input()
    print("Listening for 'hey nova'...", flush=True)


try:
    while True:
        # Idle: nothing is being recorded, so this is a safe moment to talk.
        if alerts.pending_count():
            speak_pending_alerts()

        raw       = audio.stream.read(CHUNK, exception_on_overflow=False)
        _wake_window.append(raw)
        audio_arr = np.frombuffer(raw, dtype=np.int16)
        prediction = audio.wake_model.predict(audio_arr)

        for _, score in prediction.items():
            if score <= WAKE_THRESHOLD:
                # Record only near misses. Twelve frames a second of 0.01 is
                # the model correctly ignoring an empty room; a 0.35 against a
                # 0.4 threshold is the case worth seeing. Rate limited so a
                # sustained near miss cannot flood the table.
                if score >= WAKE_LOG_FLOOR:
                    now = time.monotonic()
                    if now - _last_wake_log[0] >= 1.0:
                        _last_wake_log[0] = now
                        log_wake_near_miss(score, WAKE_THRESHOLD)

                # Captured on a lower floor than the log, because the misses
                # that matter most may be the ones scoring near zero, and those
                # are precisely what WAKE_LOG_FLOOR hides.
                if score >= WAKE_MISS_FLOOR:
                    now = time.monotonic()
                    if now - _last_wake_capture[0] >= 1.5:
                        _last_wake_capture[0] = now
                        audio.save_wake_miss(list(_wake_window), float(score))
                continue

            # Measured before anything else so that both sides of the gate
            # land in the journal, not only the rejections. wake_log recorded
            # failures alone and read as a cluster of attempts under the line,
            # which is how the threshold got lowered to 0.3 on bad evidence.
            level = wake_gate.window_peak_dbfs(list(_wake_window))
            level_text = f"{level:.1f} dBFS" if level is not None else "no level"

            # The score cannot separate a real wake from the neighbours: on 49
            # labelled hits the two overlap from 0.40 to 0.94. Level can, because
            # speech through a wall arrives quiet and the model never looks at
            # loudness. See docs/AUDIO_PIPELINE.md.
            #
            # Fails open on a window that is not yet full, which is the first
            # 2.5 seconds after a restart. Judging a real wake from a short
            # buffer is how a gate goes deaf for a reason nobody can
            # reconstruct afterwards.
            #
            # Second reason, the same consequences: a low score must also be
            # loud (WAKE_RULE). Two thirds of his labelled array wakes were not
            # him, and this is what his labels say can go without losing his.
            reason = None
            if wake_gate.too_quiet(level, len(_wake_window), _wake_window.maxlen):
                reason = f"under {WAKE_LEVEL_FLOOR:.1f} dBFS"
            elif wake_gate.low_score_needs_louder(score, level, len(_wake_window),
                                                  _wake_window.maxlen):
                reason = (f"a score under {WAKE_RULE_SCORE_BELOW} needs a peak "
                          f"over {WAKE_RULE_PEAK_ABOVE:.0f} dBFS")
            if reason:
                print(f"Wake ignored: {score:.2f} at {level_text}, {reason}.",
                      flush=True)
                # Rate limited on the same grounds as the miss capture: talking
                # through a wall is continuous, and one conversation should not
                # evict the directory it is being recorded into.
                now = time.monotonic()
                if now - _last_wake_gated[0] >= 1.5:
                    _last_wake_gated[0] = now
                    audio.save_wake_gated(list(_wake_window), float(score))
                # Clears the prediction smoothing so one rejected event does not
                # keep reporting above threshold on the frames after it.
                audio.wake_model.reset()
                continue

            print(f"Wake word detected! ({score:.2f} at {level_text})", flush=True)

            # Saved before anything else touches the stream. _wake_window holds
            # the audio that produced this score, and it is the only record of
            # what actually triggered a wake: the recording archive starts
            # after this point and so contains the command, not the trigger.
            # Without it a false wake is a number with nothing behind it, which
            # is exactly the hole wake_log had on the other side.
            audio.save_wake_hit(list(_wake_window), float(score))

            # Flush the buffer so the command starts clean after the wake word
            for _ in range(int(audio.RATE / CHUNK * 0.5)):
                audio.stream.read(CHUNK, exception_on_overflow=False)
            audio.wake_model.reset()

            # The chime is a tone, and webrtcvad at mode 2 does not read a tone
            # as speech, so it overlaps capture harmlessly. A spoken ack does
            # read as speech: overlapped, it endpoints the recording on Nova's
            # own voice and puts her at the head of the clip Resemblyzer scores
            # against his voiceprint. So it plays to completion and the buffer
            # is flushed before the mic opens, which costs its own length.
            #
            # play() returning False when nothing is rendered is what makes the
            # chime the fallback: an unrendered bank degrades to today.
            if (random.random() < ACK_SPOKEN_CHANCE
                    and phrasebank.play('ack') is not None):
                audio.flush_input()
            else:
                tts.play_chime()

            timing.begin_turn('initial')
            wav_path  = audio.record_command(on_wake=tts.play_chime)
            recording = audio.archive_recording(wav_path, 'initial')
            user_text = audio.transcribe(wav_path)

            # Only the wake word, usually said again because the first recording
            # caught the room. She is listening; record once more for the
            # command instead of sending "Hey Nova. Hey Nova." to Claude.
            said_wake, remainder = split_after_wake_phrase(user_text)
            if said_wake and not remainder:
                print("Only the wake word, listening again.", flush=True)
                tts.play_chime()
                wav_path  = audio.record_command(on_wake=tts.play_chime)
                recording = audio.archive_recording(wav_path, 'initial')
                user_text = audio.transcribe(wav_path)
                said_wake, remainder = split_after_wake_phrase(user_text)
            if said_wake and remainder:
                user_text = remainder

            if is_noise_transcript(user_text):
                print(f"No speech detected (transcript: {user_text!r}).\n", flush=True)
                timing.abandon_turn()
                continue

            print(f"You: {user_text}", flush=True)

            verify_result = audio.verify_voice(wav_path, transcript=user_text,
                                                turn_type='initial', wake_confidence=float(score),
                                                recording_path=recording)

            # No voiced audio is not an authorization failure, so it does not
            # get the intruder response.
            # An ambiguous score is the model saying it does not know, and the
            # honest response is to ask rather than to silently ignore him.
            # Three of five real rejections sat within 0.05 of the threshold,
            # and the repeat is usually longer than the original, which is
            # itself the thing that fixes the score.
            if verify_result == audio.RETRY:
                print("Ambiguous voice match, asking to repeat.\n", flush=True)
                tts.speak("Sorry, say that again?")
                timing.abandon_turn()
                print("Listening for 'hey nova'...", flush=True)
                continue

            if verify_result == audio.NO_AUDIO:
                print("Nothing to verify.\n", flush=True)
                timing.abandon_turn()
                print("Listening for 'hey nova'...", flush=True)
                continue

            # Not recognized is not refused. Until Sep 26 2026 this spoke "I
            # don't recognize your voiceprint" and stopped; now someone else in
            # his home gets a guest's answer: general knowledge, nothing above
            # genin, and none of his history.
            session_is_him = verify_result != audio.REJECTED
            if not session_is_him:
                print("Voice not recognized, answering as a guest.", flush=True)
            speculative_turn.set_session(session_is_him)
            # Taken before this turn saves anything, so a guest's history is
            # this conversation and nothing before it.
            conversation_start = latest_message_id()

            # Late at night, his first turn starts the evening check in. It
            # fetches while she answers, and is said after, so his answer
            # never waits on it. Never for a guest: it reads his health data.
            checkin_job = checkin.start() if session_is_him else None

            start  = time.time()
            # He said the wake word to start this, by definition.
            result = run_turn(user_text, named=True,
                              tier=None if session_is_him else GUEST_TIER,
                              conversation_start=conversation_start)
            checkin.deliver(checkin_job)

            # Nothing to say and nothing to save. Back to the wake word rather
            # than into the follow up window, which would only collect more
            # speech that cannot be answered either.
            if result is None:
                audio.flush_input()
                print("Listening for 'hey nova'...", flush=True)
                continue

            # Not addressed to her. Say nothing and go back to standby rather
            # than announcing that she was not part of it, which is itself a
            # way of joining in and costs him a wait to hear.
            if result.ignored or reads_as_not_addressed(result.text):
                print("Not addressed to Nova, staying quiet.\n", flush=True)
                timing.end_turn(transcript=user_text, response=None)
                audio.flush_input()
                print("Listening for 'hey nova'...", flush=True)
                continue

            nova_response = result.text
            print(f"Nova: {nova_response}", flush=True)
            print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)
            timing.end_turn(transcript=user_text, response=nova_response)

            # Nova's own voice is buffered on the mic by now. Left in place it
            # trips the follow up window immediately. Unless he cut her off: then
            # what is buffered is him, kept by the barge in carrier.
            interruption = audio.claim_interruption()
            if interruption is None:
                audio.flush_input()

            # ── Follow up conversation loop ──
            in_conversation = True
            followup_turns  = 0
            while in_conversation:
                if followup_turns >= MAX_FOLLOWUP_TURNS:
                    print(f"Follow up limit reached ({MAX_FOLLOWUP_TURNS} turns). "
                          "Returning to wake word.\n", flush=True)
                    break

                # Between turns, before the window opens. Without this an
                # alert would wait for the whole conversation to end. Not over
                # an interruption: the flush after it would throw away what he
                # is still saying, so the alert waits one turn.
                if interruption is None:
                    for alert in alerts.take_for_speech():
                        tts.speak(alert.text)
                        audio.flush_input()

                print(f"Listening for follow up... ({FOLLOWUP_TIMEOUT}s timeout)", flush=True)
                timing.begin_turn('followup')
                followup_path = audio.listen_for_followup(timeout=FOLLOWUP_TIMEOUT,
                                                          on_wake=tts.play_chime,
                                                          seed=interruption)
                interrupted_turn, interruption = interruption is not None, None
                followup_turns += 1
                recording = (audio.archive_recording(followup_path, 'followup')
                             if followup_path else None)

                if followup_path is None:
                    print("No follow up. Returning to wake word.\n", flush=True)
                    timing.abandon_turn()
                    in_conversation = False
                    break

                followup_text = audio.transcribe(followup_path)

                # Noise ends the session instead of reopening the window. The
                # `continue` that used to be here is what made the loop self
                # sustaining: room noise tripped capture, transcribed to a
                # hallucinated token, drew a response, and opened another
                # window to be tripped again.
                if is_noise_transcript(followup_text):
                    timing.abandon_turn()
                    if interrupted_turn:
                        # She has already stopped for it. Ending here leaves him
                        # in silence; a cough that cut her off just times out.
                        print(f"Nothing usable in his interruption ({followup_text!r}); "
                              "still listening.", flush=True)
                        interruption = resume_after_empty_stop()
                        continue
                    print(f"No speech detected (transcript: {followup_text!r}). "
                          "Returning to wake word.\n", flush=True)
                    break

                # Saying the wake word during the window starts a fresh turn
                # rather than being transcribed into the middle of one. He gets
                # the chime that confirms she is listening, and the turn is
                # verified properly instead of accepted on session state, since
                # "hey nova" plus a command is long enough to embed.
                said_wake, remainder = split_after_wake_phrase(followup_text)
                if said_wake:
                    tts.play_chime()
                    if not remainder:
                        # Only the wake phrase. She is listening; wait for the
                        # rest rather than sending an empty turn to Claude.
                        print("Wake word during follow up, listening again.", flush=True)
                        continue
                    followup_text = remainder
                    print(f"Wake word during follow up, treating as a new turn.",
                          flush=True)

                print(f"You: {followup_text}", flush=True)

                # The session already authenticated on the initial command, so
                # a follow up too short to embed reliably is trusted rather
                # than scored. Only follow ups long enough to judge are judged.
                #
                # A wake word restart is not session trusted: it is a fresh
                # request and long enough to score, so it gets scored.
                #
                # Only his conversation is trusted on the session. A guest's
                # follow up is scored, so a guest conversation stays a guest's
                # unless he speaks long enough to be recognized.
                followup_result = audio.verify_voice(followup_path, transcript=followup_text,
                                                      turn_type='followup',
                                                      session_trusted=(not said_wake
                                                                       and session_is_him),
                                                      recording_path=recording,
                                                      learn=not interrupted_turn)

                if followup_result == audio.NO_AUDIO:
                    timing.abandon_turn()
                    if interrupted_turn:
                        # "Wait." and then a pause to think, Oct 2 2026: 0.48s
                        # of voice, too short to verify, and she went back to
                        # waiting for her name with him mid thought. She stopped
                        # for him, so she keeps listening for the rest.
                        print("His interruption was too short to answer; still listening.",
                              flush=True)
                        interruption = resume_after_empty_stop()
                        continue
                    print("Nothing to verify. Returning to wake word.\n", flush=True)
                    break

                if followup_result == audio.RETRY:
                    tts.speak("Sorry, say that again?")
                    continue

                # Whose conversation it is follows the latest judgement: a
                # rejected follow up is a guest's turn, not the end of the
                # conversation, and the short ones after it inherit that.
                session_is_him = followup_result != audio.REJECTED
                if not session_is_him:
                    print("Voice not recognized on follow up, answering as a guest.",
                          flush=True)
                speculative_turn.set_session(session_is_him)
                if said_wake:
                    # A fresh turn, so a guest's history starts over too.
                    conversation_start = latest_message_id()

                start  = time.time()
                result = run_turn(followup_text, named=said_wake, followup=True,
                                  tier=None if session_is_him else GUEST_TIER,
                                  conversation_start=conversation_start)

                if result is None:
                    audio.flush_input()
                    in_conversation = False
                    break

                # Overheard speech during the follow up window, which is where
                # this happens most: the window is open and the room is not.
                # Ends the conversation rather than reopening it, so an ongoing
                # exchange nearby cannot hold her attention turn after turn.
                if result.ignored or reads_as_not_addressed(result.text):
                    print("Not addressed to Nova, returning to standby.\n", flush=True)
                    timing.end_turn(transcript=followup_text, response=None)
                    audio.flush_input()
                    in_conversation = False
                    break

                nova_response = result.text
                print(f"Nova: {nova_response}", flush=True)
                print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)
                timing.end_turn(transcript=followup_text, response=nova_response)

                interruption = audio.claim_interruption()
                if interruption is None:
                    audio.flush_input()

                # Nova decided the conversation was over and said so in her own
                # words. She has already spoken the farewell, so there is
                # nothing to add here.
                if result.dismissed:
                    print("Conversation ended by user.\n", flush=True)
                    in_conversation = False
                    break

            print("Listening for 'hey nova'...", flush=True)

except KeyboardInterrupt:
    print("\nNova is going to sleep.", flush=True)
    audio.stream.stop_stream()
    audio.stream.close()
    audio._audio.terminate()
