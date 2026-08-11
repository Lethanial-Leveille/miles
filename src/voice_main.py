import time
import numpy as np

# audio import triggers mic/wake word hardware init and ALSA silencing
import audio
import timing
import tts
import actions
from brain import ask_nova
from database import init_db
from parsing import is_noise_transcript
from config import CHUNK, WAKE_THRESHOLD, MAX_FOLLOWUP_TURNS

# Wire the speak callback so timer/reminder alerts play audio
actions.set_speak_fn(tts.speak)

print("Starting M.I.L.E.S. v0.7...", flush=True)
audio.log_mic_gain()
print("Initializing database...", flush=True)
init_db()

print("\n=== M.I.L.E.S. v0.7 — Nova is online ===", flush=True)
print("Listening for 'hey nova'... (Ctrl+C to stop)\n", flush=True)

try:
    while True:
        raw       = audio.stream.read(CHUNK, exception_on_overflow=False)
        audio_arr = np.frombuffer(raw, dtype=np.int16)
        prediction = audio.wake_model.predict(audio_arr)

        for _, score in prediction.items():
            if score <= WAKE_THRESHOLD:
                continue

            print(f"Wake word detected! ({score:.2f})", flush=True)

            # Flush the buffer so the command starts clean after the wake word
            for _ in range(int(audio.RATE / CHUNK * 0.5)):
                audio.stream.read(CHUNK, exception_on_overflow=False)
            audio.wake_model.reset()

            tts.play_chime()
            timing.begin_turn('initial')
            wav_path  = audio.record_command()
            user_text = audio.transcribe(wav_path)

            if is_noise_transcript(user_text):
                print(f"No speech detected (transcript: {user_text!r}).\n", flush=True)
                timing.abandon_turn()
                continue

            print(f"You: {user_text}", flush=True)

            verify_result = audio.verify_voice(wav_path, transcript=user_text,
                                                turn_type='initial', wake_confidence=float(score))

            # No voiced audio is not an authorization failure, so it does not
            # get the intruder response.
            if verify_result == audio.NO_AUDIO:
                print("Nothing to verify.\n", flush=True)
                timing.abandon_turn()
                print("Listening for 'hey nova'...", flush=True)
                continue

            if verify_result == audio.REJECTED:
                print("Voice not recognized.", flush=True)
                timing.abandon_turn()
                tts.speak("[calmly] That capability requires voice authorization. I don't recognize your voiceprint.")
                audio.flush_input()
                print("Listening for 'hey nova'...", flush=True)
                continue

            start  = time.time()
            result = ask_nova(user_text)
            nova_response = result.text
            print(f"Nova: {nova_response}", flush=True)
            print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)
            timing.end_turn(transcript=user_text, response=nova_response)

            # Nova's own voice is buffered on the mic by now. Left in place it
            # trips the follow up window immediately.
            audio.flush_input()

            # ── Follow up conversation loop ──
            in_conversation = True
            followup_turns  = 0
            while in_conversation:
                if followup_turns >= MAX_FOLLOWUP_TURNS:
                    print(f"Follow up limit reached ({MAX_FOLLOWUP_TURNS} turns). "
                          "Returning to wake word.\n", flush=True)
                    break

                print("Listening for follow up... (10s timeout)", flush=True)
                timing.begin_turn('followup')
                followup_path = audio.listen_for_followup(timeout=10)
                followup_turns += 1

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
                    print(f"No speech detected (transcript: {followup_text!r}). "
                          "Returning to wake word.\n", flush=True)
                    timing.abandon_turn()
                    break

                print(f"You: {followup_text}", flush=True)

                # The session already authenticated on the initial command, so
                # a follow up too short to embed reliably is trusted rather
                # than scored. Only follow ups long enough to judge are judged.
                followup_result = audio.verify_voice(followup_path, transcript=followup_text,
                                                      turn_type='followup',
                                                      session_trusted=True)

                if followup_result == audio.NO_AUDIO:
                    print("Nothing to verify. Returning to wake word.\n", flush=True)
                    timing.abandon_turn()
                    break

                if followup_result == audio.REJECTED:
                    print("Voice not recognized on follow up.", flush=True)
                    timing.abandon_turn()
                    tts.speak("[calmly] I don't recognize that voice. Returning to standby.")
                    audio.flush_input()
                    in_conversation = False
                    break

                start  = time.time()
                result = ask_nova(followup_text)
                nova_response = result.text
                print(f"Nova: {nova_response}", flush=True)
                print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)
                timing.end_turn(transcript=followup_text, response=nova_response)

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
