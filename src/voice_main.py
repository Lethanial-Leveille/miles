import time
import numpy as np

# audio import triggers mic/wake word hardware init and ALSA silencing
import audio
import tts
import actions
from brain import ask_nova
from database import init_db
from config import CHUNK, WAKE_THRESHOLD, EXIT_PHRASES

# Wire the speak callback so timer/reminder alerts play audio
actions.set_speak_fn(tts.speak)

print("Starting M.I.L.E.S. v0.7...", flush=True)
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
            wav_path  = audio.record_command()
            user_text = audio.transcribe(wav_path)

            if not user_text or "BLANK" in user_text or "silence" in user_text.lower():
                print("Didn't catch that.\n", flush=True)
                continue

            print(f"You: {user_text}", flush=True)

            if not audio.verify_voice(wav_path):
                print("Voice not recognized.", flush=True)
                tts.speak("[calmly] That capability requires voice authorization. I don't recognize your voiceprint.")
                print("Listening for 'hey nova'...", flush=True)
                continue

            start        = time.time()
            nova_response = ask_nova(user_text)
            print(f"Nova: {nova_response}", flush=True)
            print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)

            # ── Follow up conversation loop ──
            in_conversation = True
            while in_conversation:
                print("Listening for follow up... (10s timeout)", flush=True)
                followup_path = audio.listen_for_followup(timeout=10)

                if followup_path is None:
                    print("No follow up. Returning to wake word.\n", flush=True)
                    in_conversation = False
                    break

                followup_text = audio.transcribe(followup_path)
                if not followup_text or "BLANK" in followup_text or "silence" in followup_text.lower():
                    print("Didn't catch that.\n", flush=True)
                    continue

                print(f"You: {followup_text}", flush=True)

                cleaned = followup_text.lower().strip().rstrip('.')
                if cleaned in EXIT_PHRASES:
                    print("Conversation ended by user.", flush=True)
                    tts.speak("[calmly] Understood. I'll be here if you need me.")
                    in_conversation = False
                    break

                if not audio.verify_voice(followup_path):
                    print("Voice not recognized on follow up.", flush=True)
                    tts.speak("[calmly] I don't recognize that voice. Returning to standby.")
                    in_conversation = False
                    break

                start        = time.time()
                nova_response = ask_nova(followup_text)
                print(f"Nova: {nova_response}", flush=True)
                print(f"(Total: {time.time() - start:.2f}s)\n", flush=True)

            print("Listening for 'hey nova'...", flush=True)

except KeyboardInterrupt:
    print("\nNova is going to sleep.", flush=True)
    audio.stream.stop_stream()
    audio.stream.close()
    audio._audio.terminate()
