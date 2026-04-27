import time
import numpy as np

# audio import triggers hardware init and ALSA silencing
import audio
import actions
from brain import ask_nova
from database import init_db
from config import CHUNK, WAKE_THRESHOLD, EXIT_PHRASES

# Wire the speak callback so timer/reminder alerts play audio
actions.set_speak_fn(audio.speak)

print("Starting M.I.L.E.S. v0.7...")
print("Initializing database...")
init_db()

print("\n=== M.I.L.E.S. v0.7 — Nova is online ===")
print("Listening for 'hey nova'... (Ctrl+C to stop)\n")

try:
    while True:
        raw       = audio.stream.read(CHUNK, exception_on_overflow=False)
        audio_arr = np.frombuffer(raw, dtype=np.int16)
        prediction = audio.wake_model.predict(audio_arr)

        for _, score in prediction.items():
            if score <= WAKE_THRESHOLD:
                continue

            print(f"Wake word detected! ({score:.2f})")

            # Flush the buffer so the command starts clean after the wake word
            for _ in range(int(audio.RATE / CHUNK * 0.5)):
                audio.stream.read(CHUNK, exception_on_overflow=False)
            audio.wake_model.reset()

            wav_path  = audio.record_command()
            user_text = audio.transcribe(wav_path)

            if not user_text or "BLANK" in user_text or "silence" in user_text.lower():
                print("Didn't catch that.\n")
                continue

            print(f"You: {user_text}")

            if not audio.verify_voice(wav_path):
                print("Voice not recognized.")
                audio.speak("[calmly] That capability requires voice authorization. I don't recognize your voiceprint.")
                print("Listening for 'hey nova'...")
                continue

            start        = time.time()
            nova_response = ask_nova(user_text)
            print(f"Nova: {nova_response}")
            print(f"(Total: {time.time() - start:.2f}s)\n")

            # ── Follow up conversation loop ──
            in_conversation = True
            while in_conversation:
                print("Listening for follow up... (10s timeout)")
                followup_path = audio.listen_for_followup(timeout=10)

                if followup_path is None:
                    print("No follow up. Returning to wake word.\n")
                    in_conversation = False
                    break

                followup_text = audio.transcribe(followup_path)
                if not followup_text or "BLANK" in followup_text or "silence" in followup_text.lower():
                    print("Didn't catch that.\n")
                    continue

                print(f"You: {followup_text}")

                cleaned = followup_text.lower().strip().rstrip('.')
                if cleaned in EXIT_PHRASES:
                    print("Conversation ended by user.")
                    audio.speak("[calmly] Understood. I'll be here if you need me.")
                    in_conversation = False
                    break

                if not audio.verify_voice(followup_path):
                    print("Voice not recognized on follow up.")
                    audio.speak("[calmly] I don't recognize that voice. Returning to standby.")
                    in_conversation = False
                    break

                start        = time.time()
                nova_response = ask_nova(followup_text)
                print(f"Nova: {nova_response}")
                print(f"(Total: {time.time() - start:.2f}s)\n")

            print("Listening for 'hey nova'...")

except KeyboardInterrupt:
    print("\nNova is going to sleep.")
    audio.stream.stop_stream()
    audio.stream.close()
    audio._audio.terminate()
