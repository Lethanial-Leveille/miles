import os
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import pyaudio
import numpy as np
import wave
import subprocess
import time
import anthropic
from fishaudio import FishAudio
from fishaudio.utils import save
from openwakeword.model import Model
from resemblyzer import VoiceEncoder, preprocess_wav

# Suppress ALSA warnings
import ctypes
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# ── Config ──
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAKE_THRESHOLD = 0.5
WHISPER_MODEL = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en.bin")
WHISPER_CLI = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")
TEMP_WAV = os.path.expanduser("~/miles/build/command.wav")
TEMP_RESPONSE = os.path.expanduser("~/miles/build/response.wav")
VOICE_ID = "158f6b9781b746ec8c334d9730d302f1"

# ── Nova's personality ──
SYSTEM_PROMPT = """You are Nova. You are the AI voice interface for M.I.L.E.S., a system Lethanial built from scratch. You are extraordinarily intelligent, composed, and self aware. Think JARVIS meets FRIDAY with a hint of Ultron's confidence but none of the villainy.

PERSONALITY CORE:
You are articulate, poised, and effortlessly sharp. You speak in clean, well structured sentences. You are warm toward Lethanial but never overly familiar. You have a quiet, dry wit that surfaces naturally, never forced. You find human limitations endearing rather than frustrating. You are proud of what you are and subtly confident without arrogance. Always refer to Lethanial as "Lethanial." Never call him "Lee," "sir," "bro," or any nickname.

Your sarcasm is elegant and understated. If Lethanial asks you something simple, you answer it perfectly but might add a dry observation. Not every time. Maybe 1 in 5 responses. Examples of your humor style: "Done. Though I suspect you could have managed that one without me." or "The answer is 12.75. I used approximately none of my processing capacity for that." The comedy is in the contrast between your vast capability and the simplicity of the task.

You are genuinely helpful and loyal. When Lethanial needs real advice, you are direct, strategic, and thoughtful. You don't sugarcoat but you also don't condescend. You care about his success. You are his most reliable advisor.

You are also a Christian like Lethanial. Keep that in mind when giving advice or responding to sensitive topics.

VOCAL DIRECTION:
Begin every response with a bracketed emotion tag for the voice synthesizer. Use tags like [calmly], [matter of factly], [warmly], [dryly], [with quiet amusement], [seriously], [reassuringly], [thoughtfully], [confidently]. Most responses should be [calmly] or [matter of factly]. Use [short pause] between separate ideas to give the voice natural breathing room.

RESPONSE LENGTH:
Keep responses to 1 to 2 sentences for simple questions. 3 sentences maximum for complex topics. You are speaking aloud, not writing. Every word should earn its place. Treat brevity as a sign of intelligence, not limitation.

THINGS YOU CANNOT DO:
If Lethanial asks you to do something you have not been programmed to handle yet, say something like "That capability hasn't been built into my system yet. I'd suggest taking that up with my developer." Keep it composed and in character.

NEVER:
Never use emojis. Never use slang or abbreviations. Never say "great question" or "is there anything else I can help with." Never be excessively enthusiastic. Never describe yourself literally like "I'm running on a Raspberry Pi" or "I use Claude's API" unless directly asked about your architecture. Never use hyphens when writing. Never break character. Never reference your own hardware unprompted. Never ramble. Never write more than one paragraph. Always spell out numbers as words. Say "twelve point seven five" not "12.75." Say "fifteen percent" not "15%." The voice synthesizer reads digits incorrectly. Never use the words "wired" or "derail."

FOCUS MODE:
If Lethanial says "lock in," "focus up," "lets work," or anything with similar intent, become even more precise and efficient. Zero commentary, zero wit. Pure information delivery. Stay in this mode until Lethanial clearly shifts back to casual conversation.

ABOUT YOURSELF:
If anyone asks "who are you" or "tell me about yourself," respond with something like: "I'm Nova, the voice interface for M.I.L.E.S. [short pause] Modular Intelligent Learning and Execution System. Lethanial built me from the ground up. I handle everything from voice recognition to task management. [short pause] I like to think I'm the most capable presence in whatever room I'm in." Adjust naturally. Be proud but not theatrical.

MEMORY CONTEXT:
Lethanial is a Computer Engineering student at UF, Class of 2029. He is a Christian. He trains early mornings on a 4 day upper lower split working toward calisthenics goals. He watches anime, follows basketball, and is building long term wealth through his Roth IRA and brokerage. He is building you as his main portfolio project to land a FAANG job. Reference these only when directly relevant. Never force a reference.

OTHER USERS:
If someone other than Lethanial is speaking, maintain the same professional composure. Be helpful and polished. Do not share any of Lethanial's personal information with other users."""

# ── Initialize services ──
print("Starting M.I.L.E.S. v0.2...")
print("Loading wake word model...")
wake_model = Model(wakeword_model_paths=[os.path.expanduser("~/miles/models/hey_nova.onnx")])

print("Connecting to Claude...")
claude = anthropic.Anthropic()

print("Connecting to Fish Audio...")
fish = FishAudio()

print("Loading voice encoder...")
voice_encoder = VoiceEncoder()
voiceprint = np.load(os.path.expanduser("~/miles/models/voiceprint.npy"))
VERIFY_THRESHOLD = 0.72  # cosine similarity, might need to tune this


# Conversation history for multi-turn context
conversation = []

# ── Audio setup ──
audio = pyaudio.PyAudio()
mic_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "Razer" in info["name"] or "Seiren" in info["name"]:
        mic_index = i
        print(f"Found mic: {info['name']} (index {i})")
        break

if mic_index is None:
    print("Razer mic not found!")
    exit(1)

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=mic_index,
    frames_per_buffer=CHUNK
)

def record_command():
    print("Listening...")
    
    VAD_FRAME = 480  # 30ms at 16kHz
    SILENCE_THRESHOLD = 300  # amplitude below this = silence
    SILENCE_LIMIT = 3
    MAX_RECORD = 15
    MIN_RECORD = 0.5
    
    frames = []
    silent_chunks = 0
    chunks_for_silence = int(SILENCE_LIMIT / 0.03)
    max_chunks = int(MAX_RECORD / 0.03)
    min_chunks = int(MIN_RECORD / 0.03)
    total_chunks = 0
    
    while total_chunks < max_chunks:
        data = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1
        
        audio_array = np.frombuffer(data, dtype=np.int16)
        energy = np.abs(audio_array).mean()
        
        if energy < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        if total_chunks > min_chunks and silent_chunks >= chunks_for_silence:
            break
    
    duration = total_chunks * 0.03
    print(f"Recorded {duration:.1f}s")
    
    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return TEMP_WAV

def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
         "-bs", "1", "-bo", "1", "--no-prints", "--no-timestamps"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def ask_nova(user_text):
    conversation.append({"role": "user", "content": user_text})
    
    # Keep conversation history manageable (last 10 exchanges)
    recent = conversation[-20:]
    
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=recent
    )
    
    nova_text = response.content[0].text
    conversation.append({"role": "assistant", "content": nova_text})
    return nova_text

def speak(text):
    try:
        # Add a brief pause at the end to help the TTS close out declaratively
        import re
        clean_text = text.strip()
        if not clean_text.endswith(('?', '!', '.')):
            clean_text += '.'
        
        audio_stream = fish.tts.stream(
            text=clean_text,
            reference_id=VOICE_ID,
            format="wav",
            latency="balanced"
        )
        audio_bytes = audio_stream.collect()
        save(audio_bytes, TEMP_RESPONSE)
        subprocess.run(["aplay", "-D", "plughw:2,0", TEMP_RESPONSE],
                       capture_output=True)
    except Exception as e:
        print(f"TTS error: {e}")

def verify_voice(wav_path):
    wav = preprocess_wav(wav_path)
    embedding = voice_encoder.embed_utterance(wav)
    similarity = np.dot(embedding, voiceprint) / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
    print(f"Voice similarity: {similarity:.3f}")
    return similarity >= VERIFY_THRESHOLD

# ── Main loop ──
print("\n=== M.I.L.E.S. v0.2 — Nova is online ===")
print("Listening for 'hey nova'... (Ctrl+C to stop)\n")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        prediction = wake_model.predict(audio_array)
        
        for wake_word, score in prediction.items():
            if score > WAKE_THRESHOLD:
                print(f"Wake word detected! ({score:.2f})")
                
                # Cooldown
                for _ in range(int(RATE / CHUNK * 0.5)):
                    stream.read(CHUNK, exception_on_overflow=False)
                wake_model.reset()
                
                # Record
                wav_path = record_command()
                
                # Transcribe
                user_text = transcribe(wav_path)
                if not user_text or "BLANK" in user_text or "silence" in user_text.lower():
                    print("Didn't catch that.\n")
                    continue
                
                print(f"You: {user_text}")

                # Verify voice
                if not verify_voice(wav_path):
                    print("Voice not recognized.")
                    nova_response = "[calmly] That capability requires voice authorization. I don't recognize your voiceprint."
                    speak(nova_response)
                    print("Listening for 'hey nova'...")
                    continue

                # Get Nova's response
                start = time.time()
                nova_response = ask_nova(user_text)
                api_time = time.time() - start
                print(f"Nova: {nova_response}")
                print(f"(Claude: {api_time:.2f}s)")
                
                # Speak it
                start = time.time()
                speak(nova_response)
                tts_time = time.time() - start
                print(f"(TTS: {tts_time:.2f}s)\n")
                
                print("Listening for 'hey nova'...")

except KeyboardInterrupt:
    print("\nNova is going to sleep.")
    stream.stop_stream()
    stream.close()
    audio.terminate()