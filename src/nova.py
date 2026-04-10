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
import sqlite3
import re
from datetime import datetime
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

# ── Database ──
DB_PATH = os.path.expanduser("~/miles/data/miles.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'explicit',
            created_at TEXT NOT NULL,
            last_referenced TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            due_at TEXT,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

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

print("Initializing database...")
init_db()

print("Connecting to Claude...")
claude = anthropic.Anthropic()

print("Connecting to Fish Audio...")
fish = FishAudio()

print("Loading voice encoder...")
voice_encoder = VoiceEncoder()
voiceprint = np.load(os.path.expanduser("~/miles/models/voiceprint.npy"))
VERIFY_THRESHOLD = 0.72  # cosine similarity, might need to tune this


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
    SILENCE_LIMIT = 2.5
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

def save_memory(content, source="explicit"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memories (content, source, created_at) VALUES (?, ?, ?)",
        (content, source, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    print(f"Memory saved ({source}): {content}")

def get_memories(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_history (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_recent_messages(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    # Rows come back newest-first, reverse to get chronological order
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]

def ask_nova(user_text):
    # Save user message to database
    save_message("user", user_text)
    
    # Build memory context
    memories = get_memories()
    memory_block = ""
    if memories:
        memory_lines = [f"- {content}" for id, content in memories]
        memory_block = "\nCURRENT MEMORIES (things you know about Lethanial):\n" + "\n".join(memory_lines) + "\n"
    
    # Build enhanced system prompt
    enhanced_prompt = SYSTEM_PROMPT + memory_block + """
MEMORY INSTRUCTION:
If Lethanial shares a personal fact, preference, habit, or anything worth remembering for future conversations, include it in your response wrapped in [MEMORY: ...] tags. Examples:
- If he says "I just started watching Naruto" you might include [MEMORY: Lethanial is currently watching Naruto]
- If he says "my exam got moved to Thursday" include [MEMORY: Lethanial's exam moved to Thursday]
Do NOT mention the memory tag out loud. It will be silently extracted. Only tag genuinely useful facts, not every detail. Do not tag things already in your current memories.

If Lethanial explicitly says "remember" something, that will also be stored separately. You do not need to tag those.
"""
    
    # Get conversation history from database
    recent = get_recent_messages(20)
    
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=enhanced_prompt,
        messages=recent
    )
    
    nova_text = response.content[0].text
    
    # Extract implicit memories from response
    clean_response, implicit_memories = extract_memories(nova_text)
    for mem in implicit_memories:
        save_memory(mem, source="implicit")
    
    # Check for explicit "remember" commands
    lower_text = user_text.lower()
    if "remember" in lower_text:
        # Store everything after "remember" as explicit memory
        remember_phrases = ["remember that ", "remember ", "remember,"]
        memory_content = user_text
        for phrase in remember_phrases:
            idx = lower_text.find(phrase)
            if idx != -1:
                memory_content = user_text[idx + len(phrase):]
                break
        if memory_content.strip():
            save_memory(memory_content.strip(), source="explicit")
    
    # Save Nova's clean response to database
    save_message("assistant", clean_response)
    
    return clean_response

def extract_memories(response_text):
    pattern = r'\[MEMORY:\s*(.+?)\]'
    memories = re.findall(pattern, response_text)
    clean_response = re.sub(pattern, '', response_text).strip()
    # Clean up any double spaces left behind
    clean_response = re.sub(r'  +', ' ', clean_response)
    return clean_response, memories

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