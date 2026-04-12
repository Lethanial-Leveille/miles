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
import re
import sqlite3
import requests
import json
import anthropic
import threading
from datetime import datetime
from fishaudio import FishAudio
from fishaudio.utils import save
from openwakeword.model import Model
from resemblyzer import VoiceEncoder, preprocess_wav

# ── Config ──
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAKE_THRESHOLD = 0.4
VERIFY_THRESHOLD = 0.65
WHISPER_MODEL = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en.bin")
WHISPER_CLI = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")
TEMP_WAV = os.path.expanduser("~/miles/build/command.wav")
TEMP_RESPONSE = os.path.expanduser("~/miles/build/response.wav")
VOICE_ID = "158f6b9781b746ec8c334d9730d302f1"
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
DEFAULT_LOCATION = "Gainesville"
DB_PATH = os.path.expanduser("~/miles/data/miles.db")
speak_lock = threading.Lock()

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

# ── Database ──

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
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
            created_at TEXT NOT NULL,
            source_device TEXT NOT NULL DEFAULT 'pi'
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

def save_message(role, content, device="pi"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_history (role, content, created_at, source_device) VALUES (?, ?, ?, ?)",
        (role, content, datetime.now().isoformat(), device)
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
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]

# ── Parsing ──

def extract_memories(response_text):
    explicit_pattern = r'\[MEMORY-EXPLICIT:\s*(.+?)\]'
    implicit_pattern = r'\[MEMORY:\s*(.+?)\]'

    explicit_memories = re.findall(explicit_pattern, response_text)
    implicit_memories = re.findall(implicit_pattern, response_text)

    clean_response = re.sub(explicit_pattern, '', response_text)
    clean_response = re.sub(implicit_pattern, '', clean_response)
    clean_response = re.sub(r'  +', ' ', clean_response).strip()

    return clean_response, explicit_memories, implicit_memories

def extract_actions(response_text):
    param_pattern = r'\[ACTION:\s*(\w+)\s*\|\s*(.+?)\]'
    simple_pattern = r'\[ACTION:\s*(\w+)\s*\]'

    actions = re.findall(param_pattern, response_text)
    clean_response = re.sub(param_pattern, '', response_text)

    simple_actions = re.findall(simple_pattern, clean_response)
    clean_response = re.sub(simple_pattern, '', clean_response)
    clean_response = re.sub(r'  +', ' ', clean_response).strip()

    parsed_actions = []
    for action_type, params_str in actions:
        params = {}
        for param in params_str.split(','):
            if ':' in param:
                key, value = param.split(':', 1)
                params[key.strip()] = value.strip()
            else:
                params['value'] = param.strip()
        parsed_actions.append({"type": action_type.lower(), "params": params})

    for action_type in simple_actions:
        parsed_actions.append({"type": action_type.lower(), "params": {}})

    return clean_response, parsed_actions

# ── Actions ──

def get_weather(location=None):
    if not location:
        location = DEFAULT_LOCATION

    try:
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        geo_response = requests.get(geo_url, params={
            "q": location,
            "limit": 1,
            "appid": WEATHER_API_KEY
        })
        geo_data = geo_response.json()

        if not geo_data:
            return f"Could not find location: {location}"

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_response = requests.get(weather_url, params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "imperial"
        })
        data = weather_response.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Location: {location}. "
            f"Currently {description}, {temp} degrees F (feels like {feels_like} degrees F). "
            f"Humidity: {humidity} percent. Wind: {wind_speed} mph."
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"

def set_timer(duration_str):
    parts = duration_str.lower().strip().split()
    if len(parts) < 2:
        return "Could not parse timer duration."

    try:
        amount = int(parts[0])
    except ValueError:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
            "forty five": 45, "fifty": 50, "sixty": 60
        }
        amount = word_to_num.get(parts[0], 0)
        if amount == 0:
            return "Could not parse timer duration."

    unit = parts[1]
    if "hour" in unit:
        seconds = amount * 3600
    elif "min" in unit:
        seconds = amount * 60
    elif "sec" in unit:
        seconds = amount
    else:
        return f"Unknown time unit: {unit}"

    def timer_thread():
        time.sleep(seconds)
        print(f"\n*** TIMER DONE: {amount} {unit} ***")
        alert_text = f"[calmly] Lethanial, your {amount} {unit} timer is up."
        speak(alert_text)
        print("Listening for 'hey nova'...")

    thread = threading.Thread(target=timer_thread, daemon=True)
    thread.start()
    return f"Timer set for {amount} {unit} ({seconds} seconds)."

def set_reminder(content, due_time=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (content, due_at, created_at) VALUES (?, ?, ?)",
        (content, due_time, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    if due_time:
        try:
            due_dt = datetime.fromisoformat(due_time)
            delay = (due_dt - datetime.now()).total_seconds()
            if delay > 0:
                def reminder_thread():
                    time.sleep(delay)
                    print(f"\n*** REMINDER: {content} ***")
                    alert_text = f"[calmly] Lethanial, a reminder. {content}."
                    speak(alert_text)
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE reminders SET completed = 1 WHERE content = ? AND due_at = ?",
                              (content, due_time))
                    conn.commit()
                    conn.close()
                    print("Listening for 'hey nova'...")

                thread = threading.Thread(target=reminder_thread, daemon=True)
                thread.start()
            else:
                return f"That time has already passed. Reminder saved but won't trigger."
        except:
            pass

    return f"Reminder saved: {content}" + (f" (due: {due_time})" if due_time else "")

def cancel_reminder(content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "DELETE FROM reminders WHERE content LIKE ? AND completed = 0",
        (f"%{content}%",)
    )
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        return f"Removed {deleted} reminder(s) matching '{content}'."
    else:
        return f"No active reminders found matching '{content}'."

def execute_actions(actions):
    results = []
    for action in actions:
        if action["type"] == "weather":
            location = action["params"].get("location") or action["params"].get("value")
            result = get_weather(location)
            results.append({"type": "weather", "data": result})
        elif action["type"] == "timer":
            duration = action["params"].get("duration") or action["params"].get("value", "")
            result = set_timer(duration)
            results.append({"type": "timer", "data": result})
        elif action["type"] == "reminder":
            content = action["params"].get("content") or action["params"].get("value", "")
            due_time = action["params"].get("due", None)
            result = set_reminder(content, due_time)
            results.append({"type": "reminder", "data": result})
        elif action["type"] == "cancel_reminder":
            content = action["params"].get("content") or action["params"].get("value", "")
            result = cancel_reminder(content)
            results.append({"type": "cancel_reminder", "data": result})
    return results

# ── Audio ──

def record_command():
    print("Listening...")

    VAD_FRAME = 480
    SILENCE_THRESHOLD = 300
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

def verify_voice(wav_path):
    wav = preprocess_wav(wav_path)
    embedding = voice_encoder.embed_utterance(wav)
    similarity = np.dot(embedding, voiceprint) / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
    print(f"Voice similarity: {similarity:.3f}")
    return similarity >= VERIFY_THRESHOLD

def listen_for_followup(timeout=10):
    VAD_FRAME = 480
    SILENCE_THRESHOLD = 300
    SPEECH_THRESHOLD = 200
    timeout_chunks = int(timeout / 0.03)
    speech_detected = False
    silent_after_speech = 0
    chunks_for_silence = int(2.5 / 0.03)
    max_chunks = int(15 / 0.03)
    min_chunks = int(0.5 / 0.03)
    
    frames = []
    total_chunks = 0
    waiting_chunks = 0
    
    # Phase 1: wait for speech to start (up to timeout)
    while waiting_chunks < timeout_chunks:
        data = stream.read(VAD_FRAME, exception_on_overflow=False)
        audio_array = np.frombuffer(data, dtype=np.int16)
        energy = np.abs(audio_array).mean()
        waiting_chunks += 1
        
        if energy > SPEECH_THRESHOLD:
            # Speech started, switch to recording mode
            frames.append(data)
            speech_detected = True
            break
    
    if not speech_detected:
        return None  # Timeout, no one spoke
    
    # Phase 2: record until silence (same as record_command)
    total_chunks = 1  # already have one frame
    silent_chunks = 0
    
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
    
    duration = (waiting_chunks + total_chunks) * 0.03
    print(f"Follow up recorded {duration:.1f}s")
    
    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return TEMP_WAV

def speak(text):
    with speak_lock:
        try:
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

# ── Nova ──

def ask_nova(user_text):
    save_message("user", user_text)

    memories = get_memories()
    memory_block = ""
    if memories:
        memory_lines = [f"- {content}" for id, content in memories]
        memory_block = "\nCURRENT MEMORIES (things you know about Lethanial):\n" + "\n".join(memory_lines) + "\n"

    enhanced_prompt = SYSTEM_PROMPT + memory_block + """
MEMORY INSTRUCTION:
When Lethanial shares a personal fact, preference, habit, schedule detail, or anything worth remembering for future conversations, include it in your response wrapped in memory tags.

Use [MEMORY-EXPLICIT: ...] when Lethanial directly asks you to store something:
- "remember that my exam is Friday"
- "remind me to push my code tonight"
- "don't forget I switched to morning classes"

Use [MEMORY: ...] when Lethanial shares something worth remembering but didn't ask you to store it:
- "I just started watching Naruto"
- "my exam got moved to Thursday"
- "I hit 225 on bench today"

Do NOT tag retrieval questions like "do you remember when my exam is" or "what did I tell you about my schedule." Those are questions, not new information.

Do NOT mention the memory tags out loud. They will be silently extracted. Only tag genuinely useful facts, not every detail. Do not tag things already in your current memories.

ACTION INSTRUCTION:
When Lethanial asks for information or tasks that require an external service, include an action tag in your response. Available actions:

[ACTION: weather | location: City] — for weather requests. If no location specified, omit the location param and the default will be used.
[ACTION: timer | duration: 10 minutes] — for timer requests. Always include the duration param with a number and unit.
[ACTION: reminder | content: push code to GitHub | due: 2026-04-11T21:00:00] — for reminder requests. Due is optional and should be ISO format. If the user says "tonight" or "in an hour," calculate the actual datetime.
[ACTION: cancel_reminder | content: push code] — for canceling reminders. Match against the reminder content.

Example responses with action tags:
- "What's the weather?" → "[ACTION: weather] [calmly] Let me check on that."
- "Set a timer for 10 minutes" → "[ACTION: timer | duration: 10 minutes] [calmly] Timer is set."
- "Remind me to push my code tonight" → "[ACTION: reminder | content: push code to GitHub | due: 2026-04-11T21:00:00] [calmly] I'll remind you."
- "Remember to study for circuits" → "[ACTION: reminder | content: study for circuits] [calmly] Noted."
- "Never mind about the code reminder" → "[ACTION: cancel_reminder | content: push code] [calmly] Reminder removed."

Always include a brief spoken response alongside the action tag. For timers, reminders, and cancellations, the spoken response IS the final response. The action will be executed silently.

Do NOT invent weather data or any external data. Always use the action tag and wait for real data.
"""

    recent = get_recent_messages(20)

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=enhanced_prompt,
        messages=recent
    )

    nova_text = response.content[0].text

    clean_response, explicit_memories, implicit_memories = extract_memories(nova_text)
    for mem in explicit_memories:
        save_memory(mem, source="explicit")
    for mem in implicit_memories:
        save_memory(mem, source="implicit")

    clean_response, actions = extract_actions(clean_response)

    if actions:
        results = execute_actions(actions)

        needs_followup = any(r["type"] in ["weather"] for r in results)

        if needs_followup:
            data_block = "\n".join([f"{r['type'].upper()} DATA: {r['data']}" for r in results])

            followup_messages = recent + [
                {"role": "assistant", "content": clean_response},
                {"role": "user", "content": f"[SYSTEM: Here is the data you requested]\n{data_block}\nDeliver this information naturally as Nova. Stay in character. Keep it concise."}
            ]

            followup = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=followup_messages
            )

            clean_response = followup.content[0].text
            clean_response, _, _ = extract_memories(clean_response)
            clean_response, _ = extract_actions(clean_response)

    save_message("assistant", clean_response)
    return clean_response

# ── Initialize ──
print("Starting M.I.L.E.S. v0.5...")
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

# ── Main loop ──
EXIT_PHRASES = ["that's all", "thats all", "thanks nova", "thank you nova",
                "we're good", "were good", "goodbye", "good night",
                "that is all", "i'm done", "im done", "you're dismissed",
                "dismissed", "peace", "later", "that's it", "thats it",
                "all good", "we're done", "were done", "i'm good", "im good",
                "that'll be all", "nothing else", "nah i'm good", "nah im good"]

print("\n=== M.I.L.E.S. v0.6 — Nova is online ===")
print("Listening for 'hey nova'... (Ctrl+C to stop)\n")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        prediction = wake_model.predict(audio_array)

        for wake_word, score in prediction.items():
            if score > WAKE_THRESHOLD:
                print(f"Wake word detected! ({score:.2f})")

                for _ in range(int(RATE / CHUNK * 0.5)):
                    stream.read(CHUNK, exception_on_overflow=False)
                wake_model.reset()

                wav_path = record_command()

                user_text = transcribe(wav_path)
                if not user_text or "BLANK" in user_text or "silence" in user_text.lower():
                    print("Didn't catch that.\n")
                    continue

                print(f"You: {user_text}")

                if not verify_voice(wav_path):
                    print("Voice not recognized.")
                    nova_response = "[calmly] That capability requires voice authorization. I don't recognize your voiceprint."
                    speak(nova_response)
                    print("Listening for 'hey nova'...")
                    continue

                start = time.time()
                nova_response = ask_nova(user_text)
                api_time = time.time() - start
                print(f"Nova: {nova_response}")
                print(f"(API: {api_time:.2f}s)")

                start = time.time()
                speak(nova_response)
                tts_time = time.time() - start
                print(f"(TTS: {tts_time:.2f}s)\n")

                # ── Follow up conversation loop ──
                in_conversation = True
                while in_conversation:
                    print("Listening for follow up... (10s timeout)")
                    followup_path = listen_for_followup(timeout=10)
                    

                    if followup_path is None:
                        print("No follow up. Returning to wake word.\n")
                        in_conversation = False
                        break

                    followup_text = transcribe(followup_path)
                    if not followup_text or "BLANK" in followup_text or "silence" in followup_text.lower():
                        print("Didn't catch that.\n")
                        continue

                    print(f"You: {followup_text}")

                    # Check for exit phrases
                    cleaned = followup_text.lower().strip().rstrip('.')
                    if cleaned in EXIT_PHRASES:
                        print("Conversation ended by user.")
                        farewell = "[calmly] Understood. I'll be here if you need me."
                        speak(farewell)
                        in_conversation = False
                        break

                    # Verify voice on follow up too
                    if not verify_voice(followup_path):
                        print("Voice not recognized on follow up.")
                        nova_response = "[calmly] I don't recognize that voice. Returning to standby."
                        speak(nova_response)
                        in_conversation = False
                        break

                    start = time.time()
                    nova_response = ask_nova(followup_text)
                    api_time = time.time() - start
                    print(f"Nova: {nova_response}")
                    print(f"(API: {api_time:.2f}s)")

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