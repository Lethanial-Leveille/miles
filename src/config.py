import os
import threading

# ── Audio hardware ──
CHUNK = 1280          # 80ms frames, required by openWakeWord
CHANNELS = 1
RATE = 16000
WAKE_THRESHOLD = 0.4
VERIFY_THRESHOLD = 0.5

# ── Paths ──
WHISPER_MODEL   = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en.bin")
WHISPER_CLI     = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")
TEMP_WAV        = os.path.expanduser("~/miles/build/command.wav")
TEMP_RESPONSE   = os.path.expanduser("~/miles/build/response.wav")
WAKE_MODEL_PATH = os.path.expanduser("~/miles/models/hey_nova.onnx")
VOICEPRINT_PATH = os.path.expanduser("~/miles/models/voiceprint.npy")
DB_PATH         = os.path.expanduser("~/miles/data/miles.db")

# ── External services ──
VOICE_ID         = "158f6b9781b746ec8c334d9730d302f1"
WEATHER_API_KEY  = os.environ.get("WEATHER_API_KEY")
DEFAULT_LOCATION = "Gainesville"
SPEAKER_DEVICE   = "plughw:0,0"

# ── Shared state ──
speak_lock = threading.Lock()

# ── Conversation exit phrases ──
EXIT_PHRASES = [
    "that's all", "thats all", "thanks nova", "thank you nova",
    "we're good", "were good", "goodbye", "good night",
    "that is all", "i'm done", "im done", "you're dismissed",
    "dismissed", "peace", "later", "that's it", "thats it",
    "all good", "we're done", "were done", "i'm good", "im good",
    "that'll be all", "nothing else", "nah i'm good", "nah im good",
]
