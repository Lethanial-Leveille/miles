import os
import threading
from elevenlabs import VoiceSettings

# ── Audio hardware ──
CHUNK = 1280          # 80ms frames, required by openWakeWord
CHANNELS = 1
RATE = 16000
WAKE_THRESHOLD = 0.4
VERIFY_THRESHOLD = 0.5

# ── Voice activity detection ──
# webrtcvad replaces the bare amplitude threshold that used to gate capture.
# Amplitude cannot tell speech from a running AC unit, which is what drove the
# runaway follow up loop on Aug 10 2026.
#
# Mode runs 0 (permissive) to 3 (aggressive about rejecting non speech).
# Resemblyzer uses 3 internally for offline trimming, where clipping a soft
# onset costs nothing. Live capture is less forgiving, so 2 here.
VAD_MODE = 2

# Audio retained from before speech onset is detected. Without this, the frames
# containing a soft leading consonant are discarded before capture starts, which
# is what truncated "What year do I graduate?" down to "year do I graduate?".
VAD_PREROLL_MS = 300

# Consecutive speech frames required to declare onset, so a single click or
# keyboard tap cannot open a recording.
VAD_ONSET_FRAMES = 2

# Discarded from the mic after Nova finishes speaking, on top of draining
# whatever accumulated during playback.
#
# aplay's ALSA buffer holds roughly 185ms after writing stops, and in a shared
# enclosure the mic will also pick up structure borne ring after the cone
# stops moving. Raise this once there is an enclosure to measure: record Nova
# speaking, find where her energy actually ends in the mic signal relative to
# when aplay exits, and set the margin above that.
TTS_FLUSH_MARGIN_MS = 250

# ── Enrollment ──
# Resemblyzer embeddings are unstable below roughly three seconds of voiced
# audio. The previous voiceprint was poisoned by a sample ("Lock in") that
# trimmed to well under a second, so this is enforced at record time.
MIN_VOICED_SECONDS    = 4.0
ENROLL_RECORD_SECONDS = 8

# ── Mic gain ──
# Tuned Aug 10 2026 to peak -10.3 dBFS on worst case close range projected
# speech, persisted with alsactl store. Checked at startup and logged, because
# a silent revert corrupts collected data in a way that only shows up days
# later as inexplicably low scores.
EXPECTED_MIC_GAIN    = 23
MIC_MIXER_CARD       = "0"
MIC_MIXER_CONTROL    = "Mic"

# ── Paths ──
WHISPER_MODEL   = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en.bin")
WHISPER_CLI     = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")
TEMP_WAV        = os.path.expanduser("~/miles/build/command.wav")
TEMP_RESPONSE   = os.path.expanduser("~/miles/build/response.wav")
WAKE_MODEL_PATH = os.path.expanduser("~/miles/models/hey_nova.onnx")
VOICEPRINT_PATH = os.path.expanduser("~/miles/models/voiceprint.npy")
# Individual enrollment embeddings plus condition labels, kept so the centroid
# can be recomputed or analyzed without re recording. The old enrollment saved
# only the mean, which is why a poisoned sample could not be identified later.
ENROLLMENT_DATA_PATH = os.path.expanduser("~/miles/models/enrollment.npz")
DB_PATH         = os.path.expanduser("~/miles/data/miles.db")

# ── External services ──

# ROLLBACK: Fish Audio config preserved for emergency rollback
# VOICE_ID = "158f6b9781b746ec8c334d9730d302f1"

# ElevenLabs TTS configuration (v0.7.1)
ELEVENLABS_API_KEY    = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID   = os.environ.get("ELEVENLABS_VOICE_ID")
DEFAULT_TTS_MODEL     = "eleven_flash_v2_5"
EXPRESSIVE_TTS_MODEL  = "eleven_v3"
TTS_OUTPUT_FORMAT     = "pcm_22050"
LOOKAHEAD_CHARS       = 50
ACTION_PREFIX         = "[ACTION:"

EMMA_NEUTRAL = VoiceSettings(
    stability=0.50, similarity_boost=0.75, style=0.00,
    use_speaker_boost=True, speed=1.00,
)

EMMA_WITTY = VoiceSettings(
    stability=0.30, similarity_boost=0.75, style=0.35,
    use_speaker_boost=True, speed=1.05,
)

EMMA_SERIOUS = VoiceSettings(
    stability=0.65, similarity_boost=0.80, style=0.00,
    use_speaker_boost=True, speed=0.95,
)

WEATHER_API_KEY  = os.environ.get("WEATHER_API_KEY")
DEFAULT_LOCATION = "Gainesville"

# ALSA card numbers shift between boots, so the speaker is resolved by name
# at runtime in tts.py rather than hardcoded here. This is just the name to
# search for in /proc/asound/cards.
SPEAKER_NAME_HINT = "AB13X"

# ── Shared state ──
speak_lock = threading.Lock()

# ── Conversation loop ──
# Hard ceiling on consecutive follow up turns before the loop returns to wake
# word state, independent of what the VAD decides. Bounds the blast radius of
# a runaway loop to a fixed number of Claude calls and TTS syntheses. Six sits
# above the deepest genuine conversation observed in production (four follow
# ups) with headroom to spare.
MAX_FOLLOWUP_TURNS = 6

# ── Conversation exit phrases ──
EXIT_PHRASES = [
    "that's all", "thats all", "thanks nova", "thank you nova",
    "we're good", "were good", "goodbye", "good night",
    "that is all", "i'm done", "im done", "you're dismissed",
    "dismissed", "peace", "later", "that's it", "thats it",
    "all good", "we're done", "were done", "i'm good", "im good",
    "that'll be all", "nothing else", "nah i'm good", "nah im good",
]
