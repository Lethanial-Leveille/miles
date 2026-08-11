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

# Silence required to decide the speaker has finished. This lands entirely on
# perceived latency: the user is already done talking and waiting.
#
# Was 3.0s, which was tuned when endpointing ran on an amplitude threshold that
# never fired, so recordings ended on the timeout rather than on speech and the
# constant was doing nothing except adding three seconds. Now that webrtcvad
# actually endpoints, that margin is unnecessary.
#
# 1.2s is set against measured behavior, not taste: frame level VAD over real
# recordings puts the longest genuine pause inside an utterance at 0.63s, with
# every other internal pause at or under 0.24s. This leaves roughly double the
# worst observed pause.
#
# Raise it if Nova starts cutting you off mid thought. That symptom means a
# real pause exceeded this value, and the fix is this number rather than
# anything in the VAD.
SILENCE_LIMIT = 1.2

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
MIN_VOICED_SECONDS = 4.0

# Calibrated against the completed enrollment run rather than estimated. The
# measured rate across twelve samples was 0.287s of voiced audio per word,
# ranging from 0.213 to 0.341 depending on delivery. Phrases are sized at
# eighteen to twenty words, which yields 5.2s of voiced audio at the average
# rate and 6.8s at the slowest, comfortably clear of MIN_VOICED_SECONDS.
#
# Window sizing works off wall clock, not voiced time: trimming removes the
# gaps between words, so a phrase yielding 6.8s voiced occupies roughly 8.7s
# of the recording. Ten seconds fits the slowest delivery of the longest
# phrase with margin to spare.
ENROLL_RECORD_SECONDS = 10

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

# Encoder audio context, in 20ms frames. The default of 1500 is a full thirty
# seconds, and whisper pads every clip to it, which is why transcription cost
# was flat at ~2000ms whether the audio was one second or fifteen. Capping the
# context skips that padded work.
#
# 1000 frames is twenty seconds, comfortably above the fifteen second
# MAX_RECORD. Validated by transcribing four real recordings of Lethanial at
# 3s, 6s, 10s, and 14s under both settings: word identical on all four, mean
# saving 785ms.
#
# Do not lower this without rerunning that validation. 750 and 900 both
# corrupted a word on reference speech ("ask not" became "asked not"), and 900
# additionally sent one noisy clip into a decode loop that took 8.5 seconds.
WHISPER_AUDIO_CTX = 1000
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

# ── Model ──
# Haiku 4.5 after a measured A/B against Sonnet 4.5 over twenty turns: 614ms
# faster on median time to first token, 31% lower, p=0.0007 on a permutation
# test. The two distributions barely overlapped. Haiku also drew more follow up
# turns, which carry more history and therefore larger prompts, so the split
# was working against it and it won anyway.
#
# A/B left in place but off. Flip MODEL_AB_TEST back on to compare a future
# model, and pin it off again before measuring anything cache related, since
# prompt caches are model scoped and alternation makes every turn a miss.
#
# ── Model A/B test ──
# Strict alternation rather than randomization. With a day of turns, random
# assignment can hand one arm a 60/40 split and pair it with a slow network
# stretch; alternating guarantees balanced counts and interleaves both arms
# against slow drifting confounds (network, API load, time of day).
#
# MODEL_A is the existing production model and serves as the control, so the
# comparison is against the known baseline rather than against a second change.
#
# Note for later: prompt caches are model scoped, so once cache_control is
# added, per turn alternation makes every turn a cache miss. Pin to one model
# before measuring caching, or the two experiments will fight.
MODEL_AB_TEST = False
MODEL_A = "claude-haiku-4-5"          # alias for claude-haiku-4-5-20251001
MODEL_B = "claude-sonnet-4-5-20250929"

# ── Prompt caching ──
# Caches the system prompt, which is the stable prefix: the persona and seed
# corpus change rarely, while conversation history changes every turn and so
# has to sit after the breakpoint. Measured effect on a cache hit was time to
# first token dropping from 1995ms to 639ms.
#
# Haiku 4.5 requires a 4096 token cacheable prefix. The assembled system
# prompt measures 4165, so the margin is only 69 tokens: deleting a handful of
# seed memories would drop it under and caching would stop working *silently*,
# with no error and no warning. cache_read_tokens is logged per turn for
# exactly that reason. A run of zeroes there means the prefix fell below the
# minimum, not that the cache expired.
#
# Turn spacing supports the default five minute TTL: 79% of observed gaps
# between turns fall inside it, median gap 50 seconds. The one hour TTL costs
# 2x on writes instead of 1.25x and would only cover another 10%.
PROMPT_CACHING = True

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
