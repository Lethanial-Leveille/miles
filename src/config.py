import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import VoiceSettings

# Loaded here rather than in auth.py alone, because config.py is the one module
# every entry point imports. systemd passes .env in through EnvironmentFile, so
# the service always had these; a foreground `python voice_main.py` did not, and
# the only symptom was a 401 from ElevenLabs on every sentence while the turn
# itself worked perfectly. The two launch paths now behave the same.
#
# load_dotenv does not override variables that are already set, so under systemd
# this is a no op and EnvironmentFile still wins.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Audio hardware ──
CHUNK = 1280          # 80ms frames, required by openWakeWord
CHANNELS = 1
RATE = 16000
WAKE_THRESHOLD = 0.4
VERIFY_THRESHOLD = 0.5

# Below VERIFY_THRESHOLD but above this, the score is genuinely ambiguous rather
# than a rejection. Three of five real rejections landed within 0.05 of the
# threshold, so a hard line there turns a coin flip into "Nova ignored me".
# Asking him to repeat converts those into a second, longer utterance, which is
# also the thing that fixes the score.
VERIFY_RETRY_THRESHOLD = 0.45

# Wake scores at or above this are logged when they fail to fire. Below it the
# model is correctly ignoring the room and there is nothing to learn.
WAKE_LOG_FLOOR = 0.15

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
# 0.9s against a measured worst internal pause of 0.63s.
#
# Caveat worth knowing: that 0.63s came from reading a scripted phrase during
# enrollment. Spontaneous conversation carries longer thinking pauses than
# read speech does, so this margin is thinner in practice than the number
# suggests. max_pause_ms is logged per turn precisely so this can be tuned
# from real conversation instead of from scripted audio: if turns start
# ending on a pause rather than on a finished thought, that column will show
# pauses landing near this value, and the fix is this number.
SILENCE_LIMIT = 0.9

# Hard ceiling on one recording. 15% of recordings were hitting the old 15s
# limit, every one of them a follow up, and every one truncated mid sentence:
# thinking out loud in conversation runs longer than issuing a command.
#
# Capped below the whisper audio context window (WHISPER_AUDIO_CTX of 1000
# frames is 20 seconds), because audio past that window is not transcribed at
# all. Raising this past 18 means raising that too, which costs transcription
# time, so the two move together or not at all.
MAX_RECORD = 18.0

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

# The seed corpus lives under data/ rather than in src/ because data/ is
# gitignored and src/ is not. seed_memories.py used to carry the corpus inline
# as a Python literal, which meant every fact about Lethanial and everyone he
# named was committed to a public repository. The loader is worth showing; the
# payload is not, and the people named in it never agreed to be published.
SEED_PATH       = os.path.expanduser("~/miles/data/seed_memories.json")

# ── External services ──

# ROLLBACK: Fish Audio config preserved for emergency rollback
# VOICE_ID = "158f6b9781b746ec8c334d9730d302f1"

# ── ElevenLabs voice ──
# The single place the voice is defined. Swapping voices means editing
# TTS_VOICE_ID here and nothing else; no call site names a voice, a model, or a
# settings object.
#
# The id was previously read from .env, which made it a deployment secret rather
# than a configuration choice. A voice id is neither secret nor deployment
# specific, and keeping it there meant changing voices required editing a
# gitignored file with no history. The API key stays in .env, because that one
# actually is a secret.
#
# Voice: Victoria.
TTS_VOICE_ID = "qSeXEcewz7tA0Q0qk9fH"

ELEVENLABS_API_KEY   = os.environ.get("ELEVENLABS_API_KEY")
EXPRESSIVE_TTS_MODEL = "eleven_v3"   # HTTP only, no WebSocket, no speaker boost
TTS_OUTPUT_FORMAT    = "pcm_22050"   # raw S16_LE mono, piped straight to aplay

# flash_v2 rather than flash_v2_5, measured Aug 11 2026.
#
# v2_5 does not merely ignore <phoneme> tags, it drops the word they wrap.
# Synthesizing "Lethanial" plain gave 0.79s of audio; wrapped in a phoneme tag
# it gave 0.23s, and deliberately absurd phonemes gave the same 0.23s. Identical
# output for different phoneme strings means the content is discarded.
#
# v2 honors them: correct phonemes matched the plain duration at 0.74s, and
# absurd phonemes stretched to 1.21s.
#
# The switch is free. Median time to first byte over five runs was 349ms on v2
# against 347ms on v2_5, which is noise. v2 is English only, and Nova speaks
# English.
DEFAULT_TTS_MODEL = "eleven_flash_v2"

# Use the arpabet column instead of the alias respelling. Requires a model that
# honors phoneme tags; on v2_5 this would delete the word.
#
# Aliases are guesses tuned by ear. Phonemes are exact.
#
# On, after an audition. Candidate 15 of the sweep, L AE0 TH AE1 N Y AH0 L, was
# judged good in five of six renditions at different seeds. That measurement is
# the point: an earlier pass ranked candidates on one rendition each, and the
# same string came out wrong once and right the next time, so the ranking was
# partly recording which generation got lucky.
#
# No respelling survived the first pass, which is what settled alias against
# phoneme. See scripts/pronounce.py.
TTS_PHONEME_TAGS = True

# ElevenLabs accepts 0.7 to 1.2. Set explicitly rather than left to the API
# default, because an unset speed is an invisible dependency on whatever the
# API decides, and this lands directly on how long a turn occupies the room.
# stability 0.90, chosen by ear across a sweep at a fixed seed, Aug 11 2026:
#   0.45 no      0.60 eh      0.75 pretty good      0.90 good      1.00 good
#
# 0.60 was production while the name sounded butchered, and it rates "eh" on its
# own. Most of that hunt was chasing phonemes when the setting was the dominant
# term: stability governs how much one rendition varies from the next, and at
# 0.60 the same phoneme string came out right once and wrong the next time.
#
# 0.90 rather than 1.00 because the gain stopped there. 0.75 to 0.90 was audible,
# 0.90 to 1.00 was not, and 1.00 spends the last of the expressiveness that the
# persona's dry wit depends on for nothing measurable.
#
# Then 0.75 turned out to read noticeably better on ordinary sentences while
# occasionally missing the name, and 0.90 the reverse. Those are the two ends of
# one trade: stability buys consistency by reducing variation, and the same
# variation is what makes delivery sound alive. 0.80 sits in the band because
# this is a continuous value, not a choice between the five points that happened
# to get sampled.
#
# Settled at 0.80 after listening to the band between them. Change it only
# after listening, not by reasoning: every step of this was decided by ear and
# the two failures it trades between are audible, not measurable.
#   python3 scripts/pronounce.py stability Lethanial 15 .78 .80 .82 .85
TTS_VOICE_SETTINGS = VoiceSettings(
    stability=0.80, similarity_boost=0.75, style=0.00,
    use_speaker_boost=True, speed=1.00,
)

# Kept for the response classification work that selects a profile per turn.
# Unused today: speak() falls back to TTS_VOICE_SETTINGS when no override is
# passed, so these are inert until something selects one.
VOICE_WITTY = VoiceSettings(
    stability=0.30, similarity_boost=0.75, style=0.35,
    use_speaker_boost=True, speed=1.05,
)

VOICE_SERIOUS = VoiceSettings(
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

# ── Recording archive ──
# Every captured command is copied here alongside its logged transcript.
#
# build/command.wav is overwritten every turn, so there has never been an
# archive of real commands to test against. That blocked validating the
# whisper audio context change on actual speech (it had to be checked against
# enrollment audio instead) and it blocks the open question of whether tiny.en
# is accurate enough to trade for its 724ms saving. It will block the same
# question again when the mic changes.
#
# Recordings live under data/, which is gitignored. They are recordings of a
# real person in a real room; treat them accordingly.
ARCHIVE_RECORDINGS = True
ARCHIVE_DIR        = os.path.expanduser("~/miles/data/recordings")

# Roughly 30KB per second of audio, so a 9 second turn is about 270KB. Six
# hundred files is on the order of 150MB, which is a few weeks of normal use.
# Oldest are pruned first.
ARCHIVE_MAX_FILES = 600

# ── Conversation history ──
# Past assistant turns are trimmed to this many words before being sent as
# context. Nothing in the database changes; this is prompt assembly only.
#
# Response length is anchored far more strongly by history than by any
# instruction: holding the prompt fixed and varying only what history contains
# gave 94 words with full history against 54 with assistant turns trimmed.
# Nova was few shot learning her own verbosity from her own transcript, and
# each long answer made the next one likelier.
#
# Trimming beats the alternatives. Sending no history at all scored worse (64
# words, and a much longer tail) because an open question with no context
# invites a survey. Sending fewer whole messages scored the same but threw
# away fourteen turns of context to get there. Thirty words keeps what was
# discussed while dropping how long it took to say.
HISTORY_ASSISTANT_WORDS = 30

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

# How many tier two memories get attached to a turn when the transcript matches
# them. Small on purpose: these land after the cache breakpoint so they are
# cheap, but they are also unranked prose in front of the model, and five
# loosely relevant facts crowd out the one that mattered. Raise it only against
# logged misses, never on the theory that more context helps.
RECALL_LIMIT = 3

# Relevance floor for retrieval, in document frequency. A row is only attached
# if the query terms it matches are at least as informative as a term appearing
# in this many memories. Raising it makes retrieval quieter and misses more;
# lowering it attaches noise. Tune against logged misses, not by feel.
RECALL_MIN_DF = 15

# Semantic retrieval. all-MiniLM-L6-v2 is 384 dimensions, around 80MB, and
# measured at 31ms per query on this Pi, which is 0.7 percent of a 4298ms turn.
# Changing this re embeds the corpus rather than mixing vectors from two
# different spaces, which would yield similarity scores that look plausible and
# mean nothing.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reciprocal rank fusion constant. Keyword scores are summed inverse document
# frequencies and semantic scores are cosines; they are not on the same scale
# and normalising one into the other is guesswork. RRF throws away the
# magnitudes and keeps only the ranks, so both methods vote without either
# having to be calibrated against the other. 60 is the value from the original
# paper and there is no reason here to depart from it.
RRF_K = 60

# How long the follow up window stays open after Nova finishes speaking.
#
# Was a bare 10 in voice_main.py. Ten seconds is a long time to stand in a quiet
# room deciding whether you are done, and every expiry costs a full window of
# dead air before the wake word comes back. Six is comfortably past a normal
# beat of hesitation without leaving the mic open on an empty room.
#
# Lower bound is set by thinking time, not by speech: the window has to survive
# the pause between deciding to say something and starting to say it.
FOLLOWUP_TIMEOUT = 6

# Conversation exit is no longer a phrase list. Twenty six exact strings could
# not match "alright thanks Nova" or "that's all for now", and seven of them
# ("later", "peace", "all good", "i'm good", "that's it", "dismissed",
# "we're good") are ordinary mid conversation utterances that would have ended
# a session by accident. Nova now recognizes the intent and emits
# [ACTION: dismiss]; see prompts.py.
