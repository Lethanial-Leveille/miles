import os
import re
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

# ── Speaker encoder ──
# Which model turns a waveform into a voiceprint. See src/speaker_encoder.py.
#
# Still resemblyzer, and that is a staging decision rather than a preference.
# ECAPA is measurably far better here (EER 4.8% against 22.3%, and an impostor
# max of 0.179 against 0.843 on 23 labelled clips of his sister), but swapping
# it invalidates the voiceprint outright: the two produce 192 and 256 dimensions
# in unrelated spaces, so the swap and a re enrollment are one operation, not
# two. Flip this only in the same session as running enroll.py.
SPEAKER_ENCODER = "resemblyzer"

# Per encoder, because the scales are not interchangeable and migrating the
# number is the single most likely way to break this.
#
# resemblyzer 0.5: in production since the beginning, well below his 0.764
# genuine median. It also accepts his sister most of the time, which is the
# whole reason for the swap.
#
# ecapa 0.30: NOT YET DERIVED FROM THIS SETUP. The measured landmark is 0.179,
# which excluded all 23 impostor clips at a 5.3% cost to him, but that was one
# person on one evening in the old room against a centroid built from archived
# clips rather than from enrollment. 0.30 sits above it with margin and below
# his 0.542 median. Re-derive it against a real enrollment before trusting it,
# and replace this comment with the numbers when you do.
VERIFY_THRESHOLDS = {
    "resemblyzer": 0.5,
    "ecapa": 0.30,
}
VERIFY_THRESHOLD = VERIFY_THRESHOLDS[SPEAKER_ENCODER]

# Below VERIFY_THRESHOLD but above this, the score is genuinely ambiguous rather
# than a rejection. Three of five real rejections landed within 0.05 of the
# threshold, so a hard line there turns a coin flip into "Nova ignored me".
# Asking him to repeat converts those into a second, longer utterance, which is
# also the thing that fixes the score.
# Expressed as a fraction of VERIFY_THRESHOLD rather than an absolute, because
# an absolute 0.45 is a narrow band under resemblyzer's 0.5 and sits far ABOVE
# the whole usable range under ecapa, where the threshold is 0.30. A ratio
# keeps "just barely short" meaning the same thing on both scales.
VERIFY_RETRY_RATIO = 0.90
VERIFY_RETRY_THRESHOLD = VERIFY_THRESHOLD * VERIFY_RETRY_RATIO

# Wake scores at or above this are logged when they fail to fire. Below it the
# model is correctly ignoring the room and there is nothing to learn.
#
# Aug 12 2026: this table was briefly used to argue WAKE_THRESHOLD down to 0.3.
# That was wrong, and the mistake is worth recording because the table alone
# invites it. wake_log holds only FAILURES, so read on its own it looks like a
# cluster of attempts sitting just under the line.
#
# Successful wakes print their score to the journal. Pulling both:
#
#   successful  n=16   min 0.520   median 0.800   max 0.970
#   failed      n=17   min 0.161   median 0.271   max 0.365
#
# The band from 0.365 to 0.520 is empty. The model separates cleanly, 0.4 already
# sits in that gap, and lowering it only moves the line into the noise cluster.
# Missed wakes are a model problem, not a threshold problem: either the attempt
# scored under this floor and was never logged, or the model scored it as noise.
#
# Never tune this threshold from wake_log alone. Compare both distributions.
WAKE_LOG_FLOOR = 0.15

# Incremental voiceprint learning. A sample is only kept when it clears this
# similarity and this duration, both well above the accept bar: folding in a
# wrong embedding poisons the profile permanently and quietly, which is exactly
# how the April voiceprint died. The accept threshold is 0.5; this is not a
# second accept bar, it is a much stricter "certain enough to learn from".
# Also a ratio of the accept bar, and for the same reason: 0.75 is a strict
# 1.5x under resemblyzer and unreachable under ecapa, whose genuine median is
# 0.542. A fixed value here would silently stop the voiceprint ever learning
# again the moment the encoder changed, with no error and no log line.
VOICEPRINT_LEARN_RATIO = 1.5
VOICEPRINT_LEARN_MIN_SIMILARITY = VERIFY_THRESHOLD * VOICEPRINT_LEARN_RATIO
VOICEPRINT_LEARN_MIN_SECONDS = 3.0
# Rolling window, oldest dropped. Tracks how he sounds now rather than
# averaging over every room he has ever been in.
VOICEPRINT_SAMPLE_CAP = 60

# Interrupt Nova mid sentence by saying the wake word again.
#
# BACKEND_TODO treats barge in as blocked on acoustic echo cancellation, and
# for the general case it is: "any speech stops her" cannot work while her own
# voice is speech reaching the microphone. But the wake word model is already a
# discriminator trained on one phrase, and Nova saying "Nova" mid sentence does
# not reach the threshold. An interrupt keyed on a keyword her echo never
# contains needs no echo cancellation at all.
#
# Off by default because it reads the shared microphone stream during playback,
# and a second reader on that stream is the kind of bug that shows up as
# occasional silence rather than an exception. Turn it on deliberately and
# watch it.
BARGE_IN = False

# Barge in has its own threshold, and it is deliberately lower than
# WAKE_THRESHOLD rather than sharing it.
#
# The prior is different. Cold, "hey nova" competes with a whole room of speech
# that is not addressed to Nova, so the bar guards against a false start. While
# she is speaking, almost nothing else explains him saying her name: people do
# not discuss their assistant during its sentences. A bare "Nova" scores lower
# than the full phrase the model was trained on, and this is what lets that
# still count, without touching the cold threshold where his own worry applies:
# talking *about* Nova should never wake her.
#
# The cost of a false interrupt is also small. She stops talking and he repeats
# himself. The cost of a false cold wake is she starts listening to a
# conversation she is not part of.
#
# Do not lower this below about 0.25 without checking wake_log. That table
# records what nearly fired and is the only evidence for where the floor is.
#
BARGE_IN_THRESHOLD = 0.30

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
# Raised 0.9 to 1.2 on Aug 12 2026, because it was demonstrably cutting him off.
# The caveat below turned out to be exactly right: max_pause_ms over 66 real
# turns ran p50 270, p75 450, p90 630, p99 840, against a 900ms limit, and live
# transcripts showed the truncation directly ("set a timer for", "Start give me
# start five minutes. I'm").
#
# The original 0.63s worst pause came from reading a scripted phrase during
# enrollment. Spontaneous conversation carries longer thinking pauses than read
# speech does, so that margin was always thinner than the number suggested.
#
# This normally lands 1:1 on perceived latency, but speculative transcription
# changes the arithmetic: the extra 300ms of waiting is also 300ms more overlap
# with Whisper, which was 1166ms and is the stage being hidden. The two very
# nearly cancel, so this costs far less than it used to. Do not read that as
# licence to raise it further; past the point where transcription is fully
# hidden it goes back to costing full price.
SILENCE_LIMIT = 1.2

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
MIC_MIXER_CONTROL    = "Mic"

# Searched for in /proc/asound/cards, mirroring SPEAKER_NAME_HINT. Update this
# when the capsule changes, not the card number below it.
MIC_NAME_HINT        = "Seiren"


def _resolve_mixer_card(name_hint, cards_path="/proc/asound/cards"):
    """The ALSA card number carrying the microphone, found by name.

    This was hardcoded to "0", and card 0 is the AB13X speaker adapter rather
    than the Razer. So every gain check ever run read the speakers' capture
    input, found 255 against an expected 23, and printed a drift warning at
    every single service start. The fix it suggested, setting card 0 to 23, is
    nine percent on that device's 0 to 255 scale and has nothing to do with the
    microphone, whose own scale is 0 to 31 where 23 is 74 percent and 7.00dB.

    The damage was not the wrong number, it was that a guard built to be
    impossible to ignore fired on every boot and so became impossible to
    notice. A check that is always failing is indistinguishable from no check.

    This is the third instance of one bug. ALSA card numbers shift between
    boots, which is why the mic is already found by name in PyAudio and the
    speaker by name in tts.py. The mixer was the one device reference left as
    an index, and it was invisible precisely because it sat next to a lookup
    that was already doing the right thing.

    Returns None rather than falling back to a card number. A fallback here
    does not fail loudly, it silently measures the wrong hardware, which is the
    failure being removed. Callers report that they cannot check instead of
    reporting a pass or a drift they did not observe.

    cards_path is a parameter so this is testable without /proc, the same way
    netcheck.default_interface takes its route table."""
    try:
        with open(cards_path) as f:
            cards = f.read()
    except OSError:
        return None

    # Each card is two lines: " N [ID   ]: driver - description" followed by an
    # indented detail line. Split on the start of each entry so the hint can
    # match either line of a block, since the model name often appears only in
    # the second.
    for block in re.split(r'\n(?=\s*\d+\s+\[)', cards):
        match = re.match(r'\s*(\d+)\s+\[', block)
        if match and name_hint in block:
            return match.group(1)
    return None


MIC_MIXER_CARD       = _resolve_mixer_card(MIC_NAME_HINT)


# PyAudio appends the ALSA hardware address to every device name, and the card
# number inside it shifts between boots. Stripping it is what makes the name
# identify a capsule rather than a boot.
#
# It had already gone wrong. voiceprint_samples held three distinct "mics" for
# one physical microphone, hw:0,0, hw:1,0 and hw:3,0, across ten samples. A
# column whose entire job is to keep samples from two capsules out of one
# centroid was instead splitting one capsule three ways, and
# get_voiceprint_samples filters on an exact string, so a recompute scoped to
# "this microphone" would quietly have used a third of the data. Migration 023
# collapsed the rows already written.
#
# Lives here rather than in audio.py so it can be tested. Importing audio opens
# PyAudio and takes the exclusive mic lock, so nothing that runs outside the
# voice process can touch it.
_HW_ADDRESS = re.compile(r'\s*\(hw:\d+,\d+\)\s*$')


def capsule_name(device_name):
    """A microphone's identity, with the boot specific address removed."""
    return _HW_ADDRESS.sub('', device_name or '').strip()

# ── Paths ──
# q8_0 rather than the fp16 base.en, quantized locally from it with
# whisper-quantize. Same model and same weights, stored as 8 bit integers with
# one fp16 scale per block of 32 instead of 16 bit floats each.
#
# Measured over 40 archived clips at 3 threads, Aug 13 2026: 1013ms against
# 1435ms, a 422ms saving, 36/40 transcripts word identical to fp16. Two of the
# four differences were mildly worse and two were garbage under both. It is a
# real accuracy cost, not a free one, and it is a far better trade than tiny.en
# (52% identical) or ac 500 (70%), both of which were rejected.
#
# The reason it wins is the Cortex-A76's native int8 dot product. q5_1 is
# smaller still and was measured at 1959ms, SLOWER than doing nothing, because
# nothing on this CPU is natively 5 bit and every weight has to be shifted,
# masked and reassembled before it can be multiplied. Smaller is fewer bytes,
# not fewer cycles. Do not assume a smaller quantization is faster; measure it.
#
# Regenerate after a whisper.cpp upgrade, since the file is derived:
#   ./build/bin/whisper-quantize models/ggml-base.en.bin \
#       models/ggml-base.en-q8_0.bin q8_0
WHISPER_MODEL   = os.path.expanduser("~/miles/whisper.cpp/models/ggml-base.en-q8_0.bin")
WHISPER_CLI     = os.path.expanduser("~/miles/whisper.cpp/build/bin/whisper-cli")

# Whisper's initial prompt, which biases its internal language model. NOT SET,
# deliberately, and this note is here so it is not rediscovered as an idea.
#
# "Pi" and "pie" are perfect homophones. No acoustic model can separate them,
# so the choice is made by the decoder's language prior, and on Aug 13 2026
# "what's the temperature of the Pi" transcribed as "pie" and Nova answered
# that she had no thermometer. A prompt naming Raspberry Pi and Nova fixes that
# clip outright, and also turned "7 times for 5 minutes." into "Set a timer for
# five minutes."
#
# It also turned "push my code tonight" into "push my coat tonight". Over 40
# clips it disagreed with fp16 base.en on 16 of them, and it was frequently
# BETTER when it disagreed, which is exactly why the agreement score cannot
# decide this: base.en is not ground truth and is often the wrong one.
#
# Blocked on hand labelled transcripts, not on implementation. Revisit with
# real labels rather than by re-running the agreement comparison.
WHISPER_INITIAL_PROMPT = None

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

# ── Wake miss capture ──
# wake_log stores a score and no audio, which makes every question about a
# missed wake unanswerable: a 0.27 could be him saying the phrase quietly, a
# fragment of unrelated speech, or the television, and those need opposite
# fixes. This keeps the audio that produced the score.
#
# It is also the only route to a real fix. models/hey_nova.onnx is dated Apr 8
# 2026, predating the mic gain tuning of Aug 10, and retraining or even
# evaluating it needs real recordings of attempts that failed. Synthetic data
# cannot tell you why this room misses.
CAPTURE_WAKE_MISSES = True
WAKE_MISS_DIR       = os.path.expanduser("~/miles/data/wake_misses")

# Was 0.05, on the theory that the failures that matter most may be the ones
# scoring near zero. The data disconfirmed the need rather than the theory.
#
# Measured Sep 6 2026 with the buffer at its 400 file cap:
#
#   <0.10       283      (71%)
#   0.10-0.20    88
#   0.20-0.30    19
#   >=0.30       10
#
# Seventy one percent of the buffer was empty room, and because pruning keeps
# the newest 400 by modification time, that noise was evicting the band that
# actually sits near the 0.4 threshold. A capture floor low enough to record
# everything is a capture floor that keeps the least interesting thing.
#
# 283 clips is already far more of the near zero case than anyone will listen
# to, so the original question is answerable from what is on disk.
WAKE_MISS_FLOOR     = 0.15
WAKE_MISS_MAX_FILES = 400

# ── Wake hit capture ──
# The audio behind a wake that DID fire.
#
# wake_misses answers "why did she not hear me". It cannot answer "why did she
# answer when I said nothing", because it only records scores below the
# threshold and a false wake is by definition above it. That case had no audio
# kept anywhere: archive_recording holds the command that followed, not the
# sound that triggered the wake.
#
# Observed Sep 6 2026 in the new apartment, every wake still in the journal:
# 0.75 "Oh shoot, this is weird.", 0.64 "(beep)", 0.50 "over here", 0.46
# "We'll see you later.", and one genuine conversation at 0.43. CLAUDE.md
# records the old room as separating cleanly, with an empty band from 0.365 to
# 0.520. It does not separate here, and it now overlaps in both directions, so
# no threshold value fixes it and the model itself is the thing to look at.
# hey_nova.onnx is dated Apr 8 2026, predating both the gain tuning and this
# room.
#
# The rolling window voice_main already keeps for near misses holds this audio
# too. Nothing read it on the firing path.
CAPTURE_WAKE_HITS   = True
WAKE_HIT_DIR        = os.path.expanduser("~/miles/data/wake_hits")
WAKE_HIT_MAX_FILES  = 300

# Audio retained before the score, so the clip contains the phrase rather than
# what followed it. The wake model reads 80ms frames, so this is 31 of them.
WAKE_MISS_PREROLL_MS = 2500

# ── Speculative transcription ──
# Endpointing and transcribing run in sequence today: 880ms of waiting for
# silence, then 1177ms of Whisper, while the CPU sits idle through the first.
# The audio Whisper needs already exists the moment speech stops, so it can
# start during the wait instead of after it.
#
# The speculation is thrown away if he speaks again, which makes this free in
# accuracy terms: the same model at the same settings on the same samples, only
# started earlier. Trailing silence is all that differs, and it carries nothing.
SPECULATIVE_TRANSCRIBE = True

# How much silence before speculating. Bounded on both sides: earlier means more
# overlap but more discarded work, and the overlap can never exceed
# SILENCE_LIMIT minus this.
#
# Set from the p75 of max_pause_ms over 66 turns (p50 270, p75 450, p90 630,
# p99 840). At 450ms about a quarter of turns speculate on a pause he then
# speaks through, and the surviving three quarters overlap 450ms of the 900ms
# wait. Re-read that distribution before changing this; it is the whole basis.
SPECULATIVE_SILENCE_MS = 450

# One core left for capture. Whisper defaults to all four, and starving the
# PyAudio read loop drops frames, which matters precisely when speech resumes
# and the recording still has to be good.
SPECULATIVE_THREADS = 3
SPECULATIVE_WAV     = os.path.expanduser("~/miles/build/speculative.wav")
TEMP_RESPONSE   = os.path.expanduser("~/miles/build/response.wav")
WAKE_MODEL_PATH = os.path.expanduser("~/miles/models/hey_nova.onnx")
VOICEPRINT_PATH = os.path.expanduser("~/miles/models/voiceprint.npy")
# Individual enrollment embeddings plus condition labels, kept so the centroid
# can be recomputed or analyzed without re recording. The old enrollment saved
# only the mean, which is why a poisoned sample could not be identified later.
ENROLLMENT_DATA_PATH = os.path.expanduser("~/miles/models/enrollment.npz")

# The raw enrollment recordings, kept alongside the embeddings derived from
# them. enroll.py used to write every sample to one temp file and overwrite it,
# saving only embeddings, and an embedding is locked to the encoder that made
# it: Resemblyzer is 256 dimensions and ECAPA is 192, and the two are not
# convertible. So changing encoder meant re recording all twelve samples, and
# would again for the encoder after that.
#
# This is the April voiceprint lesson taken one step further. That failure was
# "only the mean was saved, so a bad sample could not be identified", and
# keeping the individual embeddings fixed it for analysis while leaving the
# same hole for re embedding. Audio is the only artifact every future encoder
# can read.
#
# Gitignored with the rest of models/. These are recordings of a real person.
ENROLLMENT_AUDIO_DIR = os.path.expanduser("~/miles/models/enrollment_audio")
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

# eleven_v3 since Sep 13 2026, chosen by ear.
#
# On flash_v2 every reply sounded like a narrator stringing words together.
# Heard side by side on the same real reply at a fixed seed, v3 was the one
# change that made an audible difference. Stability on flash, speed, and
# splitting a reply into sentences did not.
#
# The cost is time to first audio: roughly 470 to 620ms on v3 against 380ms on
# flash_v2, from single renders that day. Re measure in timing_log before
# quoting either number.
#
# Why never flash_v2_5, measured Aug 11 2026: it drops any word wrapped in a
# phoneme tag, which would delete his name. v3 keeps it. "Morning, Lethanial."
# with the tag ran 1.04s against 0.72s for "Morning." alone, and deliberately
# wrong phonemes ran longer at 1.28s, so the tag's content is used rather than
# thrown away.
DEFAULT_TTS_MODEL = "eleven_v3"

# Use the arpabet column instead of the alias respelling. Requires a model that
# honors phoneme tags: flash_v2 and v3 do, v2_5 deletes the word.
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

# Tuned for eleven_v3 by ear, Sep 13 2026, on one real reply at a fixed seed.
#
# stability 1.0 over 0.5. At 0.5 v3 sounded natural but a little exaggerated
# and slow; 1.0 kept the natural delivery with less of both, and ran shorter,
# 12.56s against 13.44s for the same reply.
#
# speed is declared but v3 ignored it: 1.0 and 1.1 rendered to exactly the same
# length at both stabilities. It stays explicit so an unset value is never an
# invisible API default, and a model that does honor it behaves as written.
#
# use_speaker_boost is unset because the renders he chose did not send it. The
# API accepts it on v3, so this matches what was heard; it is not a limit of
# the model.
#
# The flash_v2 tuning history, 0.60 through 0.90, is in docs/VOICE_OUTPUT.md.
# Those values were found on a different model and do not carry over.
#
# Changing any of this means re rendering the phrase bank, or cached clips and
# live speech drift apart: python3 scripts/render_phrases.py render --force
TTS_VOICE_SETTINGS = VoiceSettings(
    stability=1.0, similarity_boost=0.75, style=0.00, speed=1.00,
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

# ── Phrase bank ──
# Victoria, rendered once while online and played from disk forever after.
#
# Two holes open when the network drops: Claude raises, and ElevenLabs fails
# silently. The first is caught in voice_main. This closes the second for the
# fixed set of things Nova says that never needed a language model anyway.
#
# The point is not speed, though playback is faster than synthesis. It is that a
# local TTS engine would be a different voice wearing her name, and these files
# are actually her.
#
# Under data/ and so gitignored: the manifest text lives in phrasebank.py, which
# is versioned, and the audio is derived from it with scripts/render_phrases.py.
PHRASE_DIR = os.path.expanduser("~/miles/data/phrases")

# How often a wake gets a spoken acknowledgement instead of the chime. The chime
# is 0.320s and overlaps recording harmlessly because webrtcvad at mode 2 does
# not read a tone as speech. A spoken ack does read as speech, so it has to
# finish before the mic opens, which puts its length on the path between the
# wake word and the command. Mixing keeps the fast case the common one.
ACK_SPOKEN_CHANCE = 0.75

# ── Local intent ──
# Cosine against the canonical phrasings in local_intent.EXAMPLES. Both the
# lexical gate and this have to agree before anything fires.
#
# Tuned high on purpose, because the costs are asymmetric. A miss falls through
# to Claude and costs about 4.4 seconds and nothing else. A false fire starts
# the wrong timer or hangs up on him mid thought.
INTENT_SIMILARITY_THRESHOLD = 0.55

# A closing phrase inside a longer sentence is not a goodbye. "Thanks, now set
# a timer for ten minutes" is nine words and means the opposite of thanks.
MAX_DISMISS_WORDS = 6

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
# How long the microphone stays open after Nova answers, waiting for speech to
# start. Six seconds was generous and the generosity had a cost: everything
# said in the room during that window reaches her, including him turning to
# talk to somebody else.
#
# This is also the practical mitigation for the session trust hole. A follow up
# too short to embed is accepted on session state rather than verified, so a
# stranger's "yeah" inside the window is accepted as his. Shortening the window
# shrinks that exposure far more cheaply than trying to verify a one second
# utterance, which today is not possible.
FOLLOWUP_TIMEOUT = 3.0

# Conversation exit is no longer a phrase list. Twenty six exact strings could
# not match "alright thanks Nova" or "that's all for now", and seven of them
# ("later", "peace", "all good", "i'm good", "that's it", "dismissed",
# "we're good") are ordinary mid conversation utterances that would have ended
# a session by accident. Nova now recognizes the intent and emits
# [ACTION: dismiss]; see prompts.py.
