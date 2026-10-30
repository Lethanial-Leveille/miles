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
# ECAPA since Sep 27 2026, switched in the same session as a fresh enrollment,
# because the two encoders produce 192 and 256 dimensions in unrelated spaces and
# a voiceprint from one is meaningless to the other. Resemblyzer could not keep
# anyone out: EER 21.5% on 28 labelled impostor clips from three people, impostor
# max 0.812 above his median. ECAPA: EER 7.7%, impostor max 0.210 against his
# median 0.516. The Resemblyzer voiceprint is kept in
# models/resemblyzer_backup_2026-09-26 for rollback.
SPEAKER_ENCODER = "ecapa"

# Per encoder, because the scales are not interchangeable and migrating the
# number is the single most likely way to break this.
#
# resemblyzer 0.5: in production since the beginning, well below his 0.764
# genuine median. It also accepts his sister most of the time, which is the
# whole reason for the swap.
#
# ecapa 0.15: derived Sep 27 2026 against the enrollment voiceprint itself.
# 27 labelled impostor clips (his sister, two friends) scored median 0.004, max
# 0.088; 82 of his clips scored median 0.436. At 0.15 no impostor gets in and
# 12.2% of his clips fall to guest, an overestimate: the archive is only
# presumed his, and its lowest clips score about 0.00, which is impostor range.
# A wrong rejection costs him a guest's answer for one turn, not a refusal.
VERIFY_THRESHOLDS = {
    "resemblyzer": 0.5,
    "ecapa": 0.15,
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

# Verification embeds at most this much trimmed speech, the start of it. ECAPA
# costs about 100ms per second of audio on this Pi, and the early run gets
# roughly a second's head start, so on Sep 28 2026 every long turn waited 1.2 to
# 2.8s on verification alone. Measured that day on 49 archived clips over 12s:
# the full clip took a median 1742ms, the first 10s about 970ms flat. Scores
# moved a median 0.027 down; 3 of 49 went from him to guest, all old clips that
# already sat at 0.18 to 0.21, and none of the Sep 27 and 28 clips moved near
# the line. 10 matches ENROLL_RECORD_SECONDS, what the voiceprint was built on.
# None embeds everything.
VERIFY_MAX_SECONDS = 10.0

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
# Off Sep 30 2026, when "hey nova" over her never woke the watcher: her echo
# crossed two USB clocks and the canceller could not lock. On Oct 1 2026, with her
# playing from the array's own jack, the wake model heard him over her 27 of 29
# times on BARGE_IN_CHANNEL. What was tried and measured is in BACKEND_TODO.md,
# Barge in on the array.
BARGE_IN = True

# Which of the array's two channels barge in listens on, while she plays.
# Sep 30 2026, "hey nova" said over her: channel 0, the processed output, chopped
# his voice into "Hello?" at both echo suppression settings tried and never woke;
# channel 1, the recognition beam, woke at 0.88 and 0.98. Only barge in reads it;
# everything else stays on MIC_CHANNEL. See AUDIO_PIPELINE.md.
BARGE_IN_CHANNEL = 1

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

# A barge in wake whose last WINDOW of channel 1 never peaks above the floor is
# not him, and she keeps talking. Oct 5 2026 the model scored 0.43 on near
# silence (about minus 50 dBFS, nobody talking, her echo cancelled) 14.9s into
# a reply, and stopped her; that window peaked at minus 35.1. His 37 "hey nova"s
# over her that fired (wake_eval 20261001T114604, three spots) peaked at minus
# 22.5 or louder. Not WAKE_LEVEL_FLOOR: at minus 42 it would have let this one
# through. One false stop and one session of his: retune from barge_in_stops,
# where every rejection keeps its audio as wakegated_*.
BARGE_IN_WAKE_FLOOR = -30.0
BARGE_IN_WAKE_WINDOW_S = 2.5

# Barge in on any speech, not only the wake word: he talks over her and she
# stops. Its own switch, so the wake word keeps working with this off. Decided by
# interrupt_speech.py, on BARGE_IN_CHANNEL, beside the wake model.
#
# Anyone's speech stops her, a TV's included; whose it was is decided afterward
# on the whole interruption. The floor is the knob for a busy room: raising it
# keeps only nearer, louder voices.
#
# -37, from three sessions replayed through this code on Oct 1 2026: him saying
# "hey nova" over her from three spots (wake_eval 20261001T114604), claps over
# her (barge_in_eval 20261001T134904), and her alone while people talked outside
# the room (20261001T141710_silence). At -40 he was caught 29 of 29, but one
# clap, one burst of her own echo and 7 outside voices stopped her too. At -37:
# 29 of 29, and none of the rest. The margin is about 1dB each way: -38 let 3
# outside voices through, -36 lost one take from his farthest spot, -35 lost 6.
# Requiring 360ms instead lost him before it lost the outside voices. Retune
# from data/barge_in_stops, not from these sessions alone.
BARGE_IN_SPEECH = True
BARGE_IN_SPEECH_FLOOR = -37.0
BARGE_IN_SPEECH_MS = 300

# Her own echo is not him. Oct 4 and 5 2026 she stopped on her own first
# syllables, leaking through the canceller at minus 30 dBFS after hours of
# quiet; their envelopes followed what she was playing at r 0.94 and 0.97,
# where his 28 takes over her (wake_eval 20261001T114604, barge_in_eval
# 20261002T004951) scored 0.69 at most. Compared over the last WINDOW of mic
# levels, at delays across LAG from when aplay started the device: 90 to 150ms
# on the eval, padded each way because the live start time is estimated. A
# wider search let his own speech line up with hers by luck (0.91 over 1.5s).
# A wrong estimate fails safe: no match, and she stops as she did before.
BARGE_IN_ECHO_CHECK = True
BARGE_IN_ECHO_SCORE = 0.8
BARGE_IN_ECHO_WINDOW_MS = 1200
BARGE_IN_ECHO_LAG_MS = (30, 300)

# The audio behind every barge in stop, wake or speech, so each one can be heard:
# a stop on a clap or the TV is invisible in a log line. Named by why it stopped
# and, for speech, where his speech began in the clip. Up to about 15s each, so
# 100 is about 50MB at most.
BARGE_IN_STOP_DIR       = os.path.expanduser("~/miles/data/barge_in_stops")
BARGE_IN_STOP_MAX_FILES = 100

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

# How long a command recording waits for him to start, before any speech. Not
# SILENCE_LIMIT, which is for knowing he has finished: a pause after the chime is
# not the end of a question he has not asked yet. The same as FOLLOWUP_TIMEOUT,
# the wait for a follow up. See endpoint.py.
COMMAND_ONSET_S = 3.0

# Hard ceiling on one recording.
#
# 30 since the night of Sep 13 2026. It was 18, just under Whisper's twenty
# second window, which cut a real scheduling request off mid thought; it went to
# 60 once long clips got the full window and splitting. That same night guests
# came over and 60 was wrong: a false wake from the room recorded 53 seconds of
# their conversation and a follow up hit the full 60, each time holding the mic
# while he said the wake word into the recording. Neither was him.
#
# 30 covers him. His longest real request was 42 words in 18 seconds, and 30 is
# about 75 words at a normal pace. It also fits one Whisper pass almost always.
# Since Sep 5 2026, half of 83 recordings were under 2.9s and nine in ten under
# 8.2s. The wake word now interrupts a recording too, so the room cannot hold
# the mic even for 30.
#
# SILENCE_LIMIT still ends a recording on a pause, so a long explanation with a
# long thinking pause ends at the pause, not here.
MAX_RECORD = 30.0

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
# Checked at startup and logged, because a silent revert corrupts collected
# data in a way that only shows up days later as inexplicably low scores.
#
# The XVF3800 runs its own automatic gain on chip, so the USB capture control
# stays at its maximum, 60 of 60 (0dB), and the chip decides the level. The
# Seiren's value was 23 on a 0 to 31 scale, tuned Aug 10 2026; it does not carry.
EXPECTED_MIC_GAIN    = 60

# A mic read blocked this long is not coming back: the process exits and systemd
# restarts it. One chunk normally returns in 80ms. See mic_watchdog.py.
MIC_STALL_S = 5.0

MIC_MIXER_CONTROL    = "Headset"

# Searched for in /proc/asound/cards and in PyAudio's device names, mirroring
# SPEAKER_NAME_HINT. Update this when the capsule changes, not the card number
# below it. The Seiren V3 Mini was the mic until Sep 30 2026.
MIC_NAME_HINT        = "XVF3800"

# The array offers two channels and nothing else; the pipeline wants one.
# mic_channel.py opens both and keeps MIC_CHANNEL. Why channel 0: that module.
MIC_CAPTURE_CHANNELS = 2
MIC_CHANNEL          = 0

# What the array delivers, which the firmware decides: 16000 on 2.0.6, 48000 on
# 2.1.1_48k2ch, which offers nothing else. mic_channel.py brings it down to RATE,
# so change this with the firmware and nothing else. Must be a multiple of RATE.
MIC_CAPTURE_RATE     = 48000


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

# ── Presence radar ──
# An LD2410C on the GPIO UART (dtparam=uart0=on). Logging only for now: nothing
# Nova does reads it yet. Wiring and tuning: BACKEND_TODO.md, Presence detection.
PRESENCE_SENSOR      = True
PRESENCE_PORT        = "/dev/ttyAMA0"
PRESENCE_BAUD        = 256000

# No valid frame for this long is logged as silent, never as absent. It reports
# about ten a second, and it stops altogether while the HLKRadarTool app is
# connected, which is the case this exists for: not seeing is not nobody.
PRESENCE_SILENT_S    = 3.0

# The energy in every gate, summarised once a window into presence_energy.
# Asleep he read 30% present against Oura's 01:59 to 10:08 on Sep 28 2026, and
# only per gate energies across a whole night can say whether a threshold can
# fix that or the sensor has to move. Turns engineering mode on at each open.
PRESENCE_ENERGY_LOG  = True
PRESENCE_ENERGY_WINDOW_S = 60.0

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

# Whisper's own full window, thirty seconds, used only for recordings longer than
# the fast window was validated for, fifteen seconds. Two fixed settings chosen by clip length, not a window
# scaled to each clip, which was tested in August and rejected for corrupting
# transcripts at every length. See audio_segments.audio_ctx_for.
WHISPER_AUDIO_CTX_LONG = 1500

# The longest piece one Whisper run is given. Past this a recording is cut at its
# quietest moment near each boundary and transcribed in pieces. Under the thirty
# second window with room to spare.
WHISPER_SEGMENT_SECONDS = 28.0

# Beam search. whisper.cpp defaults to 5 and this pipeline passed 1 until Sep 25
# 2026, so every WER figure measured here before that date was greedy and no
# comparison had ever varied it. Greedy commits to the first plausible word and
# never reconsiders, which is the error this set is full of: "what's the date"
# heard as "was the beat" on clean audio.
#
# Measured on the 58 hand labelled clips: 19.8% to 17.5% WER, exact 36/58 to
# 38/58, and it is the only change tried that improves initial commands and
# follow ups at the same time. It also beat small.en running greedy, at a third
# of the compute. Cost is 113ms of a 990ms head start, p50 1024 to 1137ms.
#
# Set to 1 to go back to greedy. Both the speculative run and the full one read
# this through _whisper_cmd, so they cannot drift apart.
WHISPER_BEAM_SIZE = 5

# ── Deepgram, the online transcript ──
#
# Measured Sep 25 2026 on the 58 hand labelled clips: 7.6% WER against 17.5% for
# base.en q8_0 at beam 5, follow ups 10.7% against 33.3%, and it invents words in
# 2 of 20 non speech clips against 5. Faster as well, p50 520ms and p90 826ms,
# both inside the 990ms speculation head start that whisper at beam 5 exceeds.
#
# **Whisper is the floor and stays.** Every failure falls back to it, so the
# offline behaviour that already exists does not regress and a bad day at Deepgram
# costs nothing more than today.
#
# Set DEEPGRAM_STT to False to run fully local again; nothing else has to change.
#
# The key is a secret and lives in .env. The model name is a configuration choice
# and does not, for the same reason TTS_VOICE_ID does not.
DEEPGRAM_STT       = True
DEEPGRAM_API_KEY   = os.environ.get("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL     = "nova-3"
# Generous against the 520ms median, because the fallback is a full whisper run
# rather than a failed turn, and cutting off a request that was about to answer
# buys nothing. Bounded so a hung socket cannot hold the turn open.
DEEPGRAM_TIMEOUT_S = 8.0
# Names Nova 3 is told to expect (its keyterm prompting). Family nicknames are
# not dictionary words, and a misheard one is a person she does not know: on
# Oct 9 2026 "Madge" and "MJ" both came back as strangers where "Maj" was his
# cousin. Names only; a common word here would bias every other transcript.
# label_transcripts.py scores with and without this list.
DEEPGRAM_KEYTERMS = ["Lethanial", "Maj", "Majelin", "Caiah", "Micaiah",
                     "Lahna", "Mylahna", "Bree", "Mabryel", "Celo", "Marcelo",
                     "Malique", "Azarieyah"]

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

# ── The level gate ──
#
# The score cannot tell a real wake from the neighbours. Measured on 49 labelled
# hits, real and false wakes overlap from 0.40 to 0.94, so 0.8 still admits a
# false wake while discarding half the real ones. Level does separate them,
# because speech through a wall arrives quiet and the model does not look at
# loudness at all. Peak beats RMS: a wake phrase has a loud consonant in it even
# when its average is low. Both tables are in docs/AUDIO_PIPELINE.md.
#
# -42 is the lowest floor that costs nothing measured: 0 of 38 real wakes lost,
# 7 of 22 false wakes cut. -40 is tempting at 13 of 22, and it is where two
# genuine attempts from the next room actually landed. Do not raise this without
# far field clips in the sample; the 27 desk distance wakes alone said -40 was
# free and they were wrong.
#
# A false wake that clears this gate is expected. 9 of the 22 arrive loud and
# nearby because they are his own voice not addressed to Nova, and no acoustic
# measure separates those.
WAKE_LEVEL_GATE  = True
WAKE_LEVEL_FLOOR = -42.0

# A wake that scores low must also be loud. On 106 array wakes he labelled from
# Oct 1 to 12 2026 (36 his, 70 not; scripts/wake_rule.py), a score under 0.5
# needing a peak louder than -9 dBFS cut 20 of the 70 and lost none of his.
# Every looser setting lost some of his: 0.6 at -9 cut 27 and lost one. Ignored
# wakes land in WAKE_GATED_DIR like the floor's. **His labelled wakes are
# mostly from his seat**: a far "hey nova" is both low scoring and quiet, which
# is exactly what this drops, so recheck with far field takes before tightening.
WAKE_RULE             = True
WAKE_RULE_SCORE_BELOW = 0.5
WAKE_RULE_PEAK_ABOVE  = -9.0

# Rejections keep their audio, because a gate that can deafen her must be
# auditable by ear rather than by a log line. label_wake.py reads this as a
# third set. A real "hey nova" landing here is the regression signal.
WAKE_GATED_DIR       = os.path.expanduser("~/miles/data/wake_gated")
WAKE_GATED_MAX_FILES = 300

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

# Open the turn's first Claude call during the endpoint wait, from the Deepgram
# transcript of the speculation, and let the real turn claim it at the endpoint.
# See speculative_turn.py. False returns to calling Claude after the endpoint.
SPECULATIVE_CLAUDE = True

# Deepgram answers the speculation in about 300ms, so whisper is only a fallback:
# started when Deepgram fails, or when it has not answered after this long. It
# used to run on every speculation and took the CPU from the ECAPA embedding
# beside it (238ms alone, 617ms beside whisper, Sep 28 2026). A slow Deepgram
# costs at most this over whisper alone; offline, Deepgram fails at once and
# whisper starts at once.
WHISPER_FALLBACK_AFTER_MS = 1000

# How much silence before speculating. Bounded on both sides: earlier means more
# overlap but more discarded work, and the overlap can never exceed
# SILENCE_LIMIT minus this.
#
# Was 450, the p75 of max_pause_ms over 66 turns (p50 270, p75 450, p90 630,
# p99 840), chosen to limit discarded work. Lowered Sep 16 2026 because the
# discarded work turned out not to be the cost that matters: Whisper takes about
# a second, so the transcript arrived a median 287ms after the recording ended.
# Starting sooner closes that gap without changing when the recording ends.
#
# Re-read over 280 turns: p50 330, p75 600, p90 900, p99 1140; 38% of turns
# already discarded a speculation at 450, and 60% will at 210. Simulated on 283
# archived recordings, speculations per recording go from a mean of 1.5 to 2.3.
# Each discarded run is killed at once; the voice embedding beside it may finish
# one stale run first, at most about 200ms.
SPECULATIVE_SILENCE_MS = 210

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

# What /speak sends the app, which is a different problem from what aplay wants.
# The room speaker gets raw PCM because it is local and a pipe is free. The
# phone is on the far side of a Cloudflare Tunnel, and the same audio as PCM is
# roughly 44KB a second, so a ten second reply is 440KB over whatever signal he
# has. mp3 at 64kbps is about 8KB a second and iOS decodes it natively.
#
# Rate and bitrate are tunable by ear. Do not drop to mp3_22050_32 without
# listening on the phone speaker first: the saving is small and the voice is the
# whole point of sending it.
TTS_APP_OUTPUT_FORMAT = "mp3_44100_64"

# A synthesis request with no audio after this many ms is sent again, and the
# first to answer is played. First byte since Sep 13 2026, 136 requests: p50
# 628ms, p95 1199ms, p99 5272ms, worst 16376ms. At 1500 about 4 percent resend,
# and they are the ones that would have left him in silence. None turns it off.
TTS_HEDGE_AFTER_MS = 1500

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

# Anthropic's own web search, run on their side inside the same Claude call, so
# there is no key here and nothing on the Pi executes it. Added Sep 27 2026
# after "What was the score of the UF game today?" was answered with "I don't
# have access to live sports scores."
#
# The basic 20250305 variant because it is the one Haiku 4.5 supports; the
# newer dynamic filtering variants need a larger model. Two searches is enough
# to retry one bad query, and caps what a single turn can spend and wait for.
# False removes the tool from every request and from the prompt's tool list.
WEB_SEARCH = True
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
WEB_SEARCH_MAX_USES = 2
WEB_SEARCH_LOCATION = {
    "type": "approximate", "city": "Gainesville", "region": "Florida",
    "country": "US", "timezone": "America/New_York",
}

# ALSA card numbers shift between boots, so the speaker is resolved by name
# at runtime in tts.py rather than hardcoded here. This is just the name to
# search for in /proc/asound/cards.
SPEAKER_NAME_HINT = "XVF3800"

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

# What Nova said, as she said it: each spoken reply's audio with the exact text
# ElevenLabs was given. "It sounds a little off" cannot be fixed until the
# moment can be replayed and pointed at, and until Sep 28 2026 her output was
# never kept anywhere. About 44KB per second at pcm_22050, so a hundred replies
# is roughly 45MB; kept small because the SD card is the weak part of this Pi.
# Labelled replies are evicted last (wake_capture.clips_to_evict).
ARCHIVE_SPOKEN   = True
SPOKEN_DIR       = os.path.expanduser("~/miles/data/spoken")
SPOKEN_MAX_FILES = 100

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

# How long a tool may run before a bridge line covers it. Below this the answer
# is close behind, and a 1.5s clip in front of it only delays it: a calendar
# addition takes about 5ms, and its read back waited behind "Let me look at your
# calendar." Tool times in tool_call_log, Sep 25 2026: reads of the calendar,
# Oura and weather run 500 to 1000ms, additions, reminders and memory under 50ms.
BRIDGE_DELAY_MS = 400

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

# History carries no times, so a reply from this afternoon reads as if it were
# said a moment ago. On Sep 28 2026 she had said at 1:46 PM that his last
# workout was six days old, and at 6:57 PM, two minutes after he saved one, she
# repeated it without calling a tool, in all 8 replays of that turn; with the
# afternoon removed from the history she called the tool 3 of 3 times. Past
# this gap the newest message says how long ago the conversation before it
# ended. Thirty minutes is far past any follow up and short of a new errand.
HISTORY_GAP_NOTE_MINUTES = 30

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

# The background pass that notices facts worth remembering; see memory_pass.py.
# Haiku, measured Sep 16 2026 on his real messages: 16 of 24 facts against 12
# for Sonnet 5, at about half the cost. It runs after the reply, so its speed
# is not on the path he waits for.
MEMORY_PASS = True
MEMORY_PASS_MODEL = "claude-haiku-4-5"

# The evening check in; see checkin.py. Haiku words the facts code already
# chose, so the same week does not sound the same every night. Like the memory
# pass it runs off his turn's clock, after her reply.
CHECKIN = True
CHECKIN_MODEL = "claude-haiku-4-5"


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

# The memory pass does not queue a fact this close in meaning to one already
# stored or waiting. Only a rewording scores this high: measured Sep 28 2026,
# the one pair at or above it was the same top three films stored twice, 0.94.
# Duplicates in general ran 0.79 to 0.94 and different facts 0.76 to 0.83, so
# lowering it toward 0.8 starts dropping real facts. The pass only, never a fact
# he asked her to remember.
MEMORY_DUPLICATE_SIMILARITY = 0.9

# Pairs listed by `scripts/memory.py similar` for merging by hand. Low enough
# that every duplicate found on Sep 28 2026 (lowest 0.79) is on the list; the
# different facts it also lists are his to leave alone.
MEMORY_SIMILAR_LIST = 0.75

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


# ── Push notifications, through Apple's push service (APNs) ──
#
# Reminders and timers used to be heard only in his room. With these set, the
# poller also sends them to his phone. The whole feature is off until the three
# variables below are in .env, and a send with any of them missing is skipped,
# so the Pi runs exactly as it did before the Apple Developer Program.
#
# APNS_TEAM_ID  developer.apple.com, Membership details. Not assumed from the
#               Xcode project: that value is whatever team the app was last
#               signed with, and enrolling can change it.
# APNS_KEY_ID   the ten character id of the key made under Certificates, IDs
#               and Profiles, Keys, with Apple Push Notifications ticked.
# APNS_KEY_PATH where its AuthKey_XXXX.p8 file sits. Apple lets it be
#               downloaded once, and the file is a secret, so it stays out of
#               the repo (*.p8 is gitignored).
PUSH_NOTIFICATIONS = True
APNS_TEAM_ID       = os.environ.get("APNS_TEAM_ID")
APNS_KEY_ID        = os.environ.get("APNS_KEY_ID")
APNS_KEY_PATH      = os.environ.get("APNS_KEY_PATH")
# The topic APNs routes by. It has to equal the app's bundle identifier.
APNS_BUNDLE_ID     = os.environ.get("APNS_BUNDLE_ID", "com.Lethanial.Nova")
# What gets a push. A timer set by voice is also heard in the room, and a
# kitchen timer buzzing a phone in another room is still the point of a timer.
PUSH_KINDS         = ("reminder", "timer")
# How long APNs holds a push for a phone that is off or has no signal. A
# reminder is worth reading late; a timer that finished an hour ago is not.
PUSH_TTL_S         = {"reminder": 6 * 3600, "timer": 300}
PUSH_DEFAULT_TTL_S = 3600
APNS_TIMEOUT_S     = 10.0
