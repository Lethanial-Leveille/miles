#!/usr/bin/env python3
"""Record the wake word evaluation set: him saying "hey nova", over her and not.

A retrained wake model can only be trusted if it beats the current one on the
same recordings, so this set is recorded once, kept, and every candidate model
is scored on it. Rerun it after a change to the room (a lower volume, the
enclosure) into a new session directory, never over an old one.

Three kinds of take, all recorded on both of the array's channels, because
which channel the barge in watcher should read is one of the questions:

    over     one of her real replies plays, with the echo reference on as barge
             in would run it, and a cue appears twice; he says "hey nova" once
             per cue. The cue times are the ground truth.
    her      her reply alone. Nothing here may fire, since a barge in that
             fires on her own voice cuts off every reply.
    quiet    no playback, one cue: the normal wake, so a model that learns her
             voice and forgets the quiet room shows up as a regression.

After every take he confirms whether he said it at each cue, and that answer
is the label. Sep 30 2026: a survey's cues never showed on his terminal and he
talked whenever she did; the log said otherwise. A cue is an intention, only
his answer says what happened.

These clips are evaluation only. They must never be used to train a model, or
the score measures memory instead of hearing.

    python3 scripts/record_wake_eval.py             # a full session, about fifteen minutes
    python3 scripts/record_wake_eval.py --takes 3   # a short try of each kind
    python3 scripts/record_wake_eval.py --clap      # claps over her, for barge in
    python3 scripts/record_wake_eval.py --sentence  # sentences over her and quiet

--sentence shows a sentence to read at each cue, over her and in a quiet room,
into data/barge_in_eval/. Barge in keeps what he says over her from channel 1,
which nothing downstream has heard before; this scores Deepgram and voice
verification on each channel against the sentence shown.

--clap records only over her, with "CLAP" cues asking for one, two or three
claps, into data/barge_in_eval/. Barge in on speech stops her on anything loud
and speech like for 300ms (interrupt_speech.py); a clap must not be that. Her
alone and his voice over her are already in the wake sessions.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

import mic_channel
import tts
from config import MIC_NAME_HINT, MIC_CAPTURE_RATE, RATE
from label_wake import _voice_service_paused

EVAL_DIR = os.path.join(ROOT, "data", "wake_eval")
CLAP_DIR = os.path.join(ROOT, "data", "barge_in_eval")

# What a clap cue asks for. Several kinds, because a single clap and a run of
# them are different shapes to the speech rule: one decays, the other may not.
CLAPS = ["CLAP ONCE", "CLAP TWICE", "CLAP THREE TIMES"]

# What a sentence cue shows. Requests he makes, and interruptions, long enough
# to verify (about three seconds and up) and known in advance, so a transcript
# can be scored against what was said rather than against the other channel.
SENTENCES = [
    "What's the weather going to be like in Gainesville tomorrow afternoon?",
    "Set a timer for twenty minutes and remind me to check the oven.",
    "Can you tell me what I have on my calendar for Thursday?",
    "How many grams of protein are in a cup of Greek yogurt?",
    "Remind me to call my mom when I get back from class tonight.",
    "What was my readiness score this morning and how did I sleep?",
    "Move my study session on Friday to four in the afternoon please.",
    "What's the difference between a pipelined processor and a superscalar one?",
    "Add a reminder to submit the lab report before midnight on Sunday.",
    "How long would it take to drive from Gainesville to Tampa right now?",
    "What did I lift during my last upper body workout on Hevy?",
    "Actually wait, I meant the other one, the one from yesterday.",
    "No no no, hold on, go back to what you said about oranges.",
    "Wait, stop, that's not what I asked, I wanted the one for Saturday.",
    "Hold on, can you say that last part again a little slower?",
    "Actually never mind that, what time does the gym close tonight?",
]
SPOKEN_DIR = os.path.join(ROOT, "data", "spoken")

README = """# Wake word evaluation set

Recorded by scripts/record_wake_eval.py. Him saying "hey nova" at cues, over
her replies and in a quiet room, plus her replies alone. Both array channels,
16kHz stereo: channel 0 processed, channel 1 the recognition beam.

EVALUATION ONLY. Never train on these clips. A model trained on its own test
set scores itself on memory, and the number stops meaning anything.

Never prune this directory. A label with no audio cannot be checked again
(INCIDENTS.md, the wake_misses eviction).

manifest.jsonl, one line per take: file, kind, spot, clip played, where
playback starts and ends in the recording, and each cue with whether he
confirmed saying it.
"""

# Where he is. Spread across the takes so the set is not one position.
SPOTS = ["your usual seat", "your other usual spot", "wherever you would naturally be"]
# --far. His labelled wakes through Oct 12 2026 were mostly from his seat, and a
# far "hey nova" is both low scoring and quiet: exactly what WAKE_RULE drops and
# what a retrained model must not lose. These takes are the evidence for both.
FAR_SPOTS = ["your bed", "the doorway", "across the room, facing away from her"]


def _clips(minimum_s=10.0, maximum_s=30.0):
    """Her replies of a length that leaves room for two cues."""
    found = []
    for name in sorted(os.listdir(SPOKEN_DIR)):
        if not name.endswith(".wav"):
            continue
        with wave.open(os.path.join(SPOKEN_DIR, name)) as w:
            seconds = w.getnframes() / w.getframerate()
        if minimum_s <= seconds <= maximum_s:
            found.append(name)
    return found


class _Clip:
    """A recorded reply, handed to play() the way a synthesis would be."""

    def __init__(self, pcm):
        self.pcm = pcm
        self.requested_at = time.monotonic()
        self.error = None

    def chunks(self):
        for i in range(0, len(self.pcm), 4096):
            yield self.pcm[i:i + 4096]


def _cue(at, t0, times, ask="SAY  \"HEY NOVA\""):
    time.sleep(max(0.0, t0 + at - time.monotonic()))
    times.append(round(time.monotonic() - t0, 3))
    # Written to the real terminal and flushed at once. Printed from a thread
    # while play() runs, which is the one place a cue went missing before.
    sys.__stdout__.write("\n" + "#" * 40 + f"\n###   {ask}  NOW   ###\n"
                         + "#" * 40 + "\n")
    sys.__stdout__.flush()


def _confirm(n_cues):
    said = []
    for i in range(n_cues):
        answer = input(f"    cue {i + 1}: did you do it? [Enter = yes, n = no] ").strip().lower()
        said.append(answer != "n")
    return said


def _take(session, number, kind, spot, clip, mic):
    path = os.path.join(session, f"{number:02d}_{kind}.wav")
    pcm, seconds = b"", 5.0
    if clip is not None:
        with wave.open(os.path.join(SPOKEN_DIR, clip)) as w:
            pcm = w.readframes(w.getnframes())
        seconds = len(pcm) / 2 / 22050

    # Cues at random moments, kept clear of the start and end and of each
    # other, so the model is heard over different parts of her sentences.
    if kind in ("say_over", "say_quiet"):
        seconds = max(seconds, 6.0)
        cues_at = [random.uniform(2.0, max(2.5, seconds - 7.0))] if kind == "say_over" else [1.5]
    elif kind in ("over", "clap"):
        first = random.uniform(2.0, seconds / 2 - 1.5)
        cues_at = [first, random.uniform(max(first + 3.0, seconds / 2), seconds - 2.0)]
    elif kind == "quiet":
        cues_at = [1.5]
    else:
        cues_at = []

    where = f" from {spot}" if spot else ""
    what = {"over": "she plays; say it at each cue", "her": "she plays; stay silent",
            "quiet": "no playback; say it at the cue",
            "clap": "she plays; clap as each cue says, say nothing",
            "say_over": "she plays; at the cue, read the sentence aloud at a normal level",
            "say_quiet": "no playback; at the cue, read the sentence aloud"}[kind]
    print(f"\n=== take {number}: {kind}{where}, {seconds:.0f}s. {what}")
    input("    press Enter when ready... ")

    # Recorded at whatever the firmware delivers (48kHz only on 2.1.1), then
    # brought to RATE by the pipeline's own filter, so a take scores what the
    # wake model in production would have heard.
    raw_path = path + ".raw.wav"
    tail = 7 if kind in ("say_over", "say_quiet") else 3   # room to finish the sentence
    rec = subprocess.Popen(["arecord", "-q", "-D", mic, "-f", "S16_LE",
                            "-r", str(MIC_CAPTURE_RATE), "-c", "2",
                            "-d", str(int(seconds + tail)), raw_path])
    t0 = time.monotonic()
    time.sleep(1.0)
    play_at = time.monotonic() - t0
    heard_at = []
    if kind == "clap":
        asks = [random.choice(CLAPS) for _ in cues_at]
    elif kind in ("say_over", "say_quiet"):
        asks = [f'READ:  "{random.choice(SENTENCES)}"' for _ in cues_at]
    else:
        asks = ['SAY  "HEY NOVA"'] * len(cues_at)
    cue_threads = [threading.Thread(target=_cue, args=(1.0 + at, t0, heard_at, ask))
                   for at, ask in zip(cues_at, asks)]
    for c in cue_threads:
        c.start()
    if pcm:
        tts.play(_Clip(pcm))
    for c in cue_threads:
        c.join()
    rec.wait()
    _to_pipeline_rate(raw_path, path)

    said = _confirm(len(cues_at))
    return {"file": os.path.basename(path), "kind": kind, "spot": spot, "clip": clip,
            "play_at": round(play_at, 3), "play_s": round(seconds, 3) if pcm else 0.0,
            "cues": [{"at": at, "said": s, "asked": a}
                     for at, s, a in zip(heard_at, said, asks)]}


class _Recorded:
    """A finished recording, read the way ChannelPick reads the live stream."""

    def __init__(self, data):
        self._data, self._at = data, 0

    def read(self, frames, **kwargs):
        size = frames * 4                       # two channels of 16 bit
        chunk = self._data[self._at:self._at + size].ljust(size, b"\0")
        self._at += size
        return chunk


def _to_pipeline_rate(raw_path, path):
    """Both channels at RATE, through mic_channel's filter, as one stereo wav."""
    with wave.open(raw_path) as w:
        data, frames = w.readframes(w.getnframes()), w.getnframes()
    out_frames = frames // mic_channel.decimation(MIC_CAPTURE_RATE, RATE)
    channels = [np.frombuffer(
        mic_channel.ChannelPick(_Recorded(data), 2, pick, capture_rate=MIC_CAPTURE_RATE,
                                rate=RATE).read(out_frames), dtype=np.int16)
        for pick in (0, 1)]
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.column_stack(channels).tobytes())
    os.remove(raw_path)


def _plan(takes, clap=False, sentence=False, spots=SPOTS):
    """The order he goes through: over her by spot, her alone, then quiet."""
    plan = []
    if clap:
        return [("clap", spot) for spot in spots for _ in range(takes)]
    if sentence:
        return ([("say_over", spot) for spot in spots for _ in range(takes)]
                + [("say_quiet", spot) for spot in spots for _ in range(max(1, takes // 2))])
    for spot in spots:
        plan += [("over", spot)] * takes
    plan += [("her", None)] * max(1, takes * 4 // 5)
    for spot in spots:
        plan += [("quiet", spot)] * max(1, takes * 2 // 3)
    return plan


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takes", type=int, default=5,
                        help="over her takes per spot (default 5)")
    parser.add_argument("--clap", action="store_true",
                        help="clap at the cues instead, into data/barge_in_eval")
    parser.add_argument("--sentence", action="store_true",
                        help="read a sentence at each cue, into data/barge_in_eval")
    parser.add_argument("--far", action="store_true",
                        help="the far spots (bed, doorway, across the room) instead of his usual ones")
    args = parser.parse_args()

    mic_card = tts._resolve_speaker_device(MIC_NAME_HINT, fallback=None)
    if mic_card is None:
        sys.exit("The array was not found; nothing recorded.")
    tts.timing.note_tts = lambda *a, **k: None
    mic = mic_card.replace("plughw", "hw")

    barge_in = args.clap or args.sentence
    root = CLAP_DIR if barge_in else EVAL_DIR
    os.makedirs(root, exist_ok=True)
    if not barge_in:
        with open(os.path.join(EVAL_DIR, "README.md"), "w") as f:
            f.write(README)
    session = os.path.join(root, time.strftime("%Y%m%dT%H%M%S") + ("_far" if args.far else ""))
    os.makedirs(session)

    clips = _clips()
    random.shuffle(clips)
    plan = _plan(args.takes, clap=args.clap, sentence=args.sentence,
                 spots=FAR_SPOTS if args.far else SPOTS)
    print(f"{len(plan)} takes into {session}. Ctrl+C stops; takes already done are kept.")

    manifest = os.path.join(session, "manifest.jsonl")
    with _voice_service_paused():
        for number, (kind, spot) in enumerate(plan, 1):
            clip = (clips[number % len(clips)] if kind in ("over", "her", "clap", "say_over")
                    else None)
            row = _take(session, number, kind, spot, clip, mic)
            # One line per take, written at once, so an interrupted session
            # keeps everything it recorded.
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(f"\nDone: {session}")


if __name__ == "__main__":
    main()
