"""Voice enrollment: build a speaker centroid from multiple varied samples.

Rewritten Aug 10 2026 after the previous voiceprint was found to be poisoned.

What went wrong before: samples were a fixed five second recording with no
check on how much of that was actually speech. The prompt list included "Lock
in", two syllables, which after silence trimming left well under a second of
voiced audio. Resemblyzer embeddings are unstable below roughly three seconds,
so that sample was close to noise, and it was averaged into the centroid with
the same weight as every good sample. A surviving enrollment recording scored
only 0.833 against the centroid it helped build, and the derived average
pairwise similarity across enrollment samples was about 0.63. One bad sample
accounts for the whole thing arithmetically.

Three defenses here:
  1. Every prompt is long enough to yield several seconds of speech.
  2. Trimmed voiced duration is measured per sample and short ones are
     rejected at record time, before they can reach the centroid.
  3. Individual embeddings are stored alongside the centroid, and pairwise
     similarity is reported, so an outlier is visible rather than silently
     averaged in.

Conditions are varied deliberately. Vocal effort is crossed with distance
because they are confounded in real use: stepping back makes you raise your
voice, and raised voice is a different production mode, not the same voice
louder. Enrolling only close and quiet trains on a condition that far field
usage never reproduces.
"""

import os
os.environ["ONNXRUNTIME_DISABLE_GPU"] = "1"
os.environ["JACK_NO_START_SERVER"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import re
import subprocess
import wave

import ctypes
import numpy as np
import pyaudio

# Suppress ALSA warnings before opening any device
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except OSError:
    pass

from resemblyzer import VoiceEncoder, preprocess_wav

from config import (
    RATE, CHANNELS, CHUNK, SPEAKER_NAME_HINT,
    VOICEPRINT_PATH, ENROLLMENT_DATA_PATH,
    MIN_VOICED_SECONDS, ENROLL_RECORD_SECONDS,
)

FORMAT = pyaudio.paInt16
TEMP_ENROLL = os.path.expanduser("~/miles/build/enroll_temp.wav")

# Outlier detection uses two tests because an absolute threshold alone is not
# enough. Resemblyzer embeddings are non negative (the encoder output is
# ReLU'd), so even unrelated audio floors around 0.75 rather than 0.0. A bad
# sample therefore does not look bad in absolute terms, it looks bad relative
# to how tightly the other samples agree.
OUTLIER_SIMILARITY = 0.80   # absolute floor: clearly wrong on its own
OUTLIER_MARGIN     = 0.08   # or this far below the median sample

# Each prompt must carry several seconds of speech on its own. Nothing here is
# short enough to trim down below the stability floor, which is the specific
# failure that ruined the previous voiceprint.
#
# Conditions cross vocal effort with distance. "Projected" means speaking as
# though across the room, which changes pitch, effort, and spectral tilt, not
# just loudness.
SAMPLES = [
    ("near, normal voice",
     "Tell me what the weather is going to be like this weekend"),
    ("near, normal voice",
     "Set a timer for twenty five minutes so I can finish this problem set"),
    ("near, normal voice",
     "Remind me to email my advisor about registration tomorrow morning"),
    ("near, projected",
     "Tell me what the weather is going to be like this weekend"),
    ("near, projected",
     "What time is my first class and how long do I have to get there"),
    ("far, projected",
     "Tell me what the weather is going to be like this weekend"),
    ("far, projected",
     "Set a timer for twenty five minutes so I can finish this problem set"),
    ("far, projected",
     "Remind me to email my advisor about registration tomorrow morning"),
    ("far, normal voice",
     "What time is my first class and how long do I have to get there"),
    ("facing away, normal voice",
     "Tell me about everything I have scheduled for tomorrow afternoon"),
    ("casual, relaxed delivery",
     "Yeah so I was thinking about that thing we talked about earlier"),
    ("careful, precise delivery",
     "Please confirm the current status of every system you are running"),
]


def resolve_speaker_device(name_hint, fallback="plughw:0,0"):
    """Card numbers shift between boots, so find the card by name.

    Duplicated from tts.py rather than imported: importing tts constructs an
    ElevenLabs client at module scope, and enrollment has no business
    requiring a TTS API key."""
    try:
        with open("/proc/asound/cards") as f:
            cards = f.read()
    except OSError:
        return fallback

    for block in re.split(r'\n(?=\s*\d+\s+\[)', cards):
        match = re.match(r'\s*(\d+)\s+\[', block)
        if match and name_hint in block:
            return f"plughw:{int(match.group(1))},0"
    return fallback


def find_mic(audio):
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if "Razer" in info["name"] or "Seiren" in info["name"]:
            return i
    return None


def record_sample(stream, audio, seconds):
    frames = []
    for _ in range(int(RATE / CHUNK * seconds)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    with wave.open(TEMP_ENROLL, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def playback(device):
    subprocess.run(["aplay", "-D", device, TEMP_ENROLL], capture_output=True)


def pairwise_matrix(embeddings):
    """Cosine similarity between every pair. Embeddings are already unit norm
    out of embed_utterance, but normalizing again costs nothing and keeps this
    correct if that ever changes."""
    n = len(embeddings)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = embeddings[i], embeddings[j]
            matrix[i][j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return matrix


def report_pairwise(embeddings, labels):
    """Print the similarity matrix and flag samples that disagree with the
    rest, so a bad recording is caught before it is averaged in rather than
    diagnosed months later from the centroid norm."""
    matrix = pairwise_matrix(embeddings)
    n = len(embeddings)

    print("\n=== Pairwise similarity between enrollment samples ===\n")
    print("      " + "".join(f"{i + 1:>7}" for i in range(n)))
    for i in range(n):
        row = "".join(f"{matrix[i][j]:>7.3f}" for j in range(n))
        print(f"  {i + 1:>3} {row}")

    # Mean similarity to every other sample, excluding self.
    means = []
    for i in range(n):
        others = [matrix[i][j] for j in range(n) if j != i]
        means.append(sum(others) / len(others))

    median = float(np.median(means))

    print("\nMean similarity to the other samples:")
    outliers = []
    for i, mean in enumerate(means):
        is_outlier = mean < OUTLIER_SIMILARITY or mean < median - OUTLIER_MARGIN
        flag = "   <-- OUTLIER, consider re recording" if is_outlier else ""
        if is_outlier:
            outliers.append(i + 1)
        print(f"  {i + 1:>3}. {mean:.3f}  [{labels[i]}]{flag}")

    overall = sum(means) / len(means)
    print(f"\nMedian sample agreement: {median:.3f}")
    print(f"Overall mean pairwise similarity: {overall:.3f}")

    if outliers:
        print(f"\n{len(outliers)} sample(s) flagged: {outliers}")
        print("A single bad sample drags the whole centroid. Re recording a")
        print("flagged sample is cheaper than living with a weak voiceprint.")
    elif overall < OUTLIER_SIMILARITY:
        print("\nWARNING: samples disagree with each other broadly. This centroid")
        print("will be weak. Consider re recording in quieter conditions.")
    else:
        print("\nSamples are mutually consistent. Centroid should be sound.")
    return means


def main():
    audio = pyaudio.PyAudio()
    mic_index = find_mic(audio)
    if mic_index is None:
        audio.terminate()
        raise SystemExit("Razer mic not found.")

    speaker = resolve_speaker_device(SPEAKER_NAME_HINT)

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=mic_index,
                        frames_per_buffer=CHUNK)

    encoder = VoiceEncoder()

    print("\n=== M.I.L.E.S. Voice Enrollment ===\n")
    print(f"{len(SAMPLES)} samples, {ENROLL_RECORD_SECONDS}s each.")
    print(f"Each needs at least {MIN_VOICED_SECONDS}s of actual speech after")
    print("silence trimming, or it will be rejected and re recorded.\n")
    print("Conditions vary on purpose. Follow the condition line exactly:")
    print("  projected  = speak as if across the room, not just louder")
    print("  far        = actually stand across the room")
    print("Keep talking for the whole recording. Repeat the phrase if you")
    print("finish early.\n")

    embeddings = []
    labels     = []
    durations  = []

    i = 0
    while i < len(SAMPLES):
        condition, phrase = SAMPLES[i]
        print(f"--- Sample {i + 1}/{len(SAMPLES)} ---")
        print(f"  Condition: {condition}")
        print(f"  Say: \"{phrase}\"")

        command = input("  Enter to record, 's' to skip, 'q' to quit: ").strip().lower()
        if command == 'q':
            print("Enrollment cancelled. Nothing was saved.")
            stream.stop_stream(); stream.close(); audio.terminate()
            return
        if command == 's':
            print("  Skipped.\n")
            i += 1
            continue

        print(f"  Recording {ENROLL_RECORD_SECONDS}s...", flush=True)
        record_sample(stream, audio, ENROLL_RECORD_SECONDS)

        wav = preprocess_wav(TEMP_ENROLL)
        voiced = len(wav) / RATE

        # The guard that would have caught "Lock in". Rejected before it can
        # reach the centroid, not flagged afterwards.
        if voiced < MIN_VOICED_SECONDS:
            print(f"  REJECTED: only {voiced:.2f}s of speech "
                  f"(need {MIN_VOICED_SECONDS}s). Speak for longer.\n")
            continue

        print(f"  Captured {voiced:.2f}s of speech. Playing back...")
        playback(speaker)

        while True:
            choice = input("  Keep (y), redo (n), replay (r)? ").strip().lower()
            if choice == 'r':
                playback(speaker)
                continue
            if choice == 'n':
                print("  Redoing.\n")
                break
            embeddings.append(encoder.embed_utterance(wav))
            labels.append(condition)
            durations.append(voiced)
            print(f"  Sample {i + 1} accepted.\n")
            i += 1
            break

    stream.stop_stream(); stream.close(); audio.terminate()

    if len(embeddings) < 6:
        raise SystemExit(
            f"Only {len(embeddings)} samples captured. Need at least 6 for a "
            "usable centroid. Nothing saved, run enrollment again."
        )

    report_pairwise(embeddings, labels)

    centroid = np.mean(embeddings, axis=0)
    np.save(VOICEPRINT_PATH, centroid)

    # Individual embeddings are kept so the centroid can be recomputed, a bad
    # sample dropped, or per condition analysis run later without re recording
    # anything. Not having these is why the previous voiceprint could only be
    # diagnosed indirectly through its norm.
    np.savez(
        ENROLLMENT_DATA_PATH,
        centroid=centroid,
        embeddings=np.array(embeddings),
        conditions=np.array(labels),
        voiced_seconds=np.array(durations),
    )

    print(f"\nCentroid saved to {VOICEPRINT_PATH}")
    print(f"Full enrollment data saved to {ENROLLMENT_DATA_PATH}")
    print(f"({len(embeddings)} samples, centroid norm {np.linalg.norm(centroid):.3f})")
    print("\nA centroid norm near 1.0 means the samples agree closely.")
    print("The previous poisoned voiceprint had a norm of 0.862.")


if __name__ == "__main__":
    main()
