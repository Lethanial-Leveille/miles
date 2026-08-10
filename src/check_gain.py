"""Measure capture level so mic gain gets tuned against numbers rather than
against how loud playback sounds.

Reports RMS and peak in dBFS, how much of the sixteen bit range is actually
in use, clipping count, and what webrtcvad makes of the result at the mode
the live capture path uses. That last number is the one that matters most:
gain is not an aesthetic setting here, it directly determines whether the
VAD fires.

Usage:
  python3 check_gain.py                      # record 5s from the mic
  python3 check_gain.py --seconds 8
  python3 check_gain.py --file ../build/command.wav

Tuning loop:
  amixer -c 0 sset Mic <value>
  python3 check_gain.py --seconds 5          # while speaking worst case:
                                             # close range, projecting
Raise until peak lands in the target band. Peak matters more than RMS,
because clipping is unrecoverable and destroys embeddings far worse than
quiet audio does.
"""

import argparse
import os
import subprocess
import sys
import wave

import numpy as np

# Peak targets. Clipping is destructive and irreversible, so the ceiling is
# a hard limit while the floor is only a quality preference.
PEAK_TARGET_LOW  = -12.0
PEAK_TARGET_HIGH = -6.0
PEAK_DANGER      = -3.0

RATE = 16000


def dbfs(x):
    """Full scale here is 1.0, so a sine at full amplitude reads 0 dBFS."""
    if x <= 0:
        return float("-inf")
    return 20 * np.log10(x)


def voice_service_running():
    result = subprocess.run(
        ["systemctl", "is-active", "--quiet", "miles-voice"], capture_output=True
    )
    return result.returncode == 0


def record(seconds):
    """Record from the Razer mic. Imports pyaudio lazily so --file works even
    when the mic is busy."""
    import pyaudio

    audio = pyaudio.PyAudio()
    mic_index = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if "Razer" in info["name"] or "Seiren" in info["name"]:
            mic_index = i
            break

    if mic_index is None:
        audio.terminate()
        sys.exit("Razer mic not found.")

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                        input=True, input_device_index=mic_index,
                        frames_per_buffer=480)

    print(f"Recording {seconds}s. Speak your worst case now: close range, "
          "projecting as if across the room.", flush=True)
    frames = []
    for _ in range(int(RATE / 480 * seconds)):
        frames.append(stream.read(480, exception_on_overflow=False))
    print("Done.\n", flush=True)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return np.frombuffer(b"".join(frames), dtype=np.int16)


def load(path):
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != RATE:
            print(f"Note: file is {wf.getframerate()}Hz, not {RATE}Hz.")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def speech_frame_ratio(samples):
    """What the live capture path would see. Runs on raw audio at the
    configured mode, which is the whole point: resemblyzer normalizes before
    its own VAD pass, but our capture VAD does not get that luxury."""
    try:
        import webrtcvad
        from config import VAD_MODE
    except ImportError:
        return None, None

    vad = webrtcvad.Vad(VAD_MODE)
    raw = samples.tobytes()
    total = speech = 0
    for i in range(0, len(raw) - 960, 960):
        total += 1
        if vad.is_speech(raw[i:i + 960], RATE):
            speech += 1
    return (speech / total if total else 0.0), VAD_MODE


def report(samples):
    if samples.size == 0:
        sys.exit("No audio.")

    normalized = samples.astype(np.float64) / 32768.0
    rms  = float(np.sqrt(np.mean(normalized ** 2)))
    peak = float(np.abs(normalized).max())

    rms_db  = dbfs(rms)
    peak_db = dbfs(peak)
    clipped = int((np.abs(samples) >= 32767).sum())

    # log2 of the peak's fraction of full scale gives the bits left unused.
    bits_used = np.log2(peak * 32768) if peak > 0 else 0

    print(f"Duration:      {samples.size / RATE:.2f}s")
    print(f"RMS level:     {rms_db:>7.1f} dBFS")
    print(f"Peak level:    {peak_db:>7.1f} dBFS")
    print(f"Crest factor:  {peak_db - rms_db:>7.1f} dB")
    print(f"Bit depth in use: {bits_used:.1f} of 16 "
          f"({16 - bits_used:.1f} bits unused)")
    print(f"Clipped samples: {clipped}"
          + ("  <-- CLIPPING, BACK OFF" if clipped else ""))

    ratio, mode = speech_frame_ratio(samples)
    if ratio is not None:
        print(f"webrtcvad mode {mode}: {ratio:.0%} of frames read as speech")

    print()
    print(f"Target peak: {PEAK_TARGET_LOW:.0f} to {PEAK_TARGET_HIGH:.0f} dBFS "
          "on worst case input")

    if clipped or peak_db >= PEAK_DANGER:
        print("VERDICT: TOO HOT. Lower the gain until clipping stops.")
    elif peak_db < PEAK_TARGET_LOW:
        short_by = PEAK_TARGET_LOW - peak_db
        print(f"VERDICT: TOO QUIET by roughly {short_by:.0f} dB. Raise the gain.")
        print(f"         Every 6 dB recovers about one bit of resolution, so "
              f"this is worth about {short_by / 6:.1f} bits.")
    elif peak_db > PEAK_TARGET_HIGH:
        print("VERDICT: hot but not clipping. Fine if this really was your "
              "loudest case, otherwise back off a step.")
    else:
        print("VERDICT: in the target band. Leave it here.")

    print()
    print("Current mixer setting:")
    subprocess.run(["amixer", "-c", "0", "sget", "Mic"])


def main():
    parser = argparse.ArgumentParser(description="Measure mic capture level.")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="recording length (default 5)")
    parser.add_argument("--file", help="analyze a wav instead of recording")
    args = parser.parse_args()

    if args.file:
        report(load(os.path.expanduser(args.file)))
        return

    # The voice loop holds the mic exclusively, so recording here would either
    # fail or fight it for the device.
    if voice_service_running():
        sys.exit(
            "miles-voice.service is running and holds the mic.\n"
            "  sudo systemctl stop miles-voice\n"
            "  python3 check_gain.py --seconds 5\n"
            "  sudo systemctl start miles-voice\n"
            "Or analyze an existing recording: "
            "python3 check_gain.py --file ../build/command.wav"
        )

    report(record(args.seconds))


if __name__ == "__main__":
    main()
