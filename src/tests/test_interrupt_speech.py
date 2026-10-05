import numpy as np

from interrupt_speech import FRAME, SpeechOnset, frame_dbfs

FLOOR, NEED_MS = -40.0, 300     # ten 30ms frames
CHUNK = 1280                    # what the watcher reads, 80ms


def frames(dbfs, n):
    """n 30ms frames of a steady tone at the given RMS level."""
    t = np.arange(FRAME * n) / 16000
    amplitude = 10 ** (dbfs / 20) * np.sqrt(2) * 32768
    return (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()


def always(_):
    return True


def never(_):
    return False


def feed_in_chunks(onset, pcm, size=CHUNK):
    fired_at = None
    for i in range(0, len(pcm), size * 2):
        if onset.feed(pcm[i:i + size * 2]) and fired_at is None:
            fired_at = i
    return fired_at


def test_the_level_is_rms():
    assert abs(frame_dbfs(np.frombuffer(frames(-30, 1), dtype=np.int16)) + 30) < 0.1


def test_loud_speech_for_the_whole_duration_fires():
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    assert feed_in_chunks(onset, frames(-30, 10)) is not None


def test_one_frame_short_does_not():
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    assert feed_in_chunks(onset, frames(-30, 9)) is None


def test_loud_but_not_speech_never_fires():
    """A door, a fan: level alone is not him."""
    onset = SpeechOnset(FLOOR, NEED_MS, never)
    assert feed_in_chunks(onset, frames(-10, 50)) is None


def test_speech_under_the_floor_never_fires():
    """Her echo, which VAD does call speech now and then."""
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    assert feed_in_chunks(onset, frames(-45, 50)) is None


def test_a_quiet_frame_restarts_the_run():
    """Her runs on the replay were 180ms at most, broken up; they must not add."""
    pcm = frames(-30, 6) + frames(-60, 1) + frames(-30, 6)
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    assert feed_in_chunks(onset, pcm) is None


def test_frames_carry_across_reads_rather_than_restarting():
    """80ms reads do not divide into 30ms frames. Dropping the remainder would
    lose a third of a frame per read and fire on audio that was never measured."""
    pcm = frames(-60, 3) + frames(-30, 10)
    whole = SpeechOnset(FLOOR, NEED_MS, always)
    whole.feed(pcm)
    chunked = SpeechOnset(FLOOR, NEED_MS, always)
    feed_in_chunks(chunked, pcm)
    assert chunked.fired and chunked.onset == whole.onset == 3 * FRAME


def test_onset_is_where_his_speech_began_not_where_it_fired():
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    onset.feed(frames(-60, 20) + frames(-30, 12))
    assert onset.onset == 20 * FRAME


def test_vad_hears_every_frame_quiet_ones_too():
    """webrtcvad is stateful. Shown only the loud frames, it took her loudest
    moments for the background and missed him 2 of 29 on the Oct 1 2026 replay."""
    asked = []
    onset = SpeechOnset(FLOOR, NEED_MS, lambda f: asked.append(f) or True)
    onset.feed(frames(-60, 20))
    assert len(asked) == 20


def test_it_stays_fired():
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    onset.feed(frames(-30, 10))
    assert onset.feed(frames(-60, 1))


# ── the watcher: structural, because audio.py cannot be imported while
# miles-voice holds the mic (see test_speculation.py) ──

def test_the_watcher_stops_her_when_the_speech_rule_fires():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "audio.py").read_text()
    watcher = next(n for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.FunctionDef) and n.name == "watch_for_interrupt")
    text = ast.unparse(watcher)
    assert "if BARGE_IN_SPEECH:" in text, "the speech rule ignores its own switch"
    assert "interrupt_speech.SpeechOnset(" in text
    fires = next(n for n in ast.walk(watcher)
                 if isinstance(n, ast.If) and "speech.feed(data)" in ast.unparse(n.test))
    assert "stop_event.set()" in ast.unparse(fires), (
        "the speech rule fires and nothing stops her")


def test_every_stop_keeps_its_audio():
    """build/ keeps four sentences; the first live session lost both stops he
    suspected were claps before they could be looked at."""
    import ast
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "audio.py").read_text()
    watcher = next(n for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.FunctionDef) and n.name == "watch_for_interrupt")
    saves = next(n for n in ast.walk(watcher)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "heard")
    assert "BARGE_IN_STOP_DIR" in ast.unparse(saves)


# ── step 2: keeping what he said ──

import threading
import time

from interrupt_speech import Carrier, seed_from


def test_the_seed_starts_a_preroll_before_his_onset():
    kept = bytes(range(200)) * 100                   # 10000 samples
    assert seed_from(kept, 5000, 4800) == kept[200 * 2:]
    assert seed_from(kept, 1000, 4800) == kept, "never before the first byte"


def test_the_carrier_keeps_reading_until_claimed():
    reads = []

    def read():
        reads.append(1)
        time.sleep(0.005)
        return b"\x01\x00" * 1280

    carrier = Carrier(b"seed", read, max_seconds=30)
    carrier.start()
    time.sleep(0.05)
    pcm = carrier.claim()
    assert pcm.startswith(b"seed") and len(reads) >= 2
    count = len(reads)
    time.sleep(0.03)
    assert len(reads) == count, "after the claim it must never read the mic again"


def test_the_carrier_stops_by_itself_at_the_cap():
    carrier = Carrier(b"", lambda: b"\x00\x00" * 1600, max_seconds=1.0)
    carrier.start()
    assert carrier._done.wait(1.0)
    assert len(carrier.claim()) == 16000 * 2


def test_a_failing_read_ends_the_carrier_with_what_it_had():
    def read():
        raise OSError("gone")
    carrier = Carrier(b"seed", read, max_seconds=30)
    carrier.start()
    assert carrier.claim() == b"seed"


def _source_of(module, name):
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parent.parent / module).read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_only_a_speech_stop_is_carried_and_only_after_the_watcher_reads_no_more():
    import ast
    watcher = _source_of("audio.py", "watch_for_interrupt")
    hand_off = next(n for n in ast.walk(watcher) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "heard == 'speech'"
                    and "Carrier(" in ast.unparse(n))
    loop = next(n for n in ast.walk(watcher) if isinstance(n, ast.While))
    assert hand_off.lineno > loop.end_lineno, "two readers on one stream"
    assert "channel=BARGE_IN_CHANNEL" in ast.unparse(hand_off)


def test_a_flush_releases_the_carrier_before_it_reads():
    import ast
    flush = ast.unparse(_source_of("audio.py", "flush_input"))
    assert flush.index("claim_interruption()") < flush.index("stream.read")


def test_the_voice_loop_claims_his_interruption_instead_of_flushing_it():
    """Oct 1 2026: what he said after cutting in was flushed as her echo, and
    "Are you?" was all that reached Claude."""
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "voice_main.py").read_text()
    assert source.count("interruption = audio.claim_interruption()\n") == 2, (
        "both replies, the first and every follow up, can be interrupted")
    assert "seed=interruption" in source
    assert "learn=not interrupted_turn" in source


def test_an_interruption_never_teaches_the_voiceprint():
    import ast
    verify = ast.unparse(_source_of("audio.py", "verify_voice"))
    keep = next(line for line in verify.splitlines() if "accepted and learn" in line)
    assert keep


def test_a_short_interruption_keeps_her_listening():
    """Oct 2 2026: "Wait." then a pause; 0.48s of voice ended the conversation
    with him mid thought."""
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "voice_main.py").read_text()
    for branch in ("if is_noise_transcript(followup_text):",
                   "if followup_result == audio.NO_AUDIO:"):
        body = source[source.index(branch):].split("break", 1)[0]
        assert "if interrupted_turn:" in body and "continue" in body, branch


# ── step 3: what she said before he cut in ──

from interrupt_speech import HER_WORDS_PER_S, words_heard

TEN = "one two three four five six seven eight nine ten"


def test_heard_by_share_of_the_audio_when_its_length_is_known():
    assert words_heard(TEN, heard_s=2.0, audio_s=4.0) == "one two three four five"


def test_heard_by_her_pace_when_the_audio_had_not_all_arrived():
    assert words_heard(TEN, heard_s=1.0) == " ".join(TEN.split()[:round(HER_WORDS_PER_S)])


def test_heard_never_more_than_was_said_nor_less_than_nothing():
    assert words_heard(TEN, heard_s=60.0) == TEN
    assert words_heard(TEN, heard_s=0.0, audio_s=4.0) == ""


# ── her own echo is not him ──

from interrupt_speech import (APLAY_START_BYTES, ENVELOPE_FLOOR, HerAudio,
                              echo_score, envelope)


def syllables(n, seed):
    """n frames of a speech like envelope: loud and quiet stretches."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) > 0.5, -25.0, -50.0) + rng.normal(0, 2, n)


def test_envelope_lines_up_across_rates():
    """Her 22050Hz audio and the 16kHz mic must give one level per 30ms each."""
    t16 = np.arange(16000) / 16000
    t22 = np.arange(22050) / 22050
    a = (3000 * np.sin(2 * np.pi * 300 * t16)).astype(np.int16)
    b = (3000 * np.sin(2 * np.pi * 300 * t22)).astype(np.int16)
    assert len(envelope(a, 16000)) == len(envelope(b, 22050)) == 33
    assert abs(envelope(a, 16000)[5] - envelope(b, 22050)[5]) < 0.1


def test_envelope_clamps_digital_silence():
    assert envelope(np.zeros(4800, dtype=np.int16), 16000).min() == ENVELOPE_FLOOR


def test_her_delayed_copy_scores_high_at_its_delay():
    hers = syllables(60, 1)
    mic = hers[60 - 40 - 5:60 - 5] - 15         # 150ms late, quieter
    assert echo_score(mic, hers, 1, 10) > 0.95


def test_a_delay_outside_the_search_is_not_found():
    """The window is the guard against luck; a match outside it is not hers."""
    hers = syllables(80, 1)
    mic = hers[80 - 40 - 30:80 - 30]            # 900ms late
    assert echo_score(mic, hers[-50:], 1, 10) < 0.5


def test_his_own_syllables_score_low():
    assert echo_score(syllables(40, 2), syllables(50, 1), 1, 10) < 0.5


def test_her_silence_never_counts_as_a_match():
    assert echo_score(syllables(40, 2), np.full(50, ENVELOPE_FLOOR), 1, 10) == -1.0


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_her_clock_starts_when_aplay_buffer_fills_not_at_the_first_write():
    clock = Clock()
    her = HerAudio(clock=clock)
    her.wrote(b"\x00\x00" * 1000)
    clock.t = 100.5
    her.wrote(b"\x00\x00" * 8000)
    assert her.playing_from == 100.5


def test_a_short_sentence_starts_when_aplay_input_closes():
    clock = Clock()
    her = HerAudio(clock=clock)
    her.wrote(b"\x00\x00" * 100)
    clock.t = 101.0
    her.closed()
    assert her.playing_from == 101.0


def test_her_envelope_is_placed_where_she_played():
    clock = Clock()
    her = HerAudio(clock=clock)
    loud = (8000 * np.sin(2 * np.pi * 300 * np.arange(22050) / 22050)).astype(np.int16)
    her.wrote(loud.tobytes())                   # 1s, past the buffer: plays from 100.0
    env = her.envelope_until(100.6, 40)         # 1.2s ending 0.6s into her
    assert (env[:20] == ENVELOPE_FLOOR).all()   # before she began
    assert (env[20:] > -20).all()


def test_nothing_played_yet_is_all_floor():
    assert (HerAudio().envelope_until(5.0, 10) == ENVELOPE_FLOOR).all()


def test_echo_does_not_stop_her_and_him_after_it_does():
    """Oct 4 and 5 2026: her first syllables leaked through and stopped her."""
    calls = []

    def echo_for_the_first_twenty(levels):
        calls.append(len(levels))
        return len(levels) <= 20

    onset = SpeechOnset(FLOOR, NEED_MS, always, is_echo=echo_for_the_first_twenty)
    onset.feed(frames(-30, 20))
    assert not onset.fired and onset.echo_frames == 11
    onset.feed(frames(-30, 1))
    assert onset.fired


def test_after_echo_his_onset_is_where_the_firing_frames_began():
    """Kept from the run's start, the next recording opened on her words: Oct 4
    the carried "interruption" transcribed as "I didn't catch the..."."""
    onset = SpeechOnset(FLOOR, NEED_MS, always, is_echo=lambda lv: len(lv) <= 20)
    onset.feed(frames(-60, 5) + frames(-30, 21))
    assert onset.onset == (21 - 10) * FRAME     # fired at frame 21; the run began at 5


def test_without_an_echo_check_nothing_changes():
    onset = SpeechOnset(FLOOR, NEED_MS, always)
    onset.feed(frames(-60, 5) + frames(-30, 12))
    assert onset.onset == 5 * FRAME and onset.echo_frames == 0


def test_the_aplay_buffer_matches_what_tts_opens():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "tts.py").read_text()
    assert f"--buffer-size={APLAY_START_BYTES // 2}" in source


def test_play_tells_her_audio_every_write_and_the_close():
    import ast
    play = ast.unparse(_source_of("tts.py", "play"))
    assert play.index("aplay.stdin.write(chunk)") < play.index("her.wrote(chunk)")
    assert play.index("aplay.stdin.close()") < play.index("her.closed()")


def test_the_watcher_and_the_player_share_one_her_audio():
    import ast
    body = ast.unparse(_source_of("brain.py", "_play_with_barge_in"))
    assert "args=(interrupt, her)" in body and "her=her" in body
