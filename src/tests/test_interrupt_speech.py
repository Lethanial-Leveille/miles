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
