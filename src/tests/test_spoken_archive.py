"""What Nova said is kept as she said it. Nothing here reaches ElevenLabs or
the speaker, and everything is written under tmp_path."""

import asyncio
import os
import threading
import time
import wave

import brain
import spoken_archive

ONE_SECOND = b"\x01\x00" * 22050


def _wavs(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".wav"))


def test_the_silence_he_heard_is_put_back_between_pieces(tmp_path):
    path = spoken_archive.save_reply(
        [("One.", ONE_SECOND, 900.0), ("Two... Three.", ONE_SECOND, 500.0)],
        directory=str(tmp_path))
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (22050, 1, 2)
        # The first piece's wait is time to first audio, not a pause in the reply.
        assert w.getnframes() == 22050 * 2 + int(22050 * 0.5)


def test_the_text_is_what_elevenlabs_was_sent(tmp_path):
    path = spoken_archive.save_reply(
        [("One.", ONE_SECOND, None), ("Two... Three.", ONE_SECOND, 12.4)],
        directory=str(tmp_path))
    with open(path[:-4] + ".txt") as f:
        assert f.read() == "One.\n[12ms before this piece]\nTwo... Three.\n"


def test_nothing_is_written_when_nothing_played(tmp_path):
    assert spoken_archive.save_reply([("One.", b"", 0)], directory=str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_labelled_replies_outlive_the_rest_and_take_their_text_with_them(tmp_path):
    first = spoken_archive.save_reply([("Off.", ONE_SECOND, 0)], directory=str(tmp_path))
    (tmp_path / "labels.csv").write_text(f"file,label\n{os.path.basename(first)},off\n")
    os.utime(first, (0, 0))   # the oldest, so only its label can save it
    for _ in range(3):
        time.sleep(0.002)   # names are stamped to the millisecond
        spoken_archive.save_reply([("Fine.", ONE_SECOND, 0)],
                                  directory=str(tmp_path), max_files=2)
    wavs = _wavs(tmp_path)
    assert len(wavs) == 2 and os.path.basename(first) in wavs
    txts = sorted(f for f in os.listdir(tmp_path) if f.endswith(".txt"))
    assert [t[:-4] for t in txts] == [w[:-4] for w in wavs]


def test_the_consumer_hands_over_each_piece_with_its_wait(monkeypatch):
    class Played:
        def __init__(self, text):
            self.text, self.audio = text, bytearray(b"\x02\x00")

    saved, done = [], threading.Event()
    monkeypatch.setattr(brain, "start_synthesis", Played)
    monkeypatch.setattr(brain, "_play_with_barge_in",
                        lambda s: (False, 1.0 if s.text == "One." else 640.0))
    monkeypatch.setattr(spoken_archive, "save_reply",
                        lambda pieces: (saved.extend(pieces), done.set()))

    async def go():
        queue = asyncio.Queue()
        for sentence in ["One.", "Two.", "Three.", None]:
            queue.put_nowait(sentence)
        await brain._tts_consumer(queue, [], set())
    asyncio.run(go())

    assert done.wait(timeout=2)
    assert saved == [("One.", b"\x02\x00", 1.0), ("Two... Three.", b"\x02\x00", 640.0)]
