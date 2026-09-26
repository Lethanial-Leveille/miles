"""What reaches the speaker, and what the journal hears about it."""

import tts


def test_only_underruns_reach_the_journal(capsys):
    """aplay prints a banner on every start; only a dry buffer is news."""
    tts._report_underruns([
        b"Playing raw data 'stdin' : Signed 16 bit Little Endian, Rate 22050 Hz, Mono\n",
        b"underrun!!! (at least 412.337 ms long)\n",
    ])
    out = capsys.readouterr().out
    assert "speaker ran dry: underrun!!! (at least 412.337 ms long)" in out
    assert "Playing raw data" not in out


def test_aplay_errors_are_read_not_discarded():
    import inspect
    source = inspect.getsource(tts._open_aplay)
    assert "stderr=subprocess.PIPE" in source
    assert "_report_underruns" in source
