from endpoint import recording_done

# 30ms chunks, as record_command counts them.
MIN, SILENCE, ONSET = 33, 40, 100


def test_a_pause_after_the_chime_is_not_the_end():
    """Oct 1 2026: 1.2s of nothing before his question ended the recording."""
    assert not recording_done(60, MIN, 60, heard_speech=False,
                              chunks_for_silence=SILENCE, chunks_for_onset=ONSET)


def test_he_never_speaks_and_it_still_ends():
    assert recording_done(100, MIN, 100, heard_speech=False,
                          chunks_for_silence=SILENCE, chunks_for_onset=ONSET)


def test_once_he_has_spoken_the_short_limit_ends_it():
    assert recording_done(200, MIN, SILENCE, heard_speech=True,
                          chunks_for_silence=SILENCE, chunks_for_onset=ONSET)


def test_a_pause_shorter_than_the_limit_does_not():
    assert not recording_done(200, MIN, SILENCE - 1, heard_speech=True,
                              chunks_for_silence=SILENCE, chunks_for_onset=ONSET)


def test_never_before_the_minimum():
    assert not recording_done(MIN, MIN, 500, heard_speech=True,
                              chunks_for_silence=SILENCE, chunks_for_onset=ONSET)
