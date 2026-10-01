"""When a command recording is finished: the stop decision alone.

Kept out of audio.py so it can be tested while miles-voice holds the microphone.

SILENCE_LIMIT is how long a pause means he has finished. It was also, until
Oct 1 2026, how long he had to start: a wake, the chime, a breath before the
question, and a recording of 1.2s of nothing went to Deepgram and came back
empty, with no answer and no sign anything had gone wrong. 3 of 21 wakes on Sep
30 and 3 of 6 the next morning, two of them his questions about Yuka and
sparkling ice. Before he speaks the wait is COMMAND_ONSET_S; after, SILENCE_LIMIT.
"""


def recording_done(total_chunks, min_chunks, silent_chunks, heard_speech,
                   chunks_for_silence, chunks_for_onset):
    if total_chunks <= min_chunks:
        return False
    limit = chunks_for_silence if heard_speech else chunks_for_onset
    return silent_chunks >= limit
