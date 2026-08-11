import re
import asyncio

# Sentence boundary: . ! ? not preceded by abbreviation (Dr.) or decimal (3.14)
_SENTENCE_END = re.compile(r'(?<![A-Z][a-z])(?<!\d)[.!?](?=\s|$)')


class StreamRouter:
    """Accumulates streaming text deltas and flushes complete sentences to a
    queue for TTS.

    Previously this also detected [ACTION:...] tags mid stream and buffered
    LOOKAHEAD_CHARS before emitting anything, so a stray "[" in prose could not
    be mistaken for the start of a tag. Native tool use removed the need for
    both: tool calls arrive as their own content blocks and never appear in the
    text stream, so text is only ever text.

    Dropping the lookahead is a latency win as well as a simplification. The
    first sentence used to wait for 50 characters to accumulate before it could
    be considered at all, and that wait sat directly on the path to first
    audio. Now a short first sentence flushes as soon as it is complete.

    Emotion cue leakage is still possible ("[calmly] Right away") and is still
    handled, but downstream: strip_leading_bracket_cue runs per sentence in the
    TTS consumer, which is where it always ran.
    """

    def __init__(self, sentence_queue: asyncio.Queue):
        self._buf            = ""
        self._sentence_queue = sentence_queue
        # Counted rather than inferred from queue size: the TTS consumer runs
        # on the same event loop and can drain the queue between feeds, so
        # qsize() is not a reliable "has anything been emitted yet" signal.
        self.sentences_emitted = 0

    async def feed(self, delta: str) -> None:
        self._buf += delta
        await self._flush_sentences()

    async def _flush_sentences(self) -> None:
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            end      = match.end()
            sentence = self._buf[:end].strip()
            if sentence:
                await self._sentence_queue.put(sentence)
                self.sentences_emitted += 1
            self._buf = self._buf[end:]

    async def finalize(self) -> None:
        """Flush whatever is left after the stream ends, then send the sentinel.

        The tail matters: a response ending without terminal punctuation, or
        cut off by max_tokens, would otherwise never be spoken at all."""
        remaining = self._buf.strip()
        if remaining:
            await self._sentence_queue.put(remaining)
            self.sentences_emitted += 1
        self._buf = ""
        await self._sentence_queue.put(None)
