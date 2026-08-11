import re
import asyncio
from config import LOOKAHEAD_CHARS, ACTION_PREFIX

# Sentence boundary: . ! ? not preceded by abbreviation (Dr.) or decimal (3.14)
_SENTENCE_END = re.compile(r'(?<![A-Z][a-z])(?<!\d)[.!?](?=\s|$)')


class StreamRouter:
    """
    Accumulates streaming text deltas, flushes complete sentences to a queue,
    and detects [ACTION:...] tags so brain.py can dispatch them without
    accidentally speaking the tag aloud.
    """

    def __init__(self, sentence_queue: asyncio.Queue):
        self._buf             = ""
        self._lookahead_done  = False
        self._sentence_queue  = sentence_queue
        self.action_tags      = []     # every [ACTION:...] found, in order
        # Counted rather than inferred from queue size: the TTS consumer runs
        # on the same event loop and can drain the queue between feeds, so
        # qsize() is not a reliable "has anything been emitted yet" signal.
        self.sentences_emitted = 0

    async def feed(self, delta: str) -> None:
        self._buf += delta

        if not self._lookahead_done:
            # The lookahead guards against a stray "[" in prose being mistaken
            # for a tag. ACTION_PREFIX is specific enough that its presence is
            # not ambiguous, so seeing it ends the wait early. Without this, a
            # response shorter than the lookahead never gets scanned at all,
            # and the prompt instructs the tag to come first.
            if len(self._buf) < LOOKAHEAD_CHARS and ACTION_PREFIX not in self._buf:
                return
            self._lookahead_done = True

        self._extract_actions()

        # An incomplete tag may still be sitting at the tail, waiting for the
        # rest of itself to arrive. Flush only what precedes it, or the tag
        # text gets spoken aloud one fragment at a time.
        idx = self._buf.find(ACTION_PREFIX)
        if idx == -1:
            await self._flush_sentences()
        elif idx > 0:
            pending, self._buf = self._buf[idx:], self._buf[:idx]
            await self._flush_sentences()
            self._buf += pending

    def _extract_actions(self) -> None:
        """Strip every complete [ACTION:...] tag out of the buffer, in order.

        A tag that has not received its closing bracket yet is left in place
        rather than committed. Deltas do not respect tag boundaries, so a tag
        routinely arrives split across two chunks; committing on the first
        chunk truncated it mid word and left the remainder to be spoken."""
        while True:
            idx = self._buf.find(ACTION_PREFIX)
            if idx == -1:
                return
            close = self._buf.find(']', idx)
            if close == -1:
                return
            self.action_tags.append(self._buf[idx:close + 1].strip())
            self._buf = self._buf[:idx] + self._buf[close + 1:]

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
        """Flush remaining buffer after stream ends, then send sentinel."""
        # Backstop: a stream that ended before the lookahead released still has
        # its tags sitting in the buffer, and the strip below would delete them
        # along with any genuinely incomplete tag.
        self._extract_actions()

        # Drop only a trailing incomplete tag. Every complete one is already
        # out, so anything left starting with ACTION_PREFIX is a fragment. The
        # old rule cut from any "[" to the end, which silently swallowed the
        # rest of a sentence containing an ordinary bracket.
        pending  = self._buf.find(ACTION_PREFIX)
        remaining = (self._buf[:pending] if pending != -1 else self._buf).strip()
        if remaining:
            await self._sentence_queue.put(remaining)
            self.sentences_emitted += 1
        self._buf = ""
        await self._sentence_queue.put(None)
