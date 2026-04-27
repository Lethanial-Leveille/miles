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
        self._action_detected = False
        self._sentence_queue  = sentence_queue
        self.action_tag       = None   # set when [ACTION:...] is found

    async def feed(self, delta: str) -> None:
        self._buf += delta

        if self._action_detected:
            # Continue flushing bridge sentences that arrive after the action tag
            await self._flush_sentences()
            return

        if not self._lookahead_done:
            if len(self._buf) < LOOKAHEAD_CHARS:
                return
            self._lookahead_done = True

        if ACTION_PREFIX in self._buf:
            await self._handle_action()
            return

        await self._flush_sentences()

    async def _handle_action(self) -> None:
        self._action_detected = True
        idx = self._buf.index(ACTION_PREFIX)

        pre = self._buf[:idx].strip()
        if pre:
            await self._sentence_queue.put(pre)

        tag_buf = self._buf[idx:]
        close   = tag_buf.find(']')

        if close == -1:
            self.action_tag = tag_buf.strip()
            self._buf = ""
        else:
            self.action_tag = tag_buf[:close + 1]
            # Set buf to post-] text so feed() keeps flushing the bridge sentence
            self._buf = tag_buf[close + 1:]

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
            self._buf = self._buf[end:]

    async def finalize(self) -> None:
        """Flush remaining buffer after stream ends, then send sentinel."""
        remaining = re.sub(r'\[.*$', '', self._buf).strip()
        if remaining:
            await self._sentence_queue.put(remaining)
        self._buf = ""
        await self._sentence_queue.put(None)
