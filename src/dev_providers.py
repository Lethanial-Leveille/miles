"""Deterministic, in-process providers selected by the isolated runtime."""

from datetime import datetime, time, timedelta
from functools import cache
from types import SimpleNamespace


CHAT = "This is a synthetic development reply."
SESSION = "This is a synthetic session reply."


class Stream:
    def __init__(self, text):
        block = SimpleNamespace(type="text", text=text)
        usage = SimpleNamespace(input_tokens=0, output_tokens=0,
                                cache_read_input_tokens=0, cache_creation_input_tokens=0)
        self.message = SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")
        middle = len(text) // 2
        self.events = [SimpleNamespace(type="content_block_start", content_block=block)]
        self.events.extend(SimpleNamespace(type="content_block_delta",
                                            delta=SimpleNamespace(type="text_delta", text=part))
                           for part in (text[:middle], text[middle:]))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


class AsyncStream(Stream):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def get_final_message(self):
        return self.message


class Claude:
    def __init__(self, *, asynchronous=False, summary="Synthetic session summary."):
        self.messages = self
        self.asynchronous = asynchronous
        self.summary = summary

    def stream(self, **kwargs):
        return AsyncStream(CHAT) if self.asynchronous else Stream(SESSION)

    def create(self, **kwargs):
        if kwargs.get("tool_choice", {}).get("name") == "record_intent":
            # The dev profile has no model to read intent, so it records none.
            return SimpleNamespace(content=[SimpleNamespace(
                type="tool_use", name="record_intent", id="intent",
                input={"action": "no_change", "target": "", "timing": "", "more_than_one_change": False})],
                usage=SimpleNamespace(input_tokens=0, output_tokens=0))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.summary)],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))


def encode(texts):
    import zlib
    import numpy as np

    rows = list(texts)
    vectors = np.zeros((len(rows), 64), dtype=np.float32)
    for text, vector in zip(rows, vectors):
        for word in text.lower().replace(".", "").split():
            vector[zlib.crc32(word.encode()) % len(vector)] += 1
        vector /= np.linalg.norm(vector) or 1
    return vectors


@cache
def calendar():
    from fake_calendar import FakeCalendar, Service, PRIMARY, MILES

    fake = FakeCalendar()
    morning = datetime.combine(datetime.now().date(), time(10))
    fake.add(PRIMARY, "fake-ordinary", "Sample appointment", morning,
             morning + timedelta(hours=1))
    fake.add(MILES, "fake-conflict", "Sample conflict", morning + timedelta(minutes=30),
             morning + timedelta(hours=1, minutes=30))
    fake.add(PRIMARY, "fake-evallo", "Alex tutoring", morning + timedelta(hours=3),
             morning + timedelta(hours=4), extendedProperties={"private": {
                 "evalloEvent": "true", "evalloSessionId": "a" * 24, "orgId": "fake-org"}})
    return Service(fake)
