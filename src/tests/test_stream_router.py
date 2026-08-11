"""StreamRouter after the tool use migration.

The action tag tests that used to live here are gone with the machinery they
covered: tag extraction, the split delta reassembly, and the LOOKAHEAD_CHARS
guard against a stray "[" in prose. Tool calls now arrive as their own content
blocks and never touch the text stream, so text is only ever text.

What remains is the part that always mattered: sentences reach TTS whole, in
order, and independently of how the network happened to chunk the deltas.
"""

import asyncio

import pytest

from stream_router import StreamRouter


async def _run(text, chunk=7):
    """Feed `text` through a router in fixed size chunks, the way streaming
    deltas actually arrive, and return the spoken sentences."""
    queue = asyncio.Queue()
    router = StreamRouter(queue)
    for i in range(0, len(text), chunk):
        await router.feed(text[i:i + chunk])
    await router.finalize()

    spoken = []
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            spoken.append(item)
    return spoken, router


def run(text, chunk=7):
    return asyncio.run(_run(text, chunk))


# ── sentence flushing ──

def test_sentences_are_flushed_whole_and_in_order():
    spoken, _ = run("Right away. The weather is clear. Nothing else to report.")
    assert spoken == ["Right away.", "The weather is clear.",
                      "Nothing else to report."]


def test_tail_without_terminal_punctuation_is_still_spoken():
    """A response cut off by max_tokens would otherwise never be spoken."""
    spoken, _ = run("Checking that now. It looks like")
    assert spoken == ["Checking that now.", "It looks like"]


def test_question_and_exclamation_end_sentences():
    spoken, _ = run("Ready? Absolutely! Done.")
    assert spoken == ["Ready?", "Absolutely!", "Done."]


def test_decimals_do_not_split_a_sentence():
    spoken, _ = run("The answer is 12.75 degrees. Nothing more.")
    assert spoken == ["The answer is 12.75 degrees.", "Nothing more."]


def test_abbreviations_do_not_split_a_sentence():
    spoken, _ = run("Dr. Chen replied. That is all.")
    assert spoken == ["Dr. Chen replied.", "That is all."]


@pytest.mark.parametrize("chunk", [1, 3, 7, 40, 500])
def test_result_is_independent_of_delta_size(chunk):
    """The network decides chunk boundaries, so nothing may depend on them."""
    text = "Right away. It is ninety one degrees. Rain is likely around 4 PM."
    spoken, _ = run(text, chunk=chunk)
    assert spoken == ["Right away.", "It is ninety one degrees.",
                      "Rain is likely around 4 PM."]


def test_sentences_emitted_counts_every_flush():
    """brain.py reads this to time the gap between first token and first
    speakable sentence. Queue size cannot be used: the TTS consumer runs on the
    same event loop and drains between feeds."""
    _, router = run("One. Two. Three.")
    assert router.sentences_emitted == 3


def test_empty_response_emits_nothing_but_still_terminates():
    spoken, router = run("")
    assert spoken == []
    assert router.sentences_emitted == 0


# ── what the migration removed ──

def test_brackets_in_prose_are_no_longer_special():
    """The lookahead existed only to stop a stray "[" being read as the start of
    an action tag. With tool calls in their own content blocks, an ordinary
    bracket is ordinary text and must survive intact."""
    spoken, _ = run("The reading was [redacted] at that point. Moving on.")
    assert spoken == ["The reading was [redacted] at that point.", "Moving on."]


def test_a_short_first_sentence_flushes_immediately():
    """Regression guard on the latency win. The router used to buffer 50
    characters before considering anything, and that wait sat directly on the
    path to first audio. A short opener must now flush on its own."""
    async def check():
        queue = asyncio.Queue()
        router = StreamRouter(queue)
        await router.feed("Right away. ")
        # Flushed on the feed itself, before finalize is ever called.
        assert router.sentences_emitted == 1
        assert await queue.get() == "Right away."

    asyncio.run(check())
