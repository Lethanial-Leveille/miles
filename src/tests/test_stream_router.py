import asyncio

import pytest

from stream_router import StreamRouter


async def _run(text, chunk=7):
    """Feed `text` through a router in fixed size chunks, the way streaming
    deltas actually arrive, and return (action_tags, spoken_sentences)."""
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
    return router.action_tags, spoken


def run(text, chunk=7):
    return asyncio.run(_run(text, chunk))


def test_two_actions_in_one_turn_are_both_captured():
    """Asking for two things at once dispatched only the first, and spoke the
    second tag aloud."""
    tags, spoken = run(
        "Setting that up. [ACTION: timer | duration: 10 minutes] "
        "[ACTION: reminder | content: call mom | due: tomorrow] Both set."
    )
    assert tags == [
        "[ACTION: timer | duration: 10 minutes]",
        "[ACTION: reminder | content: call mom | due: tomorrow]",
    ]
    assert spoken == ["Setting that up.", "Both set."]


def test_tag_split_across_deltas_is_not_truncated():
    """Deltas do not respect tag boundaries. Committing on the first chunk cut
    the tag mid word and left the remainder to be spoken aloud."""
    tags, spoken = run("Okay. [ACTION: timer | duration: 10 minutes] Done.", chunk=3)
    assert tags == ["[ACTION: timer | duration: 10 minutes]"]
    assert not any("ACTION" in s or "inutes" in s for s in spoken)


def test_action_tag_before_any_speech_survives():
    """The prompt instructs the tag to come first, so a short response never
    reached the lookahead and finalize deleted the tag along with the text."""
    tags, spoken = run("[ACTION: timer | duration: 5 minutes] Timer set.")
    assert tags == ["[ACTION: timer | duration: 5 minutes]"]
    assert spoken == ["Timer set."]


def test_two_tags_before_any_speech():
    tags, spoken = run(
        "[ACTION: timer | duration: 5 minutes]"
        "[ACTION: weather | location: Miami] On it."
    )
    assert len(tags) == 2
    assert spoken == ["On it."]


def test_ordinary_brackets_in_prose_are_not_treated_as_tags():
    """Stripping from any "[" to end swallowed the rest of the sentence."""
    tags, spoken = run("I read chapter [four] of the book. It was long.")
    assert tags == []
    assert " ".join(spoken) == "I read chapter [four] of the book. It was long."


def test_no_action_still_flushes_sentences():
    tags, spoken = run("The capital of Japan is Tokyo. It has been for a long time.")
    assert tags == []
    assert spoken == ["The capital of Japan is Tokyo.",
                      "It has been for a long time."]


@pytest.mark.parametrize("chunk", [1, 3, 7, 40, 500])
def test_result_is_independent_of_delta_size(chunk):
    """Whether a tag is captured must not depend on how the stream happens to
    be chunked, which is what the truncation bug came down to."""
    tags, spoken = run(
        "Checking now. [ACTION: weather | location: Gainesville] One moment.",
        chunk=chunk,
    )
    assert tags == ["[ACTION: weather | location: Gainesville]"]
    assert "Checking now." in spoken


def test_sentences_emitted_counts_every_queued_sentence():
    """Used to time the gap between the first token and first speakable
    sentence, so it has to track the queue exactly."""
    async def go():
        queue = asyncio.Queue()
        router = StreamRouter(queue)
        await router.feed("One sentence here. And a second one follows. ")
        await router.finalize()
        drained = 0
        while not queue.empty():
            if await queue.get() is not None:
                drained += 1
        return router.sentences_emitted, drained

    emitted, drained = asyncio.run(go())
    assert emitted == drained
