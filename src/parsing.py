import re

_LEADING_BRACKET = re.compile(r'^\s*\[([^\]]*)\]\s*')
_RECOGNIZED_TAG   = re.compile(r'^(ACTION|MEMORY-EXPLICIT|MEMORY):', re.IGNORECASE)

# Whisper wraps non speech events in brackets or parentheses: [BLANK_AUDIO],
# (beep), [ Silence ].
_SOUND_ANNOTATION = re.compile(r'[\[(][^\])]*[\])]')

# Transcripts that mean "there was no speech here". Matched against the whole
# normalized transcript, never as a substring, because every one of these is a
# legitimate word inside a real sentence.
#
# whisper.cpp does not reliably emit [BLANK_AUDIO] for non speech input. Fed
# room noise it hallucinates a short plausible token instead, drawn from its
# training data. The runaway conversation loop on Aug 10 2026 was driven by
# exactly this: an empty room transcribed as "over." three separate times.
_NOISE_TRANSCRIPTS = frozenset({
    # Explicit whisper.cpp markers
    "blank audio", "silence",
    # Observed in production logs from an empty room, Aug 10 2026
    "over", "beep", "enola",
    # Documented whisper hallucinations on non speech input
    "you", "thank you", "thanks for watching", "bye", "so",
    "uh", "um", "ah", "oh", "mm", "hmm",
    "applause", "music", "click", "cough", "laughter", "noise",
})


def is_noise_transcript(text):
    """True when a transcript carries no real speech.

    Two independent failure modes to catch. Whisper either annotates the non
    speech event, which leaves nothing behind once the annotation is removed,
    or it hallucinates a short real word, which only an exact match can catch
    without eating legitimate short replies like "yes" or "why".

    Deliberately conservative: a three word hallucination such as "He know
    what." still gets through. The consecutive follow up cap is the backstop
    for those, not this function."""
    if not text:
        return True

    without_annotations = _SOUND_ANNOTATION.sub(' ', text)

    # Keep apostrophes so contractions stay intact, drop everything else so
    # trailing punctuation cannot hide a match ("over." must equal "over").
    normalized = re.sub(r"[^a-z0-9' ]", ' ', without_annotations.lower())
    normalized = ' '.join(normalized.split())

    if not normalized:
        return True
    return normalized in _NOISE_TRANSCRIPTS


def strip_leading_bracket_cue(text, seen=None):
    """Strip a leading bracketed cue Nova should not be emitting, e.g. "[clearly]".

    SURVIVED THE TOOL USE MIGRATION. extract_actions and _parse_action_tag
    were deleted; bracket handling as a whole was not. This exists for emotion
    cues, which are a rendering hint with no side effect and were never part of
    the action system.

    Leaves ACTION/MEMORY tags alone (defensive: they should already be gone
    by the time this runs, but this must never eat a real tag).

    The same leak commonly gets caught twice in one turn: once on the
    sentence text before TTS, once on the accumulated response text before
    it is returned. Pass a shared `seen` set (scoped to one turn by the
    caller) so the strip still happens both times but only the first
    occurrence of a given leaked phrase logs, keeping a leak counter built
    from these logs accurate rather than double counted."""
    match = _LEADING_BRACKET.match(text)
    if not match:
        return text
    inner = match.group(1).strip()
    if _RECOGNIZED_TAG.match(inner):
        return text
    if seen is None or inner not in seen:
        print(f"Bracket leakage stripped: [{inner}]", flush=True)
        if seen is not None:
            seen.add(inner)
    return text[match.end():]


def extract_memories(response_text):
    """SURVIVED THE TOOL USE MIGRATION. Not deleted with extract_actions.

    Memory tags were considered for migration to a `remember` tool and
    deliberately deferred. The reason is not round trip cost, since a tool with
    returns_to_model=False costs no extra Claude call. The reason is that
    nothing in the memory system can correct a memory: `superseded_by` is
    declared and never written, `volatile` and `references_date` are written and
    never read. Automating writes into a store that cannot be corrected is worse
    than manual capture, because the errors are permanent.

    See docs/BACKEND_TODO.md, "Memory system: correction before automation".
    This becomes a tool once SUPERSEDE and expiry exist, and not before."""
    explicit_pattern = r'\[MEMORY-EXPLICIT:\s*(.+?)\]'
    implicit_pattern = r'\[MEMORY:\s*(.+?)\]'

    explicit_memories = re.findall(explicit_pattern, response_text)
    implicit_memories = re.findall(implicit_pattern, response_text)

    clean = re.sub(explicit_pattern, '', response_text)
    clean = re.sub(implicit_pattern, '', clean)
    clean = re.sub(r'  +', ' ', clean).strip()

    return clean, explicit_memories, implicit_memories


# The wake phrase as Whisper actually renders it. "Hey Nova" comes back with
# varying punctuation and casing, and occasionally as "Hey, Nova."
# "hey" is required rather than optional. A bare "Nova" would also catch him
# saying her name to somebody else in the room, which is the same addressing
# problem the follow up window already has, and he was explicit about being
# sure of the full phrase and unsure of the short one. Dropping the requirement
# is one word here if the logs later say it is safe.
_WAKE_LEAD = re.compile(r"^\s*hey[\s,]+nova\b[\s,.!?]*", re.I)


def split_wake_phrase(transcript):
    """Return (said_wake_word, remainder) for a follow up transcript.

    Saying the wake word during a follow up should start a fresh turn rather
    than being transcribed into the middle of one. Without this, "hey nova what
    is the weather" reaches Claude with the wake phrase attached, and the turn
    is still session trusted rather than verified.

    Treating it as a fresh turn is the better behaviour twice over: he gets the
    chime that tells him she is listening, and the utterance is long enough to
    verify properly instead of being accepted on session state.

    The full phrase is required. A bare "Nova" would also match him saying her
    name to somebody else during the window, which is the addressing problem
    rather than a solution to it."""
    if not transcript:
        return False, transcript
    match = _WAKE_LEAD.match(transcript)
    if not match or not match.group(0).strip():
        return False, transcript
    return True, transcript[match.end():].strip()


# ── Number words ──
# Both directions are needed. Whisper writes numbers as digits or as words
# depending on the phrasing, so parsing a transcript has to accept either, and
# Nova spells them out when speaking, so building a phrase bank sentence has to
# go the other way.

_ONES = ("one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 2, "thirty": 3, "forty": 4, "fifty": 5, "sixty": 6}


def _build_number_words():
    words = {word: value + 1 for value, word in enumerate(_ONES)}
    for tens_word, tens in _TENS.items():
        words[tens_word] = tens * 10
        for index, ones_word in enumerate(_ONES[:9]):
            words[f"{tens_word} {ones_word}"] = tens * 10 + index + 1
    return words


# One through sixty. Past that a spoken duration is said in hours, and a timer
# longer than an hour is rare enough to be worth a fall through to Claude
# rather than sixty more entries that are never played.
NUMBER_WORDS = _build_number_words()


# Output only, and deliberately wider than _TENS. NUMBER_WORDS is the input
# side and stays capped at sixty, because widening it would teach the duration
# parser to read "eighty minutes" as a timer it has no rendered clip for. This
# table is only ever read when spelling a number back out.
_TENS_SPOKEN = {**_TENS, "seventy": 7, "eighty": 8, "ninety": 9}

_SPELL_MIN, _SPELL_MAX = -60, 130


def words_for_number(n):
    """Spell a number the way Nova says it aloud.

    Durations use one through sixty. Temperatures use the rest: Gainesville
    reaches the nineties every summer, and a local weather answer is composed
    here rather than by Claude, so the speller has to cover what a thermometer
    can actually read.

    Output for one through sixty is frozen. phrasebank.timer_text and time_text
    build rendered WAV filenames out of these strings, so changing one would
    orphan a file on disk and silently fall back to live synthesis."""
    if not _SPELL_MIN <= n <= _SPELL_MAX:
        raise ValueError(
            f"only {_SPELL_MIN} through {_SPELL_MAX} are spelled: {n}")

    if n < 0:
        return f"minus {words_for_number(-n)}"
    if n == 0:
        return "zero"
    if n <= 19:
        return _ONES[n - 1]

    if n >= 100:
        hundreds, rest = divmod(n, 100)
        spoken = f"{_ONES[hundreds - 1]} hundred"
        return spoken if rest == 0 else f"{spoken} {words_for_number(rest)}"

    tens, ones = divmod(n, 10)
    tens_word = next(w for w, t in _TENS_SPOKEN.items() if t == tens)
    return tens_word if ones == 0 else f"{tens_word} {_ONES[ones - 1]}"
