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

    SURVIVES THE TOOL USE MIGRATION. Phase 2 deletes `extract_actions` and
    `_parse_action_tag`, not bracket handling as a whole. This function exists
    for emotion cues, which are a rendering hint with no side effect and were
    never part of the action system. They are not migrating to tool use and
    this stays regardless of what the Phase 2 commit message says.

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
    """SURVIVES THE TOOL USE MIGRATION. Do not delete alongside extract_actions.

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


def extract_actions(response_text):
    """DELETED IN PHASE 2 of the tool use migration, along with
    brain._parse_action_tag. These two are the only casualties in this module:
    strip_leading_bracket_cue and extract_memories both stay, for reasons
    documented on each."""
    param_pattern  = r'\[ACTION:\s*(\w+)\s*\|\s*(.+?)\]'
    simple_pattern = r'\[ACTION:\s*(\w+)\s*\]'

    actions = re.findall(param_pattern, response_text)
    clean   = re.sub(param_pattern, '', response_text)

    simple_actions = re.findall(simple_pattern, clean)
    clean = re.sub(simple_pattern, '', clean)
    clean = re.sub(r'  +', ' ', clean).strip()

    parsed = []
    for action_type, params_str in actions:
        params = {}
        for param in params_str.split(','):
            if ':' in param:
                key, value = param.split(':', 1)
                params[key.strip()] = value.strip()
            else:
                params['value'] = param.strip()
        parsed.append({"type": action_type.lower(), "params": params})

    for action_type in simple_actions:
        parsed.append({"type": action_type.lower(), "params": {}})

    return clean, parsed
