#!/usr/bin/env python3
"""Audition and set pronunciations, by ear, without editing any code.

Pronunciations live in the database, and tts.speak reads them fresh on every
sentence, so a change here takes effect on the very next thing Nova says. No
restart, no deploy, no file to edit.

    python3 scripts/pronounce.py list
    python3 scripts/pronounce.py sweep Lethanial          # audition all 25
    python3 scripts/pronounce.py sweep Lethanial 12       # resume at 12
    python3 scripts/pronounce.py sweep Lethanial 1,6,8    # finals, these only
    python3 scripts/pronounce.py spread Lethanial 15      # one candidate, many seeds
    python3 scripts/pronounce.py stability Lethanial 15            # coarse bracket
    python3 scripts/pronounce.py stability Lethanial 15 .78 .82 .86  # narrow band
    python3 scripts/pronounce.py demo                     # real responses, current settings
    python3 scripts/pronounce.py demo 0.75 8              # real responses, 8 of them, at 0.75
    python3 scripts/pronounce.py pick Lethanial 7         # commit number 7
    python3 scripts/pronounce.py try Lethanial Luthanyull "La-thanyull"
    python3 scripts/pronounce.py tryph Lethanial "L AA1 TH AE0 N Y AH0 L"
    python3 scripts/pronounce.py set Lethanial Luthanyull
    python3 scripts/pronounce.py phonemes Lethanial "L AH0 TH AE1 N Y AH0 L"
    python3 scripts/pronounce.py say "Good morning Lethanial."

`try` speaks each candidate in a sentence and changes nothing, so you can pick
by ear before committing to one. `set` writes the winner.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import config                                            # noqa: E402
import tts                                               # noqa: E402
from database import (get_pronunciations,                # noqa: E402
                      upsert_pronunciation, init_db)

# Said around each candidate so it is judged in context. A name in isolation
# sounds wrong even when it is right.
FRAME = "Good morning {}. Your timer is up."


def cmd_list():
    rows = get_pronunciations()
    if not rows:
        print("no pronunciations set")
        return
    print(f"{'grapheme':<24} {'alias':<24} arpabet")
    for grapheme, alias, arpabet in rows:
        print(f"{grapheme:<24} {alias:<24} {arpabet or ''}")
    print(f"\nmodel {config.DEFAULT_TTS_MODEL}, "
          f"phoneme tags {'ON' if config.TTS_PHONEME_TAGS else 'OFF'}")


def cmd_try(grapheme, candidates):
    """Speak each candidate. Writes nothing."""
    print(f"auditioning {len(candidates)} candidates for {grapheme!r}")
    print("nothing is saved; run `set` with the one you want\n")

    print("  0. (as written, no alias)")
    tts.speak(FRAME.format(grapheme))
    time.sleep(0.6)

    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. {candidate}")
        # Bypass the table: speak the candidate directly in place of the name.
        tts.speak(FRAME.format(candidate))
        time.sleep(0.6)

    print(f"\npick one:  python3 scripts/pronounce.py set {grapheme} <alias>")


def cmd_tryph(grapheme, arpabets):
    """Speak candidate ARPAbet strings directly. Writes nothing, changes no config.

    Separate from `try` because phonemes and respellings fail differently. A
    respelling is at the mercy of how the model reads English spelling, which is
    what makes an over annunciated Y hard to spell away. Phonemes name the
    sound, so a glide stays a glide."""
    if not config.TTS_PHONEME_TAGS:
        print("note: TTS_PHONEME_TAGS is OFF, so this auditions phonemes that")
        print("      production is not using yet. Turn it on in src/config.py")
        print("      once one of these sounds right.\n")

    print(f"auditioning {len(arpabets)} phoneme strings for {grapheme!r}\n")
    print("  0. (as written, no tag)")
    tts.speak(FRAME.format(grapheme))
    time.sleep(0.6)

    for i, ph in enumerate(arpabets, 1):
        print(f"  {i}. {ph}")
        tagged = f'<phoneme alphabet="cmu-arpabet" ph="{ph}">{grapheme}</phoneme>'
        # Bypasses the table and the config flag: the tag is built here and
        # handed straight to the model, which is the only way to compare
        # candidates without a restart between each one.
        tts.speak(FRAME.format(tagged))
        time.sleep(0.6)

    print(f"\npick one:  python3 scripts/pronounce.py phonemes {grapheme} \"<string>\"")


# Candidate sweep, built from Lethanial's own description of the sounds:
# "Luh" as in love, "than" with the unvoiced th of thighs, "yull" as in dull
# with a y in front.
#
# That maps to L + AH + TH + AE + N + Y + AH + L. The unvoiced th is the reason
# the phoneme rows exist at all: English spells voiced and unvoiced th
# identically, so "than" is voiced and "thigh" is not, and no respelling picks
# one reliably. TH is unvoiced, DH is voiced.
#
# Stress digits are 0 unstressed, 1 primary, 2 secondary. Most of the variation
# below is stress placement, because a name that sounds butchered is usually
# stressed wrong rather than voweled wrong.
SWEEP = [
    # ── phonemes: stress placement on the target sounds ──
    ("arpabet", "L AH0 TH AE1 N Y AH0 L",  "current stored, stress on THAN"),
    ("arpabet", "L AH1 TH AE1 N Y AH1 L",  "every syllable stressed"),
    ("arpabet", "L AH0 TH AE1 N Y AH1 L",  "stress THAN and YULL"),
    ("arpabet", "L AH1 TH AE1 N Y AH0 L",  "stress LUH and THAN"),
    ("arpabet", "L AH1 TH AE0 N Y AH1 L",  "stress LUH and YULL"),
    ("arpabet", "L AH1 TH AE0 N Y AH0 L",  "stress LUH only"),
    ("arpabet", "L AH0 TH AE2 N Y AH1 L",  "secondary THAN, primary YULL"),
    ("arpabet", "L AH2 TH AE1 N Y AH0 L",  "secondary LUH, primary THAN"),
    # ── phonemes: the last vowel, yull like dull ──
    ("arpabet", "L AH0 TH AE1 N Y UH1 L",  "YULL closer to book"),
    ("arpabet", "L AH0 TH AE1 N Y AO1 L",  "YULL closer to all"),
    ("arpabet", "L AH0 TH AE1 N Y OW0 L",  "YULL closer to yolk"),
    # ── phonemes: controls, to confirm the diagnosis ──
    ("arpabet", "L AH0 DH AE1 N Y AH0 L",  "voiced th, should sound wrong"),
    ("arpabet", "L AH0 TH AE1 N AH0 L",    "no Y at all, should lose the glide"),
    ("arpabet", "L AA0 TH AE1 N Y AH0 L",  "LA as in father"),
    ("arpabet", "L AE0 TH AE1 N Y AH0 L",  "LA as in cat"),
    ("arpabet", "L AH0 TH AA1 N Y AH0 L",  "THAN closer to thon"),
    # ── respellings, for comparison ──
    ("alias", "Luh-thanyull",   "your description, split once"),
    ("alias", "Luhthanyull",    "no split"),
    ("alias", "Luh-than-yull",  "split twice"),
    ("alias", "Luthanyull",     "single h"),
    ("alias", "Luh-thann-yull", "double n to harden the a"),
    ("alias", "Luh-thinyull",   "thin spelling to force unvoiced th"),
    ("alias", "Luh-thanyul",    "single l"),
    ("alias", "Luhth-anyull",   "split inside the th"),
    ("alias", "Luh-thawnyull",  "aw vowel in the middle"),
]


# Every comparison runs at this seed so two candidates differ by the thing
# being compared and nothing else. Lethanial heard the same phoneme string come
# out "no" on one repeat and "yes" on the next, which means the spread between
# renditions of identical input was as large as the spread between candidates.
# Any ranking collected without pinning this was measuring luck.
COMPARE_SEED = 4242

# Below this many candidates, each is spoken twice. A finals round is decided on
# small differences, and hearing something once is not enough to separate two
# that are close; hearing it twice in a row is.
REPEAT_BELOW = 9


def cmd_sweep(grapheme, start=0, only=None):
    """Speak candidates in order, numbered. Writes nothing.

    Numbering always refers to the full SWEEP list, even when a subset is
    selected, so `pick 11` still means the same candidate in a finals round as
    it did in the first pass. Renumbering a shortlist would silently invalidate
    every number written down during the first listen."""
    chosen = [(i, c) for i, c in enumerate(SWEEP, 1)
              if (i in only if only else i >= start)]
    repeats = 2 if len(chosen) < REPEAT_BELOW else 1

    print(f"{len(chosen)} candidates for {grapheme!r}"
          + (f", each spoken {repeats} times" if repeats > 1 else ""))
    print("nothing is saved; note the number you like, then `pick <n>`\n")

    if not only:
        print("  0. (as written, no substitution)")
        tts.speak(FRAME.format(grapheme))
        time.sleep(0.7)

    for i, (kind, value, note) in chosen:
        print(f"  {i:2}. [{kind:7}] {value:28} {note}", flush=True)
        spoken = value if kind == "alias" else (
            f'<phoneme alphabet="cmu-arpabet" ph="{value}">{grapheme}</phoneme>')
        for _ in range(repeats):
            tts.speak(FRAME.format(spoken), seed=COMPARE_SEED)
            time.sleep(0.7)
        time.sleep(0.5)

    print(f"\npick one:  python3 scripts/pronounce.py pick {grapheme} <number>")


def cmd_spread(grapheme, number, runs=6):
    """Speak one candidate at several seeds, to hear how much it varies.

    This is the measurement that should have come first. A candidate is only
    worth choosing if it is reliably right, and a phoneme string that lands
    correctly half the time is worse than a duller one that always lands. The
    numbers to compare are how many of these sound acceptable, not whether any
    single one does."""
    kind, value, note = SWEEP[number - 1]
    spoken = value if kind == "alias" else (
        f'<phoneme alphabet="cmu-arpabet" ph="{value}">{grapheme}</phoneme>')
    print(f"candidate {number}: {value}   {note}")
    print(f"{runs} different seeds, same input. Count how many sound right.\n")
    for i in range(runs):
        print(f"  seed {1000 + i * 111}")
        tts.speak(FRAME.format(spoken), seed=1000 + i * 111)
        time.sleep(0.7)


# Coarse bracket, five points across the usable range. Once the good region is
# known, pass explicit values to search inside it: stability is a continuous
# float, not a set of presets, and the interesting differences are usually in a
# band narrower than these steps.
STABILITY_STEPS = (0.45, 0.60, 0.75, 0.90, 1.00)


def cmd_stability(grapheme, number, steps=None):
    """Speak one candidate across stability values, at a fixed seed.

    stability is the setting that governs how much a rendition varies. If a
    candidate sounds right at 0.85 and inconsistent at 0.60, the fix is the
    setting rather than the phonemes, and no amount of further phoneme hunting
    will help."""
    from elevenlabs import VoiceSettings
    kind, value, note = SWEEP[number - 1]
    spoken = value if kind == "alias" else (
        f'<phoneme alphabet="cmu-arpabet" ph="{value}">{grapheme}</phoneme>')
    print(f"candidate {number}: {value}   {note}")
    print("current production stability is "
          f"{config.TTS_VOICE_SETTINGS.stability}\n")
    for stability in (steps or STABILITY_STEPS):
        print(f"  stability {stability:.2f}")
        settings = VoiceSettings(
            stability=stability,
            similarity_boost=config.TTS_VOICE_SETTINGS.similarity_boost,
            style=config.TTS_VOICE_SETTINGS.style,
            use_speaker_boost=True, speed=1.00,
        )
        # Two renditions per value: the point is consistency, and one sample
        # cannot show it.
        for seed in (1000, 2000):
            tts.speak(FRAME.format(spoken), voice_settings=settings, seed=seed)
            time.sleep(0.5)
        time.sleep(0.4)
    print("\nset the winner in src/config.py TTS_VOICE_SETTINGS, then restart")


def cmd_demo(stability=None, count=6):
    """Speak real past responses, to judge delivery over more than one sentence.

    Every other command here uses a single short frame, which is right for
    comparing a name and useless for deciding whether a voice sounds flat.
    Flatness shows up across sentences, in a response that carries an aside or a
    joke, not in five words about a timer.

    Pulls actual assistant turns out of conversation_history rather than
    invented lines, so what is judged is what Nova really says."""
    import sqlite3
    from elevenlabs import VoiceSettings

    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute(
        "SELECT content FROM conversation_history WHERE role = 'assistant' "
        "AND LENGTH(content) BETWEEN 80 AND 400 ORDER BY id DESC LIMIT ?",
        (count,)
    ).fetchall()
    conn.close()

    if not rows:
        print("no past responses long enough to judge; talk to Nova first")
        return

    settings = None
    if stability is not None:
        settings = VoiceSettings(
            stability=stability,
            similarity_boost=config.TTS_VOICE_SETTINGS.similarity_boost,
            style=config.TTS_VOICE_SETTINGS.style,
            use_speaker_boost=True, speed=1.00,
        )

    shown = stability if stability is not None else config.TTS_VOICE_SETTINGS.stability
    print(f"{len(rows)} real responses at stability {shown}")
    print("listening for flatness, not for the name\n")
    for i, (content,) in enumerate(reversed(rows), 1):
        print(f"  {i}. {content[:90]}{'...' if len(content) > 90 else ''}", flush=True)
        # No seed: production does not pin one, and the question here is how the
        # voice behaves in normal use rather than how two candidates compare.
        tts.speak(content, voice_settings=settings)
        time.sleep(0.5)


def cmd_pick(grapheme, number):
    """Commit whichever sweep candidate won."""
    if not 1 <= number <= len(SWEEP):
        print(f"pick a number between 1 and {len(SWEEP)}")
        return
    kind, value, note = SWEEP[number - 1]
    print(f"{number}. [{kind}] {value}   {note}")
    if kind == "alias":
        cmd_set(grapheme, value)
    else:
        cmd_phonemes(grapheme, value)


def cmd_set(grapheme, alias):
    existing = {g: (a, p) for g, a, p in get_pronunciations()}
    arpabet = existing.get(grapheme, (None, None))[1]
    upsert_pronunciation(grapheme, alias, arpabet=arpabet, verified=True)
    print(f"{grapheme} -> {alias}")
    print("live on the next thing Nova says; no restart needed")
    tts.speak(FRAME.format(grapheme))


def cmd_phonemes(grapheme, arpabet):
    existing = {g: a for g, a, _ in get_pronunciations()}
    alias = existing.get(grapheme, grapheme)
    upsert_pronunciation(grapheme, alias, arpabet=arpabet, verified=True)
    print(f"{grapheme} arpabet -> {arpabet}")
    if not config.TTS_PHONEME_TAGS:
        print("\nTTS_PHONEME_TAGS is OFF, so this is stored but unused.")
        print("Set it True in src/config.py and restart to use phonemes")
        print("instead of the alias respelling.")
    else:
        tts.speak(FRAME.format(grapheme))


def cmd_say(text):
    print("sending to ElevenLabs:", tts.normalize_pronunciation(text))
    tts.speak(text)


def main():
    init_db()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    command, rest = args[0], args[1:]
    if command == "list":
        cmd_list()
    elif command == "sweep" and rest:
        spec = rest[1] if len(rest) > 1 else ""
        if "," in spec:
            cmd_sweep(rest[0], only={int(n) for n in spec.split(",") if n.strip()})
        else:
            cmd_sweep(rest[0], start=int(spec) if spec else 0)
    elif command == "spread" and len(rest) >= 2:
        cmd_spread(rest[0], int(rest[1]), int(rest[2]) if len(rest) > 2 else 6)
    elif command == "demo":
        cmd_demo(float(rest[0]) if rest else None,
                 int(rest[1]) if len(rest) > 1 else 6)
    elif command == "stability" and len(rest) >= 2:
        # Trailing values narrow the search: stability Lethanial 15 .78 .82 .86
        steps = tuple(float(v) for v in rest[2:]) or None
        cmd_stability(rest[0], int(rest[1]), steps)
    elif command == "pick" and len(rest) == 2:
        cmd_pick(rest[0], int(rest[1]))
    elif command == "try" and len(rest) >= 2:
        cmd_try(rest[0], rest[1:])
    elif command == "tryph" and len(rest) >= 2:
        cmd_tryph(rest[0], rest[1:])
    elif command == "set" and len(rest) == 2:
        cmd_set(rest[0], rest[1])
    elif command == "phonemes" and len(rest) == 2:
        cmd_phonemes(rest[0], rest[1])
    elif command == "say" and rest:
        cmd_say(" ".join(rest))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
