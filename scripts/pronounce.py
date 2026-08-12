#!/usr/bin/env python3
"""Audition and set pronunciations, by ear, without editing any code.

Pronunciations live in the database, and tts.speak reads them fresh on every
sentence, so a change here takes effect on the very next thing Nova says. No
restart, no deploy, no file to edit.

    python3 scripts/pronounce.py list
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
