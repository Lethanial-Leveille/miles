"""Last pass over a finished reply, in code, because the prompt alone did not hold.

Nov 21 2026: typed replies in the app still came back with "ninety six active
applications" and em dashes, though NUMBER_FORMAT_TEXT, TEXT_TURN_NOTE and the
no hyphens rule all say otherwise. The history she reads is mostly spoken
replies with every number written out, and that outweighs a rule. So the reply
is fixed after it is written rather than argued with before.

Two passes, kept deliberately narrow so they never change what a reply means:
  undash   every channel. An em or en dash between words becomes a comma; one
           between two numbers becomes "to". Hyphens inside words, dates and
           URLs are left alone.
  numerals text channel only. A spelled out number becomes digits when it is
           10 or more, a decimal, or a percentage. One to nine stay as words
           ("four reps", "one of those"), as most style guides write them, and
           a run that reads as two numbers ("eleven fifty eight", "nineteen
           eighty four") is left exactly as written rather than guessed at.
"""
import re

_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMBER_WORDS = set(_SMALL) | set(_TENS) | {"hundred"} | set(_SCALES)

_WORD = re.compile(r"[A-Za-z]+")
# Only spaces or a hyphen may sit between words of one number: a comma or a
# full stop ends it, so two numbers in a list are never read as one.
_JOIN = re.compile(r"^(?:\s+|\s*-\s*)$")
_PERCENT = re.compile(r"^\s+percent\b", re.I)


def _value(words):
    """The integer a run of number words spells, or None if it is not exactly one."""
    total = current = 0
    last = None
    for w in words:
        if w in _SMALL:
            n = _SMALL[w]
            if last == "small" or (last == "tens" and not 1 <= n <= 9):
                return None
            current += n
            last = "small"
        elif w in _TENS:
            if last in ("small", "tens") or current % 100:
                return None
            current += _TENS[w]
            last = "tens"
        elif w == "hundred":
            if last != "small" or not 1 <= current <= 9:
                return None
            current *= 100
            last = "hundred"
        elif w in _SCALES:
            if current == 0 or last == "scale":
                return None
            total += current * _SCALES[w]
            current = 0
            last = "scale"
        elif w == "and":
            if last != "hundred":
                return None
        else:
            return None
    return total + current


def _runs(text):
    """Spans of consecutive number words joined only by spaces or hyphens.

    "and" and "point" are taken inside a run only where a number continues
    after them, so "two and three" stays two numbers."""
    tokens = list(_WORD.finditer(text))
    i = 0
    while i < len(tokens):
        if tokens[i].group().lower() not in _NUMBER_WORDS:
            i += 1
            continue
        j = i
        while j + 1 < len(tokens) and _JOIN.match(text[tokens[j].end():tokens[j + 1].start()]):
            nxt = tokens[j + 1].group().lower()
            if nxt in _NUMBER_WORDS:
                j += 1
            elif nxt in ("and", "point") and j + 2 < len(tokens) \
                    and tokens[j + 2].group().lower() in _NUMBER_WORDS \
                    and _JOIN.match(text[tokens[j + 1].end():tokens[j + 2].start()]):
                j += 2
            else:
                break
        yield tokens[i:j + 1]
        i = j + 1


def numerals(text):
    out, cursor = [], 0
    for run in _runs(text):
        words = [t.group().lower() for t in run]
        if "point" in words:
            k = words.index("point")
            whole, frac = words[:k], words[k + 1:]
            if not frac or any(w not in _SMALL or _SMALL[w] > 9 for w in frac):
                continue
            value = _value(whole)
            if value is None:
                continue
            number = f"{value}." + "".join(str(_SMALL[w]) for w in frac)
        else:
            value = _value(words)
            if value is None:
                continue
            number = str(value)
        start, end = run[0].start(), run[-1].end()
        percent = _PERCENT.match(text[end:])
        if percent:
            number += "%"
            end += percent.end()
        elif "." not in number and value < 10:
            continue
        out.append(text[cursor:start])
        out.append(number)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


_RANGE_DASH = re.compile(r"(\d)\s*[–—]\s*(\d)")
_WORD_DASH = re.compile(r"\s*[–—]+\s*|\s+--?\s+")


def undash(text):
    text = _RANGE_DASH.sub(r"\1 to \2", text)
    text = _WORD_DASH.sub(", ", text)
    # A dash that ended a clause right before other punctuation
    return re.sub(r",\s*([,.!?;:])", r"\1", text).strip(" ,")


def for_display(text, channel):
    """What the reply should read as once it is finished, for this channel."""
    if not text:
        return text
    text = undash(text)
    return numerals(text) if channel == "text" else text
