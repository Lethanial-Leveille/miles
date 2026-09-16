"""Answer the commands that never needed a language model, without one.

"Set a timer for ten minutes" costs a median 4404ms today, and 1609ms of that
is Claude's time to first token plus ElevenLabs' time to first byte. Neither is
buying anything: there is no reasoning in parsing a duration and starting a
thread, and the confirmation is one of a few dozen fixed sentences already on
disk. Matching locally and playing a rendered clip replaces that 1609ms with
about 85ms.

Offline capability falls out of this rather than motivating it. Everything
before the API call is already local, so an intent handled here works with the
wifi down. That is a consequence of the latency work, not the reason for it.

## Why both signals have to agree

Memory retrieval fuses keyword and semantic results by reciprocal rank, because
it is ranking candidates and a hit either list finds is worth surfacing. This
is the opposite problem. It is a gate, not a ranking, and the two signals are
combined with AND rather than fused.

The reason is that semantic similarity alone is unsafe here. "Set a ten minute
timer" and "cancel the ten minute timer" are near neighbours in embedding space
and opposite in meaning, so any threshold loose enough to catch real phrasing
variety is also loose enough to confuse them. The lexical gate carries the
distinctions that are exact, and the embedding carries the ones that are fuzzy.

## Why it is deliberately reluctant

The costs are asymmetric. Falling through to Claude costs about 4.4 seconds and
is otherwise correct. Firing wrongly starts the wrong timer or hangs up on him
mid thought. So every gate is written to decline when unsure, and an intent
whose confirmation is not rendered declines too rather than improvising.
"""

import random
import re
from datetime import datetime

import numpy as np
import requests

import actions
import embeddings
import netcheck
import phrasebank
from database import active_reminder_count
from parsing import NUMBER_WORDS, words_for_number
from config import INTENT_SIMILARITY_THRESHOLD, MAX_DISMISS_WORDS


class Match:
    """A matched intent, its extracted slots, and how confident the match was."""

    def __init__(self, name, slots, score):
        self.name = name
        self.slots = slots
        self.score = score

    def __repr__(self):
        return f"Match({self.name!r}, {self.slots!r}, {self.score:.3f})"


# Canonical phrasings, not an exhaustive list. The embedding generalizes across
# wording; these only have to span the shapes a request takes, so adding a near
# duplicate of one already here buys nothing.
EXAMPLES = {
    'set_timer': [
        "set a timer for ten minutes",
        "start a twenty minute timer",
        "timer for five minutes",
        "give me fifteen minutes",
        "wake me up in half an hour",
        "count down thirty seconds",
    ],
    'time_of_day': [
        "what time is it",
        "what's the time",
        "do you know the time",
        "got the time",
        "what time is it right now",
    ],
    'cancel_reminder': [
        "cancel that reminder",
        "never mind that reminder",
        "forget the reminder",
        "cancel my reminder",
        "cancel the timer",
        "stop my timer",
        "never mind the timer",
    ],
    'dismiss': [
        "that's all thanks",
        "goodnight nova",
        "never mind",
        "we're done here",
        "that's it for now",
        "alright bye",
        "I'm good thanks",
    ],
    'weather': [
        "what's the weather",
        "what's it like outside",
        "how hot is it",
        "is it going to rain",
        "do I need a jacket",
        "what's the temperature outside",
    ],
}

_example_vectors = None


def _vectors():
    """Encoded once, on first use. Falls back to no semantic signal if the
    embedding model failed to load, which means every intent declines and every
    turn goes to Claude. Degrading to today is the right failure here."""
    global _example_vectors
    if _example_vectors is None:
        _example_vectors = {name: embeddings.encode(texts)
                            for name, texts in EXAMPLES.items()}
    return _example_vectors


def warm():
    """Encode the examples now rather than on the first real command.

    Loading the model and encoding the examples are separate costs and both are
    lazy. Warming only the model leaves the second one to land on a turn he is
    waiting through: measured at 10.8s cold against 100ms warm."""
    _vectors()


def _scores(text):
    query = embeddings.encode([text])[0]
    # encode returns unit length vectors, so a dot product is the cosine.
    return {name: float(np.max(vecs @ query)) for name, vecs in _vectors().items()}


# ── Slot extraction ──
# Exact, and so a regex. An embedding can tell you he wants a timer; it cannot
# tell you ten from fifteen.

_SHORTHAND = (
    (re.compile(r'\bhalf an hour\b'), '30 minutes'),
    (re.compile(r'\ban hour\b'),      '1 hour'),
    (re.compile(r'\ba minute\b'),     '1 minute'),
    (re.compile(r'\ba second\b'),     '1 second'),
)

# Longest first, so "twenty five" is preferred over "twenty".
_NUMBER_ALT = "|".join(sorted((re.escape(w) for w in NUMBER_WORDS),
                              key=len, reverse=True))
_DURATION = re.compile(
    rf"\b(?:(\d{{1,3}})|({_NUMBER_ALT}))\s*(second|minute|hour)s?\b")

_CANCELS = re.compile(r'\b(cancel|stop|kill|delete|remove|clear|scrap|forget)\b')
_TIMER_VERB = re.compile(
    r'\b(timer|countdown|count down|set|start|give me|wake me|remind me in)\b')
_DISMISS_WORD = re.compile(
    r"\b(thanks|thank you|bye|goodbye|goodnight|good night|night|"
    r"never mind|nevermind|that's all|thats all|that's it|thats it|"
    r"i'm good|im good|i'm done|im done|we're done|were done|all set|"
    r"later|see ya|see you|nope|nothing|that'll be all|thatll be all)\b")

_THANKS = re.compile(r"\b(thanks|thank you|appreciate it|cheers)\b")

# A closing word aimed at something is not aimed at the conversation.
_DISMISS_OBJECT = re.compile(r'\b(timer|reminder|alarm|weather|about|instead)\b')

_QUESTIONISH = re.compile(
    r'\b(what|when|how|why|who|where|which|whats|hows|can you|could you|'
    r'do you|did you|are you|is it|tell me|play|call|text)\b')

TIMER_VALUES = set(phrasebank.TIMER_VALUES)


# Spelling variants that the embedding does not treat as variants. "actually
# nevermind" scored 0.55 against a 0.55 threshold and was declined by a hair,
# while "actually never mind" scored 0.89, so the turn went to Claude, which
# called cancel_reminder with no reminder to cancel and answered "Done."
# Collapsing the spelling is the fix; lowering the threshold would have been
# the symptom fix and would have loosened every other intent too.
_VARIANTS = (
    (re.compile(r'\bnevermind\b'), 'never mind'),
    (re.compile(r'\bgoodnite\b'),  'goodnight'),
    (re.compile(r'\bgotta\b'),     'got to'),
)


def normalize(text):
    text = text.lower().replace('-', ' ')
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern, replacement in _VARIANTS:
        text = pattern.sub(replacement, text)
    return text


def parse_duration(text):
    """(amount, unit) with unit plural, or None."""
    for pattern, replacement in _SHORTHAND:
        text = pattern.sub(replacement, text)
    match = _DURATION.search(text)
    if not match:
        return None
    digits, word, unit = match.groups()
    amount = int(digits) if digits else NUMBER_WORDS[word]
    return amount, f"{unit}s"


# ── Lexical gates ──
# Each returns slots on a pass and None on a decline. Cheap on purpose: they
# run before the embedding, so most transcripts are rejected without paying the
# 34ms encode at all.

def _gate_set_timer(text):
    if _CANCELS.search(text):
        return None
    if not _TIMER_VERB.search(text):
        return None
    duration = parse_duration(text)
    if duration is None:
        return None
    # Not enumerated means nothing rendered to say it with. Declining sends the
    # turn to Claude, which is slower and right, rather than improvising a
    # confirmation in some other voice.
    if duration not in TIMER_VALUES:
        return None
    return {'amount': duration[0], 'unit': duration[1]}


def _gate_dismiss(text):
    # A positive requirement, not just guards. Written first with only the
    # negative checks, this gate passed anything short that was not a question,
    # so "stop the timer" and "set a timer for ninety minutes" both arrived at
    # dismiss once their own intents declined. Leaving the embedding to sort
    # that out is the exact mistake the module docstring argues against.
    if not _DISMISS_WORD.search(text):
        return None

    # The dismiss tool description warns Claude not to fire when a closing
    # phrase sits inside a larger thought. A word cap is the structural version
    # of that warning: "thanks, now set a timer for ten minutes" is not a
    # goodbye, and it is long.
    if len(text.split()) > MAX_DISMISS_WORDS:
        return None

    # A closing word with an object after it is about the object, not the
    # conversation. "Never mind" ends the turn; "never mind about the cold
    # reminder" cancels a reminder, and that one is a real transcript.
    if _DISMISS_OBJECT.search(text):
        return None

    if _QUESTIONISH.search(text) or _DURATION.search(text):
        return None
    # Carried so the farewell can answer what he actually said. "Never mind"
    # earning "Any time at all" is what this exists to stop.
    return {'thanked': bool(_THANKS.search(text))}


_TIME_QUESTION = re.compile(
    r"\b(what time is it|what's the time|whats the time|what time|"
    r"do you (have|know) the time|got the time|the time right now)\b")

# A clock question is about now. Any of these makes it about something else:
# "what time is my reminder", "what time does it open".
_TIME_OBJECT = re.compile(
    r'\b(timer|reminder|alarm|does|did|will|should|tomorrow|yesterday|'
    r'meeting|class|game|open|close|start|end)\b')


def _gate_time_of_day(text):
    if not _TIME_QUESTION.search(text) or _TIME_OBJECT.search(text):
        return None
    return {}


_REMINDER_WORD = re.compile(r'\b(reminder|reminders)\b')
# Timers are rows too since Sep 16 2026, so they cancel the same way.
_TIMER_WORD = re.compile(r'\b(timer|timers|alarm|countdown)\b')

# Wider than _CANCELS, which guards set_timer and must not treat "never mind"
# as cancelling. Aimed at a reminder, "never mind" plainly does. "stop" is here
# for timers: "stop the timer" is the commonest way to say it.
_CANCEL_REMINDER_WORD = re.compile(
    r'\b(cancel|forget|delete|remove|scrap|drop|stop|kill|clear|never mind|nevermind)\b')


def _gate_cancel_reminder(text):
    if not _CANCEL_REMINDER_WORD.search(text):
        return None
    timer, reminder = bool(_TIMER_WORD.search(text)), bool(_REMINDER_WORD.search(text))
    if timer == reminder:
        # Neither named, or both: nothing says which to cancel.
        return None
    kind = "timer" if timer else "reminder"
    # Only when there is nothing to disambiguate, counted within that kind, so
    # "cancel the timer" with one reminder outstanding never cancels the
    # reminder. Zero or several both go to Claude, because cancelling the wrong
    # one is worse than spending four seconds cancelling the right one.
    try:
        if active_reminder_count(kind) != 1:
            return None
    except Exception:
        return None
    return {"kind": kind}


_WEATHER_WORD = re.compile(
    r'\b(weather|forecast|rain|raining|snow|snowing|sunny|cloudy|overcast|'
    r'humid|humidity|windy|jacket|umbrella|muggy|freezing|outside|'
    r'how (?:hot|cold|warm|chilly)|temperature|degrees)\b')

# The Pi has a core temperature too and get_system_state answers that one.
# Without this, "what's the core temperature" reads as weather. Both spellings
# of Pi are here because on Aug 13 2026 whisper transcribed "what's the
# temperature of the Pi" as "pie" and Nova answered that she had no
# thermometer. "your" but not "you", since "can you tell me the weather" is one
# of the most common phrasings there is and must still fire.
_WEATHER_INDOOR = re.compile(
    r'\b(pi|pie|cpu|core|chip|processor|system|server|inside|in here|your)\b')

# The tool answers now, plus twelve hours of precipitation outlook. Anything
# further out is a forecast it cannot make.
_WEATHER_ELSEWHEN = re.compile(
    r'\b(tomorrow|tonight|yesterday|weekend|next week|this week|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b')

# Any "in" is read as naming a place. This declines on "will it rain in the
# morning" too, which costs a fall through to Claude and buys never answering
# about Gainesville when he asked about Miami. Extracting the place instead
# would be guessing, and a wrong city stated confidently is the failure this
# module is written to avoid.
_WEATHER_ELSEWHERE = re.compile(r'\bin\b')


def _gate_weather(text):
    if not _WEATHER_WORD.search(text):
        return None
    if (_WEATHER_INDOOR.search(text) or _WEATHER_ELSEWHEN.search(text)
            or _WEATHER_ELSEWHERE.search(text)):
        return None
    return {}


GATES = {
    'set_timer': _gate_set_timer,
    'time_of_day': _gate_time_of_day,
    'cancel_reminder': _gate_cancel_reminder,
    'dismiss': _gate_dismiss,
    'weather': _gate_weather,
}


def classify(text):
    """The matched intent, or None meaning send it to Claude."""
    text = normalize(text)
    if not text:
        return None

    passed = {}
    for name, gate in GATES.items():
        slots = gate(text)
        if slots is not None:
            passed[name] = slots
    if not passed:
        return None

    try:
        scores = _scores(text)
    except Exception as exc:
        # The embedding model is loaded in a daemon thread at boot and is
        # allowed to fail. Declining here means the turn goes to Claude, which
        # is exactly what happens today.
        print(f"Intent scoring unavailable, deferring to Claude: {exc}", flush=True)
        return None

    name = max(passed, key=lambda n: scores[n])
    if scores[name] < INTENT_SIMILARITY_THRESHOLD:
        return None
    return Match(name, passed[name], scores[name])


# ── Weather, composed here rather than by Claude ──
# The only local intent whose answer is not a fixed sentence. Temperature times
# condition is not a space that can be enumerated and rendered, so this one
# alone falls through to live ElevenLabs. That is still worth doing: it skips
# claude_ttft and the second call that reads the tool result back, which
# measured 1324ms and 720ms, and keeps only the 628ms fetch and the 382ms of
# synthesis. It does NOT work offline. Nothing here makes weather local, only
# the language model.

_WEATHER_TEMPLATES = (
    "It's {temp} degrees with {condition}.",
    "{temp} degrees, {condition}.",
    "Right now, {temp} degrees with {condition}.",
)

# "3 PM" out of _precip_outlook, said as words. Nova spells numbers aloud
# everywhere else and a bare digit here would be the one place she does not.
_CLOCK_HOUR = re.compile(r'\b(\d{1,2}) ?(AM|PM)\b', re.I)


def _spoken(n):
    """Spell it, or leave the digits if it is off the end of the speller.

    A thermometer reading outside the spelled range is not a reason to fail a
    turn that has already committed to being answered locally."""
    try:
        return words_for_number(n)
    except ValueError:
        return str(n)


def _spoken_clock(phrase):
    return _CLOCK_HOUR.sub(
        lambda m: f"{_spoken(int(m.group(1)))} {m.group(2).upper()}", phrase)


# OpenWeatherMap's wording is written to be read, not spoken. "It's seventy two
# degrees with clear sky" is not a sentence anyone says. Only the awkward ones
# are here; "light rain" and "few clouds" already sit correctly after "with".
_CONDITION_SPOKEN = {
    'clear sky': 'clear skies',
    'overcast clouds': 'full overcast',
    'heavy intensity rain': 'heavy rain',
    'very heavy rain': 'heavy rain',
    'shower rain': 'showers',
}


def _sentence_case(phrase):
    """Upper the first letter and leave the rest alone.

    str.capitalize lowercases the remainder, which turned "rain likely around
    three PM" into "...three pm" and undid the spelling work one line earlier."""
    return phrase[:1].upper() + phrase[1:] if phrase else phrase


def _weather_sentence(data):
    condition = _CONDITION_SPOKEN.get(data['condition'], data['condition'])
    parts = [random.choice(_WEATHER_TEMPLATES).format(
        temp=_spoken(data['temp']), condition=condition)]

    # Only when it disagrees enough to be worth a sentence. Announcing that
    # eighty four feels like eighty five is noise.
    if abs(data['feels_like'] - data['temp']) >= 4:
        parts.append(f"Feels like {_spoken(data['feels_like'])}.")

    if data.get('precip'):
        parts.append(_sentence_case(_spoken_clock(data['precip'])) + ".")

    return " ".join(parts)


def _weather_response():
    """(spoken text, phrase bank key, dismissed) for a local weather turn.

    A failure here cannot fall through to Claude: classify already committed
    this turn to the local path. So it diagnoses instead, the same way a failed
    Claude turn does, and says which thing is broken rather than apologizing
    vaguely.

    Only network errors reach netcheck. A bare `except Exception` here caught an
    AttributeError from a wrong function name and announced "it's the model I
    can't reach", which is the exact false statement netcheck exists to stop it
    making. Anything that is not a failed request is a bug, and a bug belongs at
    run_turn's boundary saying so, not dressed up as a network problem."""
    try:
        data = actions.fetch_weather()
    except requests.RequestException as exc:
        cause = netcheck.diagnose()
        print(f"Local weather failed, {cause}: {exc}", flush=True)
        return phrasebank.PHRASES[cause][0], cause, False

    # Reached OpenWeatherMap and it declined. Not a network fault, so it does
    # not get a network explanation.
    if 'error' in data:
        print(f"Local weather unavailable: {data['error']}", flush=True)
        return phrasebank.PHRASES['error'][0], 'error', False

    # No key, so phrasebank.play finds nothing and voice_main speaks this
    # through ElevenLabs. Passed explicitly rather than relying on a key that
    # happens to have no rendered files.
    return _weather_sentence(data), None, False


def execute(match):
    """Run it. Returns (spoken text, phrase bank key, whether to end the turn).

    The spoken text is returned even though the phrase bank normally plays it,
    because it is what gets written to history and shown in the app, and
    because it is the fallback if nothing is rendered yet."""
    if match.name == 'set_timer':
        amount, unit = match.slots['amount'], match.slots['unit']
        # The same function the tool calls, so the parsing and threading that
        # were already tested stay the tested ones.
        actions.set_timer(f"{amount} {unit}")
        return (phrasebank.timer_text(amount, unit),
                phrasebank.timer_key(amount, unit), False)

    if match.name == 'time_of_day':
        # Read at execution rather than at classification, so a slow turn
        # cannot report a time that has already passed.
        key = phrasebank.time_key_for(datetime.now())
        return phrasebank.PHRASES[key][0], key, False

    if match.name == 'cancel_reminder':
        # Empty text matches the single outstanding one of that kind.
        actions.cancel_reminder('', kind=match.slots["kind"])
        return phrasebank.PHRASES['cancelled'][0], 'cancelled', False

    if match.name == 'dismiss':
        return phrasebank.PHRASES['dismiss'][0], 'dismiss', True

    if match.name == 'weather':
        return _weather_response()

    raise ValueError(f"no handler for intent {match.name!r}")
