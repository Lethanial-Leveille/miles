import pytest

import local_intent as li
import phrasebank


# The semantic half is stubbed throughout. Loading sentence_transformers costs
# about seven seconds, the whole suite runs in under three, and what needs
# asserting here is the gate logic and the threshold, both of which are exact.
# The semantic half is validated by hand against real archive transcripts.
@pytest.fixture
def confident(monkeypatch):
    monkeypatch.setattr(li, "_scores", lambda text: {n: 1.0 for n in li.EXAMPLES})


@pytest.fixture
def unconfident(monkeypatch):
    monkeypatch.setattr(li, "_scores", lambda text: {n: 0.0 for n in li.EXAMPLES})


@pytest.mark.parametrize("text,amount,unit", [
    ("set a timer for ten minutes", 10, "minutes"),
    ("start a twenty minute timer", 20, "minutes"),
    ("timer for 5 minutes", 5, "minutes"),
    ("set a timer for twenty five minutes", 25, "minutes"),
    ("give me thirty seconds", 30, "seconds"),
    ("wake me up in half an hour", 30, "minutes"),
    ("set a timer for an hour", 1, "hours"),
    ("timer for a minute", 1, "minutes"),
])
def test_durations_parse(text, amount, unit, confident):
    match = li.classify(text)
    assert match is not None, text
    assert match.slots == {"amount": amount, "unit": unit}


@pytest.mark.parametrize("text", [
    "cancel the ten minute timer",
    "stop the timer",
    "forget the ten minute timer",
])
def test_cancelling_is_not_setting(text, confident, monkeypatch):
    """Semantically these sit right next to setting a timer, which is why the
    lexical gate carries the distinction rather than the embedding."""
    monkeypatch.setattr(li, "active_reminder_count", lambda kind=None: 0)
    assert li.classify(text) is None


@pytest.mark.parametrize("text", [
    "how long is ten minutes",
    "how much time is left",
    "what's the weather like today",
])
def test_no_timer_verb_no_timer(text, confident):
    """A duration in the sentence is not a request to start one.

    Asserts on the intent rather than on None because "what's the weather like
    today" legitimately matches weather now. The claim being pinned is that
    none of these start a timer, which is what it always was."""
    match = li.classify(text)
    assert match is None or match.name != 'set_timer'


def test_unenumerated_duration_defers(confident):
    """Ninety minutes has no rendered confirmation. Declining sends it to
    Claude, which is slower and correct; improvising would mean a different
    voice saying it."""
    assert li.classify("set a timer for ninety minutes") is None


def test_enumerated_boundary_still_fires(confident):
    assert li.classify("set a timer for sixty minutes") is not None


@pytest.mark.parametrize("text", [
    "that's all thanks",
    "goodnight",
    "never mind",
    "alright bye",
])
def test_dismissals_fire(text, confident):
    match = li.classify(text)
    assert match is not None and match.name == "dismiss"


def test_closing_phrase_inside_a_longer_thought_is_not_a_goodbye(confident):
    """The dismiss tool warns Claude about exactly this. Here it is structural:
    a goodbye is short, and this is not."""
    match = li.classify("thanks but first tell me what the weather is doing")
    assert match is None or match.name != "dismiss"


def test_a_request_that_thanks_first_is_still_a_request(confident):
    match = li.classify("thanks now set a timer for ten minutes")
    assert match is not None and match.name == "set_timer"


def test_low_similarity_defers_even_when_the_gate_passes(unconfident):
    """Both signals have to agree. The gate alone is not enough."""
    assert li.classify("set a timer for ten minutes") is None


def test_scoring_failure_defers_rather_than_raising(monkeypatch):
    """The embedding model loads in a daemon thread that is allowed to fail.
    When it has, every turn should go to Claude, which is what happens today."""
    def boom(text):
        raise RuntimeError("model never loaded")
    monkeypatch.setattr(li, "_scores", boom)
    assert li.classify("set a timer for ten minutes") is None


def test_execute_returns_a_rendered_key(confident, monkeypatch):
    started = []
    monkeypatch.setattr(li.actions, "set_timer", lambda spec: started.append(spec))

    match = li.classify("set a timer for ten minutes")
    spoken, key, dismissed = li.execute(match)

    assert started == ["10 minutes"]
    assert not dismissed
    assert spoken == "Timer set for ten minutes."
    assert key in phrasebank.PHRASES


def test_execute_dismiss_ends_the_turn(confident):
    spoken, key, dismissed = li.execute(li.classify("goodnight"))
    assert dismissed and key == "dismiss"
    assert spoken in phrasebank.PHRASES["dismiss"]


def test_singular_units_are_not_pluralized():
    assert phrasebank.timer_text(1, "minutes") == "Timer set for one minute."
    assert phrasebank.timer_text(2, "minutes") == "Timer set for two minutes."


def test_every_enumerated_timer_has_text():
    for amount, unit in phrasebank.TIMER_VALUES:
        assert phrasebank.PHRASES[phrasebank.timer_key(amount, unit)]


def test_goodnight_is_night_only():
    """A conversation ends at any hour. "Goodnight" at two in the afternoon is
    worse than no farewell, so it is filtered by clock rather than left to
    chance."""
    night = dict(phrasebank.rendered("dismiss", hour=23))
    day   = dict(phrasebank.rendered("dismiss", hour=14))
    assert 0 in night
    assert 0 not in day


def test_daytime_still_has_farewells():
    assert phrasebank.rendered("dismiss", hour=14)


def test_play_reports_what_it_played(monkeypatch):
    """History records the spoken variant, so play returns text rather than a
    bool. Returning a bool meant the caller guessed index zero and could log
    "Goodnight, Lethanial." on a turn that actually said "Talk soon."."""
    monkeypatch.setattr(phrasebank.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(phrasebank, "rendered",
                        lambda key, hour=None, thanked=False: [(2, "/nonexistent.wav")])
    assert phrasebank.play("dismiss") == phrasebank.PHRASES["dismiss"][2]


def test_play_returns_none_when_nothing_rendered(monkeypatch):
    monkeypatch.setattr(phrasebank, "rendered",
                        lambda key, hour=None, thanked=False: [])
    assert phrasebank.play("dismiss") is None


@pytest.mark.parametrize("text", [
    "what time is it",
    "what's the time",
    "do you have the time",
    "got the time",
])
def test_clock_questions_fire(text, confident):
    match = li.classify(text)
    assert match is not None and match.name == "time_of_day"


@pytest.mark.parametrize("text", [
    "what time is my reminder",
    "what time does the game start",
    "what time should I leave tomorrow",
])
def test_a_time_question_about_something_else_is_not_the_clock(text, confident):
    """"What time is it" asks about now. These ask about an event, and only
    Claude knows anything about the event."""
    match = li.classify(text)
    assert match is None or match.name != "time_of_day"


def test_clock_reads_at_execution_not_classification(confident):
    """A slow turn must not report a time that has already gone by."""
    from datetime import datetime
    match = li.classify("what time is it")
    spoken, key, dismissed = li.execute(match)
    assert not dismissed
    assert key == phrasebank.time_key_for(datetime.now())
    assert spoken.startswith("It's")


@pytest.mark.parametrize("minute,expected", [
    (0,  "It's four o'clock."),
    (5,  "It's four oh five."),
    (15, "It's four fifteen."),
    (55, "It's four fifty five."),
])
def test_clock_phrasing(minute, expected):
    assert phrasebank.time_text(4, minute) == expected


def test_clock_rounds_to_nearest_five_and_rolls_the_hour():
    from datetime import datetime
    assert phrasebank.time_key_for(datetime(2026, 8, 12, 16, 17)) == "time_4_15"
    assert phrasebank.time_key_for(datetime(2026, 8, 12, 16, 58)) == "time_5_0"
    assert phrasebank.time_key_for(datetime(2026, 8, 12,  0, 3))  == "time_12_5"


def test_cancel_fires_when_exactly_one_reminder_is_outstanding(confident, monkeypatch):
    monkeypatch.setattr(li, "active_reminder_count", lambda kind=None: 1)
    match = li.classify("cancel that reminder")
    assert match is not None and match.name == "cancel_reminder"


@pytest.mark.parametrize("count", [0, 2, 5])
def test_cancel_defers_when_ambiguous(count, confident, monkeypatch):
    """Cancelling the wrong reminder is worse than spending four seconds
    cancelling the right one, so anything but exactly one goes to Claude."""
    monkeypatch.setattr(li, "active_reminder_count", lambda kind=None: count)
    assert li.classify("cancel that reminder") is None


def test_cancel_defers_when_the_count_query_fails(confident, monkeypatch):
    def boom(kind=None):
        raise RuntimeError("database locked")
    monkeypatch.setattr(li, "active_reminder_count", boom)
    assert li.classify("cancel that reminder") is None


def test_never_mind_alone_is_a_dismissal_not_a_cancel(confident, monkeypatch):
    monkeypatch.setattr(li, "active_reminder_count", lambda: 1)
    match = li.classify("never mind")
    assert match is not None and match.name == "dismiss"


def test_every_enumerated_clock_reading_has_text():
    for hour, minute in phrasebank.TIME_VALUES:
        assert phrasebank.PHRASES[phrasebank.time_key(hour, minute)]


def test_never_mind_does_not_get_a_gratitude_reply(confident):
    """A retraction is not a thank you. "Any time at all" in reply to "never
    mind" reads as not having listened, which is what THANKS_ONLY prevents."""
    match = li.classify("never mind")
    assert match is not None and match.slots.get("thanked") is False

    playable = [phrasebank.PHRASES["dismiss"][i]
                for i, _ in phrasebank.rendered("dismiss", hour=14, thanked=False)]
    assert "Anytime." not in playable
    assert "Any time at all." not in playable
    assert playable, "a dismissal still needs something to say"


def test_thanking_her_unlocks_the_gratitude_replies(confident):
    match = li.classify("that's all thanks")
    assert match is not None and match.slots.get("thanked") is True

    playable = [phrasebank.PHRASES["dismiss"][i]
                for i, _ in phrasebank.rendered("dismiss", hour=14, thanked=True)]
    assert "Anytime." in playable


# ── weather ──
# The only local intent that composes its answer instead of choosing a rendered
# one, and the only one that still needs the network. Both of those make it the
# easiest to get wrong, so the gate is pinned in both directions.

@pytest.mark.parametrize("text", [
    "what's the weather",
    "what's it like outside",
    "how hot is it",
    "is it going to rain",
    "do I need a jacket",
    "can you tell me the weather",
    "what's the temperature outside",
    "how cold is it",
])
def test_weather_fires(text, confident):
    match = li.classify(text)
    assert match is not None and match.name == "weather"


@pytest.mark.parametrize("text", [
    # The Pi has a core temperature and get_system_state answers that one.
    "what's the core temperature",
    "how hot is the pi",
    "what's the temperature of the raspberry pi",
    # Whisper transcribes "Pi" as "pie", which is what sent this to the wrong
    # answer on Aug 13 2026. Both spellings have to decline.
    "what's the temperature of the pie right now",
    "what about the pie the raspberry pie",
    "what's your temperature",
    # A named place would need geocoding a guess out of free text.
    "what's the weather in miami",
    # Further out than the twelve hour outlook the tool can actually make.
    "what's the weather tomorrow",
    "will it rain this weekend",
])
def test_weather_declines(text, confident):
    match = li.classify(text)
    assert match is None or match.name != "weather"


def test_weather_returns_no_phrase_key(confident, monkeypatch):
    """Temperature times condition cannot be enumerated and rendered, so this
    intent alone returns a null key and is spoken live."""
    monkeypatch.setattr(li.actions, "fetch_weather", lambda: {
        "temp": 84, "feels_like": 91, "condition": "broken clouds",
        "precip": None})
    text, key, dismissed = li.execute(li.classify("what's the weather"))
    assert key is None
    assert dismissed is False
    assert "eighty four" in text


def test_weather_spells_every_number_it_says(confident, monkeypatch):
    monkeypatch.setattr(li.actions, "fetch_weather", lambda: {
        "temp": 101, "feels_like": 108, "condition": "clear sky",
        "precip": "rain likely around 3 PM"})
    text, _, _ = li.execute(li.classify("what's the weather"))
    assert not any(ch.isdigit() for ch in text), text
    assert "three PM" in text, text


def test_weather_network_failure_diagnoses(confident, monkeypatch):
    """A failed fetch cannot fall through to Claude, since classify already
    committed the turn. It names the cause instead."""
    def boom():
        raise li.requests.ConnectionError("down")
    monkeypatch.setattr(li.actions, "fetch_weather", boom)
    monkeypatch.setattr(li.netcheck, "diagnose", lambda: "no_wifi")
    text, key, _ = li.execute(li.classify("what's the weather"))
    assert key == "no_wifi"
    assert text in li.phrasebank.PHRASES["no_wifi"]


def test_a_bug_is_not_reported_as_a_network_problem(confident, monkeypatch):
    """A bare `except Exception` here caught an AttributeError and announced
    "it's the model I can't reach". Being wrong about its own state is worse
    than saying less, so anything that is not a failed request propagates."""
    def boom():
        raise AttributeError("typo in a function name")
    monkeypatch.setattr(li.actions, "fetch_weather", boom)
    with pytest.raises(AttributeError):
        li.execute(li.classify("what's the weather"))


# ── timers cancel locally too (Sep 16 2026) ──

def _counts(monkeypatch, timers, reminders):
    monkeypatch.setattr(li, "active_reminder_count",
                        lambda kind=None: {"timer": timers, "reminder": reminders}[kind])


@pytest.mark.parametrize("text", ["cancel the timer", "stop the timer", "never mind the timer",
                                  "kill the timer"])
def test_the_one_running_timer_is_cancelled_locally(text, confident, monkeypatch):
    _counts(monkeypatch, timers=1, reminders=0)
    match = li.classify(text)
    assert match is not None and match.name == "cancel_reminder"
    assert match.slots == {"kind": "timer"}


def test_cancelling_a_timer_never_touches_the_one_reminder(confident, monkeypatch):
    _counts(monkeypatch, timers=0, reminders=1)
    assert li.classify("cancel the timer") is None


def test_naming_neither_goes_to_claude(confident, monkeypatch):
    _counts(monkeypatch, timers=1, reminders=1)
    assert li.classify("cancel that") is None
    assert li.classify("cancel the timer and the reminder") is None
