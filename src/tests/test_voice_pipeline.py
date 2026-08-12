"""End to end wiring of the speech path.

Every test here exists because something in this chain broke silently and was
only found by listening. The chain is: text comes out of brain, gets bracket
cues stripped, gets the acronym normalized, gets pronunciation aliases applied,
and is handed to ElevenLabs with a specific voice id and settings object.

Each link has failed at least once. The voice id lived in .env where a missing
variable produced a 401 per sentence while the turn itself looked healthy. The
settings object was named after a voice that had been replaced. An alias was
being used as a regex replacement template. None of these raised anywhere a
test was looking, and none changed the printed transcript, which is exactly why
they are pinned here.
"""

import re

import pytest

import config
import tts


@pytest.fixture
def captured(monkeypatch, db):
    """Run speak() against a fake ElevenLabs and a fake aplay, returning every
    keyword argument the API was called with."""
    sent = {}

    class FakeTTS:
        @staticmethod
        def stream(**kwargs):
            sent.update(kwargs)
            return iter(())

    monkeypatch.setattr(tts, "_elevenlabs",
                        type("C", (), {"text_to_speech": FakeTTS()})())
    monkeypatch.setattr(tts, "get_pronunciations", db.get_pronunciations)
    # Alias mode pinned: these assert the text handed to the API, and the
    # phoneme tag is asserted separately in test_pronunciation.py.
    monkeypatch.setattr(tts, "TTS_PHONEME_TAGS", False)
    monkeypatch.setattr(tts.subprocess, "Popen", lambda *a, **k: type("P", (), {
        "stdin": type("S", (), {"write": lambda s, c: None,
                                "flush": lambda s: None,
                                "close": lambda s: None})(),
        "wait": lambda s: None,
    })())
    return sent


# ── the voice is configured in exactly one place ──

def test_speak_uses_the_configured_voice_id(captured):
    tts.speak("hello")
    assert captured["voice_id"] == config.TTS_VOICE_ID


def test_speak_uses_the_configured_settings_object(captured):
    tts.speak("hello")
    assert captured["voice_settings"] is config.TTS_VOICE_SETTINGS


def test_speak_uses_the_configured_model_and_format(captured):
    tts.speak("hello")
    assert captured["model_id"] == config.DEFAULT_TTS_MODEL
    assert captured["output_format"] == config.TTS_OUTPUT_FORMAT


def test_an_explicit_override_still_wins(captured):
    tts.speak("hello", voice_settings=config.VOICE_SERIOUS,
              model=config.EXPRESSIVE_TTS_MODEL)
    assert captured["voice_settings"] is config.VOICE_SERIOUS
    assert captured["model_id"] == config.EXPRESSIVE_TTS_MODEL


def test_the_voice_id_is_present_and_plausible():
    """A missing or empty id is the failure that produced a 401 on every
    sentence while the turn itself completed and printed the right answer."""
    assert config.TTS_VOICE_ID
    assert re.fullmatch(r"[A-Za-z0-9]{20}", config.TTS_VOICE_ID), config.TTS_VOICE_ID


def test_no_module_reads_the_voice_id_from_the_environment():
    """It moved to config.py deliberately. Reading it from the environment again
    would reintroduce a voice that changes with deployment and has no history."""
    import inspect
    for module in (tts, config):
        assert "ELEVENLABS_VOICE_ID" not in inspect.getsource(module)


def test_settings_carry_the_values_that_were_chosen():
    """Pinned so a change is deliberate rather than incidental. stability in
    particular was chosen by ear across a sweep, 0.45 no through 0.90 good, and
    0.60 is the value that was live while the voice sounded inconsistent.

    Expect to update this deliberately. 0.75 read better on ordinary sentences
    and 0.90 held the name better, so the value is a live trade rather than a
    settled fact."""
    s = config.TTS_VOICE_SETTINGS
    assert (s.stability, s.similarity_boost, s.style) == (0.80, 0.75, 0.00)
    assert s.use_speaker_boost is True
    assert s.speed == 1.00, "an unset speed is an invisible API default"


# ── the text handed to the synthesizer ──

def test_bracket_cues_never_reach_the_synthesizer(captured):
    tts.speak("[calmly] Your timer is up.")
    assert "[" not in captured["text"]
    assert captured["text"] == "Your timer is up."


def test_the_acronym_is_spoken_as_a_word(captured):
    """M.I.L.E.S. read letter by letter is not the name of the system."""
    tts.speak("M.I.L.E.S. is online.")
    assert captured["text"] == "Miles is online."


def test_pronunciation_aliases_are_applied(captured):
    tts.speak("Good morning Lethanial.")
    assert captured["text"] == "Good morning Luhthanyul."


def test_the_whole_chain_runs_in_order(captured):
    """Bracket strip, then acronym, then alias, then a terminal period."""
    tts.speak("[warmly] Lethanial, M.I.L.E.S. is online")
    assert captured["text"] == "Luhthanyul, Miles is online."


def test_text_with_nothing_left_after_stripping_is_not_sent(captured):
    tts.speak("[calmly]")
    assert captured == {}


# ── aliases are data, not code ──

def test_an_alias_containing_a_backslash_is_substituted_literally(captured, db):
    """re.sub interprets backslash escapes in a replacement template. An alias
    of "back\\1slash" raised invalid group reference, and because normalization
    runs before the try block in speak(), that exception killed the entire turn
    rather than just the pronunciation."""
    db.upsert_pronunciation("Testword", r"back\1slash")
    tts.speak("say Testword now")
    assert captured["text"] == r"say back\1slash now."


def test_an_alias_containing_a_newline_escape_is_not_expanded(captured, db):
    db.upsert_pronunciation("Testword", r"a\nb")
    tts.speak("say Testword now")
    assert "\n" not in captured["text"]
    assert captured["text"] == r"say a\nb now."


def test_a_broken_lookup_speaks_the_text_unchanged(monkeypatch, captured):
    """Mispronouncing a word is a smaller problem than a bad row silencing
    every turn."""
    def boom():
        raise RuntimeError("table gone")

    monkeypatch.setattr(tts, "get_pronunciations", boom)
    tts.speak("Good morning Lethanial.")
    assert captured["text"] == "Good morning Lethanial."


# ── the text channel must never receive an alias ──

def test_normalization_lives_only_inside_speak():
    """Structural rather than conditional. The text channel never calls speak(),
    so there is no flag that can be set wrong."""
    import inspect
    import brain
    import server

    assert "normalize_pronunciation" in inspect.getsource(tts.speak)
    for module in (brain, server):
        assert "normalize_pronunciation" not in inspect.getsource(module)


def test_speaking_does_not_mutate_the_callers_string(captured):
    original = "Good morning Lethanial."
    tts.speak(original)
    assert original == "Good morning Lethanial."
    assert captured["text"] != original
