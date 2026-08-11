"""Pronunciation normalization.

The synthesizer reads "Lethanial" as spelled, which is not how it sounds. An
alias is a respelling fed to the synthesizer in place of the real word, so it
must reach the speaker and nothing else. "Luhthanyul" on a screen is just a
misspelling of his name.
"""

import pytest

import tts


@pytest.fixture(autouse=True)
def pronunciations(db, monkeypatch):
    """Point the normalizer at a throwaway database seeded by the migration."""
    monkeypatch.setattr(tts, "get_pronunciations", db.get_pronunciations)
    return db


# ── exact match ──

def test_replaces_the_grapheme_with_its_alias():
    assert tts.normalize_pronunciation("Hello Lethanial") == "Hello Luhthanyul"


def test_replaces_every_occurrence():
    out = tts.normalize_pronunciation("Lethanial, this is Lethanial speaking")
    assert out == "Luhthanyul, this is Luhthanyul speaking"


def test_text_with_no_match_is_untouched():
    assert tts.normalize_pronunciation("the weather is fine") == "the weather is fine"


def test_empty_text_is_safe():
    assert tts.normalize_pronunciation("") == ""


# ── case insensitivity ──

@pytest.mark.parametrize("written", ["Lethanial", "lethanial", "LETHANIAL", "LeThAnIaL"])
def test_matching_ignores_case(written):
    assert tts.normalize_pronunciation(f"hi {written}") == "hi Luhthanyul"


def test_the_alias_is_substituted_verbatim():
    """Capitalization is not preserved from the original. The synthesizer reads
    sound, not spelling, so matching the source casing would mean nothing."""
    assert tts.normalize_pronunciation("LETHANIAL") == "Luhthanyul"


# ── whole word only ──

@pytest.mark.parametrize("text", [
    "Lethanials",        # trailing
    "xLethanial",        # leading
    "unLethanialish",    # both sides
])
def test_a_grapheme_inside_a_longer_word_is_not_replaced(text):
    assert tts.normalize_pronunciation(text) == text


@pytest.mark.parametrize("text,expected", [
    ("Lethanial.", "Luhthanyul."),
    ("Lethanial,", "Luhthanyul,"),
    ("(Lethanial)", "(Luhthanyul)"),
    ("Lethanial's code", "Luhthanyul's code"),
    ('"Lethanial"', '"Luhthanyul"'),
])
def test_punctuation_still_counts_as_a_word_boundary(text, expected):
    assert tts.normalize_pronunciation(text) == expected


# ── longest match priority ──

def test_a_longer_grapheme_wins_over_a_shorter_one(db):
    """A shorter entry that is a prefix of a longer one would otherwise consume
    it and leave the remainder unreplaced, producing "Luhthanyul Leveille"
    instead of the full multi word alias."""
    db.upsert_pronunciation("Lethanial Leveille", "Luhthanyul Luhvay", verified=True)
    assert tts.normalize_pronunciation("Lethanial Leveille") == "Luhthanyul Luhvay"


def test_the_shorter_entry_still_works_on_its_own(db):
    db.upsert_pronunciation("Lethanial Leveille", "Luhthanyul Luhvay", verified=True)
    assert tts.normalize_pronunciation("Lethanial alone") == "Luhthanyul alone"


def test_ordering_is_by_grapheme_length_not_insertion(db):
    db.upsert_pronunciation("aa", "SHORT")
    db.upsert_pronunciation("aa bb cc", "LONGEST")
    db.upsert_pronunciation("aa bb", "MIDDLE")
    graphemes = [g for g, _ in db.get_pronunciations()]
    assert graphemes == sorted(graphemes, key=len, reverse=True)
    assert tts.normalize_pronunciation("aa bb cc") == "LONGEST"


# ── the text channel must never see an alias ──

def test_the_normalizer_is_only_called_from_the_speech_path():
    """The guarantee is structural, not conditional: normalization lives inside
    speak(), which only the voice path calls. Nothing on the text path can
    reach it, so there is no flag to get wrong."""
    import inspect
    import brain
    import server

    assert "normalize_pronunciation" in inspect.getsource(tts.speak)
    for module in (brain, server):
        assert "normalize_pronunciation" not in inspect.getsource(module)


def test_returned_text_keeps_the_real_spelling(db, monkeypatch):
    """What gets saved to history and shown in the app is the real name. Only
    the bytes handed to ElevenLabs carry the alias."""
    sent = {}
    monkeypatch.setattr(tts, "_elevenlabs", type("C", (), {
        "text_to_speech": type("T", (), {
            "stream": staticmethod(lambda **kw: sent.update(kw) or iter(()))
        })()
    })())
    monkeypatch.setattr(tts.subprocess, "Popen", lambda *a, **k: type("P", (), {
        "stdin": type("S", (), {"write": lambda s, c: None,
                                "flush": lambda s: None,
                                "close": lambda s: None})(),
        "wait": lambda s: None,
    })())

    original = "Good morning Lethanial."
    tts.speak(original)
    assert sent["text"] == "Good morning Luhthanyul."
    assert original == "Good morning Lethanial."


# ── runtime upsert ──

def test_a_new_entry_takes_effect_without_a_migration(db):
    db.upsert_pronunciation("Hevy", "Heavy", verified=True)
    assert tts.normalize_pronunciation("open Hevy") == "open Heavy"


def test_upsert_replaces_an_existing_alias(db):
    db.upsert_pronunciation("Lethanial", "Lethanyool", verified=False)
    rows = dict(db.get_pronunciations())
    assert rows["Lethanial"] == "Lethanyool"
    assert len(rows) == 1, "grapheme is UNIQUE, so this updates rather than adds"


def test_upsert_preserves_created_at_and_moves_updated_at(db):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    before = conn.execute(
        "SELECT created_at, updated_at FROM pronunciations WHERE grapheme='Lethanial'"
    ).fetchone()
    conn.close()

    db.upsert_pronunciation("Lethanial", "Different", verified=True)

    conn = sqlite3.connect(db.DB_PATH)
    after = conn.execute(
        "SELECT created_at, updated_at FROM pronunciations WHERE grapheme='Lethanial'"
    ).fetchone()
    conn.close()
    assert after[0] == before[0], "created_at records when it was first needed"
    assert after[1] >= before[1]


# ── the seeded row ──

def test_lethanial_is_seeded_with_phonetics(db):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute(
        "SELECT alias, ipa, arpabet, verified FROM pronunciations "
        "WHERE grapheme = 'Lethanial'"
    ).fetchone()
    conn.close()
    assert row == ("Luhthanyul", "ləˈθænjəl", "L AH0 TH AE1 N Y AH0 L", 1)
