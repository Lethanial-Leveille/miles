"""Editing routines by voice: names matched in code, nothing written until a yes
on the next turn, and only the asked for exercise changed."""

import copy

import pytest

import hevy_edit
import hevy_tools
import pending_action as pa
from tools import ToolNotice

LB = hevy_tools._KG_PER_LB

TEMPLATES = [
    {"id": "HSPN", "title": "Handstand Push Up Negative", "type": "reps_only"},
    {"id": "PIKE", "title": "Pike Pushup", "type": "reps_only"},
    {"id": "HSH", "title": "Handstand Hold", "type": "duration"},
    {"id": "SQB", "title": "Squat (Barbell)", "type": "weight_reps"},
    {"id": "RDL", "title": "Romanian Deadlift (Barbell)", "type": "weight_reps"},
    {"id": "RDLD", "title": "Romanian Deadlift (Dumbbell)", "type": "weight_reps"},
    {"id": "LEXT", "title": "Leg Extension (Machine)", "type": "weight_reps"},
    {"id": "CALF", "title": "Calf Raise (Machine)", "type": "weight_reps"},
]


def _routine():
    return {"id": "r-lower", "title": "Lower Strength", "updated_at": "2026-11-03T20:00:00Z",
            "exercises": [
                {"index": 0, "title": "Squat (Barbell)", "exercise_template_id": "SQB",
                 "superset_id": None, "rest_seconds": 180, "notes": "belt on top sets",
                 "sets": [
                     {"index": 0, "type": "warmup", "weight_kg": 135 * LB, "reps": 5},
                     {"index": 1, "type": "normal", "weight_kg": 225 * LB, "reps": None,
                      "rep_range": {"start": 4, "end": 6}},
                     {"index": 2, "type": "normal", "weight_kg": 225 * LB, "reps": None,
                      "rep_range": {"start": 4, "end": 6}},
                 ]},
                {"index": 1, "title": "Handstand Push Up Negative", "exercise_template_id": "HSPN",
                 "superset_id": None, "rest_seconds": 120, "notes": None,
                 "sets": [{"index": i, "type": "normal", "weight_kg": None, "reps": 4}
                          for i in range(3)]},
                {"index": 2, "title": "Leg Extension (Machine)", "exercise_template_id": "LEXT",
                 "superset_id": None, "rest_seconds": 90, "notes": None,
                 "sets": [{"index": i, "type": "normal", "weight_kg": 160 * LB, "reps": None,
                           "rep_range": {"start": 12, "end": 15}} for i in range(2)]},
            ]}


class FakeHevy:
    """Routines that change when written, so a second confirmed change sees the first."""

    def __init__(self):
        self.routines = {"r-lower": _routine(),
                         "r-upper": {"id": "r-upper", "title": "Upper Strength", "exercises": [
                             {"title": "Pike Pushup", "exercise_template_id": "PIKE",
                              "superset_id": None, "rest_seconds": 90, "notes": None,
                              "sets": [{"type": "normal", "reps": 8}]}]}}
        self.puts, self.backups = [], []

    def put(self, routine_id, body):
        self.puts.append((routine_id, copy.deepcopy(body)))
        stored = self.routines[routine_id]
        by_id = {t["id"]: t["title"] for t in TEMPLATES}
        stored["exercises"] = [{**e, "title": by_id[e["exercise_template_id"]]}
                               for e in copy.deepcopy(body["routine"]["exercises"])]


@pytest.fixture
def hevy(monkeypatch):
    fake = FakeHevy()
    monkeypatch.setattr(hevy_edit, "_routines", lambda: copy.deepcopy(list(fake.routines.values())))
    monkeypatch.setattr(hevy_edit, "_all_templates", lambda: TEMPLATES)
    monkeypatch.setattr(hevy_edit, "_put", fake.put)
    monkeypatch.setattr(hevy_edit, "_backup", lambda r: fake.backups.append(copy.deepcopy(r)))
    pa._pending, pa._turn = None, 0
    pa.begin_turn()
    yield fake
    pa._pending, pa._turn = None, 0


def _confirm(approved=True):
    pa.begin_turn()
    return pa.resolve(approved)


def _sent(fake, title):
    _, body = fake.puts[-1]
    return next(e for e in body["routine"]["exercises"]
                if e["exercise_template_id"] == {t["title"]: t["id"] for t in TEMPLATES}[title])


# ── Names ──

@pytest.mark.parametrize("said, found", [
    ("pike pushups", ["Pike Pushup"]),
    ("pike push ups", ["Pike Pushup"]),
    ("Pike Pushup", ["Pike Pushup"]),
    ("handstand push up negatives", ["Handstand Push Up Negative"]),
    ("romanian deadlift", ["Romanian Deadlift (Barbell)", "Romanian Deadlift (Dumbbell)"]),
    ("romanian deadlift dumbbell", ["Romanian Deadlift (Dumbbell)"]),
    ("bench press", []),
])
def test_names_match_his_words(said, found):
    assert hevy_edit.match_names(said, [t["title"] for t in TEMPLATES]) == found


# ── Nothing is written until he says yes, on the next turn ──

def test_the_nov_3_swap_is_asked_then_done_on_yes(hevy):
    reply = hevy_edit.edit_routine("lower strength", "swap",
                                   exercise="handstand push up negative",
                                   new_exercise="pike pushups")
    assert "In Lower Strength, swap Handstand Push Up Negative for Pike Pushup, " \
           "3 sets of 4 reps, resting 2 minutes?" in reply
    assert reply.startswith("Nothing has changed yet.")
    assert hevy.puts == []

    done = _confirm()
    assert done.startswith("Done: in Lower Strength, swap Handstand Push Up Negative for Pike Pushup")
    assert "refresh his routines in the Hevy app" in done
    sent = _sent(hevy, "Pike Pushup")
    assert [s["reps"] for s in sent["sets"]] == [4, 4, 4]
    assert len(hevy.backups) == 1


def test_a_yes_on_the_same_turn_does_nothing(hevy):
    hevy_edit.edit_routine("lower strength", "remove", exercise="leg extension")
    assert "has not answered yet" in pa.resolve(True)
    assert hevy.puts == []


def test_a_no_does_nothing(hevy):
    hevy_edit.edit_routine("lower strength", "remove", exercise="leg extension")
    assert _confirm(approved=False).startswith("Cancelled. Nothing was done")
    assert hevy.puts == []


# ── Only the asked for exercise changes ──

def test_everything_else_goes_back_exactly_as_read(hevy):
    hevy_edit.edit_routine("lower strength", "adjust", exercise="leg extension", sets=3)
    _confirm()
    _, body = hevy.puts[-1]
    squat = body["routine"]["exercises"][0]
    assert squat["notes"] == "belt on top sets" and squat["rest_seconds"] == 180
    warm = squat["sets"][0]
    assert (warm["type"], warm["weight_kg"], warm["reps"]) == ("warmup", 135 * LB, 5)
    assert squat["sets"][1]["rep_range"] == {"start": 4, "end": 6}
    assert body["routine"]["title"] == "Lower Strength"


def test_rep_range_null_is_never_sent(hevy):
    hevy_edit.edit_routine("lower strength", "adjust", exercise="squat", reps=5)
    _confirm()
    for e in hevy.puts[-1][1]["routine"]["exercises"]:
        for s in e["sets"]:
            assert s.get("rep_range", "absent") is not None


def test_adjusting_sets_copies_the_last_working_set_and_keeps_warm_ups(hevy):
    reply = hevy_edit.edit_routine("lower strength", "adjust", exercise="squat", sets=4)
    assert "Change Squat (Barbell) in Lower Strength to 4 sets of 4 to 6 reps at 225 pounds" in reply
    _confirm()
    sets = _sent(hevy, "Squat (Barbell)")["sets"]
    assert [s["type"] for s in sets] == ["warmup", "normal", "normal", "normal", "normal"]


def test_adjusting_weight_and_range(hevy):
    reply = hevy_edit.edit_routine("lower strength", "adjust", exercise="leg extension",
                                   weight_lb=170, rep_min=10, rep_max=12)
    assert "to 2 sets of 10 to 12 reps at 170 pounds, resting 1 minute 30 seconds?" in reply


def test_a_swap_clears_the_old_lifts_weight(hevy):
    reply = hevy_edit.edit_routine("lower strength", "swap", exercise="leg extension",
                                   new_exercise="calf raise")
    assert "swap Leg Extension (Machine) for Calf Raise (Machine), 2 sets of 12 to 15 reps," in reply
    _confirm()
    assert all("weight_kg" not in s or s["weight_kg"] is None
               for s in _sent(hevy, "Calf Raise (Machine)")["sets"])


def test_a_swap_to_a_timed_hold_needs_seconds(hevy):
    with pytest.raises(ToolNotice, match="timed"):
        hevy_edit.edit_routine("lower strength", "swap", exercise="handstand push up negative",
                               new_exercise="handstand hold")
    reply = hevy_edit.edit_routine("lower strength", "swap", exercise="handstand push up negative",
                                   new_exercise="handstand hold", seconds=30)
    assert "3 sets of 30 seconds" in reply
    _confirm()
    assert all("reps" not in s or s["reps"] is None for s in _sent(hevy, "Handstand Hold")["sets"])


def test_adding_needs_reps_and_goes_where_asked(hevy):
    with pytest.raises(ToolNotice, match="how many reps"):
        hevy_edit.edit_routine("lower strength", "add", new_exercise="calf raise")
    reply = hevy_edit.edit_routine("lower strength", "add", new_exercise="calf raise",
                                   exercise="squat", rep_min=12, rep_max=15)
    assert "Add Calf Raise (Machine) to Lower Strength after Squat (Barbell), " \
           "3 sets of 12 to 15 reps?" in reply
    _confirm()
    titles = [e["exercise_template_id"] for e in hevy.puts[-1][1]["routine"]["exercises"]]
    assert titles == ["SQB", "CALF", "HSPN", "LEXT"]


def test_removing_the_last_exercise_is_refused(hevy):
    with pytest.raises(ToolNotice, match="only exercise"):
        hevy_edit.edit_routine("upper strength", "remove", exercise="pike pushup")


# ── Several changes said together, confirmed together ──

def test_two_changes_confirmed_together_both_land(hevy):
    hevy_edit.edit_routine("lower strength", "swap", exercise="handstand push up negative",
                           new_exercise="pike pushups")
    reply = hevy_edit.edit_routine("lower strength", "adjust", exercise="leg extension",
                                   weight_lb=170)
    assert "and change Leg Extension (Machine)" in reply
    done = _confirm()
    assert done.count("Done:") == 2
    final = hevy.routines["r-lower"]["exercises"]
    assert [e["title"] for e in final] == ["Squat (Barbell)", "Pike Pushup", "Leg Extension (Machine)"]
    assert final[2]["sets"][0]["weight_kg"] == pytest.approx(170 * LB)


def test_a_change_made_in_the_app_meanwhile_is_kept(hevy):
    hevy_edit.edit_routine("lower strength", "adjust", exercise="leg extension", weight_lb=170)
    # He renames nothing but adds a note to squats in the app before saying yes.
    hevy.routines["r-lower"]["exercises"][0]["notes"] = "pause the first rep"
    _confirm()
    assert hevy.puts[-1][1]["routine"]["exercises"][0]["notes"] == "pause the first rep"


def test_an_exercise_gone_by_the_time_of_the_yes_fails_out_loud(hevy):
    hevy_edit.edit_routine("lower strength", "adjust", exercise="leg extension", weight_lb=170)
    hevy.routines["r-lower"]["exercises"].pop()
    assert _confirm().startswith("Failed, not done:")
    assert hevy.puts == []


# ── Ambiguity and nonsense are asked about, never guessed ──

def test_several_matches_ask_which(hevy):
    with pytest.raises(ToolNotice, match="Romanian Deadlift \\(Barbell\\), Romanian Deadlift \\(Dumbbell\\)"):
        hevy_edit.edit_routine("lower strength", "swap", exercise="squat",
                               new_exercise="romanian deadlift")
    assert pa._pending is None


def test_an_unknown_exercise_is_asked_about(hevy):
    with pytest.raises(ToolNotice, match="no Hevy exercise called bench press"):
        hevy_edit.edit_routine("lower strength", "add", new_exercise="bench press", reps=5)


@pytest.mark.parametrize("args", [
    {"sets": 0}, {"sets": 11}, {"rep_min": 8}, {"rep_min": 8, "rep_max": 6},
    {"reps": 5, "rep_min": 4, "rep_max": 6}, {"weight_lb": -5}, {"reps": 0},
])
def test_impossible_numbers_are_refused_before_anything_is_read(hevy, args):
    with pytest.raises(ToolNotice):
        hevy_edit.edit_routine("lower strength", "adjust", exercise="squat", **args)


def test_adjust_with_nothing_to_change_asks(hevy):
    with pytest.raises(ToolNotice, match="what to change"):
        hevy_edit.edit_routine("lower strength", "adjust", exercise="squat")
