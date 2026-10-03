"""The automatic raise. Hevy is faked; the database is a throwaway one."""

import datetime

import hevy_progress
import hevy_tools

LB = hevy_tools._KG_PER_LB
NOW = datetime.datetime(2026, 10, 1, 12, 0, tzinfo=datetime.timezone.utc)


def _set(lb=None, reps=None, kind="normal", low=None, high=None, seconds=None):
    s = {"type": kind, "weight_kg": lb * LB if lb else None, "reps": reps,
         "distance_meters": None, "duration_seconds": seconds, "custom_metric": None}
    if low:
        s["rep_range"] = {"start": low, "end": high}
    return s


def _routine():
    return {"id": "r1", "title": "Lower Strength", "notes": None, "exercises": [
        {"title": "Squat (Barbell)", "exercise_template_id": "SQ", "superset_id": None,
         "rest_seconds": 150, "notes": None,
         "sets": [_set(225, 5, "warmup")] + [_set(295, low=5, high=8) for _ in range(3)]},
        {"title": "Leg Extension (Machine)", "exercise_template_id": "LE", "superset_id": None,
         "rest_seconds": 75, "notes": "Permanent, for your knees.",
         "sets": [_set(150, low=12, high=15) for _ in range(2)]},
        {"title": "L-Sit Hold", "exercise_template_id": "LS", "superset_id": None,
         "rest_seconds": 45, "notes": None, "sets": [_set(seconds=15)]},
    ]}


def _workout(squat_reps, extension_reps, wid="w1", end="2026-10-01T04:04:26+00:00"):
    return {"id": wid, "routine_id": "r1", "start_time": "2026-10-01T02:22:07+00:00",
            "end_time": end, "exercises": [
                {"title": "Squat (Barbell)", "exercise_template_id": "SQ",
                 "sets": [_set(225, 5, "warmup")] + [_set(295, r) for r in squat_reps]},
                {"title": "Leg Extension (Machine)", "exercise_template_id": "LE",
                 "sets": [_set(150, r) for r in extension_reps]},
                {"title": "L-Sit Hold", "exercise_template_id": "LS", "sets": [_set(seconds=15)]},
            ]}


def test_only_the_lift_that_earned_it_is_raised():
    """Oct 1 2026, his real session: squat 5, 6, 7 of 8 stays; leg extension
    15, 15 of 15 goes up ten."""
    raises = hevy_progress.raises_for(_workout([5, 6, 7], [15, 15]), _routine())
    assert raises == [("Leg Extension (Machine)", 150, 160, "LE")]


def test_a_routine_already_at_the_new_weight_is_left_alone():
    routine = _routine()
    routine["exercises"][1]["sets"] = [_set(160, low=12, high=15) for _ in range(2)]
    assert hevy_progress.raises_for(_workout([5, 6, 7], [15, 15]), routine) == []


def test_only_the_raised_working_sets_change():
    routine = _routine()
    body = hevy_progress.raised_routine(routine, [("Leg Extension (Machine)", 150, 160, "LE")])
    squat, extension, hold = body["routine"]["exercises"]
    assert [round(hevy_tools.pounds(s["weight_kg"])) for s in extension["sets"]] == [160, 160]
    assert all(s["rep_range"] == {"start": 12, "end": 15} for s in extension["sets"])
    assert extension["notes"] == "Permanent, for your knees."
    assert [s["weight_kg"] for s in squat["sets"]] == [s["weight_kg"] for s in routine["exercises"][0]["sets"]]
    assert "rep_range" not in hold["sets"][0], "Hevy rejects rep_range: null"


def _fake_hevy(monkeypatch, workouts, routine):
    puts = []
    monkeypatch.setattr(hevy_progress.hevy_tools, "_get",
                        lambda path, **params: {"workouts": workouts})
    monkeypatch.setattr(hevy_progress.hevy_tools, "_all_pages", lambda *a: [routine])
    monkeypatch.setattr(hevy_progress, "_put", lambda r, body: puts.append(body))
    monkeypatch.setattr(hevy_progress, "_backup", lambda r: None)
    return puts


def test_a_workout_is_judged_once(db, monkeypatch):
    puts = _fake_hevy(monkeypatch, [_workout([5, 6, 7], [15, 15])], _routine())
    assert hevy_progress.run(NOW) == [("raised", "Lower Strength", "Leg Extension (Machine)", 150, 160)]
    assert hevy_progress.run(NOW) == []
    assert len(puts) == 1
    assert db.hevy_raises_after("2026-10-01T02:22:07+00:00") == {"Leg Extension (Machine)": (150, 160)}


def test_a_workout_that_earned_nothing_is_marked_and_not_written(db, monkeypatch):
    puts = _fake_hevy(monkeypatch, [_workout([5, 6, 7], [15, 14])], _routine())
    assert hevy_progress.run(NOW) == []
    assert puts == []
    assert "w1" in db.hevy_workouts_seen()


def test_old_workouts_are_never_judged(db, monkeypatch):
    """The table starts empty; months of history must not raise anything."""
    old = _workout([8, 8, 8], [15, 15], end="2026-09-20T04:00:00+00:00")
    puts = _fake_hevy(monkeypatch, [old], _routine())
    assert hevy_progress.run(NOW) == []
    assert puts == []


# ── putting back a raise his phone undid ──

SAVED = "2026-10-02T23:58:38.787Z"


def _overwritten(updated_at, extension_lb=150):
    routine = _routine()
    routine["updated_at"] = updated_at
    routine["exercises"][1]["sets"] = [_set(extension_lb, low=12, high=15) for _ in range(2)]
    return routine


def _saved_workout():
    """A session that earned nothing, so only the restore is under test."""
    return {**_workout([5, 6, 7], [12, 12], wid="w2"), "updated_at": SAVED}


def test_a_raise_undone_by_a_workout_save_is_found():
    """Oct 2 2026: the routine changed 0.2 seconds before the workout saved."""
    routine = _overwritten("2026-10-02T23:58:38.625Z")
    undone = hevy_progress.undone_raises(routine, {"Leg Extension (Machine)": 160}, [_saved_workout()])
    assert undone == [("Leg Extension (Machine)", 150, 160, "LE")]


def test_a_weight_he_lowered_on_purpose_is_left_alone():
    """An edit in the app hours from any save is his decision, not a stale copy."""
    routine = _overwritten("2026-10-03T14:00:00.000Z")
    assert hevy_progress.undone_raises(routine, {"Leg Extension (Machine)": 160}, [_saved_workout()]) == []


def test_a_weight_still_where_miles_set_it_is_not_touched():
    routine = _overwritten("2026-10-02T23:58:38.625Z", extension_lb=160)
    assert hevy_progress.undone_raises(routine, {"Leg Extension (Machine)": 160}, [_saved_workout()]) == []


def test_run_restores_and_records_it_once(db, monkeypatch):
    routine = _overwritten("2026-10-02T23:58:38.625Z")
    db.record_hevy_progression({"id": "w1", "start_time": "2026-10-01T02:22:07+00:00"}, routine,
                               [("Leg Extension (Machine)", 150, 160)])
    saved = _saved_workout()
    puts = _fake_hevy(monkeypatch, [saved], routine)
    now = datetime.datetime(2026, 10, 3, 0, 10, tzinfo=datetime.timezone.utc)
    done = hevy_progress.run(now)
    assert ("restored", "Lower Strength", "Leg Extension (Machine)", 150, 160) in done
    assert db.hevy_raises_after(saved["start_time"], kind="restore") == {"Leg Extension (Machine)": (150, 160)}
    assert len(puts) == 1

    routine["updated_at"] = "2026-10-03T00:10:00.000Z"  # the restore moved it off the save
    assert hevy_progress.run(now) == []
