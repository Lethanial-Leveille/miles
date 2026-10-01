"""The pure parts of the Hevy tools. Nothing here touches the network.

The kilogram figures are copied from his real log on Sep 28 2026, so the
rounding is tested against what Hevy actually sends, not a tidy number.
"""

import datetime

import pytest

import hevy_tools

EASTERN = datetime.timezone(datetime.timedelta(hours=-4))
TODAY = datetime.date(2026, 9, 28)


@pytest.mark.parametrize("kg,lb", [
    (20.41168092460379, 45),
    (102.05840462301894, 225),
    (83.91468824559335, 185),
    (11.339822735890995, 25),
    (1.1339809, 2.5),
])
def test_pounds_come_back_as_he_logged_them(kg, lb):
    assert hevy_tools.pounds(kg) == lb


def test_zero_or_missing_weight_is_no_weight():
    assert hevy_tools.pounds(0) is None
    assert hevy_tools.pounds(None) is None


def _set(**fields):
    base = {"type": "normal", "weight_kg": None, "reps": None,
            "distance_meters": None, "duration_seconds": None}
    return {**base, **fields}


def test_a_barbell_set():
    s = _set(weight_kg=83.91468824559335, reps=6)
    assert hevy_tools.set_words(s, "Bench Press (Barbell)") == "6 reps at 185 pounds"


def test_zeros_in_older_sets_are_not_spoken():
    """Regression for his May 2026 bench sets, which carry 0 for distance and
    duration. Read as values they became "for 0 seconds over 0 meters"."""
    s = _set(weight_kg=70.30690096252415, reps=8, distance_meters=0, duration_seconds=0)
    assert hevy_tools.set_words(s, "Bench Press (Barbell)") == "8 reps at 155 pounds"


def test_a_weighted_pull_up_says_the_weight_was_added():
    s = _set(weight_kg=11.339822735890995, reps=5)
    assert hevy_tools.set_words(s, "Pull Up (Weighted)") == "5 reps with 25 pounds added"


def test_bodyweight_and_timed_sets():
    assert hevy_tools.set_words(_set(reps=10), "Hanging Leg Raise") == "10 reps"
    assert hevy_tools.set_words(_set(duration_seconds=90), "Dead Hang") == "1 minute 30 seconds"
    assert hevy_tools.set_words(_set(reps=1), "Pull Up") == "1 rep"


def test_a_set_that_is_not_normal_says_so():
    s = _set(weight_kg=83.91468824559335, reps=3, type="failure")
    assert hevy_tools.set_words(s, "Bench Press (Barbell)") == "3 reps at 185 pounds (failure)"


def test_a_late_session_is_filed_under_his_day_not_utc():
    """02:30 UTC on the 23rd is 22:30 Eastern on the 22nd. "Did I lift
    Tuesday" has to find it."""
    when = hevy_tools._when("2026-09-23T02:30:00+00:00", TODAY, EASTERN)
    assert when == {"date": "2026-09-22", "weekday": "Tuesday", "days_ago": 6}


def test_a_workout_summary():
    w = {"title": "Upper (A)", "description": "",
         "start_time": "2026-09-22T21:59:16+00:00", "end_time": "2026-09-22T22:33:08+00:00",
         "exercises": [{"title": "Bench Press (Barbell)", "notes": "",
                        "sets": [_set(weight_kg=83.91468824559335, reps=6)]}]}
    out = hevy_tools.summarize_workout(w, TODAY, EASTERN)
    assert out["weekday"] == "Tuesday"
    assert out["duration"] == "33 minutes 52 seconds"
    assert out["exercises"] == [{"exercise": "Bench Press (Barbell)",
                                 "sets": ["6 reps at 185 pounds"]}]
    assert "his_notes" not in out


TITLES = ["Bench Dip", "Bench Press (Barbell)", "Bench Press (Smith Machine)",
          "Incline Bench Press (Dumbbell)", "Pull Up", "Pull Up (Weighted)"]


def test_an_exact_name_wins_alone():
    assert hevy_tools.match_title("pull up", TITLES) == ["Pull Up"]
    assert hevy_tools.match_title("Bench Press (Barbell)", TITLES) == ["Bench Press (Barbell)"]


def test_a_vague_name_returns_every_match_rather_than_choosing():
    assert len(hevy_tools.match_title("bench", TITLES)) == 4
    assert hevy_tools.match_title("barbell bench", TITLES) == ["Bench Press (Barbell)"]


def test_no_match():
    assert hevy_tools.match_title("deadlift", TITLES) == []
    assert hevy_tools.match_title("  ", TITLES) == []


def _hist(workout, start, kg, reps, set_type="normal"):
    return {"workout_id": workout, "workout_title": "Upper (A)",
            "workout_start_time": start, "weight_kg": kg, "reps": reps,
            "duration_seconds": None, "distance_meters": None, "set_type": set_type}


LB185 = 83.91468824559335
LB155 = 70.30690096252415


def test_history_top_sets_best_set_and_estimate():
    sets = [
        _hist("a", "2026-08-25T21:00:00+00:00", LB185, 7),
        _hist("a", "2026-08-25T21:00:00+00:00", LB185, 5),
        _hist("b", "2026-09-22T21:59:16+00:00", LB185, 6),
        _hist("b", "2026-09-22T21:59:16+00:00", LB155, 20, set_type="warmup"),
    ]
    out = hevy_tools.summarize_history("Bench Press (Barbell)", sets, TODAY, EASTERN)
    assert [r["date"] for r in out["recent_sessions_newest_first"]] == ["2026-09-22", "2026-08-25"]
    assert out["recent_sessions_newest_first"][0]["working_sets"] == 1
    assert out["best_set_ever"]["set"] == "7 reps at 185 pounds"
    # 185 x (1 + 7/30), to the half pound
    assert out["estimated_one_rep_max"]["pounds"] == 228
    assert out["estimated_one_rep_max"]["from_set"] == "7 reps at 185 pounds"


def test_a_high_rep_set_cannot_win_the_estimate():
    sets = [_hist("a", "2026-09-22T21:00:00+00:00", LB155, 30),
            _hist("a", "2026-09-22T21:00:00+00:00", LB185, 5)]
    out = hevy_tools.summarize_history("Bench Press (Barbell)", sets, TODAY, EASTERN)
    assert out["estimated_one_rep_max"]["from_set"] == "5 reps at 185 pounds"


def test_no_estimate_for_added_weight():
    """Regression for Sep 28 2026: Epley on the 25 pounds of a weighted pull up
    alone said his one rep max was 31 pounds."""
    sets = [_hist("a", "2026-08-25T21:00:00+00:00", 11.339822735890995, 5)]
    out = hevy_tools.summarize_history("Pull Up (Weighted)", sets, TODAY, EASTERN)
    assert "estimated_one_rep_max" not in out
    assert out["best_set_ever"]["set"] == "5 reps with 25 pounds added"


def test_history_of_only_warm_ups_is_nothing():
    sets = [_hist("a", "2026-09-22T21:00:00+00:00", LB155, 10, set_type="warmup")]
    assert hevy_tools.summarize_history("Bench Press (Barbell)", sets, TODAY, EASTERN) is None


def test_routines_carry_when_he_last_did_them():
    routines = [{"id": "r1", "title": "Upper (A)", "exercises": [
                    {"title": "Bench Press (Barbell)", "notes": "pause the first rep",
                     "sets": [_set(weight_kg=LB185, reps=6)]}]},
                {"id": "r2", "title": "Lower (B)", "exercises": []}]
    workouts = [{"routine_id": "r1", "start_time": "2026-08-25T21:00:00+00:00"},
                {"routine_id": "r1", "start_time": "2026-09-22T21:59:16+00:00"},
                {"routine_id": None, "start_time": "2026-09-07T21:00:00+00:00"}]
    out = hevy_tools.summarize_routines(routines, workouts, TODAY, EASTERN)
    assert out[0]["last_done"]["date"] == "2026-09-22"
    assert out[0]["exercises"][0] == {"exercise": "Bench Press (Barbell)",
                                      "planned_sets": ["6 reps at 185 pounds"],
                                      "his_notes": "pause the first rep"}
    assert out[1]["last_done"] == "not in his last 3 workouts"


def test_working_weight_ignores_a_single_heavy_test():
    """Regression for Sep 22 2026: four sets at 185 and one single at 225. The
    heaviest set made the target one more rep at 225."""
    sets = [_set(weight_kg=LB185, reps=r) for r in (6, 6, 5, 3)] + [_set(weight_kg=102.05840462301894, reps=1)]
    assert hevy_tools.pounds(hevy_tools.working_weight(sets)) == 185


def test_working_weight_tie_goes_to_the_heavier():
    sets = [_set(weight_kg=LB155, reps=8), _set(weight_kg=LB185, reps=6)]
    assert hevy_tools.pounds(hevy_tools.working_weight(sets)) == 185


def _workout(start, *exercises):
    return {"start_time": start, "exercises": [
        {"title": title, "exercise_template_id": tid, "sets": sets}
        for title, tid, sets in exercises]}


def _bench(start, *reps):
    return _workout(start, ("Bench Press (Barbell)", "B",
                            [_set(weight_kg=LB185, reps=r) for r in reps]))


ROUTINE = {"title": "Upper (A)", "exercises": [
    {"title": "Bench Press (Barbell)", "exercise_template_id": "B",
     "sets": [_set(weight_kg=LB185, reps=6)]},
    {"title": "Pull Up (Weighted)", "exercise_template_id": "P", "sets": [_set(reps=5)]},
]}


def test_a_plan_says_last_time_and_the_target():
    workouts = [_bench("2026-09-22T21:59:16+00:00", 6, 6, 5)]
    plan = hevy_tools.plan_session(ROUTINE, hevy_tools.lift_sessions(workouts), TODAY, EASTERN)
    bench, pull = plan["lifts"]
    assert bench["last_time"]["sets"] == ["6 reps at 185 pounds"] * 2 + ["5 reps at 185 pounds"]
    assert bench["target"] == "Stay at 185 pounds and add one rep to any set"
    assert bench["stalled"] == "too few sessions to tell"
    assert pull["last_time"] == "not in his recent logged workouts"


def test_two_sessions_without_beating_the_one_before_is_a_stall():
    workouts = [_bench("2026-08-18T21:00:00+00:00", 6, 6, 5),
                _bench("2026-08-25T21:00:00+00:00", 6, 6, 5),
                _bench("2026-09-22T21:00:00+00:00", 6, 5, 5)]
    plan = hevy_tools.plan_session(ROUTINE, hevy_tools.lift_sessions(workouts), TODAY, EASTERN)
    assert plan["lifts"][0]["stalled"] is True


def test_one_more_rep_is_progress():
    workouts = [_bench("2026-08-18T21:00:00+00:00", 6, 6, 5),
                _bench("2026-08-25T21:00:00+00:00", 6, 6, 5),
                _bench("2026-09-22T21:00:00+00:00", 6, 6, 6)]
    plan = hevy_tools.plan_session(ROUTINE, hevy_tools.lift_sessions(workouts), TODAY, EASTERN)
    assert plan["lifts"][0]["stalled"] is False


def test_a_heavy_single_does_not_make_the_next_session_a_stall():
    heavy = _bench("2026-08-25T21:00:00+00:00", 6, 6, 5)
    heavy["exercises"][0]["sets"].append(_set(weight_kg=102.05840462301894, reps=1))
    workouts = [_bench("2026-08-18T21:00:00+00:00", 6, 6, 4), heavy,
                _bench("2026-09-22T21:00:00+00:00", 6, 6, 6)]
    plan = hevy_tools.plan_session(ROUTINE, hevy_tools.lift_sessions(workouts), TODAY, EASTERN)
    assert plan["lifts"][0]["stalled"] is False


def test_warm_ups_are_not_a_session():
    workouts = [_workout("2026-09-22T21:00:00+00:00", ("Bench Press (Barbell)", "B",
                                                        [_set(weight_kg=LB155, reps=10, type="warmup")]))]
    assert hevy_tools.lift_sessions(workouts) == {}


def _ranged(n, low, high, kg=None, **fields):
    return [_set(weight_kg=kg, rep_range={"start": low, "end": high}, **fields) for _ in range(n)]


def test_a_planned_set_says_its_range():
    s = _ranged(1, 4, 6, LB185)[0]
    assert hevy_tools.set_words(s, "Bench Press (Barbell)") == "4 to 6 reps at 185 pounds"


def test_a_hold_with_no_time_is_not_an_empty_set():
    assert hevy_tools.set_words(_set(), "Handstand Hold") == "no set target"


def test_every_set_at_the_top_earns_the_jump():
    last = [_set(weight_kg=LB185, reps=6) for _ in range(3)]
    assert (hevy_tools.target(last, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185))
            == "Go up to 190 pounds and start at 4 reps")


def test_one_set_short_of_the_top_stays():
    last = [_set(weight_kg=LB185, reps=r) for r in (6, 6, 5)]
    assert (hevy_tools.target(last, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185))
            == "Stay at 185 pounds and add one rep to any set; go up when every set hits 6")


def test_fewer_sets_than_planned_has_not_earned_the_jump():
    """Two sets at the top of a three set lift is a short day, not a jump."""
    last = [_set(weight_kg=LB185, reps=6) for _ in range(2)]
    assert hevy_tools.target(last, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185)).startswith("Stay")


def test_a_heavy_single_does_not_count_against_the_top():
    last = [_set(weight_kg=LB185, reps=6) for _ in range(3)] + [_set(weight_kg=102.05840462301894, reps=1)]
    assert hevy_tools.target(last, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185)).startswith("Go up to 190")


@pytest.mark.parametrize("title,expected", [
    ("Squat (Barbell)", "Go up to 305 pounds"),
    ("Romanian Deadlift (Barbell)", "Go up to 305 pounds"),
    ("Incline Bench Press (Dumbbell)", "Go up to 300 pounds"),
    ("Pull Up (Weighted)", "Go up to 300 pounds added"),
    ("Leg Extension (Machine)", "Go up to 305 pounds, or the nearest pin, and start"),
    ("Face Pull", "Go up to 305 pounds, or the nearest pin, and start"),
])
def test_the_jump_he_agreed_to(title, expected):
    kg295 = 133.8097
    last = [_set(weight_kg=kg295, reps=8)]
    assert hevy_tools.target(last, title, _ranged(1, 5, 8, kg295)).startswith(expected)


def test_bodyweight_at_the_top_is_told_to_make_it_harder():
    last = [_set(reps=10) for _ in range(3)]
    assert hevy_tools.target(last, "Straight Bar Dip", _ranged(3, 6, 10)).startswith("Every set hit 10")
    assert (hevy_tools.target([_set(reps=8)] * 3, "Straight Bar Dip", _ranged(3, 6, 10))
            == "Add one rep to any set; when every set hits 10, make it harder")


def test_a_fixed_drill_is_told_to_hit_the_plan():
    """Three muscle up attempts are not progressed by a fourth rep."""
    planned = [_set(reps=1) for _ in range(3)]
    assert hevy_tools.target([_set(reps=1)], "Muscle Up", planned) == "Hit the planned sets"


def test_warm_ups_in_the_routine_do_not_count_as_working_sets():
    planned = [_set(weight_kg=LB155, reps=5, type="warmup")] + _ranged(2, 5, 8, LB185)
    last = [_set(weight_kg=LB185, reps=8) for _ in range(2)]
    assert hevy_tools.target(last, "Squat (Barbell)", planned).startswith("Go up to 195")


def test_the_jump_is_data_as_well_as_words():
    """hevy_progress reads earned_jump; Nova reads target. Same decision."""
    last = [_set(weight_kg=LB185, reps=6) for _ in range(3)]
    assert hevy_tools.earned_jump(last, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185)) == (185, 190)
    short = [_set(weight_kg=LB185, reps=r) for r in (6, 6, 5)]
    assert hevy_tools.earned_jump(short, "Bench Press (Barbell)", _ranged(3, 4, 6, LB185)) is None


def test_bodyweight_and_unranged_lifts_have_no_jump():
    assert hevy_tools.earned_jump([_set(reps=10)] * 3, "Straight Bar Dip", _ranged(3, 6, 10)) is None
    assert hevy_tools.earned_jump([_set(weight_kg=LB185, reps=6)], "Bench Press (Barbell)",
                                  [_set(weight_kg=LB185, reps=6)]) is None


def test_a_plan_says_when_the_routine_was_raised():
    workouts = [_bench("2026-09-22T21:59:16+00:00", 6, 6, 6)]
    raised = {"2026-09-22T21:59:16+00:00": {"Bench Press (Barbell)": (185, 190)}}
    plan = hevy_tools.plan_session(ROUTINE, hevy_tools.lift_sessions(workouts), TODAY, EASTERN,
                                   raised_after=lambda start: raised.get(start, {}))
    assert plan["lifts"][0]["raised_in_routine_after_last_time"] == "185 to 190 pounds"
    assert "raised_in_routine_after_last_time" not in plan["lifts"][1]
