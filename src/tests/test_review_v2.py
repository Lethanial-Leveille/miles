"""Nightly review v2: the proposal, the placement, the guard on the model's
words, blocks, the backlog, and the confirm flow. Each bug found in the live
review has its own tests. Everything runs on a fake calendar and a fake model."""

import json
import random
import re
from datetime import date, datetime, time, timedelta

import pytest

import calendar_tools
import fake_google
import lists_tasks
import nightly_review as nr
import review_plan as rp
import session_types
import sessions
from conftest import text_block, tool_block
from fake_google import ACM, MILES, NSBE, PRIMARY
from tools import ToolNotice

NOW = datetime(2026, 11, 24, 21, 30)          # Tuesday night; the 4am day is the 24th
TODAY, TOMORROW = "2026-11-24", "2026-11-25"


def at(day, hour, minute=0, month=11):
    return datetime(2026, month, day, hour, minute)


@pytest.fixture
def world(db, monkeypatch):
    """A database, a fake calendar, a fixed clock and hokage clearance."""
    fake = fake_google.install(monkeypatch)
    monkeypatch.setattr(nr, "_clock", lambda: NOW)
    monkeypatch.setattr(sessions, "effective_tier", lambda: "hokage")
    return fake


def _task(db, text, day, must_do=False, **extra):
    [task_id] = db.add_planned_tasks(
        [{"text": text, "scheduled_for": day, "must_do": must_do, **extra}],
        datetime(2026, 11, 24, 8, 0))
    return task_id


def _backlog(db, text, due=None, minutes=None):
    return db.add_task(text, now=datetime(2026, 11, 20, 9, 0), due_date=due, minutes=minutes)


def _session(now=NOW):
    session, _ = sessions.create("nightly_review", now)
    return session


def _in_turn(session_id):
    token, state = session_types.begin_turn(session_id)
    return token, state


# ── the fixed night ──

def fixed_night(db, fake):
    """Today: three done and one moved to Friday. Tomorrow: DSA on a break day
    and tutoring. A test, a conflict on Friday, an old backlog item, and one
    optional task already planned."""
    done = [_task(db, title, TODAY) for title in ("Lab report", "Gym", "Read chapter 4")]
    essay = _task(db, "Essay", TODAY)
    for task_id in done:
        nr.update_task_status(task_id, "done", now=NOW)
    nr.update_task_status(essay, "moved", move_to="2026-11-27", now=NOW)
    _task(db, "Order NVMe HAT and SSD", TOMORROW)

    fake.add(PRIMARY, "dsa", "DSA", at(25, 10, 40), at(25, 11, 30), recurringEventId="dsa")
    fake.add(PRIMARY, "jor", "Jordan <> Lethanial Leveille | Test Prep | 1-1",
             at(25, 12, 30), at(25, 14))
    fake.add(PRIMARY, "alex", "Test Prep | Lethanial <> Alex", at(27, 16), at(27, 17))
    fake.add(PRIMARY, "dentist", "Dentist", at(27, 16, 30), at(27, 17, 30))
    _backlog(db, "Email professor")
    _backlog(db, "Study for linear algebra test", due="2026-11-30")


FIXED_OPENING = "\n".join([
    "Today: 3 of 4 done, 1 moved to Friday.",
    "",
    "Tomorrow, Wednesday Nov 25",
    "• 10:40 to 11:30 AM  DSA (likely cancelled, Thanksgiving break)",
    "• 12:30 to 2 PM  Test Prep with Jordan",
    "",
    "Suggested must dos",
    "1. Study for linear algebra test, 10 AM to 12 PM",
    "   Due Monday",
    "2. Fix Friday: Test Prep with Alex overlaps Dentist, 12 to 12:30 PM",
    "   Conflict to fix",
    "3. Email professor, 2 to 3 PM",
    "   Oldest in backlog",
    "",
    "Optional",
    "• Order NVMe HAT and SSD",
    "",
    "Good, or change something?",
])


def test_the_opening_is_exact_on_a_fixed_fixture(db, world):
    fixed_night(db, world)
    text, plan = nr.opening(NOW)
    assert text == FIXED_OPENING
    assert plan["day"] == TOMORROW
    assert [m["slot"] for m in plan["must_dos"]] == [["10:00", "12:00"], ["12:00", "12:30"],
                                                    ["14:00", "15:00"]]


def test_the_opening_is_stored_as_code_and_the_plan_is_saved(db, world):
    fixed_night(db, world)
    session = _session()
    [row] = db.session_messages(session["id"])
    assert row["author"] == "code"
    assert row["content"] == [{"type": "text", "text": FIXED_OPENING}]
    assert db.session_plan(session["id"])["must_dos"][0]["title"] == "Study for linear algebra test"


def test_an_empty_day_and_an_unreadable_calendar_still_open(db, monkeypatch):
    # no_real_google refuses every calendar call, as an outage would
    text, plan = nr.opening(NOW)
    assert text.startswith("Nothing was planned today.\n\nTomorrow, Wednesday Nov 25\n"
                           "Could not read the calendar")
    assert text.endswith("Good, or change something?")


# ── bug 1: a time shown is a time on the calendar ──

def test_a_deleted_booking_reads_not_booked(db, world):
    task_id = _task(db, "Study", TOMORROW, must_do=True)
    event_id = world.add(MILES, "blk", "Study", at(25, 14), at(25, 16))
    db.set_task_calendar_event(task_id, event_id)
    assert rp.render_plan(TOMORROW).splitlines()[1] == "1. Study, 2 to 4 PM"
    world.remove(MILES, event_id)               # he deleted the block in Google
    assert rp.render_plan(TOMORROW).splitlines()[1] == "1. Study, not booked"


def test_a_cancelled_event_and_a_never_booked_must_do_read_not_booked(db, world):
    cancelled = _task(db, "Gym", TOMORROW, must_do=True)
    never = _task(db, "Read", TOMORROW, must_do=True)
    world.add(MILES, "gone", "Gym", at(25, 14), at(25, 15), status="cancelled")
    db.set_task_calendar_event(cancelled, "gone")
    lines = rp.render_plan(TOMORROW).splitlines()
    assert lines[1:] == ["1. Gym, not booked", "2. Read, not booked"]
    assert never


def test_every_time_in_the_final_plan_is_a_time_on_the_calendar(db, world):
    rng = random.Random(2026)
    token_re = re.compile(r"\d{1,2}(?::\d{2})? (?:AM|PM)")
    for trial in range(80):
        day = date(2027, 1, 1) + timedelta(days=trial)
        expected = {}
        for i in range(rng.randint(0, 3)):
            title = f"Job {trial} {i}"
            task_id = _task(db, title, day.isoformat(), must_do=True)
            mode = rng.choice(["booked", "deleted", "cancelled", "never", "booked"])
            if mode != "never":
                start = datetime.combine(day, time(rng.randint(8, 20), rng.choice([0, 15, 30, 45])))
                end = start + timedelta(minutes=rng.choice([30, 45, 60, 90, 120]))
                event_id = f"e{trial}_{i}"
                world.add(MILES, event_id, title, start, end,
                          **({"status": "cancelled"} if mode == "cancelled" else {}))
                db.set_task_calendar_event(task_id, event_id)
                if mode == "deleted":
                    world.remove(MILES, event_id)
                expected[title] = rp.range_words(start, end) if mode == "booked" else None
        lines = rp.render_plan(day).splitlines()[1:]
        shown = {}
        for line in lines:
            match = re.match(r"\d+\. (Job \d+ \d+), (.*)$", line)
            if match:
                shown[match[1]] = match[2]
        for title, want in expected.items():
            assert shown[title] == (want or "not booked")
        calendar_times = {t for want in expected.values() if want for t in token_re.findall(want)}
        for line in lines:
            for token in token_re.findall(line):
                assert token in calendar_times


def test_the_guard_drops_clock_times_and_keeps_one_line():
    guard = sessions.guard_text
    assert guard("Study is 2 to 4 PM.\nSounds good, I moved it.") == "Sounds good, I moved it."
    assert guard("Here you go. Anything else you want to change?") == "Here you go."
    assert guard("Starts at 2:30.\nAlso 14:00.\nAt 9am sharp.") == ""
    assert guard("Moved it to 9am.") == ""
    assert guard("It is from noon.") == ""
    assert guard("") == ""
    long = "word " * 60
    assert len(guard(long)) <= sessions.GUARD_MAX_CHARS
    assert "\n" not in guard("one\ntwo\nthree")


@pytest.mark.parametrize("phrase", ["2 PM", "2pm", "2:30", "14:00", "2 to 4", "9am", "9 a.m.",
                                    "noon", "from 3 until 5", "10:40"])
def test_every_clock_time_shape_is_caught(phrase):
    assert sessions.has_clock_time(f"Study at {phrase} today")


def test_a_turn_where_code_posted_holds_the_model_to_one_clockless_line(db, world, fake_claude):
    session = _session()
    fake_claude([text_block("Study is at 2 PM.\nHere is the week. Enjoy it, truly."),
                 tool_block("show_week", {}, "toolu_1")])
    events = []
    reply = sessions.run_turn(session["id"], "show me the week",
                              on_text=lambda kind, text: events.append((kind, text)), now=NOW)

    assert events[0] == ("stage", "Reading your week")
    kinds = [k for k, _ in events]
    assert kinds == ["stage", "delta", "delta"]
    assert events[1] == ("delta", "Here is the week.")
    assert events[2][1].startswith("\n\nThe week ahead\n\nWednesday Nov 25")
    assert reply == "Here is the week.\n\n" + events[2][1][2:]

    rows = db.session_messages(session["id"])
    assert [(r["role"], r["author"]) for r in rows] == [
        ("assistant", "code"), ("user", "model"), ("assistant", "model"),
        ("user", "model"), ("assistant", "code")]
    # Stored blocks are API valid: the guarded text and the call, nothing else.
    assert rows[2]["content"] == [
        {"type": "text", "text": "Here is the week."},
        {"type": "tool_use", "id": "toolu_1", "name": "show_week", "input": {}}]
    assert rows[4]["content"] == [{"type": "text", "text": events[2][1][2:]}]
    # What the app reads after a reload matches what streamed.
    shown = [m["text"] for m in sessions.public_messages(rows)]
    assert shown[1:] == ["show me the week", "Here is the week.", events[2][1][2:]]


def test_the_next_request_after_a_posted_message_is_well_formed(db, world, fake_claude):
    session = _session()
    fake = fake_claude([text_block("Sure."), tool_block("show_week", {}, "toolu_1")],
                       [text_block("Anything else?")])
    sessions.run_turn(session["id"], "week please", now=NOW)
    sessions.run_turn(session["id"], "thanks", now=NOW)
    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert roles[0] == "user" and roles[-1] == "user"


def test_a_turn_with_no_post_streams_the_models_text_whole(db, world, fake_claude):
    session = _session()
    fake_claude([text_block("Marked it. Study is 2 to 4 PM, you said.")])
    events = []
    reply = sessions.run_turn(session["id"], "hi",
                              on_text=lambda k, t: events.append((k, t)), now=NOW)
    assert reply == "Marked it. Study is 2 to 4 PM, you said."
    assert events == [("delta", reply)]


# ── bug 2: blocks live on MILES, and a deleted one is forgotten ──

def test_move_and_remove_refuse_an_event_that_is_not_on_miles(db, world):
    world.add(PRIMARY, "prim", "Test Prep", at(25, 12), at(25, 13))
    task_id = _task(db, "Study", TOMORROW)
    db.set_task_calendar_event(task_id, "prim")
    with pytest.raises(ToolNotice, match="no block on the MILES calendar"):
        nr.move_block(task_id, "15:00", now=NOW)
    db.set_task_calendar_event(task_id, "prim")
    with pytest.raises(ToolNotice, match="no block on the MILES calendar"):
        nr.remove_block(task_id, now=NOW)
    assert world.written() == []
    assert "prim" in world.events[PRIMARY]


def test_move_and_remove_work_on_a_miles_block(db, world):
    task_id = _task(db, "Study", TOMORROW)
    nr.schedule_block(task_id, 60, start_time="15:00", now=NOW)
    said = nr.move_block(task_id, "17:00", now=NOW)
    assert said == "Moved Study to 5 PM to 6 PM on Wednesday November 25."
    [(_, _, event_id)] = [w for w in world.writes if w[0] == "patch"]
    assert world.events[MILES][event_id]["start"]["dateTime"].startswith("2026-11-25T17:00")
    assert nr.remove_block(task_id, now=NOW).startswith("Removed the block for Study")
    assert event_id not in world.events[MILES]
    assert db.get_task(task_id)["calendar_event_id"] is None


def test_a_move_onto_busy_time_is_refused(db, world):
    world.add(PRIMARY, "busy", "Class", at(25, 17), at(25, 18))
    task_id = _task(db, "Study", TOMORROW)
    nr.schedule_block(task_id, 60, start_time="15:00", now=NOW)
    with pytest.raises(ToolNotice, match="overlaps"):
        nr.move_block(task_id, "17:30", now=NOW)
    assert [w for w in world.writes if w[0] == "patch"] == []


def test_a_block_he_deleted_does_not_refuse_a_new_booking(db, world):
    task_id = _task(db, "Study", TOMORROW)
    nr.schedule_block(task_id, 60, start_time="15:00", now=NOW)
    [(_, old_id)] = [(w[1], w[2]) for w in world.writes if w[0] == "insert"]
    world.remove(MILES, old_id)                  # the live failure: deleted by hand
    said = nr.schedule_block(task_id, 60, start_time="15:00", now=NOW)
    assert said.startswith("Blocked Study on Wednesday November 25 from 3 PM to 4 PM")
    new_id = db.get_task(task_id)["calendar_event_id"]
    assert new_id and new_id != old_id and new_id in world.events[MILES]


# ── bug 3: placement ──

def item(title, minutes, start=None):
    return {"title": title, "minutes": minutes, "start": start}


DAY = date(2026, 11, 25)


def test_three_must_dos_in_order_from_ten():
    slots, unplaced = rp.place_blocks([], DAY, [item("A", 60), item("B", 60), item("C", 120)],
                                      (10, 0), (21, 0))
    assert unplaced == []
    assert [(s["item"]["title"], s["start"], s["end"]) for s in slots] == [
        ("A", at(25, 10), at(25, 11)), ("B", at(25, 11), at(25, 12)),
        ("C", at(25, 12), at(25, 14))]


def test_the_default_window_starts_at_ten():
    from config import SESSION_BLOCK_EARLIEST, SESSION_BLOCK_LATEST
    assert (SESSION_BLOCK_EARLIEST, SESSION_BLOCK_LATEST) == ((10, 0), (21, 0))


def test_placement_avoids_busy_time_and_each_other():
    busy = [(at(25, 10, 40), at(25, 11, 30)), (at(25, 12, 30), at(25, 14))]
    slots, unplaced = rp.place_blocks(busy, DAY, [item("A", 120), item("B", 30), item("C", 60)],
                                      (10, 0), (21, 0))
    assert unplaced == [] and rp.overlaps(busy, *busy[0]) is True
    spans = [(s["start"], s["end"]) for s in slots]
    # A needs two hours: only after 2 PM. B takes 10:00 to 10:30 before the
    # class. C takes the hour 11:30 to 12:30 between the class and tutoring.
    assert spans == [(at(25, 14), at(25, 16)), (at(25, 10), at(25, 10, 30)),
                     (at(25, 11, 30), at(25, 12, 30))]


def test_an_item_that_does_not_fit_is_reported():
    busy = [(at(25, 10), at(25, 20))]
    slots, unplaced = rp.place_blocks(busy, DAY, [item("A", 60), item("B", 120)], (10, 0), (21, 0))
    assert [s["item"]["title"] for s in slots] == ["A"]
    assert [u["title"] for u in unplaced] == ["B"]


def test_the_window_is_respected():
    slots, unplaced = rp.place_blocks([], DAY, [item("A", 60), item("B", 60), item("C", 60)],
                                      (10, 0), (12, 0))
    assert [s["item"]["title"] for s in slots] == ["A", "B"]
    assert [u["title"] for u in unplaced] == ["C"]
    assert all(at(25, 10) <= s["start"] and s["end"] <= at(25, 12) for s in slots)


def test_a_fixed_start_is_kept_and_a_busy_or_outside_one_is_unplaced():
    slots, unplaced = rp.place_blocks(
        [(at(25, 15), at(25, 16))], DAY,
        [item("A", 60), item("Fixed", 60, time(13, 0)), item("Clash", 60, time(15, 30)),
         item("Early", 30, time(8, 0))], (10, 0), (21, 0))
    by_title = {s["item"]["title"]: (s["start"], s["end"]) for s in slots}
    assert by_title["Fixed"] == (at(25, 13), at(25, 14))
    assert by_title["A"] == (at(25, 10), at(25, 11))      # a fixed time is never taken by a flexible one
    assert [u["title"] for u in unplaced] == ["Clash", "Early"]


def test_placement_properties_over_random_days():
    rng = random.Random(11)
    for _ in range(300):
        busy = []
        for _ in range(rng.randint(0, 5)):
            start = at(25, rng.randint(6, 21), rng.choice([0, 15, 30, 45]))
            busy.append((start, start + timedelta(minutes=rng.choice([30, 60, 90, 180]))))
        items = [item(f"I{n}", rng.choice([30, 45, 60, 120])) for n in range(rng.randint(1, 4))]
        earliest, latest = (rng.choice([(9, 0), (10, 0)]), rng.choice([(18, 0), (21, 0)]))
        slots, unplaced = rp.place_blocks(busy, DAY, items, earliest, latest)
        assert len(slots) + len(unplaced) == len(items)
        lo, hi = datetime.combine(DAY, time(*earliest)), datetime.combine(DAY, time(*latest))
        for i, s in enumerate(slots):
            assert lo <= s["start"] and s["end"] <= hi
            assert s["end"] - s["start"] == timedelta(minutes=s["item"]["minutes"])
            assert not rp.overlaps(busy, s["start"], s["end"])
            for other in slots[i + 1:]:
                assert s["end"] <= other["start"] or other["end"] <= s["start"]


def test_default_minutes_are_sixty_and_two_hours_for_an_exam():
    assert rp.default_minutes("Email professor") == 60
    assert rp.default_minutes("Study for linear algebra test") == 120
    assert rp.default_minutes("Study for DSA Final Exam") == 120
    assert rp.default_minutes("Test Prep with Jordan") == 60
    assert rp.default_minutes("Study for the exam", 45) == 45


def test_schedule_block_books_a_list_in_order_without_overlap(db, world):
    world.add(PRIMARY, "busy", "Class", at(25, 11), at(25, 12))
    ids = [_task(db, name, TOMORROW, must_do=True) for name in ("A", "B", "Study for the exam")]
    said = nr.schedule_block(tasks=[{"task_id": i} for i in ids], now=NOW)
    spans = [(b["start"]["dateTime"][11:16], b["end"]["dateTime"][11:16])
             for _, b in world.inserts]
    assert spans == [("10:00", "11:00"), ("12:00", "13:00"), ("13:00", "15:00")]
    assert said.count("Blocked") == 3


def test_schedule_block_takes_earliest_and_latest(db, world):
    task_id = _task(db, "Gym", TOMORROW)
    nr.schedule_block(task_id, 60, earliest="18:00", latest="20:00", now=NOW)
    [(_, body)] = world.inserts
    assert body["start"]["dateTime"][11:16] == "18:00"


# ── bug 4's other half: suggestions ──

def _event(title, day, hour, calendar=PRIMARY, **extra):
    from test_calendar_roles import ev
    return ev(title, at(day, hour), at(day, hour + 1), calendar=calendar, **extra)


def test_suggestions_come_in_the_decided_order_and_stop_at_three():
    import calendar_roles
    backlog = [
        {"id": 1, "text": "Oldest thing", "due_date": None, "minutes": None},
        {"id": 2, "text": "Pay the fee", "due_date": "2026-11-27", "minutes": None},
        {"id": 3, "text": "Late form", "due_date": "2026-11-22", "minutes": None},
        {"id": 4, "text": "Study for chem exam", "due_date": "2026-11-30", "minutes": None},
    ]
    events = calendar_roles.annotate([
        _event("Midterm 2", 26, 9), _event("Test Prep | Lethanial <> Alex", 27, 16),
        _event("Dentist", 27, 16)])
    picked = rp.suggest(date(2026, 11, 24), backlog, events)
    # Overdue and due within three days first (earliest due first), then exams.
    assert [(p["title"], p["reason"]) for p in picked] == [
        ("Late form", "Overdue"), ("Pay the fee", "Due Friday"), ("Study for Midterm 2", "Exam Thursday")]

    picked = rp.suggest(date(2026, 11, 24), backlog[:1] + backlog[3:], events, limit=3)
    assert [(p["title"], p["reason"]) for p in picked] == [
        ("Study for Midterm 2", "Exam Thursday"), ("Study for chem exam", "Due Monday"),
        ("Fix Friday: Dentist overlaps Test Prep with Alex", "Conflict to fix")]

    picked = rp.suggest(date(2026, 11, 24), backlog[:1], [])
    assert [(p["title"], p["reason"]) for p in picked] == [("Oldest thing", "Oldest in backlog")]


def test_tutoring_is_not_an_exam_and_a_task_is_not_suggested_twice():
    import calendar_roles
    events = calendar_roles.annotate([
        _event("Jordan <> Lethanial Leveille | Test Prep | 1-1", 26, 9),
        _event("DSA Final Exam", 30, 9)])
    backlog = [{"id": 5, "text": "Study for DSA Final Exam", "due_date": "2026-11-26", "minutes": None}]
    picked = rp.suggest(date(2026, 11, 24), backlog, events)
    # In the due soon group, so not again as an exam, and the event's own
    # study item is covered by the backlog item.
    assert [p["title"] for p in picked] == ["Study for DSA Final Exam"]


def test_nothing_already_planned_for_tomorrow_is_suggested():
    backlog = [{"id": 1, "text": "Email professor", "due_date": None, "minutes": None},
               {"id": 2, "text": "Laundry", "due_date": None, "minutes": None}]
    picked = rp.suggest(date(2026, 11, 24), backlog, [], planned_titles=["email professor"])
    assert [p["title"] for p in picked] == ["Laundry"]


def test_the_proposal_counts_what_is_already_planned_before_suggesting(db, world):
    _task(db, "Gym", TOMORROW, must_do=True)
    _task(db, "Lab", TOMORROW, must_do=True)
    for name in ("One", "Two", "Three"):
        _backlog(db, name)
    plan = rp.build_proposal(TODAY, NOW)
    assert [m["title"] for m in plan["must_dos"]] == ["Gym", "Lab", "One"]


# ── the backlog ──

def test_the_backlog_is_not_todays_list(db):
    daily = db.add_task("Vitamins", "daily")
    backlog = db.add_task("Email professor")
    today = db.add_task("Call the bank", scheduled_for=TODAY)
    assert [t[0] for t in db.tasks_today(NOW)] == [daily, today]
    assert [t["id"] for t in db.tasks_for_day(TODAY, today=TODAY)] == [daily, today]
    assert [t["id"] for t in db.backlog_tasks()] == [backlog]


def test_voice_task_add_lands_on_todays_list(db):
    lists_tasks.task_add_tool("call the bank")
    lists_tasks.task_add_tool("take vitamins", "daily")
    [(task_id, _, _), (_, _, recurrence)] = db.tasks_today()
    assert db.get_task(task_id)["scheduled_for"] == db.task_day()
    assert recurrence == "daily"
    assert db.backlog_tasks() == []


def test_adding_to_the_backlog_keeps_the_due_date(db, world):
    said = nr.add_to_backlog("order a NVMe HAT", "2026-12-01", now=NOW)
    [task] = db.backlog_tasks()
    assert (task["text"], task["due_date"], task["scheduled_for"]) == (
        "Order a NVMe HAT", "2026-12-01", None)
    assert "backlog" in said
    with pytest.raises(ToolNotice, match="not a date"):
        nr.add_to_backlog("x", "soon", now=NOW)


def test_a_moved_task_keeps_its_due_date_and_minutes(db):
    task_id = _task(db, "Essay", TODAY, due_date="2026-12-01", minutes=90)
    new_id = db.move_task(task_id, "2026-11-27")
    copy = db.get_task(new_id)
    assert (copy["due_date"], copy["minutes"]) == ("2026-12-01", 90)


def test_push_to_backlog_removes_the_block_the_day_and_the_plan_item(db, world):
    fixed_night(db, world)
    session = _session()
    plan = db.session_plan(session["id"])
    study_id = plan["must_dos"][0]["task_id"]
    nr_token = _in_turn(session["id"])
    try:
        db.save_day_plan(TOMORROW, [{"task_id": study_id, "text": "x", "must_do": True}], NOW)
        nr.schedule_block(study_id, 120, start_time="10:00", now=NOW)
        block_id = db.get_task(study_id)["calendar_event_id"]
        assert block_id in world.events[MILES]

        said = nr.push_to_backlog(task_id=study_id, now=NOW)

        task = db.get_task(study_id)
        assert (task["scheduled_for"], task["must_do"], task["calendar_event_id"]) == (None, 0, None)
        assert block_id not in world.events[MILES]
        assert study_id in [t["id"] for t in db.backlog_tasks()]
        new_plan = db.session_plan(session["id"])
        assert "Study for linear algebra test" not in [m["title"] for m in new_plan["must_dos"]]
        assert "backlog" in said
        state = nr_token[1]
        assert state.posted[-1].endswith("Good, or change something?")
    finally:
        session_types.end_turn(nr_token[0])


def test_push_by_title_makes_a_backlog_task_for_a_proposed_item(db, world):
    fixed_night(db, world)
    session = _session()
    token, state = _in_turn(session["id"])
    try:
        nr.push_to_backlog(title="Fix Friday", now=NOW)
    finally:
        session_types.end_turn(token)
    assert "Fix Friday: Test Prep with Alex overlaps Dentist" in [t["text"] for t in db.backlog_tasks()]
    assert "Fix Friday" not in state.posted[-1]


# ── the flow ──

def test_a_change_posts_the_plan_in_the_openings_format(db, world):
    fixed_night(db, world)
    session = _session()
    token, state = _in_turn(session["id"])
    try:
        nr.set_plan(must_dos=[{"title": "Email professor"},
                              {"title": "Study for linear algebra test", "start": "15:00"},
                              "Laundry"],
                    optional=["Order NVMe HAT and SSD"], now=NOW)
    finally:
        session_types.end_turn(token)
    [posted] = state.posted
    assert posted == "\n".join([
        "Today: 3 of 4 done, 1 moved to Friday.",
        "",
        "Tomorrow, Wednesday Nov 25",
        "• 10:40 to 11:30 AM  DSA (likely cancelled, Thanksgiving break)",
        "• 12:30 to 2 PM  Test Prep with Jordan",
        "",
        "Suggested must dos",
        "1. Email professor, 10 to 11 AM",
        "   Oldest in backlog",
        "2. Study for linear algebra test, 3 to 5 PM",
        "   Due Monday",
        "3. Laundry, 11 AM to 12 PM",
        "   Your choice",
        "",
        "Optional",
        "• Order NVMe HAT and SSD",
        "",
        "Good, or change something?"])
    saved = db.session_plan(session["id"])
    assert [m["title"] for m in saved["must_dos"]] == [
        "Email professor", "Study for linear algebra test", "Laundry"]
    assert saved["must_dos"][0]["task_id"] is not None      # code kept the id


def test_set_plan_refuses_a_fourth_must_do_and_changes_nothing(db, world):
    fixed_night(db, world)
    session = _session()
    before = db.session_plan(session["id"])
    token, state = _in_turn(session["id"])
    try:
        with pytest.raises(ToolNotice, match="At most 3 must dos"):
            nr.set_plan(must_dos=["A", "B", "C", "D"], now=NOW)
    finally:
        session_types.end_turn(token)
    assert db.session_plan(session["id"]) == before and state.posted == []


def test_set_plan_reports_an_item_that_does_not_fit(db, world):
    session = _session()
    world.add(PRIMARY, "all", "Busy", at(25, 9), at(25, 21))
    token, state = _in_turn(session["id"])
    try:
        said = nr.set_plan(must_dos=["Gym"], now=NOW)
    finally:
        session_types.end_turn(token)
    assert "No time found for: Gym" in said
    assert "1. Gym, no time found" in state.posted[0]


def test_good_books_everything_in_order_and_closes(db, world, fake_claude):
    fixed_night(db, world)
    session = _session()
    fake_claude([text_block("Booked. Good night."), tool_block("confirm_plan", {}, "toolu_1")],
                summary="He planned tomorrow.")
    events = []
    reply = sessions.run_turn(session["id"], "good",
                              on_text=lambda k, t: events.append((k, t)), now=NOW)

    planned = [t for t in db.tasks_between(TOMORROW, TOMORROW) if t["must_do"]]
    assert sorted(t["text"] for t in planned) == sorted([
        "Study for linear algebra test", "Fix Friday: Test Prep with Alex overlaps Dentist",
        "Email professor"])
    assert all(t["calendar_event_id"] for t in planned)
    spans = [(b["start"]["dateTime"][11:16], b["end"]["dateTime"][11:16])
             for c, b in world.inserts if c == MILES]
    assert spans == [("10:00", "12:00"), ("12:00", "12:30"), ("14:00", "15:00")]
    assert [t["text"] for t in db.tasks_between(TOMORROW, TOMORROW) if not t["must_do"]] == [
        "Order NVMe HAT and SSD"]
    assert db.backlog_tasks() == []                       # they left the backlog

    closed = db.get_session(session["id"])
    assert (closed["status"], closed["closed_by"]) == ("closed", "user")
    final = [t for k, t in events if k == "delta"][-1]
    assert final.startswith("\n\nTomorrow's plan, Wednesday Nov 25\n"
                            "1. Study for linear algebra test, 10 AM to 12 PM\n"
                            "2. Fix Friday: Test Prep with Alex overlaps Dentist, 12 to 12:30 PM\n"
                            "3. Email professor, 2 to 3 PM")
    assert [k for k, _ in events][-2:] == ["delta", "closed"] or events[-1][0] == "closed"
    # The model's "Booked. Good night." is held to its first sentence.
    assert reply.startswith("Booked.\n\nTomorrow's plan")


def test_confirming_places_again_what_the_calendar_took_since_21_30(db, world, fake_claude):
    fixed_night(db, world)
    session = _session()
    # At 21:45 a new class lands on the first slot the proposal showed.
    world.add(PRIMARY, "new", "Review session", at(25, 9, 30), at(25, 12))
    fake_claude([text_block("Good night."), tool_block("confirm_plan", {}, "toolu_1")])
    sessions.run_turn(session["id"], "good", now=NOW)
    booked = {b["summary"]: (b["start"]["dateTime"][11:16], b["end"]["dateTime"][11:16])
              for c, b in world.inserts if c == MILES}
    # The proposal had Study at 10, which is taken now, so it is placed again.
    # The fix at 12 and Email at 2 are still free, so they keep their times.
    assert booked["Email professor"] == ("14:00", "15:00")
    assert booked["Study for linear algebra test"] == ("15:00", "17:00")
    assert booked["Fix Friday: Test Prep with Alex overlaps Dentist"] == ("12:00", "12:30")
    assert all(start >= "12:00" for start, _ in booked.values())


def test_a_deleted_block_reads_not_booked_in_the_posted_final_plan(db, world, fake_claude,
                                                                   monkeypatch):
    fixed_night(db, world)
    session = _session()
    fake_claude([text_block("Good night."), tool_block("confirm_plan", {}, "toolu_1")])
    original = calendar_tools.add_miles_event

    def book_then_he_deletes_it(summary, start, end):
        calendar_id, event_id = original(summary, start, end)
        if summary.startswith("Fix Friday"):
            world.remove(MILES, event_id)
        return calendar_id, event_id
    monkeypatch.setattr(calendar_tools, "add_miles_event", book_then_he_deletes_it)
    reply = sessions.run_turn(session["id"], "good", now=NOW)
    assert "2. Fix Friday: Test Prep with Alex overlaps Dentist, not booked" in reply


def test_confirming_without_a_calendar_books_nothing(db, monkeypatch, fake_claude):
    monkeypatch.setattr(nr, "_clock", lambda: NOW)
    monkeypatch.setattr(sessions, "effective_tier", lambda: "hokage")
    session = _session()                                  # the calendar is down
    _backlog(db, "Email professor")
    fake_claude([text_block("One moment."), tool_block("confirm_plan", {}, "toolu_1")],
                [text_block("The calendar is down; try again soon.")])
    sessions.run_turn(session["id"], "good", now=NOW)
    assert db.get_session(session["id"])["status"] == "open"
    assert db.tasks_between(TOMORROW, TOMORROW) == []


def test_mark_attending_makes_a_series_count(db, world):
    world.add(NSBE, "d1", "Design Team Meeting (Members Only)", at(30, 17), at(30, 18),
              recurringEventId="design")
    said = nr.mark_attending("design team", "2026-11-30", now=NOW)
    assert "every repeat" in said
    assert db.attending_set() == {(NSBE, "design")}
    assert nr.mark_attending("design team", "2026-11-30", now=NOW).endswith("already marked as attended.")
    with pytest.raises(ToolNotice, match="No club event"):
        nr.mark_attending("poker", "2026-11-30", now=NOW)


def test_mark_attending_one_occurrence_only(db, world):
    world.add(ACM, "fish1", "Friday Fishbowl", at(20, 13), at(20, 14), recurringEventId="fish")
    nr.mark_attending("fishbowl", "2026-11-20", whole_series=False, now=NOW)
    assert db.attending_set() == {(ACM, "fish1")}


def test_upcoming_items_include_conflicts_and_notes(db, world):
    fixed_night(db, world)
    result = nr.get_upcoming_items(now=NOW)
    assert result["conflicts"] == [{"date": "2026-11-27",
                                    "between": ["Test Prep | Lethanial <> Alex", "Dentist"],
                                    "overlap": "4:30 to 5 PM"}]
    dsa = [e for e in result["events"] if e["title"] == "DSA"][0]
    assert dsa["note"] == "(likely cancelled, Thanksgiving break)" and dsa["busy"] is False


def test_the_prompt_has_no_hyphens_and_names_every_tool(db):
    prompt = nr.system_prompt({"for_day": TODAY, "id": 1})
    prose = re.sub(r"\d{4}-\d{2}-\d{2}|^- ", "", prompt.replace("YYYY-MM-DD", ""), flags=re.M)
    assert "-" not in prose
    for name in nr.registry.names():
        if name not in ("schedule_block", "close_session", "get_upcoming_items"):
            assert name in prompt


def test_the_stored_plan_is_json(db, world):
    fixed_night(db, world)
    session = _session()
    json.dumps(db.session_plan(session["id"]))
