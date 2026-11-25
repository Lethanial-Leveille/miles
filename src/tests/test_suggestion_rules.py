"""The nightly review's suggestion rules (owner's rule, Nov 25): the real due
date in every label, soonest due first, exam prep before projects before the
rest on one date, and a far off item never fills a slot only for being old."""

from datetime import date

import review_plan as rp

TODAY = date(2026, 11, 25)


def _t(i, text, due=None):
    return {"id": i, "text": text, "due_date": due, "minutes": None}


LIVE = [
    _t(1, "Final prep", "2026-12-07"),
    _t(2, "Edugator practice", "2026-12-03"),
    _t(3, "Project 3", "2026-11-30"),
    _t(4, "Exam 3 prep", "2026-11-30"),
]


def _pairs(picked):
    return [(p["title"], p["reason"]) for p in picked]


def test_the_live_backlog_example():
    picked = rp.suggest(TODAY, LIVE, [])
    assert _pairs(picked) == [("Exam 3 prep", "Due Mon Nov 30"),
                              ("Project 3", "Due Mon Nov 30"),
                              ("Edugator practice", "Due Thu Dec 3")]


def test_final_prep_is_held_back_with_a_higher_limit_too():
    picked = rp.suggest(TODAY, LIVE, [], limit=10)
    assert "Final prep" not in [p["title"] for p in picked]


def test_a_far_item_is_suggested_when_nothing_is_due_sooner():
    picked = rp.suggest(TODAY, [_t(1, "Final prep", "2026-12-07")], [])
    assert _pairs(picked) == [("Final prep", "Due Mon Dec 7")]


def test_the_seven_day_edge():
    # counted from the day being planned (tomorrow, Nov 26): Dec 3 is 7 days, Dec 4 is 8
    backlog = [_t(1, "Soon thing", "2026-11-27"), _t(2, "Exactly seven", "2026-12-03"),
               _t(3, "Eight days", "2026-12-04")]
    titles = [p["title"] for p in rp.suggest(TODAY, backlog, [], limit=10)]
    assert titles == ["Soon thing", "Exactly seven"]


def test_undated_items_still_fill_after_dated_ones_oldest_first():
    backlog = [_t(1, "Old undated"), _t(2, "Dated", "2026-12-04"), _t(3, "Newer undated")]
    assert _pairs(rp.suggest(TODAY, backlog, [], limit=10)) == [
        ("Dated", "Due Fri Dec 4"), ("Old undated", "Oldest in backlog"),
        ("Newer undated", "Oldest in backlog")]


def test_a_far_item_is_held_back_even_when_an_undated_one_is_older():
    backlog = [_t(1, "Old undated"), _t(2, "Near", "2026-12-01"), _t(3, "Far", "2026-12-20")]
    assert [p["title"] for p in rp.suggest(TODAY, backlog, [], limit=10)] == [
        "Near", "Old undated"]


def test_same_date_order_exam_then_project_then_other():
    backlog = [_t(1, "Other thing", "2026-11-30"), _t(2, "Project 3", "2026-11-30"),
               _t(3, "Midterm review", "2026-11-30")]
    assert [p["title"] for p in rp.suggest(TODAY, backlog, [], limit=10)] == [
        "Midterm review", "Project 3", "Other thing"]


def test_labels_carry_the_real_date():
    backlog = [_t(1, "Late", "2026-11-22"), _t(2, "Today", "2026-11-25"),
               _t(3, "Tomorrow", "2026-11-26"), _t(4, "Friday", "2026-11-27")]
    assert _pairs(rp.suggest(TODAY, backlog, [], limit=10)) == [
        ("Late", "Overdue, due Sun Nov 22"), ("Today", "Due today, Wed Nov 25"),
        ("Tomorrow", "Due tomorrow, Thu Nov 26"), ("Friday", "Due Fri Nov 27")]


def test_project_and_exam_detection():
    assert rp.is_project_title("COP3530 Project 3: Campus Compass")
    assert rp.is_project_title("finish projects")
    assert not rp.is_project_title("Projector repair")
    assert rp.kind_rank("Project exam") == 0       # an exam title is never a project
    assert (rp.kind_rank("Exam 3 prep"), rp.kind_rank("Project 3"), rp.kind_rank("Lab")) == (0, 1, 2)
