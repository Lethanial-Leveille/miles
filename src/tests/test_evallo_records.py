"""The failure records and the unlogged list. Files only, no browser."""

import json
import os
import time
from datetime import datetime, timezone

import evallo_failures
import evallo_unlogged
from evallo_matching import EVALLO_TZ


class FakePage:
    url = "https://app.evallo.org/calendar"

    def screenshot(self, path, full_page):
        open(path, "wb").write(b"png")

    def content(self):
        return "<html>page</html>"


def test_failure_record_has_everything_a_repair_needs(tmp_path):
    folder = evallo_failures.record(
        "open_status", TimeoutError("Timeout 20000ms exceeded"), page=FakePage(),
        selector_key="status_current", selector={"kind": "text", "text": "Scheduled"},
        request={"student": "Taylor"}, base=tmp_path,
        now=datetime(2026, 11, 23, 7, 0, tzinfo=timezone.utc))
    assert folder.name == "20261123T070000Z_open_status"
    data = json.loads((folder / "failure.json").read_text())
    assert data["schema"] == 1
    assert data["step"] == "open_status"
    assert data["selector_key"] == "status_current"
    assert data["selector"] == {"kind": "text", "text": "Scheduled"}
    assert data["error_type"] == "TimeoutError"
    assert data["url"] == FakePage.url
    assert sorted(data["files"]) == ["page.html", "screenshot.png"]
    assert (folder / "page.html").read_text() == "<html>page</html>"


def test_failure_record_without_a_page_still_saves_the_json(tmp_path):
    folder = evallo_failures.record("start", RuntimeError("no browser"), base=tmp_path)
    assert json.loads((folder / "failure.json").read_text())["files"] == []


def test_records_older_than_30_days_are_deleted(tmp_path):
    old, new = tmp_path / "old_step", tmp_path / "new_step"
    old.mkdir(), new.mkdir()
    month_ago = time.time() - 31 * 86400
    os.utime(old, (month_ago, month_ago))
    assert evallo_failures.prune(base=tmp_path) == 1
    assert not old.exists() and new.exists()


def test_unlogged_list_keeps_one_row_per_session(tmp_path):
    path = tmp_path / "unlogged.json"
    start = datetime(2026, 11, 23, 16, 15, tzinfo=EVALLO_TZ)
    evallo_unlogged.add("Zelda", start, 90, evallo_unlogged.NOT_MAPPED, path=path)
    evallo_unlogged.add("zelda", start, 90, evallo_unlogged.NO_SESSION, path=path)
    rows = evallo_unlogged.load(path)
    assert len(rows) == 1
    assert rows[0] | {"recorded_at": None} == {
        "student": "zelda", "date": "2026-11-23", "time": "16:15",
        "length_minutes": 90, "reason": "no_session", "recorded_at": None}
    assert evallo_unlogged.remove("Zelda", start, path=path)
    assert evallo_unlogged.load(path) == []


def test_unlogged_list_starts_empty(tmp_path):
    assert evallo_unlogged.load(tmp_path / "missing.json") == []
