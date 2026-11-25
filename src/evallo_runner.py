"""Mark an Evallo tutoring session Completed, No Show or Canceled, by driving
the Evallo website the way Lethanial does by hand. Evallo has no API.

    set_session_status("Taylor", datetime(2026, 11, 24, 16, 15), "Completed",
                       dry_run=True)

The steps, each named so a failure says where it broke:

    check_student     the students list (/users) has this first name; if not,
                      the student is not mapped to him yet and nothing is logged
    open_calendar     the calendar loads, logged in, in the week view
    go_to_week        the small month calendar moves to the session's week
    read_week         every card in the week is read; struck through
                      (canceled) cards are left out before matching
    open_event        exactly one card matched; it is clicked
    verify_panel      the panel's title, date, start time and student all agree
                      with the request, before anything is touched
    open_status       the status menu opens; what it offers is read
    pick_status       the requested status is clicked
    confirm_popup     Evallo asks "Are you sure...?"
    go_back           dry run: "No, go back"
    verify_unchanged  dry run: page reloaded, the status is still what it was
    confirm           real run: "Yes, confirm"
    verify_changed    real run: the toast shows and the status reads the new one

A dry run walks every step a real one does and clicks "No, go back" at the
end. A real run is only ever started by Lethanial: from the command line with
--real, or by Nova after he says yes (evallo_tools.py).

Two things are reported rather than treated as breakage: the student not being
on his list, and no session at that time. Both go on the unlogged list
(evallo_unlogged.py) on a real run, and both send him a push.
"""

import fcntl
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import evallo_failures
import evallo_matching as M
import evallo_session
import evallo_unlogged
from evallo_selectors import KNOWN_STATUSES, describe, locate

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data" / "evallo.lock"
STEP_TIMEOUT_MS = 20_000
# Short on purpose: if Evallo ever stops asking, the script must notice fast.
POPUP_TIMEOUT_MS = 8_000
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

# JavaScript run in the page to read every card of the week at once: its day
# column's date, its place in that column, its text, and whether it is struck
# through. Reading styles is the only way to see "canceled": Evallo gives the
# card no word or label for it.
READ_WEEK_JS = """(cols) => [...document.querySelectorAll(cols.column)].flatMap(col =>
  [...col.querySelectorAll(cols.event)].map((ev, index) => {
    const struck = [...ev.querySelectorAll('*')].some(el =>
      el.childElementCount === 0 && el.textContent.includes('<>') &&
      getComputedStyle(el).textDecorationLine.includes('line-through'));
    return {day: col.dataset.date, index, text: ev.innerText, canceled: struck};
  }))"""


# True once every card has text and the cards have read the same on five
# checks in a row, a quarter second apart. An empty week settles too.
SETTLED_JS = """(sel) => {
  const cards = [...document.querySelectorAll(sel)];
  const w = window.__evalloSettle || (window.__evalloSettle = {sig: null, same: 0});
  // Blocked time (the hatched boxes) is a card with no text, ever (Nov 23
  // 2026), so an empty card can't mean "still loading". Settled = the same
  // cards, empty ones included, for five polls in a row. parse_card drops
  // the empty ones afterwards because they have no time.
  const sig = cards.length + ':' + cards.map(c => c.innerText.trim()).join('|');
  if (sig === w.sig) { w.same += 1; } else { w.sig = sig; w.same = 0; }
  return w.same >= 4;
}"""


@dataclass
class Result:
    outcome: str      # see OUTCOMES
    message: str      # one or two plain sentences, fit to be read aloud
    step: str | None = None
    failure_dir: str | None = None
    details: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.outcome in GOOD

    def to_dict(self):
        return {**asdict(self), "ok": self.ok}


OUTCOMES = {
    "done":          "real run: status changed and confirmed by Evallo",
    "dry_run_ok":    "dry run: walked to the popup, went back, nothing changed",
    "check_ok":      "canary: the menu opened, closed, nothing changed",
    "already_done":  "it already had the requested status",
    "not_mapped":    "the student is not on his Evallo list",
    "no_session":    "the student is mapped but no session at that time",
    "ambiguous":     "more than one live session matched",
    "other_status":  "it already has a different status; left alone",
    "not_offered":   "Evallo does not offer that status yet (before the session)",
    "login_expired": "the saved login ran out",
    "failed":        "a step broke; a failure record was saved",
    "invalid":       "the request itself was wrong",
}
GOOD = {"done", "dry_run_ok", "check_ok", "already_done"}


class _Stop(Exception):
    """Ends a run early with a finished Result."""
    def __init__(self, result):
        self.result = result


class _Run:
    """One browser session. Remembers the current step and the selector last
    used, so whatever breaks, the record says where and with what."""

    def __init__(self, page, request):
        self.page = page
        self.request = request
        self.step_name = None
        self.key = None
        self.fill = {}

    @contextmanager
    def step(self, name):
        self.step_name, self.key, self.fill = name, None, {}
        yield

    def loc(self, key, **fill):
        self.key, self.fill = key, fill
        return locate(self.page, key, **fill)

    # ── Steps shared by every run ────────────────────────────────────────

    def ensure_logged_in(self, ready_key):
        """Wait for the page's own landmark, or the sign in page."""
        ready = self.loc(ready_key)
        signin = locate(self.page, "signin_marker")
        ready.or_(signin).first.wait_for()
        if signin.count() and signin.first.is_visible():
            raise evallo_session.LoginExpired("Evallo showed its sign in page.")

    def student_names(self):
        with self.step("check_student"):
            self.page.goto(describe("students_url")["url"])
            self.ensure_logged_in("students_count")
            # The count reads "Showing 0 - 0 of 0" until the list arrives, and
            # the first version believed it (Nov 23 2026). Wait for the rows,
            # then for as many rows as the count says.
            self.loc("student_link").first.wait_for()
            shown = self.loc("students_count").first.inner_text()
            a, b, n = map(int, re.findall(r"\d+", shown)[-3:])
            if b < n:
                raise RuntimeError(f"The students list has more than one page ({shown}); "
                                   "reading further pages is not built yet.")
            self.page.wait_for_function(
                "([sel, n]) => document.querySelectorAll(sel).length >= n",
                arg=[describe("student_link")["css"], b - a + 1])
            return [x.strip() for x in self.loc("student_link").all_inner_texts()]

    def open_calendar(self):
        with self.step("open_calendar"):
            self.page.goto(describe("calendar_url")["url"])
            self.ensure_logged_in("weekly_view")

    def go_to_week(self, day):
        """Move the small month calendar to day's month, click the day, and
        prove the week grid now shows it. The mini calendar opens on the
        current month, so the number of clicks is known in advance."""
        with self.step("go_to_week"):
            today = datetime.now(M.EVALLO_TZ)
            months = (day.year - today.year) * 12 + (day.month - today.month)
            arrow = "mini_next_month" if months > 0 else "mini_prev_month"
            for _ in range(abs(months)):
                self.loc(arrow).click()
            month = self.loc("mini_month", month=MONTHS[day.month - 1], year=day.year)
            month.wait_for()
            cells = month.get_by_role("gridcell", name=str(day.day), exact=True)
            self.key, self.fill = "mini_day", {"day": day.day}
            # The grid shows a few days of the months either side. Early days of
            # the month repeat only at the end (next month), late days only at
            # the start (last month), so first or last is always the right one.
            (cells.first if day.day < 15 else cells.last).click()
            self.loc("week_day_header", month=MONTHS[day.month - 1],
                     day=day.day, year=day.year).wait_for()

    def read_week(self):
        with self.step("read_week"):
            column = describe("day_column")["css"]
            event = describe("event")["css"]
            self.key = "event"
            # Cards draw a moment after the data arrives, blank at first, and
            # the page never goes network quiet on the Pi (analytics and
            # Stripe keep connections open), so waiting for "networkidle"
            # timed out there (Nov 23 2026). Wait instead until every card
            # has text and the set has not changed for a second.
            self.page.evaluate("() => { window.__evalloSettle = null; }")
            self.page.wait_for_function(SETTLED_JS, arg=event, polling=250)
            cards = self.page.evaluate(READ_WEEK_JS, {"column": column, "event": event})
            events = [M.parse_card(c["day"], c["text"], c["canceled"], c["index"]) for c in cards]
            return [e for e in events if e is not None]

    def open_event(self, event):
        with self.step("open_event"):
            column = describe("day_column")["css"].replace(
                "[data-date]", f'[data-date="{event.start.date().isoformat()}"]')
            self.loc("event")
            self.page.locator(column).locator(describe("event")["css"]).nth(event.index).click()
            self.loc("panel_title").wait_for()

    def verify_panel(self, first_name, start):
        """Everything must agree before anything is clicked."""
        with self.step("verify_panel"):
            title = self.loc("panel_title").input_value()
            values = [b.input_value() for b in self.loc("panel_textboxes").all()]
            start_box = self.loc("panel_times").first.input_value()
            label = self.loc("panel_student_label").first
            student = label.evaluate("e => e.parentElement.innerText")
            student = re.sub(r"^\s*Student\*?\s*", "", student).strip()
            problems = []
            if not M.title_matches(title, first_name):
                problems.append(f"title is {title!r}")
            if M.panel_date(start) not in values:
                problems.append(f"no start date {M.panel_date(start)!r} in {values!r}")
            if _clock(start_box) != (start.hour, start.minute):
                problems.append(f"start time is {start_box!r}")
            if not M.starts_with_name(student, first_name):
                problems.append(f"student is {student!r}")
            if problems:
                raise RuntimeError("The open panel does not match the request: "
                                   + "; ".join(problems) + ".")
            return {"title": title, "student": student}

    def current_status(self):
        with self.step("verify_panel"):
            for status in KNOWN_STATUSES:
                if self.loc("status_current", status=status).filter(visible=True).count():
                    return status
            raise RuntimeError("No known status is showing on the panel.")

    def find(self, first_name, start):
        """Calendar, week, match, open, verify. Returns (event, status)."""
        self.open_calendar()
        self.go_to_week(start.date())
        events = self.read_week()
        with self.step("read_week"):
            event = M.find_event(events, first_name, start)
        self.open_event(event)
        self.verify_panel(first_name, start)
        return event, self.current_status()


def _clock(text):
    t = datetime.strptime(re.sub(r"\s+", " ", text.strip().upper()), "%I:%M %p")
    return t.hour, t.minute


def _when(start):
    return f"{start:%A %B} {start.day} at {start:%-I:%M %p}"


def set_session_status(student_first_name, start_datetime, status, dry_run=True,
                       minutes=None, check_only=False, headless=True, slow_mo=0):
    """Set one session's status in Evallo. Returns a Result; never raises.

    dry_run       walk every step, then "No, go back" and prove nothing changed
    check_only    the nightly canary: open the status menu, read it, close it
                  without clicking an option. See scripts/evallo_canary.py.
    minutes       the lesson's length, kept on the unlogged list if it comes to that
    """
    request = {"student": student_first_name, "start": str(start_datetime),
               "status": status, "dry_run": dry_run, "check_only": check_only}
    try:
        status = M.check_status(status)
    except ValueError as exc:
        return Result("invalid", str(exc))
    if not student_first_name.strip():
        return Result("invalid", "No student name was given.")
    start = M.to_eastern(start_datetime)
    evallo_failures.prune()

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lock:
        # One run at a time: the canary and a request must not share a page.
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = _run(student_first_name.strip(), start, status, dry_run, minutes,
                      check_only, headless, slow_mo, request)
    if result.outcome == "login_expired":
        # Once a day, with the command that fixes it. Nothing retries the login.
        evallo_session.login_expired(f"marking {student_first_name.strip()}'s session "
                                     f"{status}" + (" (dry run)" if dry_run else ""))
    elif result.outcome in ("not_mapped", "no_session", "ambiguous", "failed") and not (
            dry_run and result.outcome in ("not_mapped", "no_session")):
        evallo_session.notify(result.message)
    return result


def _run(name, start, status, dry_run, minutes, check_only, headless, slow_mo, request):
    from playwright.sync_api import sync_playwright

    try:
        state = evallo_session.load_state()
    except evallo_session.LoginExpired:
        return Result("login_expired", "Evallo login expired, log in again")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = browser.new_context(
            storage_state={"cookies": state.get("cookies", []),
                           "origins": state.get("origins", [])},
            timezone_id="America/New_York", viewport={"width": 1400, "height": 900})
        context.set_default_timeout(STEP_TIMEOUT_MS)
        run = _Run(context.new_page(), request)
        try:
            return _steps(run, name, start, status, dry_run, minutes, check_only)
        except _Stop as stop:
            return stop.result
        except evallo_session.LoginExpired:
            return Result("login_expired", "Evallo login expired, log in again",
                          step=run.step_name)
        except Exception as exc:
            folder = evallo_failures.record(
                run.step_name or "start", exc, page=run.page, selector_key=run.key,
                selector=describe(run.key, **run.fill) if run.key else None,
                request=request)
            return Result("failed", f"Evallo script broke at step {run.step_name}.",
                          step=run.step_name, failure_dir=str(folder),
                          details={"error": f"{type(exc).__name__}: {exc}"})
        finally:
            browser.close()


def _steps(run, name, start, status, dry_run, minutes, check_only):
    when = _when(start)
    real = not dry_run and not check_only

    names = run.student_names()
    if not M.student_on_list(names, name):
        if real:
            evallo_unlogged.add(name, start, minutes, evallo_unlogged.NOT_MAPPED)
        return Result("not_mapped",
                      f"{name} isn't mapped to you in Evallo yet. Session not logged.")

    try:
        event, current = run.find(name, start)
    except M.MatchError as exc:
        if str(exc).startswith("No session"):
            if real:
                evallo_unlogged.add(name, start, minutes, evallo_unlogged.NO_SESSION)
            return Result("no_session", f"No Evallo session for {name} at {when}. "
                          "Book it or check the time.", step="read_week",
                          details={"seen": str(exc)})
        return Result("ambiguous", str(exc), step="read_week")

    details = {"title": event.title, "was": current}
    if current == status and not check_only:
        return Result("already_done", f"{name}'s session {when} is already {status}.",
                      details=details)
    if current != "Scheduled":
        return Result("other_status", f"{name}'s session {when} is marked {current} "
                      f"already, so I left it alone.", details=details)

    with run.step("open_status"):
        run.loc("status_current", status=current).filter(visible=True).first.click()
        run.loc("status_menu_open").filter(visible=True).first.wait_for()
        offered = [s for s in KNOWN_STATUSES
                   if locate(run.page, "status_option", status=s).filter(visible=True).count()]
        details["offered"] = offered

    if check_only:
        # No Escape: in Evallo it closes the whole panel, not just the menu.
        # verify_unchanged loads the calendar from scratch, which drops the
        # open menu without choosing anything.
        verify_unchanged(run, name, start, current)
        return Result("check_ok", f"Checked {name}'s session {when}: the status menu "
                      f"opens and offers {', '.join(offered)}. Nothing changed.",
                      details=details)

    if status not in offered:
        # Leave the menu open; nothing was chosen, and the browser closes next.
        if start > datetime.now(M.EVALLO_TZ):
            return Result("not_offered", f"Evallo doesn't offer {status} for {name}'s "
                          f"session {when} yet; it opens once the session has started.",
                          details=details)
        with run.step("pick_status"):
            run.loc("status_option", status=status)
            raise RuntimeError(f"The status menu offers {offered}, not {status!r}.")

    with run.step("pick_status"):
        run.loc("status_option", status=status).filter(visible=True).first.click()

    with run.step("confirm_popup"):
        try:
            run.loc("confirm_heading").first.wait_for(timeout=POPUP_TIMEOUT_MS)
            run.loc("confirm_question", status=status).first.wait_for(timeout=POPUP_TIMEOUT_MS)
        except Exception:
            # The one dangerous failure: an option was clicked and the expected
            # popup did not appear. Either it is worded differently now (the
            # page is left without confirming, which changes nothing) or Evallo
            # stopped asking. Find out at once whether anything changed.
            now = _status_after_reload(run, name, start)
            raise RuntimeError(
                f"The \"Update changes\" popup asking about {status} did not appear "
                f"as expected after clicking {status}. The status now reads "
                f"{now!r}" + (" — IT MAY HAVE CHANGED WITHOUT ASKING." if now != current else
                              "; nothing changed."))

    if dry_run:
        with run.step("go_back"):
            run.loc("confirm_no").click()
            run.loc("confirm_question", status=status).first.wait_for(state="hidden")
        verify_unchanged(run, name, start, current)
        return Result("dry_run_ok", f"Dry run for {name}'s session {when} worked: Evallo "
                      f"asked to confirm {status}, I went back, and it is still {current}.",
                      details=details)

    with run.step("confirm"):
        run.loc("confirm_yes").click()
    with run.step("verify_changed"):
        run.loc("success_toast").first.wait_for()
    now = _status_after_reload(run, name, start)
    if now != status:
        with run.step("verify_changed"):
            raise RuntimeError(f"Evallo said it saved, but the status reads {now!r}.")
    evallo_unlogged.remove(name, start)
    return Result("done", f"Marked {name}'s session {when} {status} in Evallo.",
                  details=details)


def _status_after_reload(run, name, start):
    """Load the calendar from scratch and read the session's status again.

    find() opens the calendar with a fresh page load, which is the reload.
    An explicit page.reload() just before it was a second load on top of the
    first, and Evallo answered the pair with its sign in page (Nov 23 2026)."""
    _, status = run.find(name, start)
    return status


def verify_unchanged(run, name, start, expected):
    now = _status_after_reload(run, name, start)
    with run.step("verify_unchanged"):
        if now != expected:
            raise RuntimeError(f"The status changed from {expected!r} to {now!r} "
                               "on a run that was not meant to change anything.")


# ── The canary's helper: which session is next ───────────────────────────

def next_session(now=None, headless=True):
    """The next live session of his after now, looking this week and next.
    Returns (first_name, CalendarEvent), None when there is none, or a failed
    Result."""
    from playwright.sync_api import sync_playwright
    now = now or datetime.now(M.EVALLO_TZ)
    try:
        state = evallo_session.load_state()
    except evallo_session.LoginExpired:
        return Result("login_expired", "Evallo login expired, log in again")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            storage_state={"cookies": state.get("cookies", []),
                           "origins": state.get("origins", [])},
            timezone_id="America/New_York", viewport={"width": 1400, "height": 900})
        context.set_default_timeout(STEP_TIMEOUT_MS)
        run = _Run(context.new_page(), {"canary": "next_session"})
        try:
            run.open_calendar()
            for week in (0, 1):
                day = (now + timedelta(days=7 * week)).date()
                run.go_to_week(day)
                ahead = sorted((e for e in run.read_week()
                                if not e.canceled and e.start > now
                                and M.TUTOR_MARKER in M.normalize(e.title)),
                               key=lambda e: e.start)
                if ahead:
                    first = re.split(r"\s*<>", ahead[0].title)[0].split()[0]
                    return first, ahead[0]
            return None
        except evallo_session.LoginExpired:
            return Result("login_expired", "Evallo login expired, log in again")
        except Exception as exc:
            folder = evallo_failures.record(
                run.step_name or "start", exc, page=run.page, selector_key=run.key,
                selector=describe(run.key, **run.fill) if run.key else None,
                request=run.request)
            return Result("failed", f"Evallo script broke at step {run.step_name}.",
                          step=run.step_name, failure_dir=str(folder),
                          details={"error": f"{type(exc).__name__}: {exc}"})
        finally:
            browser.close()
