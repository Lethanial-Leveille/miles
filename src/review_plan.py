"""The nightly review's plan: how tomorrow is proposed, placed and shown.

The review proposes instead of interrogating. Code reads the database and the
calendar, picks at most three things worth doing tomorrow, finds times for them
inside his free gaps, and writes the whole message. The model only reads his
answer and calls tools. Every word about a time in a message he sees was
written here from a real calendar read, never by the model (the model's own
text is held to one line, see sessions.guard_text).

Pure where it can be: place_blocks, the suggestion order, the week view and
the plan text take plain data, so they are tested without Google or a model.
The functions that read the calendar or the database say so.
"""

import datetime
import re

import calendar_roles
import calendar_tools
from config import EXAM_EXCLUDE, EXAM_WORDS, SESSION_BLOCK_EARLIEST, SESSION_BLOCK_LATEST
from database import backlog_tasks, get_task, moved_destination, tasks_between, tasks_for_day

MUST_DO_CAP = 3
DEFAULT_MINUTES = 60
EXAM_MINUTES = 120
FIX_MINUTES = 30
DUE_SOON_DAYS = 3
EXAM_DAYS = 7
ASK = "Good, or change something?"


# ── words ──

def _day(value):
    return datetime.date.fromisoformat(value) if isinstance(value, str) else value


def day_words(day):
    """"Wednesday November 25"."""
    return _day(day).strftime("%A %B %-d")


def short_day(day):
    """"Wednesday Nov 25", the header of a day in the week view."""
    return _day(day).strftime("%A %b %-d")


def clock_words(moment):
    """"4 PM", "4:30 PM"."""
    return moment.strftime("%-I:%M %p").replace(":00", "")


def range_words(start, end):
    """"2 to 4 PM", "11 AM to 1 PM", "10:40 to 11:30 AM". The first time drops
    its AM or PM when both ends share it."""
    left, right = clock_words(start), clock_words(end)
    if start.strftime("%p") == end.strftime("%p") and start.date() == end.date():
        left = left.rsplit(" ", 1)[0]
    return f"{left} to {right}"


def _hhmm(moment):
    return moment.strftime("%H:%M")


def _count(n, one, many):
    return f"{n} {one if n == 1 else many}"


_TUTOR = re.compile(
    r"^(?:(?P<a>[^|<>]+?)\s*<>\s*Lethanial(?: Leveille)?\s*\|\s*(?P<kind>[^|]+?)\s*\|\s*[\d-]+"
    r"|(?P<kind2>[^|]+?)\s*\|\s*Lethanial\s*<>\s*(?P<b>.+?))\s*$")


def short_title(title):
    """"Jordan <> Lethanial Leveille | Test Prep | 1-1" reads "Test Prep with
    Jordan". Any other title is returned as it is."""
    match = _TUTOR.match(title.strip())
    if not match:
        return title.strip()
    if match["a"]:
        return f"{match['kind'].strip()} with {match['a'].strip()}"
    return f"{match['kind2'].strip()} with {match['b'].strip()}"


def is_exam_title(title):
    """Whether a title is an exam to study for: an exam word as a whole word,
    and not tutoring ("Test Prep" has the word test in it)."""
    low = title.casefold()
    if any(x in low for x in EXAM_EXCLUDE):
        return False
    words = set(re.findall(r"[a-z0-9]+", low))
    return any(w in words for w in EXAM_WORDS)


_FILLER = {"the", "a", "an", "to", "for", "of", "and", "my", "about", "with", "on", "in", "at"}
BLOCK_TITLE_WORDS = 4


def block_title(title, short_title=None):
    """The short name a MILES block carries, 2 to 4 words, so the calendar stays
    readable ("Linear algebra study", not the whole task). A task may carry its
    own (short_title), capped at forty characters. Otherwise it is derived the same way every time:
      "Study for linear algebra test"  becomes "Linear algebra study"
      "Fix Friday: Test Prep ... Dentist"  becomes "Fix Friday" (before the colon)
      anything else drops filler words and keeps the first four words.
    The full text, the reason and any notes go in the event description."""
    if short_title and short_title.strip():
        # A supplied name is capped at forty characters, cut at a word boundary.
        # No word limit: "Exam 3 prep: Lessons 19 to 21" is the name he wants.
        capped = " ".join(short_title.split())
        if len(capped) > 40:
            capped = capped[:41].rsplit(" ", 1)[0] if " " in capped[:41] else capped[:40]
        return capped
    text = title.strip()
    head = text.split(":", 1)[0].strip() if ":" in text else text
    study = re.match(r"(?i)^study for (?:the |a |an |my )?(.+?)(?: (?:exam|test|quiz|midterm|final)s?)?$",
                     head)
    if study and study.group(1).split():
        words = study.group(1).split()[:BLOCK_TITLE_WORDS - 1] + ["study"]
    else:
        words = [w for w in head.split() if w.casefold() not in _FILLER] or head.split()
        words = words[:BLOCK_TITLE_WORDS]
    if len(words) < 2:
        # Too little left (": notes", "Study for"): the task's own words, four
        # at most, so a block is never named by one stray word or nothing.
        fallback = text.strip(" :").split()[:BLOCK_TITLE_WORDS]
        if len(fallback) > len(words):
            words = fallback
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def block_description(task, reason=None):
    """What goes in the block's description: the full task text, why it is on
    the plan (a due date or an exam), his instructions (the task's notes), and
    for a study plan item its topic and source material folder."""
    from database import topic_info
    lines = [task["text"]]
    if reason and reason not in ("Your choice", "Already planned"):
        lines.append(f"Why: {reason}")
    elif task.get("due_date"):
        lines.append(f"Why: Due {short_day(task['due_date'])}")
    if task.get("notes"):
        lines.append(f"Notes: {task['notes']}")
    if task.get("source") == "study_plan" and task.get("topic_id"):
        info = topic_info(task["topic_id"])
        if info and info[0]:
            lines.append(f"Topic: {info[0]}")
        if info and info[1]:
            lines.append(f"Source material: {info[1]}")
    return "\n".join(lines)


def default_minutes(title, minutes=None):
    """60 minutes, 120 for studying for an exam, unless he said."""
    if minutes:
        return int(minutes)
    return EXAM_MINUTES if is_exam_title(title) else DEFAULT_MINUTES


# ── finding a slot (pure) ──

_STEP = datetime.timedelta(minutes=15)


def _round_up(moment):
    """Up to the next quarter hour, so a block picked at 9:07 starts at 9:15."""
    base = moment.replace(minute=0, second=0, microsecond=0)
    while base < moment:
        base += _STEP
    return base


def overlaps(busy, start, end):
    return any(a < end and start < b for a, b in busy)


def fitting_gaps(busy, window_start, window_end, minutes, not_before=None):
    """Free gaps inside the window at least `minutes` long, earliest first.

    not_before trims the window's start (to now, for a block today) and is
    rounded up to a quarter hour. A gap that starts mid quarter hour (a class
    ending at 11:50) starts its block on the next quarter."""
    start = window_start
    if not_before is not None and not_before > start:
        start = _round_up(not_before)
    if start >= window_end:
        return []
    need = datetime.timedelta(minutes=minutes)
    clipped = [(max(a, start), min(b, window_end)) for a, b in busy
               if a < window_end and b > start]
    gaps = calendar_tools.free_gaps(calendar_tools.merge_busy(clipped), start, window_end)
    rounded = [(_round_up(a), b) for a, b in gaps]
    return [(a, b) for a, b in rounded if b - a >= need]


def find_slot(busy, window_start, window_end, minutes, not_before=None):
    """(start, end) of the first free stretch that fits, or None."""
    gaps = fitting_gaps(busy, window_start, window_end, minutes, not_before)
    if not gaps:
        return None
    start = gaps[0][0]
    return start, start + datetime.timedelta(minutes=minutes)


def _at(day, clock):
    if isinstance(clock, tuple):
        clock = datetime.time(*clock)
    return datetime.datetime.combine(day, clock)


def place_blocks(busy, day, items, earliest, latest, not_before=None):
    """Find a time for each item, in order, on `day`.

    busy is a list of naive (start, end). items are dicts with title, minutes
    and optionally start, a fixed time (a datetime.time, or a datetime). The
    window is earliest to latest (datetime.time or (hour, minute)).

    Items with a fixed start are placed first, so a flexible item can never
    take a time he chose. Then each flexible item, in order, takes the first
    free gap that fits. Nothing overlaps busy time or another item, and nothing
    leaves the window. Returns (slots, unplaced): slots are
    {"item", "start", "end"} in item order; unplaced are the items that did not
    fit (a fixed one that is busy or outside the window is unplaced, never
    moved, because he chose it)."""
    window_start, window_end = _at(day, earliest), _at(day, latest)
    taken = list(busy)
    placed = {}

    def length(item):
        return datetime.timedelta(minutes=item["minutes"])

    for i, item in enumerate(items):
        fixed = item.get("start")
        if fixed is None:
            continue
        start = fixed if isinstance(fixed, datetime.datetime) else datetime.datetime.combine(day, fixed)
        end = start + length(item)
        if (start >= window_start and end <= window_end and not overlaps(taken, start, end)
                and (not_before is None or start >= not_before)):
            placed[i] = (start, end)
            taken.append((start, end))
    for i, item in enumerate(items):
        if item.get("start") is not None:
            continue
        slot = find_slot(taken, window_start, window_end, item["minutes"], not_before)
        if slot:
            placed[i] = slot
            taken.append(slot)
    slots = [{"item": item, "start": placed[i][0], "end": placed[i][1]}
             for i, item in enumerate(items) if i in placed]
    unplaced = [item for i, item in enumerate(items) if i not in placed]
    return slots, unplaced


# ── events as lines (pure) ──

def events_on(events, day):
    """The events that fall on a date, all day ones first, then by start."""
    found = []
    for e in events:
        first, last = calendar_roles.event_span(e)
        if e["all_day"]:
            if first.date() <= day < max(last.date(), first.date() + datetime.timedelta(days=1)):
                found.append((0, first, e))
        elif first.date() == day:
            found.append((1, first, e))
    found.sort(key=lambda t: (t[0], t[1], t[2]["title"]))
    return [e for _, _, e in found]


def _title_with_mark(event):
    title = short_title(event["title"])
    return f"{title} {event['mark']}" if event.get("mark") else title


def event_line(event):
    if event["all_day"]:
        return f"• All day  {_title_with_mark(event)}"
    start, end = calendar_roles.event_span(event)
    return f"• {range_words(start, end)}  {_title_with_mark(event)}"


def conflict_line(conflict):
    a, b = short_title(conflict["a"]["title"]), short_title(conflict["b"]["title"])
    return f"CONFLICT: {a} overlaps {b}, {range_words(conflict['start'], conflict['end'])}"


def day_lines(events, day, conflicts=()):
    """One line per event on `day`, then a CONFLICT line for each overlap that
    starts that day, or "Open" for a day with nothing. The one formatter the
    opening's Tomorrow section and the week view share."""
    lines = [event_line(e) for e in events_on(events, day)]
    lines += [conflict_line(c) for c in conflicts if c["start"].date() == day]
    return lines or ["Open"]


def week_view(events, first_day, days=7):
    """A header per day with its date, then that day's lines."""
    first = _day(first_day)
    conflicts = calendar_roles.find_conflicts(events)
    blocks = []
    for n in range(days):
        day = first + datetime.timedelta(days=n)
        blocks.append("\n".join([short_day(day)] + day_lines(events, day, conflicts)))
    return "\n\n".join(blocks)


# ── suggestions (pure) ──

def due_words(due, today):
    """"Overdue", "Due today", "Due tomorrow" or "Due Monday"."""
    due, today = _day(due), _day(today)
    if due < today:
        return "Overdue"
    if due == today:
        return "Due today"
    if due == today + datetime.timedelta(days=1):
        return "Due tomorrow"
    return f"Due {due.strftime('%A')}"


def suggest(today, backlog, events, planned_titles=(), limit=MUST_DO_CAP):
    """At most `limit` things worth doing tomorrow, in this order:
      1. backlog items overdue or due within DUE_SOON_DAYS days of today
      2. exams in the next EXAM_DAYS days, as studying for them: calendar
         events on his own and attended calendars (not MILES, not club or
         holiday context), and backlog items whose title says exam or test and
         are due inside that
      3. conflicts to fix, from find_conflicts over the same days
      4. the oldest backlog items
    Each is {title, task_id, reason, minutes}. Nothing is suggested twice, and
    nothing already planned for tomorrow (planned_titles) is suggested at all."""
    today = _day(today)
    tomorrow = today + datetime.timedelta(days=1)
    horizon = today + datetime.timedelta(days=EXAM_DAYS)
    seen_titles = {t.casefold() for t in planned_titles}
    seen_ids = set()
    picked = []

    def take(title, task_id, reason, minutes):
        key = title.casefold()
        if len(picked) >= limit or key in seen_titles or (task_id and task_id in seen_ids):
            return
        seen_titles.add(key)
        if task_id:
            seen_ids.add(task_id)
        picked.append({"title": title, "task_id": task_id, "reason": reason,
                       "minutes": minutes})

    def due_of(task):
        return _day(task["due_date"]) if task.get("due_date") else None

    # 1. due soon or overdue
    soon = [t for t in backlog if due_of(t) and due_of(t) <= today + datetime.timedelta(days=DUE_SOON_DAYS)]
    for task in sorted(soon, key=lambda t: (due_of(t), t["id"])):
        take(task["text"], task["id"], due_words(task["due_date"], today), task.get("minutes"))

    # 2. exams, soonest first
    exams = []
    for e in events:
        # Only events that are his: a club's "Final GBM of the Semester" is not
        # an exam. An attended club event is blocking after annotate, so it counts.
        if (e.get("editable") or e.get("cancelled") or e.get("role", "blocking") != "blocking"
                or not is_exam_title(e["title"])):
            continue
        first, _ = calendar_roles.event_span(e)
        if tomorrow <= first.date() <= horizon:
            exams.append((first.date(), e["title"].strip(), None, f"Exam {first.strftime('%A')}"
                          if first.date() != tomorrow else "Exam tomorrow"))
    for task in backlog:
        due = due_of(task)
        if due and due <= horizon and is_exam_title(task["text"]):
            exams.append((due, task["text"], task["id"], due_words(due, today)))
    backlog_titles = [t["text"].casefold() for t in backlog]
    for day, title, task_id, reason in sorted(exams, key=lambda x: (x[0], x[1])):
        if task_id is None:
            # A backlog item for this exam already says it.
            if any(title.casefold() in b for b in backlog_titles):
                continue
            title = f"Study for {title}"
        take(title, task_id, reason, EXAM_MINUTES)

    # 3. conflicts, as a job to fix
    window = [e for e in events
              if tomorrow <= calendar_roles.event_span(e)[0].date() <= horizon]
    for c in calendar_roles.find_conflicts(window):
        a, b = short_title(c["a"]["title"]), short_title(c["b"]["title"])
        take(f"Fix {c['start'].strftime('%A')}: {a} overlaps {b}", None, "Conflict to fix",
             FIX_MINUTES)

    # 4. the oldest backlog items
    for task in sorted(backlog, key=lambda t: t["id"]):
        take(task["text"], task["id"], "Oldest in backlog", task.get("minutes"))
    return picked


# ── the plan: saved as JSON in sessions.plan ──
# {"day": "2026-11-25", "earliest": "10:00", "latest": "21:00", "unchecked": false,
#  "must_dos": [{"title", "task_id", "minutes", "start", "reason", "slot"}],
#  "optional": [{"title", "task_id"}]}
# start is a time he fixed ("14:00") or None; slot is the placed ["14:00",
# "16:00"] or None.

def window_of(plan):
    return (datetime.time.fromisoformat(plan["earliest"]),
            datetime.time.fromisoformat(plan["latest"]))


def default_window():
    return (_hhmm(datetime.datetime.combine(datetime.date.min, datetime.time(*SESSION_BLOCK_EARLIEST))),
            _hhmm(datetime.datetime.combine(datetime.date.min, datetime.time(*SESSION_BLOCK_LATEST))))


def local_busy(day, exclude_ids=()):
    """(busy blocks as naive local times, calendars not read) for a day, from
    the real calendar. Raises when no calendar could be read."""
    midnight = datetime.datetime.combine(day, datetime.time(0))
    busy, unreadable = calendar_tools.busy_blocks(
        midnight, midnight + datetime.timedelta(days=1), exclude_ids=exclude_ids)
    local = [(a.astimezone().replace(tzinfo=None), b.astimezone().replace(tzinfo=None))
             for a, b in busy]
    return local, unreadable


def existing_block(task):
    """The MILES event a task points at, on any day, or None. The remembered id
    is cleared only when the event is gone (he deleted it, or it is cancelled),
    which is a fact about Google. A block merely on another day is left alone."""
    from database import clear_task_calendar_event
    if not task or not task.get("calendar_event_id"):
        return None
    event = calendar_tools.get_miles_event(task["calendar_event_id"])
    if event is None:
        clear_task_calendar_event(task["id"])
    return event


def live_block(task, day=None):
    """existing_block, but with day only a block on that day counts. A block on
    another day returns None WITHOUT clearing its id: previewing a plan writes
    nothing, so remove_block still finds the block and confirming moves it
    instead of leaving it behind and booking a second one."""
    event = existing_block(task)
    if event is not None and day is not None and event["start"].date() != _day(day):
        return None
    return event


def place_plan(plan, now=None, keep_slots=False):
    """Fill in each must do's slot from the real calendar and return the plan.

    A must do whose task already has a live block on the day keeps that block's
    time (its own block is left out of the busy time, so it does not collide
    with itself). Everything else is placed by place_blocks, in order. With
    keep_slots, a slot shown to him earlier is kept if it is still free, and
    only an item whose slot is no longer free is placed again; that is what
    confirming does, since the calendar may have changed since the proposal.
    If the calendar cannot be read, plan["unchecked"] is true and no times are
    given."""
    day = _day(plan["day"])
    earliest, latest = window_of(plan)
    blocks, exclude = {}, set()
    for i, item in enumerate(plan["must_dos"]):
        if item.get("task_id"):
            block = live_block(get_task(item["task_id"]), day)
            if block:
                blocks[i] = block
                exclude.add(block["id"])
    try:
        busy, _ = local_busy(day, exclude)
    except Exception:
        plan["unchecked"] = True
        for item in plan["must_dos"]:
            item["slot"] = None
        return plan
    plan["unchecked"] = False
    not_before = now if now and now.date() >= day else None
    window_start, window_end = _at(day, earliest), _at(day, latest)

    items = []
    for i, item in enumerate(plan["must_dos"]):
        minutes = default_minutes(item["title"], item.get("minutes"))
        fixed = datetime.time.fromisoformat(item["start"]) if item.get("start") else None
        if i in blocks:
            block = blocks[i]
            minutes = int((block["end"] - block["start"]).total_seconds() // 60)
            fixed = block["start"]
        elif keep_slots and fixed is None and item.get("slot"):
            start = datetime.datetime.combine(day, datetime.time.fromisoformat(item["slot"][0]))
            end = datetime.datetime.combine(day, datetime.time.fromisoformat(item["slot"][1]))
            if (window_start <= start and end <= window_end and not overlaps(busy, start, end)
                    and (not_before is None or start >= not_before)):
                fixed, minutes = start, int((end - start).total_seconds() // 60)
        items.append({"title": item["title"], "index": i, "minutes": minutes, "start": fixed})
    slots, _ = place_blocks(busy, day, items, earliest, latest, not_before)
    by_index = {s["item"]["index"]: s for s in slots}
    for i, item in enumerate(plan["must_dos"]):
        slot = by_index.get(i)
        item["slot"] = [_hhmm(slot["start"]), _hhmm(slot["end"])] if slot else None
        item["booked"] = i in blocks
    return plan


# ── the messages (read the database, so not pure) ──

def today_line(today):
    """"Today: 3 of 4 done, 1 moved to Friday." from the database."""
    today = _day(today).isoformat()
    tasks = tasks_for_day(today, today=today)
    if not tasks:
        return "Nothing was planned today."
    done = sum(1 for t in tasks if t["status"] == "done")
    parts = [f"{done} of {len(tasks)} done"]
    moved = {}
    for task in tasks:
        if task["status"] == "moved":
            moved.setdefault(moved_destination(task), 0)
            moved[moved_destination(task)] += 1
    for dest, n in sorted(moved.items(), key=lambda kv: kv[0] or ""):
        parts.append(f"{n} moved to {_day(dest).strftime('%A')}" if dest else f"{n} moved")
    skipped = sum(1 for t in tasks if t["status"] == "skipped")
    if skipped:
        parts.append(f"{skipped} skipped")
    return "Today: " + ", ".join(parts) + "."


def read_events(first_day, days):
    """(annotated events for the days, error text or None)."""
    first = datetime.datetime.combine(_day(first_day), datetime.time(0))
    try:
        window = calendar_tools.reviewed_events(first, first + datetime.timedelta(days=days))
    except Exception as exc:
        return [], f"could not read the calendar: {exc}"
    return window["events"], None


def _slot_words(item, plan):
    if item.get("slot"):
        day = _day(plan["day"])
        a = datetime.datetime.combine(day, datetime.time.fromisoformat(item["slot"][0]))
        b = datetime.datetime.combine(day, datetime.time.fromisoformat(item["slot"][1]))
        return f", {range_words(a, b)}"
    return "" if plan.get("unchecked") else ", no time found"


def plan_text(today, plan, now=None):
    """The opening's format, used for the opening and for every updated plan:

        Today: ...
        Tomorrow, <day>  (its events)
        Suggested must dos / Optional
        Good, or change something?
    """
    tomorrow = _day(plan["day"])
    events, error = read_events(tomorrow, 1)
    conflicts = calendar_roles.find_conflicts(events)
    tomorrow_lines = ([f"Could not read the calendar ({error.split(': ', 1)[-1]})."]
                      if error else day_lines(events, tomorrow, conflicts))
    lines = [today_line(today), "", f"Tomorrow, {short_day(tomorrow)}", *tomorrow_lines, "",
             "Suggested must dos"]
    if plan["must_dos"]:
        for n, item in enumerate(plan["must_dos"], 1):
            lines.append(f"{n}. {item['title']}{_slot_words(item, plan)}")
            if item.get("reason"):
                lines.append(f"   {item['reason']}")
    else:
        lines.append("Nothing stands out. What matters tomorrow?")
    if plan["optional"]:
        lines += ["", "Optional"] + [f"• {o['title']}" for o in plan["optional"]]
    lines += ["", ASK]
    return "\n".join(lines)


def build_proposal(today, now=None):
    """The first plan for tomorrow: what is already planned for it, then up to
    three suggestions in total, placed. Reads the database and the calendar."""
    today = _day(today)
    tomorrow = today + datetime.timedelta(days=1)
    planned = tasks_between(tomorrow.isoformat(), tomorrow.isoformat())
    events, _ = read_events(tomorrow, EXAM_DAYS + 1)
    earliest, latest = default_window()
    must = [{"title": t["text"], "task_id": t["id"], "minutes": t.get("minutes"), "start": None,
             "reason": "Already planned"} for t in planned if t["must_do"]][:MUST_DO_CAP]
    optional = [{"title": t["text"], "task_id": t["id"]} for t in planned if not t["must_do"]]
    room = MUST_DO_CAP - len(must)
    if room > 0:
        for s in suggest(today, backlog_tasks(), events,
                         planned_titles=[t["text"] for t in planned], limit=room):
            must.append({"title": s["title"], "task_id": s["task_id"], "minutes": s["minutes"],
                         "start": None, "reason": s["reason"]})
    plan = {"day": tomorrow.isoformat(), "earliest": earliest, "latest": latest,
            "unchecked": False, "must_dos": must, "optional": optional}
    return place_plan(plan, now)


# ── the final plan, from the database and the real calendar ──

def render_plan(day, order=()):
    """The plan as it really stands: must dos and optional tasks from the
    database, and each must do's time from fetching its event on the MILES
    calendar, never from anything remembered or said. An event that is gone or
    cancelled, or a must do never booked, reads "not booked"."""
    day = _day(day)
    tasks = [t for t in tasks_between(day.isoformat(), day.isoformat())
             if t["status"] == "pending"]
    # Priority is the plan's order; the table has no column for it, so a caller
    # who has the plan passes its task ids. Anything else follows by id.
    rank = {task_id: n for n, task_id in enumerate(order)}
    tasks.sort(key=lambda t: (rank.get(t["id"], len(rank)), t["id"]))
    lines = [f"Tomorrow's plan, {short_day(day)}"]
    musts = [t for t in tasks if t["must_do"]]
    for n, task in enumerate(musts, 1):
        words = "not booked"
        if task["calendar_event_id"]:
            try:
                event = calendar_tools.get_miles_event(task["calendar_event_id"])
            except Exception:
                event, words = None, "could not check the calendar"
            # Only a block on this plan's own day is its time.
            if event and not event["all_day"] and event["start"].date() == day:
                words = range_words(event["start"], event["end"])
        lines.append(f"{n}. {task['text']}, {words}")
    if not musts:
        lines.append("No must dos.")
    optional = [t for t in tasks if not t["must_do"]]
    if optional:
        lines += ["", "Optional"] + [f"• {t['text']}" for t in optional]
    return "\n".join(lines)
