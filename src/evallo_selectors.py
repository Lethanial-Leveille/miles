"""Every way the Evallo script finds something on the page, in one place.

When Evallo changes its page and a run breaks, the failure record names the
key below that was in use (failure.json, "selector_key"). Fix that entry here
and nowhere else. Each entry is plain data, so a later phase can read this
file, the failure record and the saved page.html, and propose a new value.

Kinds, in order of preference:

    role   page.get_by_role(role, name=name, exact=exact)
    text   page.get_by_text(text, exact=exact)
    css    page.locator(css). Only for FullCalendar's own documented names
           (fc-event, data-date), which the calendar library sets and keeps
           stable across versions. Never Evallo's generated styling classes
           such as "font-[500] text-[10px]": those change on any redesign.

Read from the live page on Nov 23 2026, logged in as Lethanial.
"""

SELECTORS = {
    # ── Pages ────────────────────────────────────────────────────────────
    "calendar_url": {"url": "https://app.evallo.org/calendar"},
    "students_url": {"url": "https://app.evallo.org/users"},

    # ── Logged in or not ─────────────────────────────────────────────────
    # Shown only on the sign in page. Its presence means the saved login ran out.
    "signin_marker": {"kind": "role", "role": "button", "name": "Continue with Google"},
    # The view picker reads "Weekly" in the week view the script works in.
    "weekly_view": {"kind": "role", "role": "button", "name": "Weekly"},

    # ── Moving to a week: the small month calendar on the left ───────────
    # Its grid is named for the month it shows, e.g. "November 2026".
    "mini_month": {"kind": "role", "role": "grid", "name_format": "{month} {year}"},
    "mini_day": {"kind": "role", "role": "gridcell", "name_format": "{day}", "exact": True},
    "mini_prev_month": {"kind": "role", "role": "button", "name": "Previous month"},
    "mini_next_month": {"kind": "role", "role": "button", "name": "Next month"},
    # The week grid's day headers carry the full date, which proves the right
    # week is showing: columnheader "November 24, 2026".
    "week_day_header": {"kind": "role", "role": "columnheader",
                        "name_format": "{month} {day}, {year}", "exact": True},

    # ── Events in the week (FullCalendar's own names) ────────────────────
    # One column per day, carrying its date as data-date="2026-11-24".
    "day_column": {"kind": "css", "css": ".fc-timegrid-col[data-date]"},
    "event": {"kind": "css", "css": ".fc-event"},
    # A card reads "Taylor <> Lethanial Leveille | Test Prep | 1-1" then
    # "04:15 PM - 05:45 PM". Canceled cards are struck through (CSS
    # text-decoration line-through) and lose their colour; read in code.

    # ── The panel that opens on an event ─────────────────────────────────
    "panel_title": {"kind": "role", "role": "textbox", "name": "Add Title"},
    # Start and end time boxes, both named for their placeholder. First = start.
    "panel_times": {"kind": "role", "role": "textbox", "name": "hh:mm"},
    # Any textbox; the start date is the one reading "24 Nov, 2026".
    "panel_textboxes": {"kind": "role", "role": "textbox"},
    # The label; its parent reads "Student* Taylor Chen".
    "panel_student_label": {"kind": "text", "text": "Student*", "exact": True},
    # The current status, shown at the top of the panel and clickable to open
    # the menu. The script looks for each known status by its exact word.
    "status_current": {"kind": "text", "text_format": "{status}", "exact": True},
    # A menu item, e.g. button "Completed".
    "status_option": {"kind": "role", "role": "button", "name_format": "{status}", "exact": True},

    # ── Confirmation popup and result ────────────────────────────────────
    # Seen by Lethanial on Nov 23 2026 marking two sessions Completed:
    #   heading   "Update changes"
    #   question  "Are you sure you want to mark this session as Completed?"
    #   note      "Note that this action can be irreversible in some situations."
    #   buttons   "No, go back"  "Yes, confirm"
    #   toast     "Session updated successfully"
    "confirm_heading": {"kind": "text", "text": "Update changes", "exact": True},
    # The whole sentence, status included, so the popup is proven to be about
    # the status that was picked and not some other one.
    "confirm_question": {"kind": "text", "exact": True,
                         "text_format": "Are you sure you want to mark this session as {status}?"},
    "confirm_yes": {"kind": "role", "role": "button", "name": "Yes, confirm", "exact": True},
    "confirm_no": {"kind": "role", "role": "button", "name": "No, go back", "exact": True},
    # Exact: booking a session shows "Session scheduled successfully", which
    # must never pass for a status change.
    "success_toast": {"kind": "text", "text": "Session updated successfully", "exact": True},

    # ── Students list ────────────────────────────────────────────────────
    # Each student is a link to their profile: link "Taylor Chen" -> /profile/student/<id>.
    "student_link": {"kind": "css", "css": "a[href*='/profile/student/']"},
    # "Showing 1 - 20 of 20", to know the whole list is on the page.
    "students_count": {"kind": "text", "regex": r"Showing \d+ - \d+ of \d+"},
}

# Every status the menu can show, including ones the script never sets, so
# the panel's current status can be read whatever it is.
KNOWN_STATUSES = ("Scheduled", "Completed", "No Show", "Canceled", "Tentative",
                  "Available (Subbing)")


def describe(key, **fill):
    """The entry as written, with any {placeholders} filled, for failure.json."""
    entry = dict(SELECTORS[key])
    for field in ("name_format", "text_format"):
        if field in entry:
            entry[field.replace("_format", "")] = entry.pop(field).format(**fill)
    return entry


def locate(page, key, **fill):
    """A Playwright locator for SELECTORS[key]."""
    import re
    entry = describe(key, **fill)
    kind = entry.get("kind")
    if kind == "role":
        return page.get_by_role(entry["role"], name=entry.get("name"),
                                exact=entry.get("exact", False))
    if kind == "text":
        if "regex" in entry:
            return page.get_by_text(re.compile(entry["regex"]))
        return page.get_by_text(entry["text"], exact=entry.get("exact", False))
    if kind == "css":
        return page.locator(entry["css"])
    raise KeyError(f"{key} is not a locator")
