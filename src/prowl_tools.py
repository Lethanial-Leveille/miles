"""Prowl, the job tracker at tracker.lethanial.com, as tools.

The contract is docs/miles-integration.md in the job tracker repo. Prowl built
a surface for MILES alone, so this module does not reshape the pipeline, it
relabels it. Every count, filter, sort and day difference arrives computed,
and nothing here computes one again. A number recounted from rows sounds
exactly as certain as the right one.

What does change is names. The Oura bug that spoke a score out of a hundred as
a hundred minutes was a missing unit, and the job search has the same trap in a
different shape: 90 of 114 deadlines on Sep 25 2026 were dates Lethanial set
himself so a row would not sink out of view, 77 of them already passed, and not
one came from a posting. Read as deadlines, that is "you missed 77 deadlines"
said to someone who missed none. So a deadline is renamed by where it came
from before the model sees it, and a field called deadline never reaches her.

Nulls with a meaning become words for the same reason. A null
days_since_applied means never applied, and a null fit score means unknown.
Either one read as a zero is a confident wrong sentence.

Every tool is hokage. A job search is private and should not be answerable to
whoever else is in the room.

Every tool reads. Accepting a discovered job was the one write until Prowl
retired its discovery inbox (Oct 26 2026): a stale inbox read aloud is worse
than none, so its two tools went with it.

Failures raise. None of Prowl's errors are transient, so nothing retries; the
executor in brain.py turns the exception into an is_error result and Nova says
what is misconfigured.
"""

import os

import requests

import config  # noqa: F401  loads .env, which holds PROWL_URL and PROWL_TOKEN
from tools import Permission, tool

_TIMEOUT_S = 10

# The applied list alone was 62 rows on Sep 25 2026, roughly 60 KB whole, which
# is too much for one voice turn. Prowl sorts the list, so the rows kept are its
# first ones, and the cut is flagged rather than counted: a count belongs to the
# summary. The cost is that a row past the cut cannot be opened by voice, which
# ends when Prowl takes an organization filter and a limit.
_MAX_APPLICATIONS = 15

# Prowl's ApplicationStatus, in pipeline order.
STATUSES = [
    "discovered", "drafting", "ready", "applied", "assessment",
    "recruiter_engaged", "phone_screen", "technical_interview", "onsite",
    "offer", "accepted", "declined", "rejected", "ghosted", "missed_deadline",
]

# Prowl answers every failure of one kind with one code, on purpose, so the
# sentence here says where to look rather than guessing which cause it was.
_FAILURES = {
    401: ("Prowl rejected the MILES token. PROWL_TOKEN in the Pi's .env and "
          "MILES_SERVICE_TOKEN on the droplet do not match, or the droplet "
          "service has not restarted since it was set"),
    404: "Prowl has no such job any more. The id is stale, so list again",
    500: "Prowl is misconfigured on the droplet, most likely OWNER_EMAIL is unset",
}

# Each deadline source gets its own pair of names, so the key itself says whose
# date it is. A source Prowl adds later falls to unknown, which never claims.
_DEADLINE_NAMES = {
    "posting": ("applications_close", "days_until_applications_close"),
    "self": ("your_own_target_date", "days_until_your_own_target"),
    None: ("date_of_unknown_origin", "days_until_that_date"),
}

_DEADLINE_WORDS = (
    " Dates are named by whose they are. applications_close is the employer's "
    "closing date. your_own_target_date is a date he set himself: say he wanted "
    "to get it in by then, never that applications close. date_of_unknown_origin: "
    "give the date without saying whose it is, and never say it was missed. A "
    "negative days_until count means the date has passed. Say he missed a "
    "deadline only for a job in the summary's overdue list, never from a date in "
    "a list of applications."
)


class ProwlError(Exception):
    pass


def _request(method, path, **params):
    """One call to Prowl's MILES surface, returning the parsed JSON."""
    # Read at call time rather than import time, so a rotated token in .env
    # takes effect on restart without anything caching the old one.
    base = os.environ.get("PROWL_URL")
    token = os.environ.get("PROWL_TOKEN")
    if not base or not token:
        raise ProwlError("PROWL_URL or PROWL_TOKEN is missing from the Pi's .env")

    response = requests.request(
        method,
        f"{base.rstrip('/')}/integrations/miles/{path}",
        params=params or None,
        headers={"X-Miles-Token": token},
        timeout=_TIMEOUT_S,
    )
    if response.status_code in _FAILURES:
        raise ProwlError(_FAILURES[response.status_code])
    response.raise_for_status()
    return response.json()


def speakable_deadline(row):
    """The row with deadline, days_until_deadline and deadline_source replaced by
    one pair of fields named for whose date it is. Values pass through as sent."""
    row = dict(row)
    date = row.pop("deadline", None)
    days = row.pop("days_until_deadline", None)
    source = row.pop("deadline_source", None)
    if date is None:
        return row
    date_key, days_key = _DEADLINE_NAMES.get(source, _DEADLINE_NAMES[None])
    row[date_key] = date
    row[days_key] = days
    return row


def _application(row):
    """One application as Nova should hear it."""
    row = speakable_deadline(row)
    # A link is useless spoken and costs tokens on every row.
    row.pop("posting_url", None)
    if row.get("days_since_applied") is None:
        row.pop("applied_at", None)
        row["days_since_applied"] = "never applied"
    return row


def _application_detail(row):
    """The detail view: an application plus the fit check scoreboard."""
    row = _application(row)
    # Null across the board means no fit check has been run, which is not the
    # same as a bad one.
    if row.get("requirements_total") is None:
        for key in ("requirements_met", "requirements_partial", "requirements_total"):
            row.pop(key, None)
        row["fit_check"] = "never run"
    return row


def speakable_summary(summary):
    summary = dict(summary)
    # Beside active_count, which leaves closed rows out, a bare counts_by_status
    # invites adding it up to answer "how many applications do I have".
    summary["counts_by_status_including_closed"] = summary.pop("counts_by_status", {})
    for key in ("upcoming_deadlines", "overdue", "stale"):
        summary[key] = [_application(row) for row in summary.get(key, [])]
    return summary


def _capped(rows, limit, key):
    return {key: rows[:limit], "more_not_shown": len(rows) > limit}


@tool(
    name="get_job_search_summary",
    description=(
        "Lethanial's job search from Prowl, his job tracker, in one read: how "
        "many applications are active, deadlines coming up, jobs whose posting "
        "deadline passed before he applied (overdue), applications that have "
        "gone quiet (stale), and status changes waiting for his review. Call this "
        "first for any general "
        "question about his job search, internships or applications, and for any "
        "question of how many. Every count here is exact; repeat it, never add "
        "anything up yourself. Only active_count counts his live applications."
        + _DEADLINE_WORDS
    ),
    input_schema={
        "type": "object",
        "properties": {
            "deadline_days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 90,
                "description": "How many days ahead counts as upcoming. Default 7.",
            },
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_job_search_summary(deadline_days=None):
    params = {} if deadline_days is None else {"deadline_days": deadline_days}
    return speakable_summary(_request("GET", "summary", **params))


@tool(
    name="list_job_applications",
    description=(
        "Lethanial's job applications in Prowl with one status, such as every "
        "company he has applied to or is interviewing with. Call this when he "
        "asks which jobs are at a stage, or to find the id of an application he "
        "named before opening it. If more_not_shown is true the list is cut off, "
        "so never count the rows; counts come from get_job_search_summary."
        + _DEADLINE_WORDS
    ),
    input_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": STATUSES},
        },
        "required": ["status"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def list_job_applications(status):
    rows = [_application(row) for row in _request("GET", "applications", status=status)]
    if not rows:
        return f"No applications in Prowl with status {status}."
    return _capped(rows, _MAX_APPLICATIONS, "applications")


@tool(
    name="get_job_application",
    description=(
        "One application from Prowl opened up: location, pay, a short summary, "
        "the key requirements, his notes, and how many requirements his resume "
        "meets. Call this when he asks about a specific job, what it requires or "
        "what it pays. Takes the id from the summary or a list of applications."
        + _DEADLINE_WORDS
    ),
    input_schema={
        "type": "object",
        "properties": {
            "application_id": {"type": "string"},
        },
        "required": ["application_id"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_job_application(application_id):
    return _application_detail(_request("GET", f"applications/{application_id}"))
