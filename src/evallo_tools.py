"""set_session_status: Nova's way to log a tutoring session in Evallo.

The browser work happens in a separate process (scripts/evallo_status.py, run
with ~/miles/.venv-evallo/bin/python), not inside the MILES server, for two
reasons. Playwright and Chromium live in their own environment so they cannot
disturb the packages the voice and server run on. And a hung web page can only
ever hang that one process, which is killed at RUN_TIMEOUT_S, never Nova.

A dry run goes straight through. A real run never does: it is staged with
pending_action.propose(), Nova reads the question back word for word, and it
runs only if confirm_pending_action is called on his very next turn, within
two minutes. That is the same code enforced gate calendar deletions use, so
the model cannot approve its own proposal.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pending_action
from evallo_matching import EVALLO_TZ
from tools import Permission, ToolNotice, tool

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv-evallo" / "bin" / "python"
SCRIPT = ROOT / "scripts" / "evallo_status.py"
# A full run loads three pages and a reload; two minutes is far past normal.
RUN_TIMEOUT_S = 150


def run_script(student, start, status, real, minutes=None):
    """Run the Evallo script once and return its result dict."""
    cmd = [str(PYTHON), str(SCRIPT), "--student", student,
           "--start", start.strftime("%Y-%m-%d %H:%M"), "--status", status, "--json"]
    if real:
        # --yes skips the typed confirmation meant for a terminal; he already
        # said yes to Nova, which is the only way a real run gets here.
        cmd += ["--real", "--yes"]
    if minutes:
        cmd += ["--minutes", str(minutes)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=RUN_TIMEOUT_S, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "outcome": "failed",
                "message": "Evallo took too long and was stopped. Nothing is confirmed."}
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "outcome": "failed",
                "message": "The Evallo script gave no answer.",
                "details": {"stderr": done.stderr[-500:]}}
    return json.loads(lines[-1])


def _parse_start(text):
    """"2026-11-23T16:15" or "2026-11-23 16:15", read as Eastern."""
    start = datetime.fromisoformat(text.replace(" ", "T"))
    return start if start.tzinfo else start.replace(tzinfo=EVALLO_TZ)


def _say(result):
    # The script's message is written to be read aloud already.
    return result["message"]


@tool(
    name="set_session_status",
    description=(
        "Log a tutoring session in Evallo, the scheduling site Lethanial's "
        "tutoring company uses: mark it Completed, No Show or Canceled. Call it "
        "when he says he finished, taught, or had a no show or cancellation for "
        "a student's session. Give the student's first name and the session's "
        "start time. A real change is never made straight away: it returns a "
        "question to ask him, and it happens only if he says yes. Set dry_run "
        "true only when he asks to test or check it; nothing changes then. It "
        "takes about half a minute, because it works the website."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "student_first_name": {"type": "string",
                                   "description": "The student's first name, e.g. Taylor."},
            "start": {"type": "string",
                      "description": "Session start, Eastern time, as YYYY-MM-DDTHH:MM."},
            "status": {"type": "string", "enum": ["Completed", "No Show", "Canceled"]},
            "dry_run": {"type": "boolean",
                        "description": "True to test without changing anything."},
            "length_minutes": {"type": "integer",
                               "description": "The lesson's length, if known from his calendar."},
        },
        "required": ["student_first_name", "start", "status", "dry_run"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
)
def set_session_status(student_first_name, start, status, dry_run, length_minutes=None):
    try:
        when = _parse_start(start)
    except ValueError:
        raise ToolNotice(f"{start!r} is not a time I can read; use YYYY-MM-DDTHH:MM.")
    name = student_first_name.strip()

    if dry_run:
        return _say(run_script(name, when, status, real=False, minutes=length_minutes))

    question = (f"Mark {name}'s session on {when:%A %B} {when.day} at "
                f"{when:%-I:%M %p} {status} in Evallo")
    staged = pending_action.propose(
        question, lambda: _say(run_script(name, when, status, real=True,
                                          minutes=length_minutes)))
    return (f"Nothing has changed yet. Ask Lethanial exactly this, with nothing "
            f"added before or after: {staged} Then end your turn. When he answers, "
            f"call confirm_pending_action.")
