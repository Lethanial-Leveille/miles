"""Nightly check that the Evallo script still works. Run by
miles-evallo-canary.timer at 2 AM; changes nothing.

It finds his next upcoming session and walks the script up to the status
menu: logged in, the week found, the card matched, the panel verified, the
menu opened and read. Then it closes the menu without clicking an option,
reloads, and proves the status is unchanged.

Why it stops at the menu rather than clicking "No, go back": the next session
has not happened yet, and Evallo only offers Completed and No Show once a
session has started (seen Nov 23 2026). The only option it could click is
Canceled, on a real student's real session, every night. If Evallo ever
dropped the confirmation popup, that click would cancel the lesson and likely
email the family. The popup itself is covered by the dry runs he starts, on
sessions that have happened.

A pass prints one line to the journal and nothing else. A failure sends
"Evallo script broke at step X". Three days before the saved login runs out,
it sends a reminder to log in again.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evallo_failures  # noqa: E402
import evallo_runner    # noqa: E402
import evallo_session   # noqa: E402


def main():
    evallo_failures.prune()
    try:
        state = evallo_session.load_state()
    except evallo_session.LoginExpired:
        evallo_session.notify("Evallo login expired, log in again")
        return 1
    left = evallo_session.days_left(state)
    if left is not None and left <= evallo_session.WARN_DAYS:
        evallo_session.notify(f"Evallo login runs out in {max(0, round(left))} days. "
                              "Log in again on the Mac and copy it over.")

    found = evallo_runner.next_session()
    if isinstance(found, evallo_runner.Result):
        evallo_session.notify(found.message if found.outcome == "login_expired"
                              else f"Evallo script broke at step {found.step}")
        print(f"canary FAIL [{found.outcome}] {found.message} {found.failure_dir or ''}")
        return 1
    if found is None:
        print("canary: no session in the next two weeks; login and calendar OK")
        return 0

    name, event = found
    result = evallo_runner.set_session_status(
        name, event.start, "Canceled", check_only=True)
    if result.outcome in ("check_ok", "other_status"):
        print(f"canary OK: {result.message}")
        return 0
    # set_session_status has already sent a push for these three.
    if result.outcome not in ("failed", "login_expired", "ambiguous"):
        # Matching trouble on a session read off the calendar a moment ago
        # still means the script cannot be trusted tonight.
        evallo_session.notify(f"Evallo canary stopped: {result.message}")
    print(f"canary FAIL [{result.outcome}] {result.message} {result.failure_dir or ''}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
