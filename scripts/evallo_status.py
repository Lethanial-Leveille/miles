"""Set an Evallo session's status from the command line. Dry run unless --real.

    # On the Pi
    ~/miles/.venv-evallo/bin/python scripts/evallo_status.py \\
        --student Taylor --start "2026-11-23 16:15" --status Completed

    --real        actually change it (asks you to type yes first)
    --watch       show the browser and slow it down, to watch it work (Mac)
    --minutes N   the lesson's length, kept if it lands on the unlogged list
    --json        print the result as JSON (how MILES calls this)
    --unlogged    list the sessions waiting to be logged, and stop

Times are Eastern, as Evallo shows them.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evallo_runner   # noqa: E402
import evallo_unlogged  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student")
    ap.add_argument("--start", help='"YYYY-MM-DD HH:MM", Eastern')
    ap.add_argument("--status", choices=["Completed", "No Show", "Canceled"])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--minutes", type=int)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--unlogged", action="store_true")
    args = ap.parse_args()

    if args.unlogged:
        rows = evallo_unlogged.load()
        print(json.dumps(rows, indent=2) if args.json else
              "\n".join(f"{r['date']} {r['time']}  {r['student']:<14} "
                        f"{r['length_minutes'] or '?'} min  {r['reason']}" for r in rows)
              or "Nothing waiting to be logged.")
        return 0
    if not (args.student and args.start and args.status):
        ap.error("--student, --start and --status are required")

    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    if args.real and not args.yes:
        # --yes is how MILES calls this after Lethanial has already said yes
        # to Nova; at a terminal, a person types it.
        answer = input(f"Really mark {args.student}'s {args.start} session "
                       f"{args.status} in Evallo? Type yes: ")
        if answer.strip().lower() != "yes":
            print("Not changed.")
            return 1

    result = evallo_runner.set_session_status(
        args.student, start, args.status, dry_run=not args.real,
        minutes=args.minutes, headless=not args.watch,
        slow_mo=700 if args.watch else 0)
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"[{result.outcome}] {result.message}")
        if result.failure_dir:
            print(f"Failure record: {result.failure_dir}")
        if result.details:
            print(json.dumps(result.details, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
