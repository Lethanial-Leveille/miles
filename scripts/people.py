#!/usr/bin/env python3
"""The people Nova knows about, their clearance, and their birthdays.

Three things live here because they are all properties of a person:
identity (what to call them), authorization (what they may ask for), and dates
that recur. A birthday is structured data that prose cannot answer, because
"is that soon" is not derivable from a sentence in the memory corpus.

Ranks in ascending authority: genin, chunin, jonin, hokage.

Recognition and authorization are independent. Enrolling someone's voice means
Nova knows who is speaking; it grants nothing on its own. A roommate can be
enrolled and left at genin, and an unrecognised voice is treated as genin, so
the unknown case needs no separate handling.

Tier changes live here and not in the voice path on purpose. Voice is
replayable from a recording and the similarity spread between speakers is
narrow enough that it is a convenience filter rather than a credential. A
spoken request may propose a change; confirming it happens at this keyboard.

    python3 scripts/people.py list
    python3 scripts/people.py list --tier jonin
    python3 scripts/people.py birthdays
    python3 scripts/people.py add "Marcus Webb" --relationship roommate --tier chunin
    python3 scripts/people.py set-birthday 12 2004-03-19
    python3 scripts/people.py promote 12 jonin
    python3 scripts/people.py restore          # clear a self imposed demotion
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import DB_PATH
from database import (TIERS, add_person, effective_tier, get_people, set_tier,
                      set_tier_override, upcoming_birthdays)

# What each rank can reach. Enforced in prompt assembly, not by asking Nova
# nicely: below jonin the corpus is not in her context at all.
CLEARANCE = {
    "genin":  "general knowledge and the weather. Nothing about Lethanial.",
    "chunin": "the above, plus timers. Still nothing about Lethanial.",
    "jonin":  "the above, plus memories explicitly marked shareable.",
    "hokage": "everything, every action, and memory writes.",
}


def cmd_list(tier):
    people = get_people(tier)
    if not people:
        print("Nobody at that tier." if tier else "No people yet.")
        return
    for rank in reversed(TIERS):
        group = [p for p in people if p["tier"] == rank]
        if not group:
            continue
        print(f"\n{rank.upper()}  ({len(group)})  {CLEARANCE[rank]}")
        for p in group:
            called = p["preferred_name"] or p["full_name"]
            extra = []
            if p["preferred_name"] and p["preferred_name"] != p["full_name"]:
                extra.append(f"({p['full_name']})")
            if p["relationship"]:
                extra.append(p["relationship"])
            if p["birthday"]:
                extra.append(f"b. {p['birthday']}")
            print(f"  [{p['id']:>3}] {called:<24} {'  '.join(extra)}")


def cmd_birthdays(days):
    due = upcoming_birthdays(days)
    if not due:
        print(f"Nothing in the next {days} days.")
    for p in due:
        called = p["preferred_name"] or p["full_name"]
        when = "today" if p["days_away"] == 0 else f"in {p['days_away']} days"
        print(f"  {p['due']}  {when:<14} {called}  ({p['relationship'] or 'unknown'})")

    missing = [p for p in get_people() if not p["birthday"]]
    if missing:
        print(f"\n{len(missing)} people have no birthday stored. "
              f"Add them as you learn them: people.py set-birthday <id> <YYYY-MM-DD>")


def cmd_add(args):
    person_id = add_person(args.name, relationship=args.relationship,
                           preferred_name=args.preferred, birthday=args.birthday,
                           tier=args.tier)
    if person_id is None:
        print(f"{args.name!r} already exists. Names are unique so that voice "
              f"enrollment cannot create a second identity for one person.")
        return
    print(f"[{person_id}] {args.name} at {args.tier}: {CLEARANCE[args.tier]}")


def cmd_set_birthday(person_id, birthday):
    conn = sqlite3.connect(DB_PATH)
    changed = conn.execute("UPDATE people SET birthday = ? WHERE id = ?",
                           (birthday, person_id)).rowcount
    conn.commit()
    conn.close()
    print(f"set to {birthday}" if changed else f"no person with id {person_id}")


def cmd_promote(person_id, tier):
    people = {p["id"]: p for p in get_people()}
    if person_id not in people:
        print(f"no person with id {person_id}")
        return
    was = people[person_id]["tier"]
    set_tier(person_id, tier)
    print(f"{people[person_id]['full_name']}: {was} -> {tier}")
    print(f"  now has: {CLEARANCE[tier]}")


def cmd_restore():
    """Clear a demotion he set by voice.

    This lives here and not in the voice path deliberately. Demoting yourself
    over a channel that can be replayed from a recording is harmless; restoring
    yourself over it is the whole attack. The asymmetry is the point."""
    in_force = effective_tier()
    set_tier_override(None)
    now = effective_tier()
    if in_force == now:
        print(f"Nothing to clear, already {now}.")
    else:
        print(f"{in_force} -> {now}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list");       l.add_argument("--tier", choices=TIERS)
    b = sub.add_parser("birthdays");  b.add_argument("--days", type=int, default=30)
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("--relationship")
    a.add_argument("--preferred")
    a.add_argument("--birthday")
    a.add_argument("--tier", choices=TIERS, default="genin")
    s = sub.add_parser("set-birthday")
    s.add_argument("id", type=int); s.add_argument("birthday")
    sub.add_parser("restore")
    m = sub.add_parser("promote")
    m.add_argument("id", type=int); m.add_argument("tier", choices=TIERS)

    args = p.parse_args()
    if args.cmd == "list":
        cmd_list(args.tier)
    elif args.cmd == "birthdays":
        cmd_birthdays(args.days)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "set-birthday":
        cmd_set_birthday(args.id, args.birthday)
    elif args.cmd == "restore":
        cmd_restore()
    elif args.cmd == "promote":
        cmd_promote(args.id, args.tier)


if __name__ == "__main__":
    main()
