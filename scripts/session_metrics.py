#!/usr/bin/env python3
"""The nightly review's numbers: completion rate, cost per review, and must
dos planned against done. Read only. Definitions are in src/session_metrics.py.

    python3 scripts/session_metrics.py            every review so far
    python3 scripts/session_metrics.py 2026-12-01 from that day on
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from database import session_metrics_rows, task_day  # noqa: E402
import session_metrics as metrics  # noqa: E402


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _money(value):
    return "unpriced" if value is None else f"${value:.4f}"


first_day = sys.argv[1] if len(sys.argv) > 1 else None
report = metrics.summarize(session_metrics_rows("nightly_review", first_day), task_day())

for night in report["nights"]:
    done = "pending" if night["must_dos_done"] is None else night["must_dos_done"]
    print(f"{night['day']}  {night['outcome']:<9}  {_money(night['cost']):>10}  "
          f"must dos {done}/{night['must_dos_planned']}")
print()
print(f"Reviews settled:   {report['settled']}")
print(f"Completion rate:   {_pct(report['completion_rate'])}")
print(f"Cost per review:   {_money(report['average_cost'])} (average, reviews he wrote in)")
print(f"Must dos done:     {report['must_dos_done']} of {report['must_dos_planned']} "
      f"({_pct(report['must_do_rate'])}, days already over)")
