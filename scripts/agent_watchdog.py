#!/usr/bin/env python3
"""Run by miles-agent-watchdog.timer every five minutes: push when the Mac
missed a job, and repeat 7 AM reminders. See src/agent_watchdog.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_watchdog  # noqa: E402

if __name__ == "__main__":
    for body in agent_watchdog.check():
        print(f"watchdog: pushed: {body}", flush=True)
