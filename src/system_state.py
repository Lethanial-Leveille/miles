"""Nova's view of her own health, exposed as a tool.

Everything here is read locally: procfs, sqlite, systemd, git. No network, no
credentials, nothing that can change state. That is why the whole thing is one
READ tool rather than several.

Each reader is individually guarded. A Pi without a thermal zone, a checkout
without a git directory, a systemd unit that has never restarted: none of those
should turn "how are you doing" into an error. A field that cannot be read comes
back None and Nova simply does not mention it.
"""

import os
import sqlite3
import statistics
import subprocess

from config import DB_PATH, WHISPER_MODEL
from tools import Permission, tool

# Recent turns to summarize. Enough to be representative of the current build,
# short enough that a fix landing an hour ago shows up rather than being
# averaged away by a week of worse numbers.
_LATENCY_WINDOW = 30


def _read(fn, default=None):
    """Run a reader, swallowing its failure. See module docstring."""
    try:
        return fn()
    except Exception:
        return default


def _uptime_hours():
    with open("/proc/uptime") as f:
        return round(float(f.read().split()[0]) / 3600, 1)


def _core_temp_f():
    """Fahrenheit, because that is the only scale Lethanial uses and the tool
    output is read aloud verbatim. Reporting Celsius here meant Nova announcing
    "fifty seven degrees Celsius" alongside a weather answer in Fahrenheit,
    which is two scales in one conversation."""
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        celsius = int(f.read().strip()) / 1000.0
    return round(celsius * 9 / 5 + 32, 1)


def _git_commit():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()


def _service_restarts(unit="miles-voice.service"):
    out = subprocess.run(
        ["systemctl", "show", unit, "--property=NRestarts", "--value"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    return int(out) if out else 0


def _db_stats():
    conn = sqlite3.connect(DB_PATH)
    try:
        turns = conn.execute("SELECT COUNT(*) FROM timing_log").fetchone()[0]
        memories = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT total_perceived_ms FROM timing_log "
            "WHERE total_perceived_ms IS NOT NULL ORDER BY id DESC LIMIT ?",
            (_LATENCY_WINDOW,)
        ).fetchall()
    finally:
        conn.close()

    latency = round(statistics.median(r[0] for r in rows)) if rows else None
    return {"turns": turns, "memories": memories, "median_latency_ms": latency}


@tool(
    name="get_system_state",
    description=(
        "Nova's own operational state: uptime, how many turns she has served, "
        "her median response latency over recent turns, the speech model she is "
        "running, how many memories she is holding, the Pi's core temperature in Fahrenheit, "
        "how many times the voice service has restarted, and the git commit she "
        "is running. Call this when Lethanial asks how she is doing, how she is "
        "running, whether she is healthy, how warm the Pi is, how fast she has "
        "been, or what version she is on. Report only the fields he asked "
        "about. A null field means that reading was unavailable, so say nothing "
        "about it rather than reporting it as zero."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.READ,
    returns_to_model=True,
)
def get_system_state():
    stats = _read(_db_stats, {}) or {}
    return {
        "uptime_hours": _read(_uptime_hours),
        "turns_served": stats.get("turns"),
        "median_latency_ms": stats.get("median_latency_ms"),
        "whisper_model": os.path.basename(WHISPER_MODEL),
        "active_memories": stats.get("memories"),
        "core_temp_f": _read(_core_temp_f),
        "voice_service_restarts": _read(_service_restarts),
        "git_commit": _read(_git_commit),
    }
