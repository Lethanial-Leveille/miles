#!/usr/bin/env python3
"""Tell me when Miles breaks, instead of me finding out by trying to use it.

The tunnel was down for an unknown length of time and the way that was
discovered was by opening the app. Nothing watches this system, so every
failure is silent until it happens to be in the way.

## What this does not cover, stated up front

A Pi that is powered off cannot report that it is powered off. This catches a
unit down, a crash loop, a dead tunnel, a reverted mic gain, a full disk. It
cannot catch total death, which needs a heartbeat somewhere else that alarms on
silence. Every failure actually hit so far is in the first list, so that is what
this is built for.

## The rule that shapes it

This must not depend on the health of what it is checking. netcheck.py makes
the same argument one level down: nothing in it may need the network it is
questioning.

The sharp consequence is that this must NEVER import audio. That module opens
PyAudio, claims the microphone, and takes an exclusive flock on mic.lock, so
importing it here would fight miles-voice for the device and then exit 1. The
mic gain check therefore carries its own copy of one small regex.

config IS imported, so thresholds have a single source of truth. If that import
fails the voice service could not have started either, so the failure is
reported rather than swallowed.

## Two severities

FAIL notifies. WARN only writes to the journal. A check that is not yet trusted
enough to wake someone up goes in as WARN until its false positive rate is
known, rather than being left out entirely.

    python3 scripts/healthcheck.py            # run every check, print, exit 1 on any FAIL
    python3 scripts/healthcheck.py --verbose  # include the checks that passed
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from collections import namedtuple
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

OK, WARN, FAIL = "OK", "WARN", "FAIL"

Result = namedtuple("Result", "level name detail")

UNITS = ("miles-voice", "miles-server", "miles-tunnel")

# Probed over the public hostname rather than localhost on purpose. Localhost
# proves uvicorn is listening and proves nothing about cloudflared, DNS, the
# certificate, or the route, which is the part that actually broke.
#
# /docs is used because FastAPI serves it without authentication, so this needs
# no token and cannot fail because a JWT expired.
PUBLIC_URL   = "https://miles.lethanial.com/docs"
HTTP_TIMEOUT = 10

# Where the previous run's counters live, so a restart can be seen as a change
# rather than as an absolute number. Under data/, which is gitignored.
STATE_PATH = os.path.expanduser("~/miles/data/health.state")

# Free space below this on the data partition is a problem worth hearing about.
DISK_MIN_GB = 2.0

# Contiguous zeroes in cache_read_tokens required before saying anything. The
# preflight in docs/SESSION_START.md is clear that scattered zeroes are the
# five minute TTL expiring and are normal, while a run covering every recent
# turn means the cacheable prefix fell under the model minimum.
#
# The boundary between those is genuinely fuzzy at low turn volume: sparse use
# means longer gaps between turns, which means more TTL expiry, and a real
# observed window held seven consecutive zeroes with nothing wrong. So this is
# set high and reports WARN rather than FAIL until it has proven itself against
# real data.
CACHE_ZERO_RUN = 10


# ── Notification ──

def notify(subject, body):
    """Where a failure goes.

    The journal is the floor and is always written, because it is the one
    channel that cannot be misconfigured. Email is layered on top and stays
    completely inert until all three variables are present, so this works today
    with no setup and starts mailing the day they are set.

    Gmail wants an app password here, not the account password, which requires
    2FA to be enabled on the account first."""
    print(f"\n{subject}\n{body}", flush=True)

    host = os.environ.get("MILES_SMTP_HOST", "smtp.gmail.com")
    user = os.environ.get("MILES_SMTP_USER")
    password = os.environ.get("MILES_SMTP_PASSWORD")
    to = os.environ.get("MILES_ALERT_EMAIL")
    if not (user and password and to):
        return

    # Imported here rather than at module scope so a broken mail configuration
    # can never stop the checks themselves from running.
    import smtplib
    from email.message import EmailMessage

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = user
        message["To"] = to
        message.set_content(body)
        with smtplib.SMTP(host, int(os.environ.get("MILES_SMTP_PORT", 587)),
                          timeout=20) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
        print(f"Alert emailed to {to}.", flush=True)
    except Exception as exc:
        # A failed notification must never mask the failure it was carrying.
        print(f"Could not send alert email ({type(exc).__name__}): {exc}",
              flush=True)


# ── State, for the checks that need a previous value ──

def read_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as exc:
        print(f"Could not write {STATE_PATH}: {exc}", flush=True)


# ── Checks ──

def _systemctl(*args):
    try:
        done = subprocess.run(("systemctl",) + args, capture_output=True,
                              text=True, timeout=10)
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_units(state):
    """Active, and not quietly restarting in a loop.

    is-active on its own is not enough and this is the failure it misses. A
    service crashing every twenty seconds under Restart=always reads as active
    at any instant you happen to look, which is exactly the shape of the outage
    described in CLAUDE.md: the room hears a chime, then silence, forever.

    NRestarts is cumulative, so the signal is the delta against the last run
    rather than the value. That also windows it to the timer interval for free:
    a restart that happened last week is history, one since the previous check
    is happening now."""
    results = []
    counts = {}

    for unit in UNITS:
        active = _systemctl("is-active", unit)
        if active != "active":
            results.append(Result(FAIL, unit, f"is {active or 'unknown'}, expected active"))
        else:
            results.append(Result(OK, unit, "active"))

        raw = _systemctl("show", "-p", "NRestarts", "--value", unit)
        try:
            restarts = int(raw)
        except ValueError:
            continue
        counts[unit] = restarts

        previous = state.get("restarts", {}).get(unit)
        if previous is None:
            # First ever run has nothing to compare against. Recording the
            # baseline is the correct outcome, not a finding.
            continue
        if restarts > previous:
            results.append(Result(
                FAIL, f"{unit} restarts",
                f"restarted {restarts - previous} time(s) since the last check "
                f"(total {restarts}). It may be crash looping while still "
                f"reporting active."))

    state.setdefault("restarts", {}).update(counts)
    return results


def check_tunnel():
    """The whole public path, end to end.

    Anything other than a 200 is a failure regardless of which layer produced
    it, because the question being asked is whether the app can reach Miles,
    not which component is to blame."""
    try:
        request = urllib.request.Request(PUBLIC_URL, method="GET",
                                         headers={"User-Agent": "miles-healthcheck"})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            if response.status == 200:
                return [Result(OK, "tunnel", f"{PUBLIC_URL} returned 200")]
            return [Result(FAIL, "tunnel",
                           f"{PUBLIC_URL} returned {response.status}")]
    except urllib.error.HTTPError as exc:
        return [Result(FAIL, "tunnel", f"{PUBLIC_URL} returned {exc.code}")]
    except Exception as exc:
        return [Result(FAIL, "tunnel",
                       f"{PUBLIC_URL} unreachable ({type(exc).__name__}): {exc}")]


# Deliberately duplicated from audio.py rather than imported. Importing audio
# opens PyAudio, claims the microphone, and takes an exclusive lock that
# miles-voice is already holding, so this process would exit 1 and report
# nothing. One regex is a cheap price for a checker that cannot fight the thing
# it is checking.
_MIXER_VALUE = re.compile(r'Capture (\d+) \[(\d+)%\](?: \[([-\d.]+)dB\])?')


def check_mic_gain(config):
    """Capture gain still at the tuned value.

    A silent revert does not raise anything. It halves the quality of every
    recording, every verification score, and every transcript taken afterward,
    and only shows up days later as inexplicably low numbers."""
    if config.MIC_MIXER_CARD is None:
        return [Result(WARN, "mic gain",
                       f"no ALSA card matching {config.MIC_NAME_HINT!r}. The mic "
                       f"may be unplugged, or the hint needs updating.")]

    try:
        done = subprocess.run(
            ["amixer", "-c", config.MIC_MIXER_CARD, "sget", config.MIC_MIXER_CONTROL],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return [Result(WARN, "mic gain", f"amixer did not run: {exc}")]

    match = _MIXER_VALUE.search(done.stdout)
    if not match:
        return [Result(WARN, "mic gain", "could not parse amixer output")]

    value = int(match.group(1))
    if value == config.EXPECTED_MIC_GAIN:
        return [Result(OK, "mic gain",
                       f"{value} as expected on card {config.MIC_MIXER_CARD} "
                       f"({config.MIC_NAME_HINT})")]
    return [Result(FAIL, "mic gain",
                   f"is {value}, expected {config.EXPECTED_MIC_GAIN}. Restore with: "
                   f"amixer -c {config.MIC_MIXER_CARD} sset {config.MIC_MIXER_CONTROL} "
                   f"{config.EXPECTED_MIC_GAIN} && sudo alsactl store")]


def check_disk(config):
    """Free space where the database and the archives live."""
    target = os.path.dirname(config.DB_PATH)
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return [Result(WARN, "disk", f"could not stat {target}: {exc}")]

    free_gb = usage.free / (1024 ** 3)
    detail = f"{free_gb:.1f}GB free on {target}"
    if free_gb < DISK_MIN_GB:
        return [Result(FAIL, "disk", detail + f", under {DISK_MIN_GB}GB")]
    return [Result(OK, "disk", detail)]


def check_cache(config):
    """Prompt caching still engaging.

    Falling under the model's minimum cacheable prefix disables caching with no
    error and no warning. cache_read_tokens is the only signal there is, which
    is why it is logged per turn.

    Reported as WARN rather than FAIL. Scattered zeroes are the normal five
    minute TTL expiring, and at low turn volume the gaps between turns get long
    enough that a real healthy window held seven consecutive zeroes. Until that
    false positive rate is measured, this is not allowed to wake anyone up."""
    try:
        connection = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT cache_read_tokens FROM timing_log "
            "ORDER BY id DESC LIMIT ?", (CACHE_ZERO_RUN,)).fetchall()
        connection.close()
    except sqlite3.Error as exc:
        return [Result(WARN, "prompt cache", f"could not read timing_log: {exc}")]

    values = [r[0] for r in rows]
    if len(values) < CACHE_ZERO_RUN:
        return [Result(OK, "prompt cache",
                       f"only {len(values)} turns logged, not enough to judge")]
    if all(v == 0 for v in values):
        return [Result(WARN, "prompt cache",
                       f"last {CACHE_ZERO_RUN} turns all read 0 cached tokens. "
                       f"The cacheable prefix may have fallen under the model "
                       f"minimum, which fails silently.")]
    return [Result(OK, "prompt cache",
                   f"{sum(1 for v in values if v)} of {len(values)} recent turns hit")]


# ── Entry point ──

def run_all():
    state = read_state()
    results = []

    # config carries the thresholds so they are not duplicated here. Its
    # failure to import is itself a finding: the voice service imports the same
    # module and could not have started either.
    try:
        import config
    except Exception as exc:
        results.append(Result(FAIL, "config import",
                              f"src/config.py did not import ({type(exc).__name__}): {exc}"))
        config = None

    results += check_units(state)
    results += check_tunnel()
    if config is not None:
        results += check_mic_gain(config)
        results += check_disk(config)
        results += check_cache(config)

    state["last_run"] = datetime.now().isoformat()
    write_state(state)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="include checks that passed")
    args = parser.parse_args()

    results = run_all()
    failures = [r for r in results if r.level == FAIL]
    warnings = [r for r in results if r.level == WARN]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for result in results:
        if result.level == OK and not args.verbose:
            continue
        print(f"[{result.level:4}] {result.name}: {result.detail}", flush=True)

    if failures:
        body = "\n".join(f"{r.name}: {r.detail}" for r in failures + warnings)
        notify(f"Miles health check FAILED ({len(failures)}) at {stamp}", body)
        return 1

    if warnings:
        print(f"{len(warnings)} warning(s), no failures.", flush=True)
        return 0

    if not args.verbose:
        print(f"All {len(results)} checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
