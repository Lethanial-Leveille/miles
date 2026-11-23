"""What gets kept when an Evallo run breaks, and for how long.

Each failure is one folder, data/evallo_failures/<UTC time>_<step>/, holding:

    failure.json   the structured record (below)
    screenshot.png what the page looked like at that moment
    page.html      the page's HTML at that moment

failure.json is written for a later phase that hands it to a model to repair
the selector, so it says exactly what was being looked for, not only that
something went wrong:

    schema           record version, bumped if the shape changes
    time             when it broke, UTC ISO 8601
    step             the named step, e.g. "open_status"
    selector_key     the key in evallo_selectors.SELECTORS being used, or null
    selector         that selector's role, name and text, as written
    error_type       the exception class, e.g. "TimeoutError"
    error            the message
    url              the page it was on
    request          what was asked: student, start, status, dry_run
    files            the snapshot file names in this folder

The snapshots hold students' names, emails and phone numbers. The folder is
ignored by git, and prune() deletes anything older than KEEP_DAYS on every run.
"""

import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES = ROOT / "data" / "evallo_failures"
SCHEMA = 1
KEEP_DAYS = 30


def record(step, error, page=None, selector_key=None, selector=None,
           request=None, now=None, base=FAILURES):
    """Write one failure folder and return its path. Never raises: a broken
    recorder must not hide the failure it was recording."""
    now = now or datetime.now(timezone.utc)
    safe_step = re.sub(r"[^a-z0-9_]", "_", step.lower())
    folder = base / f"{now:%Y%m%dT%H%M%SZ}_{safe_step}"
    files = []
    try:
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o700)
        if page is not None:
            try:
                page.screenshot(path=str(folder / "screenshot.png"), full_page=True)
                files.append("screenshot.png")
            except Exception:
                pass
            try:
                (folder / "page.html").write_text(page.content())
                files.append("page.html")
            except Exception:
                pass
        url = None
        try:
            url = page.url if page is not None else None
        except Exception:
            pass
        data = {
            "schema": SCHEMA,
            "time": now.isoformat(timespec="seconds"),
            "step": step,
            "selector_key": selector_key,
            "selector": selector,
            "error_type": type(error).__name__,
            "error": str(error),
            "url": url,
            "request": request,
            "files": files,
        }
        (folder / "failure.json").write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        print(f"Could not save the Evallo failure record: {exc}", flush=True)
    return folder


def prune(base=FAILURES, keep_days=KEEP_DAYS, now=None):
    """Delete failure folders older than keep_days, by the folder's own age.
    Returns how many went."""
    if not base.exists():
        return 0
    cutoff = (now or time.time()) - keep_days * 86400
    gone = 0
    for folder in base.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            gone += 1
    return gone
