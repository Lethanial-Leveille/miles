"""One time Evallo login. Opens your real Google Chrome in a throwaway profile,
you log in by hand, and the Evallo session is saved to data/evallo_state.json.

Why real Chrome and not Playwright's own browser: Google refuses "Sign in with
Google" in any browser Playwright launches ("This browser or app may not be
secure"), because that browser announces it is automated. So this starts
Chrome as an ordinary program, with no automation switches, and only after you
have logged in does Playwright attach to it to copy the session out. Seen and
fixed Nov 23 2026.

Your password never passes through this script: you type it into Google's own
page. What gets saved is the session Evallo hands back afterwards. That file is
still a credential, since anyone holding it is logged in as you, so it is
readable by you alone and ignored by git. It lasts 30 days; the nightly
canary warns three days before it runs out. The throwaway Chrome profile, which
also holds your Google login, is deleted once the session is saved.

The Pi has no screen, so run this on the Mac and copy the file over:

    .venv-evallo/bin/python scripts/evallo_login.py
    scp data/evallo_state.json theycallmelee@100.99.248.127:~/miles/data/

It saves on its own once you are past the sign in page, so there is nothing to press.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "evallo_state.json"
PROFILE = ROOT / "data" / "evallo_login_profile"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
EVALLO = "https://app.evallo.org"
CALENDAR = f"{EVALLO}/calendar"
PORT = 9223
LOGIN_WINDOW_S = 5 * 60


def _evallo_tab_url():
    """The URL of Chrome's Evallo tab, read from Chrome's own tab list. Nothing
    attaches to Chrome until the login is done, so nothing about the login
    looks automated."""
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/json/list", timeout=2) as r:
            tabs = json.load(r)
    except Exception:
        return None
    for tab in tabs:
        if tab.get("type") == "page" and tab.get("url", "").startswith("https://app.evallo.org"):
            return tab["url"]
    return None


def evallo_only(state):
    """Keep Evallo's cookies and storage and drop everything else.

    The Chrome profile is signed in to Google, so the raw cookies include the
    whole Google account's, which would make this file a key to Gmail and Drive
    as well. Seen in the first save on Nov 23 2026; that file was deleted."""
    def ours(domain):
        return domain.lstrip(".").endswith("evallo.org")
    return {
        "cookies": [c for c in state["cookies"] if ours(c["domain"])],
        "origins": [o for o in state["origins"]
                    if ours(o["origin"].split("//", 1)[-1].split("/")[0])],
    }


def describe(state):
    """Names only, never values, plus when the login runs out."""
    import evallo_session
    names = [i["name"] for o in state["origins"] for i in o["localStorage"]]
    expires = evallo_session.token_expiry(state)
    return (f"Saved: {', '.join(names)}\n"
            f"The login lasts until {expires:%a %b %-d %-I:%M %p}." if expires
            else f"Saved: {', '.join(names)}\nWARNING: no Evallo token found.")


def _logged_in(url):
    # After logging in Evallo lands on its home page, not the calendar, so any
    # Evallo page that is not the sign in page counts.
    return url is not None and "signin" not in url and "signup" not in url


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    chrome = subprocess.Popen([
        CHROME, f"--user-data-dir={PROFILE}", f"--remote-debugging-port={PORT}",
        "--no-first-run", "--no-default-browser-check", CALENDAR,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Log in to Evallo in the Chrome window that just opened. "
          "This saves by itself and closes the window.", flush=True)
    try:
        # Waiting on a person, not on a page, so a plain poll is right here.
        # Logged in has to hold three times running: the page sits on its first URL
        # for an instant before Evallo redirects to sign in.
        seen = 0
        deadline = time.monotonic() + LOGIN_WINDOW_S
        while time.monotonic() < deadline:
            seen = seen + 1 if _logged_in(_evallo_tab_url()) else 0
            if seen >= 3:
                break
            time.sleep(2)
        else:
            print("Gave up after 5 minutes without a login. Nothing saved.")
            return 1
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{PORT}")
            context = browser.contexts[0]
            page = next(pg for pg in context.pages if pg.url.startswith(EVALLO))
            # The login is a token in localStorage ("evalloToken"), sent as
            # the Authorization header. storage_state() leaves localStorage out
            # when attached to a Chrome that was already running, so the first
            # two saves held no login at all (Nov 23 2026); it is read here
            # directly instead.
            page.wait_for_function("localStorage.getItem('evalloToken')",
                                   timeout=60_000)
            local = page.evaluate(
                "Object.entries(localStorage).map(([name, value]) => ({name, value}))")
            state = evallo_only({
                "cookies": context.cookies(),
                "origins": [{"origin": EVALLO, "localStorage": local}],
            })
            STATE.write_text(json.dumps(state))
            os.chmod(STATE, 0o600)
            print(describe(state))
            browser.close()
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(PROFILE, ignore_errors=True)
    print(f"Saved the session to {STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
