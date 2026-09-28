#!/usr/bin/env python3
"""Mark which of Nova's replies sounded off, and where.

Each spoken reply is kept in data/spoken as a wav and a txt with the same name
(src/spoken_archive.py).

The usual way is a listening session, heard on the laptop and not the room:

    python3 scripts/label_spoken.py listen

It prints a localhost link. Ctrl+click it in the VS Code terminal; VS Code
forwards the port over Remote SSH and the page opens in the Mac's browser. It
plays each unlabelled reply, newest first, beside the text she was reading.
F fine, O off, R replay, S skip.

The same verdicts from the command line:

    python3 scripts/label_spoken.py list          # the last 15, newest first
    python3 scripts/label_spoken.py list 40
    python3 scripts/label_spoken.py show 20260928T013012_512
    python3 scripts/label_spoken.py mark 20260928T013012_512 off "hitch at the dash"
    python3 scripts/label_spoken.py mark 20260928T013012_512 fine

The note is the valuable part. "off" alone says a reply was wrong; "pause after
maybe" says which of the suspects in the Sep 28 investigation did it.

A labelled reply is pruned only after every unlabelled one, so marking one keeps
it. Marking "fine" keeps it too, which is deliberate: a good reply is the
comparison a fix is judged against.
"""

import csv
import json
import os
import re
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from config import SPOKEN_DIR  # noqa: E402

LABELS = ("off", "fine")
FIELDS = ["file", "label", "note"]
MANIFEST = os.path.join(SPOKEN_DIR, "labels.csv")

# The only names the page may ask for. Anything else, "../.env" included, is
# refused before it is joined to a path: the server hands out files, so the
# name is the whole of its access control.
_STEM = re.compile(r"^\d{8}T\d{6}_\d{3}$")


def _load():
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def _save(rows):
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


def _stem(name):
    """Accept the stem, the wav name, or a path, so a name pasted from anywhere
    works."""
    return os.path.basename(name).removesuffix(".wav").removesuffix(".txt")


def _text(stem):
    try:
        with open(os.path.join(SPOKEN_DIR, stem + ".txt")) as f:
            return f.read().strip()
    except OSError:
        return "(no text)"


def _seconds(stem):
    try:
        with wave.open(os.path.join(SPOKEN_DIR, stem + ".wav")) as w:
            return w.getnframes() / w.getframerate()
    except (OSError, wave.Error):
        return 0.0


def _stems():
    """Every kept reply, newest first."""
    if not os.path.isdir(SPOKEN_DIR):
        return []
    return sorted((f[:-4] for f in os.listdir(SPOKEN_DIR) if f.endswith(".wav")),
                  reverse=True)


def mark(name, label, note=""):
    """Record one verdict. Raises ValueError rather than exiting, because the
    listening page calls this too and a typo must not stop its server."""
    if label not in LABELS:
        raise ValueError(f"Label is one of {', '.join(LABELS)}, not {label!r}.")
    stem = _stem(name)
    if not _STEM.match(stem) or not os.path.exists(os.path.join(SPOKEN_DIR, stem + ".wav")):
        raise ValueError(f"No reply named {stem} in {SPOKEN_DIR}.")
    rows = _load()
    rows[stem + ".wav"] = {"file": stem + ".wav", "label": label, "note": note.strip()}
    _save(rows)
    return stem


def cmd_list(count=15):
    stems = _stems()[:count]
    if not stems:
        print(f"Nothing yet in {SPOKEN_DIR}.")
        return
    labels = _load()
    for stem in stems:
        row = labels.get(stem + ".wav")
        verdict = f"[{row['label']}] " if row else ""
        first = _text(stem).splitlines()[0]
        print(f"{stem}  {_seconds(stem):5.1f}s  {verdict}{first[:70]}")


def cmd_show(name):
    stem = _stem(name)
    print(os.path.join(SPOKEN_DIR, stem + ".wav"))
    print(_text(stem))
    row = _load().get(stem + ".wav")
    if row:
        print(f"\nlabelled {row['label']}: {row['note']}")


def cmd_mark(name, label, note=""):
    try:
        stem = mark(name, label, note)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"{stem}: {label}" + (f", {note}" if note else ""))


# ── Listening session ──

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nova Listening</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --dim:#6e6e73; --line:#e3e1dc;
          --ok:#2f7d4f; --bad:#b3403a; --accent:#3a5ba0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161618; --card:#1f1f22; --ink:#ececef; --dim:#9a9aa1; --line:#2e2e33;
            --ok:#5fbf85; --bad:#e2766f; --accent:#8aa6e6; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.5 -apple-system, system-ui, sans-serif; }
  main { max-width:680px; margin:0 auto; padding:32px 16px; }
  .top { display:flex; justify-content:space-between; color:var(--dim); font-size:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:24px; margin-top:12px; }
  .when { color:var(--dim); font-size:14px; }
  .text { font-size:19px; margin:12px 0 20px; white-space:pre-wrap; }
  .gap { display:block; color:var(--accent); font-size:14px; margin:6px 0; }
  audio { width:100%; }
  input { width:100%; padding:10px 12px; margin-top:16px; border-radius:10px;
          border:1px solid var(--line); background:var(--bg); color:var(--ink); font:inherit; }
  .row { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
  button { flex:1; min-width:120px; padding:12px; border-radius:10px; border:1px solid var(--line);
           background:var(--bg); color:var(--ink); font:inherit; cursor:pointer; }
  button.fine { border-color:var(--ok); color:var(--ok); }
  button.off  { border-color:var(--bad); color:var(--bad); }
  kbd { font-size:12px; color:var(--dim); }
  .done { text-align:center; padding:48px 24px; }
  .start { font-size:18px; padding:16px; }
</style></head><body><main>
<div class="top"><span>Did Nova sound right?</span><span id="count"></span></div>
<div id="view"></div>
</main><script>
let queue = [], i = 0, started = false;
const view = document.getElementById('view');
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function stamp(stem) {
  const m = stem.match(/^(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})/);
  return new Date(+m[1], m[2]-1, +m[3], +m[4], +m[5]).toLocaleString([], {weekday:'short',
    month:'short', day:'numeric', hour:'numeric', minute:'2-digit'});
}

function render() {
  document.getElementById('count').textContent = queue.length ? `${Math.min(i+1, queue.length)} of ${queue.length}` : '';
  if (!queue.length) { view.innerHTML = '<div class="card done">Nothing to label yet. Talk to her, then reload.</div>'; return; }
  if (!started) {
    view.innerHTML = `<div class="card done"><p>${queue.length} replies waiting.</p>
      <button class="start" onclick="started=true;render()">Start listening</button></div>`;
    return;
  }
  if (i >= queue.length) { view.innerHTML = '<div class="card done">All done. Thank you.</div>'; return; }
  const r = queue[i];
  const text = esc(r.text).replace(/\\[(\\d+)ms before this piece\\]/g, '<span class="gap">$1ms of silence</span>');
  view.innerHTML = `<div class="card">
    <div class="when">${stamp(r.stem)} · ${r.seconds.toFixed(1)}s</div>
    <div class="text">${text}</div>
    <audio id="a" controls autoplay src="audio/${r.stem}.wav"></audio>
    <input id="note" placeholder="What sounded off? Where? (optional)">
    <div class="row">
      <button class="fine" onclick="label('fine')">Sounds fine <kbd>F</kbd></button>
      <button class="off" onclick="label('off')">Sounds off <kbd>O</kbd></button>
      <button onclick="replay()">Replay <kbd>R</kbd></button>
      <button onclick="i++;render()">Skip <kbd>S</kbd></button>
    </div></div>`;
}

function replay() { const a = document.getElementById('a'); a.currentTime = 0; a.play(); }

async function label(verdict) {
  const r = queue[i], note = document.getElementById('note').value;
  const res = await fetch('label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({stem:r.stem, label:verdict, note})});
  if (!res.ok) { alert(await res.text()); return; }
  i++; render();
}

document.addEventListener('keydown', e => {
  if (!started || i >= queue.length) return;
  if (document.activeElement && document.activeElement.id === 'note') {
    if (e.key === 'Escape') document.activeElement.blur();
    if (e.key === 'Enter') label('off');
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 'f') label('fine');
  else if (k === 'o') { const n = document.getElementById('note'); n.value ? label('off') : n.focus(); e.preventDefault(); }
  else if (k === 'r') replay();
  else if (k === 's') { i++; render(); }
});

fetch('replies').then(r => r.json()).then(q => { queue = q; render(); });
</script></body></html>
"""


def _unlabelled():
    labels = _load()
    return [{"stem": s, "text": _text(s), "seconds": _seconds(s)}
            for s in _stems() if s + ".wav" not in labels]


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/replies":
            return self._send(200, json.dumps(_unlabelled()).encode(), "application/json")
        if self.path.startswith("/audio/") and self.path.endswith(".wav"):
            stem = self.path[len("/audio/"):-len(".wav")]
            if _STEM.match(stem):
                try:
                    with open(os.path.join(SPOKEN_DIR, stem + ".wav"), "rb") as f:
                        return self._send(200, f.read(), "audio/wav")
                except OSError:
                    pass
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/label":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            mark(str(body["stem"]), str(body["label"]), str(body.get("note", "")))
        except (ValueError, KeyError, TypeError) as exc:
            return self._send(400, str(exc).encode(), "text/plain")
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args):
        pass   # one line per request would bury the link he needs to click


def make_server(port=8765):
    # 127.0.0.1 only. These are recordings made in his room; the SSH tunnel
    # VS Code forwards through is the only way in.
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


def cmd_listen(port=8765):
    waiting = len(_unlabelled())
    try:
        server = make_server(port)
    except OSError as exc:
        sys.exit(f"Could not open port {port} ({exc}). Try: listen {port + 1}")
    print(f"{waiting} unlabelled replies.")
    print(f"Open http://localhost:{port}  (Ctrl+click in the VS Code terminal)")
    print("Ctrl+C here when done. Verdicts are saved as you go.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv):
    if not argv or argv[0] not in ("list", "show", "mark", "listen"):
        sys.exit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "list":
        cmd_list(int(rest[0]) if rest else 15)
    elif command == "listen":
        cmd_listen(int(rest[0]) if rest else 8765)
    elif command == "show" and len(rest) == 1:
        cmd_show(rest[0])
    elif command == "mark" and len(rest) in (2, 3):
        cmd_mark(*rest)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
