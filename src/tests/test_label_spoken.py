"""The listening page labels replies, and hands out nothing but replies.
Runs a real server on a free port on 127.0.0.1, over files under tmp_path."""

import importlib.util
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

import spoken_archive

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "label_spoken.py")
_spec = importlib.util.spec_from_file_location("label_spoken", _PATH)
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "SPOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(ls, "MANIFEST", str(tmp_path / "labels.csv"))
    (tmp_path.parent / "secret.txt").write_text("not yours")
    path = spoken_archive.save_reply([("Got it.", b"\x01\x00" * 2205, 0)],
                                     directory=str(tmp_path))
    server = ls.make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", os.path.basename(path)[:-4]
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status


def test_a_verdict_is_saved_and_the_reply_leaves_the_queue(served):
    base, stem = served
    queue = json.loads(_get(base + "/replies")[1])
    assert [r["stem"] for r in queue] == [stem] and queue[0]["text"] == "Got it."
    assert _get(f"{base}/audio/{stem}.wav")[1][:4] == b"RIFF"

    assert _post(base + "/label", {"stem": stem, "label": "off", "note": "flat"}) == 200
    assert ls._load()[stem + ".wav"] == {"file": stem + ".wav", "label": "off", "note": "flat"}
    assert json.loads(_get(base + "/replies")[1]) == []


@pytest.mark.parametrize("path", ["/audio/../secret.txt.wav", "/audio/..%2Fsecret.wav",
                                  "/audio/labels.wav", "/../secret.txt"])
def test_nothing_but_a_reply_is_handed_out(served, path):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base + path)
    assert err.value.code == 404


def test_a_bad_verdict_is_refused_without_stopping_the_server(served):
    base, stem = served
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base + "/label", {"stem": stem, "label": "meh"})
    assert err.value.code == 400
    assert _get(base + "/replies")[0] == 200
