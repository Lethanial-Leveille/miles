"""Tests for scripts/loop using stub codex and claude programs. No model is called."""
import importlib.machinery, importlib.util, json, os, stat, subprocess, tempfile, unittest
from pathlib import Path

LOOP = Path(__file__).resolve().parents[1] / "loop"
loader = importlib.machinery.SourceFileLoader("loop", str(LOOP))
spec = importlib.util.spec_from_loader("loop", loader)
loop = importlib.util.module_from_spec(spec)
loader.exec_module(loop)


class Rules(unittest.TestCase):
    def test_default_reviewer_by_tier(self):
        self.assertEqual(loop.default_reviewer("small"), "claude")
        self.assertEqual(loop.default_reviewer("medium"), "claude")
        self.assertEqual(loop.default_reviewer("large"), "codex")

    def test_blocker_without_real_harm_is_downgraded(self):
        self.assertEqual(loop.normalize({"rating": "Blocker", "harm": "none"})["rating"], "Should fix")
        self.assertEqual(loop.normalize({"rating": "Blocker", "harm": "data_loss"})["rating"], "Blocker")

    def test_ask_only_when_fixes_differ_visibly(self):
        a = loop.normalize({"rating": "Should fix", "obvious_fix": False, "changes_visible_behavior": True})
        b = loop.normalize({"rating": "Should fix", "obvious_fix": False, "changes_visible_behavior": False})
        n = loop.normalize({"rating": "Note"})
        fix, ask, defer = loop.partition([a, b, n])
        self.assertEqual((fix, ask, defer), ([b], [a], [n]))


def stub(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class EndToEnd(unittest.TestCase):
    def run_loop(self, round1, round2, args=("--reviewer", "codex")):
        d = Path(tempfile.mkdtemp())
        repo = d / "repo"; repo.mkdir()
        g = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
        g("init", "-b", "main"); g("config", "user.email", "a@example.com"); g("config", "user.name", "Alex")
        (repo / "a.py").write_text("x = 1\n"); g("add", "."); g("commit", "-m", "base")
        g("checkout", "-b", "feat"); (repo / "a.py").write_text("x = 2\n"); g("commit", "-am", "change")
        stub(d / "usage.py", "print('ok')\n")
        # Stub codex: first call returns round1, later calls round2. Writes the -o file.
        stub(d / "codex", f"""import sys, json, pathlib
a = sys.argv; out = a[a.index('-o') + 1]
marker = pathlib.Path({str(d / 'seen')!r})
first = not marker.exists(); marker.write_text('1')
pathlib.Path(out).write_text(json.dumps({round1!r} if first else {round2!r}) if False else ({json.dumps(json.dumps(round1))} if first else {json.dumps(json.dumps(round2))}))
""")
        stub(d / "claude", f"""import sys, os, json, pathlib
log = pathlib.Path({str(d / 'claude_calls.jsonl')!r})
with log.open('a') as fh:
    fh.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
if 'opus' in sys.argv:
    sys.stdin.read()
    marker = pathlib.Path({str(d / 'seen_claude')!r})
    first = not marker.exists(); marker.write_text('1')
    print({json.dumps(json.dumps(round1))} if first else {json.dumps(json.dumps(round2))})
else:
    pathlib.Path({str(repo / 'a.py')!r}).write_text('x = 3\\n')
""")
        env = dict(os.environ, LOOP_CODEX=str(d / "codex"), LOOP_CLAUDE=str(d / "claude"), LOOP_USAGE=str(d / "usage.py"))
        r = subprocess.run([str(LOOP), "--base", "main", "--tests", "true", *args], cwd=repo, env=env, capture_output=True, text=True)
        self.d = d
        return r, repo

    def test_fixed_blocker_finishes_clean(self):
        f1 = {"findings": [{"id": "F1", "rating": "Blocker", "harm": "data_loss", "file": "a.py", "line": 1, "summary": "loses rows"},
                           {"id": "F2", "rating": "Note", "file": "a.py", "line": 1, "summary": "naming"}]}
        r, repo = self.run_loop(f1, {"results": [{"id": "F1", "status": "resolved"}], "introduced": []})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("naming", (repo / "docs/KNOWN_LIMITS.md").read_text())
        self.assertIn("## Fixed", r.stdout)

    def test_surviving_blocker_asks(self):
        f1 = {"findings": [{"id": "F1", "rating": "Blocker", "harm": "wrong_contact", "file": "a.py", "line": 1, "summary": "texts wrong person"}]}
        r, _ = self.run_loop(f1, {"results": [{"id": "F1", "status": "unresolved"}], "introduced": []})
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("Blocker still open", r.stdout)

    def test_claude_reviewer_is_fresh_read_only_and_outside_the_repo(self):
        f1 = {"findings": [{"id": "F1", "rating": "Blocker", "harm": "data_loss", "file": "a.py", "line": 1, "summary": "loses rows"}]}
        r, repo = self.run_loop(f1, {"results": [{"id": "F1", "status": "resolved"}], "introduced": []},
                                args=("--reviewer", "claude"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = [json.loads(x) for x in (self.d / "claude_calls.jsonl").read_text().splitlines()]
        reviews = [c for c in calls if "opus" in c["argv"]]
        fixes = [c for c in calls if "sonnet" in c["argv"]]
        self.assertEqual(len(reviews), 2)  # round one and the verify round
        self.assertEqual(len(fixes), 1)
        for c in reviews:
            self.assertIn("--no-session-persistence", c["argv"])
            self.assertIn("Read,Grep,Glob", c["argv"])
            self.assertNotIn("acceptEdits", c["argv"])
            self.assertNotIn("--resume", c["argv"])
            self.assertNotEqual(os.path.realpath(c["cwd"]), os.path.realpath(str(repo)))
        self.assertFalse((self.d / "seen").exists())  # codex was never called
        self.assertIn("fresh Opus session", r.stdout)

    def test_tier_picks_the_reviewer_and_missing_choice_stops(self):
        f1 = {"findings": []}
        r, _ = self.run_loop(f1, {"results": [], "introduced": []}, args=("--tier", "medium"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("fresh Opus session", r.stdout)
        r, _ = self.run_loop(f1, {"results": [], "introduced": []}, args=("--tier", "large"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Codex, Astra High", r.stdout)
        r, _ = self.run_loop(f1, {"results": [], "introduced": []}, args=())
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Say who reviews", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
