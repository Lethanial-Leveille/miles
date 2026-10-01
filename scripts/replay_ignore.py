#!/usr/bin/env python3
"""Replay his past turns through the real request and count when she stays silent.

Run before changing anything that decides whether a turn is for her: the
ignore tool, WHAT_REACHES_YOU, NOT_ADDRESSED_TO_YOU, or the notes brain adds to
a voice turn. Each case is a real turn (a timing_log row, or a
conversation_history id), sent with the turn before it through
brain.build_request with the live prompt and tools, RUNS times (default 3).

    python3 scripts/replay_ignore.py                  # the code as it is
    python3 scripts/replay_ignore.py candidate.py     # with a candidate patched in

A candidate is a module with apply(prompts, brain) that patches the constants.
Oct 1 2026: 16 of 63 silent on room talk before the change, 44 after, with
22 of 24 answered. The cases were used to choose that wording, so a new
wording should also be tried on turns they do not include. Sends his turns to
the Claude API, as every live turn does: about 100 Haiku calls.
"""
import importlib.util, os, sqlite3, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import anthropic
import prompts
import brain
from config import MODEL_A

if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
    cand.apply(prompts, brain)

RUNS = int(os.environ.get("RUNS", 3))
db = sqlite3.connect(os.path.join(ROOT, "data", "miles.db")); db.row_factory = sqlite3.Row

# (timing_log id, should she stay silent)
SILENT = [511, 421, 419, 417, 413, 410, 408, 401, 363, 359, 358, 356, 293, 290,
          209, 187, 175, 161]
ANSWER = [407, 406, 405, 404, 337]
# today's, from conversation_history: (user id, named, silent)
TODAY = [(1555, True, True), (1557, False, True), (1559, False, True)]
TODAY_ANSWER_TEXT = [  # (previous user, previous reply, this, named)
    ("What are the chemical components of an apple?",
     "Apples are mostly water and carbohydrates, with the carbs split between fiber and natural sugars like fructose.",
     "Can you tell me about bananas?", False),
    ("Can you tell me about bananas?",
     "Bananas are about seventy three percent water and twenty three percent carbs, mostly natural sugars and some fiber.",
     "Actually, never mind oranges.", False),
    ("If I wanted to make guac, what could I put in there?",
     "The base is avocado, lime juice, and salt. Most people add diced onion, cilantro, and tomato.",
     "If I don't have garlic, is garlic powder okay?", False),
]


def timing_case(tid):
    row = db.execute("select * from timing_log where id=?", (tid,)).fetchone()
    prev = db.execute("select transcript, response from timing_log where id<? and "
                      "response is not null order by id desc limit 1", (tid,)).fetchone()
    hist = [{"role": "user", "content": prev["transcript"]},
            {"role": "assistant", "content": prev["response"]}] if prev else []
    return hist + [{"role": "user", "content": row["transcript"]}], row["turn_type"] == "initial"


def history_case(uid):
    rows = db.execute("select role, content from conversation_history where id<=? "
                      "order by id desc limit 3", (uid,)).fetchall()[::-1]
    return [{"role": r["role"], "content": r["content"]} for r in rows]


cases = []
for t in SILENT:
    h, named = timing_case(t); cases.append((f"t{t}", h, named, True))
for t in ANSWER:
    h, named = timing_case(t); cases.append((f"t{t}", h, named, False))
for uid, named, silent in TODAY:
    cases.append((f"h{uid}", history_case(uid), named, silent))
for i, (pu, pa, u, named) in enumerate(TODAY_ANSWER_TEXT):
    cases.append((f"a{i}", [{"role": "user", "content": pu}, {"role": "assistant", "content": pa},
                            {"role": "user", "content": u}], named, False))

client = anthropic.Anthropic()


def run(case):
    name, hist, named, silent = case
    req = brain.build_request(hist[-1]["content"], "voice", "hokage", MODEL_A, hist, [],
                              log_recall=False, named=named, followup=not named)
    out = []
    for _ in range(RUNS):
        msg = client.messages.create(**brain.first_call_kwargs(req))
        tools = [b.name for b in msg.content if b.type == "tool_use"]
        text = " ".join(b.text for b in msg.content if b.type == "text").strip()
        silent_now = ("ignore" in tools or "dismiss" in tools) and not text
        out.append((silent_now, tools, text[:90]))
    return name, named, silent, out


with ThreadPoolExecutor(6) as ex:
    results = list(ex.map(run, cases))

score = Counter()
for name, named, silent, out in results:
    right = sum(o[0] == silent for o in out)
    score["silent_right" if silent else "answer_right"] += right
    score["silent_total" if silent else "answer_total"] += len(out)
    flag = "" if right == len(out) else "  <-- "
    want = "SILENT" if silent else "ANSWER"
    print(f"{name:6} {'named' if named else 'follow':6} want {want}: {right}/{len(out)}{flag}"
          f"{'' if right == len(out) else [o for o in out if o[0] != silent][0][1:]}")
print(f"\nstayed silent when not for her: {score['silent_right']}/{score['silent_total']}"
      f"   answered when for her: {score['answer_right']}/{score['answer_total']}")
