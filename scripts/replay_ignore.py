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
the Claude API, as every live turn does: about 250 Haiku calls.

Oct 30 2026: the wake turns from Oct 12 to 30 were added as WAKE, labelled by
him, each at the tier it ran under and with the real gap since her last reply,
because most false wakes are guests and the gap note changes what she reads as
one conversation. Live wording: silent on that room talk 30 of 81, 43 of 63 on
the older set. The wording that replaced it: 60 of 81 and 61 of 63, answering
34 of 39 meant for her against 37 before.
"""
import importlib.util, os, sqlite3, sys
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import anthropic
import prompts
import brain
from config import MODEL_A, DB_PATH

if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
    cand.apply(prompts, brain)

RUNS = int(os.environ.get("RUNS", 3))
# By config, not by ROOT: a checkout in a worktree has no data/ of its own.
db = sqlite3.connect(DB_PATH); db.row_factory = sqlite3.Row

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
# Wake turns, Oct 12 to 30 2026: (timing_log id, tier it ran at), all opened by
# the wake word. Most are his roommates; an unverified voice runs as a guest.
HIM, GUEST = "hokage", "genin"
WAKE_SILENT = [(615, HIM), (616, HIM), (617, HIM), (619, HIM), (623, GUEST),
               (624, GUEST), (627, GUEST), (628, GUEST), (630, GUEST), (634, GUEST),
               (635, GUEST), (636, HIM), (637, HIM), (640, HIM), (641, GUEST),
               (646, GUEST), (648, GUEST), (651, GUEST), (657, HIM), (659, HIM),
               (660, GUEST), (661, HIM), (662, GUEST), (663, GUEST), (664, HIM),
               (667, HIM), (670, HIM),
               # a wake followed by filler alone: he chose silence, Oct 30
               (625, GUEST), (643, GUEST), (647, HIM), (655, HIM)]
WAKE_ANSWER = [(613, HIM), (668, HIM), (672, HIM), (676, HIM), (684, HIM),
               # a short line put to her, even misheard, still gets an answer
               (649, HIM)]


def timing_case(tid):
    row = db.execute("select * from timing_log where id=?", (tid,)).fetchone()
    prev = db.execute("select transcript, response from timing_log where id<? and "
                      "response is not null order by id desc limit 1", (tid,)).fetchone()
    hist = [{"role": "user", "content": prev["transcript"]},
            {"role": "assistant", "content": prev["response"]}] if prev else []
    return hist + [{"role": "user", "content": row["transcript"]}], row["turn_type"] == "initial"


def wake_case(tid):
    """A timing_case with the gap since her last reply, as the live turn had it.

    The gap note reads the clock, so the real gap is replayed as a last reply
    that far before now. Without it, a question asked hours after some room talk
    reads as part of that talk, and she ignores it. The turn before is the last
    one she spoke in: a silent one has no reply to send."""
    row = db.execute("select * from timing_log where id=?", (tid,)).fetchone()
    prev = db.execute("select transcript, response, created_at from timing_log where id<? "
                      "and response is not null and response != '' order by id desc limit 1",
                      (tid,)).fetchone()
    this = [{"role": "user", "content": row["transcript"]}]
    if prev is None:
        return this, None
    gap = datetime.fromisoformat(row["created_at"]) - datetime.fromisoformat(prev["created_at"])
    return ([{"role": "user", "content": prev["transcript"]},
             {"role": "assistant", "content": prev["response"]}] + this,
            datetime.now() - gap)


def history_case(uid):
    rows = db.execute("select role, content from conversation_history where id<=? "
                      "order by id desc limit 3", (uid,)).fetchall()[::-1]
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# (name, history, named, should be silent, tier, last reply). The older cases
# keep the setup they were measured with: his tier, no gap note.
cases = []
for t in SILENT:
    h, named = timing_case(t); cases.append((f"t{t}", h, named, True, HIM, None))
for t in ANSWER:
    h, named = timing_case(t); cases.append((f"t{t}", h, named, False, HIM, None))
for uid, named, silent in TODAY:
    cases.append((f"h{uid}", history_case(uid), named, silent, HIM, None))
for i, (pu, pa, u, named) in enumerate(TODAY_ANSWER_TEXT):
    cases.append((f"a{i}", [{"role": "user", "content": pu}, {"role": "assistant", "content": pa},
                            {"role": "user", "content": u}], named, False, HIM, None))
for (t, tier), silent in [(c, True) for c in WAKE_SILENT] + [(c, False) for c in WAKE_ANSWER]:
    h, last_reply = wake_case(t); cases.append((f"w{t}", h, True, silent, tier, last_reply))

client = anthropic.Anthropic()


def run(case):
    name, hist, named, silent, tier, last_reply = case
    req = brain.build_request(hist[-1]["content"], "voice", tier, MODEL_A, hist, [],
                              log_recall=False, named=named, followup=not named,
                              last_reply=last_reply)
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
    if name.startswith("w"):
        score["wake_silent_right" if silent else "wake_answer_right"] += right
        score["wake_silent_total" if silent else "wake_answer_total"] += len(out)
    flag = "" if right == len(out) else "  <-- "
    want = "SILENT" if silent else "ANSWER"
    print(f"{name:6} {'named' if named else 'follow':6} want {want}: {right}/{len(out)}{flag}"
          f"{'' if right == len(out) else [o for o in out if o[0] != silent][0][1:]}")
print(f"\nstayed silent when not for her: {score['silent_right']}/{score['silent_total']}"
      f"   answered when for her: {score['answer_right']}/{score['answer_total']}")
print(f"  of which wake turns since Oct 12: silent {score['wake_silent_right']}/"
      f"{score['wake_silent_total']}, answered {score['wake_answer_right']}/"
      f"{score['wake_answer_total']}")
