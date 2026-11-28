"""The request intent step: decide what a turn may change before anything runs.

Before this, the model acted first and word lists in code checked afterward, so
any phrasing the lists did not know ("in four days", "cancel reading") got
through. Now a turn that could change something first gets one small forced
schema call whose only output is an intent: an action, a target and the timing
words. Code then validates that intent against the database and the committed
words, and only tool calls that match it may run (see IntentGate and
TurnSafety.preflight). A turn that fails the cheap check below never sees a
mutating tool at all, so a miss changes nothing.

The approved rules are in docs/work/tasks-intent-contract.md.
"""

import inspect
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import database
import task_scheduling
import timing
from config import MODEL_A
from tools import permits, registry

# ── Which turns could change something ──

# Wide on purpose. A false positive costs one small model call and a "none"
# answer. A false negative costs the request, because the turn then has no
# mutating tool, so the cost of a miss is a refusal and never a wrong change.
_COULD_CHANGE = re.compile(
    r"\b(?:add|adds|adding|create|put|new|set|start|cancel|cancell?ed|stop|archive|delete|remove|clear|drop|skip|"
    r"complete|completed|finish|finished|done|mark|check|cross|tick|log|save|saved|remember|forget|note|"
    r"undo|rename|move|reschedule|schedule|book|plan|text|send|message|imessage|email|"
    r"remind|reminder|reminders|timer|timers|alarm|task|tasks|todo|to do|list|lists|"
    r"calendar|event|events|appointment|meeting|session|sessions|routine|workout|lower|access|"
    r"approve|memory|memories|pending|i took|i got|i bought|i paid|i called|i did|i had|"
    r"i need|need to|got the|bought the|took my|stretch|vitamins?)\b", re.I)

_COMPOUND = re.compile(
    r"\b(?:and|then|also|plus)\s+(?:add|create|archive|complete|mark|stop|cancel|remind|set|put|check|skip)\b", re.I)


def could_change(utterance, choice_pending=False):
    """True when the turn might ask for a change. Cheap and offline."""
    return bool(choice_pending or _COULD_CHANGE.search(utterance or ""))


# ── The intent ──

# no_change is the model saying the message asks for nothing to be changed:
# questions, reads and chat. There is deliberately no "other change" action.
# Calendar writes, remembered facts and every other mutating tool outside the
# four kinds below keep their own confirm first flow and are not reachable from
# this step as an action.
ACTIONS = ("task_add", "task_complete", "task_archive", "task_skip_today",
           "list_add", "list_check", "timer", "reminder", "no_change")

# The one registered tool each action may call.
ACTION_TOOL = {
    "task_add": "task_add", "task_complete": "task_complete", "task_archive": "archive_task",
    "task_skip_today": "task_skip_today", "list_add": "list_add", "list_check": "list_check",
    "timer": "set_timer", "reminder": "set_reminder",
}
TASK_ACTIONS = {"task_add", "task_complete", "task_archive", "task_skip_today"}
# Actions whose outcome is spoken by code from the tool result, not by the model.
# Every covered change (task, list, timer, reminder) is in it, so no model
# sentence about a stored change is ever spoken in place of the result.
AUTHORITATIVE = TASK_ACTIONS | {"list_add", "list_check", "timer", "reminder"}
# The tools the intent step governs. Any other mutating tool is outside it.
COVERED_TOOLS = frozenset(ACTION_TOOL.values())

INTENT_TOOL = {
    "name": "record_intent",
    "description": "Record the one change the user's message asks for. Nothing runs from this.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS),
                       "description": "task_add, task_complete, task_archive (stop tracking or cancel a "
                                      "task), task_skip_today, list_add, list_check, timer, reminder, or "
                                      "no_change when the message asks for nothing to be changed: a "
                                      "question, a request to read or look something up, or chat."},
            "target": {"type": "string",
                       "description": "The task, list item or thing named, in the user's words, without "
                                      "timing words. Empty when none."},
            "timing": {"type": "string",
                       "description": "The exact words that say when or how often, copied from the "
                                      "message, like 'every day' or 'in four days'. Empty when none."},
            "more_than_one_change": {"type": "boolean",
                                     "description": "True when the message asks for more than one change."},
        },
        "required": ["action", "target", "timing", "more_than_one_change"],
    },
}

_SYSTEM = (
    "You read one message from the user and record what change, if any, it asks his assistant to make "
    "to a task, a list, a timer or a reminder. Call record_intent. You never carry anything out. Use "
    "no_change for chat, questions, requests to look something up and any change that is not a task, "
    "list, timer or reminder. Pick one action only. If the message asks for two changes, set "
    "more_than_one_change. Copy the timing words exactly as said and never invent any.")


@dataclass(frozen=True)
class Intent:
    action: str
    target: str = ""
    timing: str = ""
    more: bool = False


_REQUIRED = {"action": str, "target": str, "timing": str, "more_than_one_change": bool}


def parse_intent(message):
    """The Intent in a forced tool call reply, or None when it is missing,
    malformed or conflicting. There must be exactly one record_intent block with
    every required field present and of the right type; a second record, even an
    identical one, is a reply that cannot be trusted."""
    records = [block for block in getattr(message, "content", None) or []
               if getattr(block, "type", None) == "tool_use" and block.name == "record_intent"]
    if len(records) != 1:
        return None
    data = records[0].input
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        return None
    for key, kind in _REQUIRED.items():
        if key not in data or type(data[key]) is not kind:
            return None
    return Intent(data["action"], data["target"].strip(), data["timing"].strip(),
                  data["more_than_one_change"])


async def request_intent(client, utterance, pending_title=None):
    """One forced schema call with no mutating tool offered. Its time is recorded
    in the turn's timing record as intent_ms, so the cost on task turns shows up.

    Returns None when the call fails or answers outside the schema. The caller
    treats that as a refusal, because nothing may run on a guess."""
    system = _SYSTEM
    if pending_title:
        system += (f" The assistant just asked whether to skip {pending_title} for today or stop "
                   f"it for good. A reply like that is about {pending_title}: put it in target.")
    started = time.monotonic()
    try:
        message = client.messages.create(
            model=MODEL_A, max_tokens=300, system=system,
            messages=[{"role": "user", "content": utterance}],
            tools=[INTENT_TOOL], tool_choice={"type": "tool", "name": "record_intent"})
        if inspect.isawaitable(message):
            message = await message
        return parse_intent(message)
    except Exception as exc:
        print(f"Intent call failed: {type(exc).__name__}", flush=True)
        return None
    finally:
        timing.note_intent((time.monotonic() - started) * 1000.0)


# ── Validation ──

def _words(text):
    return re.sub(r"[^a-z0-9']+", " ", (text or "").casefold()).strip()


_STOP_FOR_GOOD = re.compile(r"\b(?:archive|delete|for good|forever|permanently|anymore|no longer|get rid|"
                            r"stop (?:tracking|doing|it)|remove)\b", re.I)
_TODAY = re.compile(r"\b(?:today|tonight|this (?:morning|afternoon|evening)|skip)\b", re.I)

@dataclass
class Verdict:
    kind: str                  # "none" (no change asked), "act", "refuse" or "ask"
    text: str = ""
    gate: "IntentGate | None" = None
    choice: tuple | None = None   # (task id, title) when the question waits for an answer


@dataclass
class IntentGate:
    """What one validated request lets the turn run. Used by TurnSafety.preflight
    on every round, so a later round cannot widen it."""
    tool: str | None = None
    action: str = "no_change"
    args_ok: object = None        # callable(args) -> bool, for tools whose arguments matter
    confirm: bool = False         # a pending approval may be answered this turn
    target_id: int | None = None
    plan: dict = field(default_factory=dict)
    # A flagged turn the model called no_change keeps the mutating tools outside
    # the covered set (calendar, remember and the rest), each under its own
    # confirm first flow and guards. The covered tools stay hidden.
    others: bool = False

    @property
    def authoritative(self):
        return self.action in AUTHORITATIVE

    @property
    def requires_action(self):
        return self.tool is not None or self.confirm

    def offers(self, name):
        return (name == self.tool or (self.confirm and name == "confirm_pending_action")
                or (self.others and name not in COVERED_TOOLS and name != "confirm_pending_action"))

    def permits(self, name, args):
        if not self.offers(name):
            return False
        return self.args_ok is None or name != self.tool or bool(self.args_ok(args))


def closed_gate(confirm=False):
    """No mutating tool at all: the gate for a turn that could not change anything."""
    return IntentGate(confirm=confirm)


def no_change_gate():
    """The gate for a flagged turn the model called no_change: the covered
    tools stay hidden, every other tool is offered as it was before this step."""
    return IntentGate(others=True)


def _refuse(text):
    return Verdict("refuse", f"Nothing was changed. {text}")


def _one_task(target):
    """(row, None) for the single active task the words name, else (None, refusal text)."""
    matches = database.active_tasks_matching(target) if target else []
    if not matches:
        return None, f"No task matches '{target}'."
    if len(matches) > 1:
        return None, (f"More than one task matches '{target}': "
                      f"{', '.join(m[1] for m in matches)}. Which one do you mean?")
    return matches[0], None


def _task_resolves_to(task_id):
    def check(args):
        matches = database.active_tasks_matching(str(args.get("task", "")))
        return len(matches) == 1 and matches[0][0] == task_id
    return check


_NOT_FOR_GOOD = re.compile(
    r"\b(?:not|never|n't|don't|do not|just not|no)\s+(?:\w+\s+){0,2}?"
    r"(?:for good|forever|permanently|anymore|any more|no longer)\b", re.I)


def _cancel_scope(utterance, recurrence, title=""):
    """archive, skip or ask, from the words he used and the kind of task.

    A clear stop request archives. Today or tonight skips only today. A bare
    cancel is ambiguous for a daily task, so it asks, and for a one time task
    its only occurrence is the task, so it archives. The task's own title is
    left out, so a task called "Stop smoking" says nothing about scope, and a
    negated phrase such as "not for good" is read as the opposite of "for good".
    Words that ask for both today only and for good conflict, so they ask."""
    text = re.sub(re.escape(title), " ", utterance, flags=re.I) if title else utterance
    only_today = bool(_NOT_FOR_GOOD.search(text))
    text = _NOT_FOR_GOOD.sub(" ", text)
    good = bool(_STOP_FOR_GOOD.search(text))
    today = only_today or bool(_TODAY.search(text))
    if good and today:
        # "skip" alone is a today word; a stop word beside it is a real conflict.
        return "ask" if recurrence else "archive"
    if good:
        return "archive"
    if today:
        return "skip" if recurrence else "archive"
    if re.search(r"\b(?:stop|quit|drop|scrap)\b", text, re.I):
        return "archive"
    return "ask" if recurrence else "archive"


def _task_add(intent, utterance, today):
    from voice_action_safety import _ADD_REQUEST, _OTHER_ADD, _normalized
    text = " ".join(utterance.split()).strip(" .!?,")
    text = re.sub(r"^please ", "", text, flags=re.I)
    text = re.sub(r" please$", "", text, flags=re.I)
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", text, flags=re.I)
    match = re.fullmatch(_ADD_REQUEST.pattern, text, re.I)
    if not match or _OTHER_ADD.match(text.casefold()):
        return _refuse("Say it like 'add call the bank to my tasks'.")
    title = intent.target
    if not title:
        return _refuse("What should the task be?")
    if intent.timing and _words(intent.timing) not in _words(utterance):
        return _refuse("I could not match the timing to what you said. Please say the task again.")
    intro = "a daily task to " if match[1] else ""
    # "to my tasks" says where, not what, so it is not part of the title or timing.
    rest = re.sub(r" (?:to|on) (?:my|the) (?:tasks|to do list)$", "", match[2], flags=re.I)
    # The start clause he said, copied by the intent step, is what the resolver
    # checks the first day against.
    starts = None
    if task_scheduling._START.search(intent.timing or ""):
        starts = task_scheduling._DAILY.sub(" ", intent.timing).strip()
    plan = task_scheduling.resolve_task_schedule(f"add {intro}{rest}", title=title, today=today,
                                                 starts=starts)
    if plan.outcome != "supported":
        said = intent.timing or "that timing"
        return Verdict("refuse",
                       f"Nothing was changed. I can't schedule '{said}' by voice. {plan.reason} "
                       f"Do you want it added for today with no timing? If so, say 'add {title}' on its own.")
    # The long standing title check: the title is exactly the words after the
    # command, with no timing word left in it, quotes and media titles aside.
    from voice_action_safety import _intent_matches
    if not _intent_matches("task_add", {"task": title, "recurrence": plan.recurrence},
                           _without_start(utterance)):
        return Verdict("refuse",
                       f"Nothing was changed. I can't tell whether '{title}' includes a time or a date. "
                       f"Do you want it added for today with no timing? If so, say 'add' and just the task.")
    wanted = _normalized(title)

    def args_ok(args):
        return (_normalized(str(args.get("task", ""))) == wanted
                and args.get("recurrence") == plan.recurrence
                and args.get("start_date") == plan.start_date)
    return Verdict("act", gate=IntentGate("task_add", "task_add", args_ok,
                                          plan=dict(recurrence=plan.recurrence, start_date=plan.start_date,
                                                    day=today.isoformat())))


# ── What each write is bound to ──
# The tool name alone is not a permission. Each gate also holds the words the
# intent was validated against, and a call whose arguments say something else
# is refused before anything runs.

_UNITS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "minute": 60, "minutes": 60, "min": 60,
          "mins": 60, "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600}
_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}


def _number(words):
    """An integer from one to ninety nine said in digits or words, else None."""
    words = words.replace("-", " ").split()
    if len(words) == 1:
        word = words[0]
        if word.isdigit():
            return int(word)
        if word in ("a", "an"):
            return 1
        return _ONES.get(word) or _TENS.get(word)
    if len(words) == 2 and words[0] in _TENS and words[1] in _ONES and 0 < _ONES[words[1]] < 10:
        return _TENS[words[0]] + _ONES[words[1]]
    return None


def spoken_seconds(timing):
    """The length in seconds of "ten minutes", "an hour", "half an hour" or
    "ninety seconds". None when the words are anything else."""
    text = _words(timing)
    text = re.sub(r"^(?:for|in|after)\s+", "", text)
    if text in ("half an hour", "half hour", "a half hour"):
        return 1800
    match = re.fullmatch(r"(.+?)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?)", text)
    if not match:
        return None
    count = _number(match[1])
    return count * _UNITS[match[2]] if count else None


def _timer_args(seconds):
    def check(args):
        unit = _UNITS.get(str(args.get("unit", "")))
        amount = args.get("amount")
        return bool(unit) and type(amount) is int and amount * unit == seconds
    return check


_CLOCK = re.compile(
    r"^(?:(?:tomorrow|today|tonight)\s+)?(?:(?:at|by)\s+)?([a-z0-9-]+)(?::(\d{2}))?\s*(?:([ap])\.?m\.?)?$")


def _due_matches(due, timing, now):
    """Whether an ISO due time is the one the timing words say. Only a length
    from now or a clock time is understood; any other wording is not accepted."""
    try:
        when = datetime.fromisoformat(due)
    except (TypeError, ValueError):
        return False
    seconds = spoken_seconds(timing)
    if seconds is not None:
        return abs((when - now).total_seconds() - seconds) <= 120
    clock = _CLOCK.match(" ".join(timing.lower().split()).replace("o'clock", "").strip())
    hour = _number(clock[1]) if clock else None
    if hour is None or hour > 24:
        return False
    minute, meridiem = int(clock[2] or 0), clock[3]
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "p" else 0)
        return (when.hour, when.minute) == (hour, minute)
    return (when.hour % 12, when.minute) == (hour % 12, minute)


def _reminder_args(intent, utterance):
    spoken = _words(utterance)
    timing = intent.timing

    def check(args):
        content = _words(re.sub(r"^to\s+", "", _words(str(args.get("content", "")))))
        if not content or content not in spoken:
            return False
        due = args.get("due")
        if due is None:
            return not timing
        return bool(timing) and _due_matches(due, timing, datetime.now())
    return check


def _list_args(intent, utterance):
    from lists_tasks import stored_list_name
    spoken = _words(utterance)
    item = _words(re.sub(r"^(?:the|my)\s+", "", _words(intent.target)))
    named = re.search(r"\b(?:to|on|from|off|in)\s+(?:my\s+|the\s+)?([a-z0-9']+)\s+list\b", spoken)
    wanted = stored_list_name(named[1]) if named and named[1] not in ("my", "the") else "grocery"

    def check(args):
        got = _words(re.sub(r"^(?:the|my)\s+", "", _words(str(args.get("item", "")))))
        if got != item:
            return False
        try:
            listed = stored_list_name(args["list_name"]) if args.get("list_name") else "grocery"
        except ValueError:
            return False
        return listed == wanted
    return check


_IDENTIFYING = ("title",)


_NEGATION = re.compile(r"\b(?:not|never|no|nothing|neither|nor|unfinished|incomplete)\b|n't\b", re.I)
_HYPOTHETICAL = re.compile(
    r"\?|\b(?:if|when|once|whether|should|would|could|might|maybe|going to|gonna|plan to|plans to|"
    r"supposed to|about to|will)\b|'ll\b", re.I)
_QUESTION_START = re.compile(r"^(?:did|do|does|have|has|am|is|was|were|what|when|how|why|who|which)\b", re.I)
_READ_START = re.compile(
    r"^(?:did|does|have|has|am|is|are|was|were|what|what's|whats|how|why|who|which|where|"
    r"when (?:did|do|does|is|was|are|were|will)|do (?:i|we|you)|any)\b", re.I)


def _asks_a_question(utterance):
    """True for a question about the user's own things ("what is on my list",
    "did I finish reading"). A polite request ("can you add eggs") is not one."""
    text = " ".join((utterance or "").split()).strip(" .!").removeprefix("please ").removeprefix("Please ")
    if re.match(r"^(?:can|could|would|will) you\b", text, re.I):
        return False
    return bool(_READ_START.match(text))


_POLITE_OPEN = re.compile(r"^(?:please\s+)?(?:(?:can|could|would|will) you\s+(?:please\s+)?)?", re.I)


def _affirmative(utterance):
    """True for a plain statement or command. Negation, a question and a
    hypothetical clause all fail, so "I didn't finish reading" never completes it."""
    text = " ".join(utterance.split()).strip(" .!")
    asked = bool(re.search(r"\b(?:can|could|would|will) you\b", text, re.I))
    text = _POLITE_OPEN.sub("", text.rstrip("?")) if asked else text
    if _QUESTION_START.match(text.strip()) or _NEGATION.search(text):
        return False
    return not _HYPOTHETICAL.search(text)


# ── The verb must be the intent's action, said about the task ──
# A completion or removal word found anywhere in the sentence is not enough:
# "I almost finished reading" holds the word finished and still is not a report
# that reading is done. The words before the task must be exactly a command or
# a first person report of the intent's action, and the words after it only
# filler that cannot change what happened.

_OPENER = r"(?:please )?(?:(?:can|could|would|will) you (?:please )?)?"
_ARTICLE = r"(?: (?:the|my|that|this))?"
_COMPLETE_BEFORE = re.compile(
    rf"{_OPENER}(?:(?:mark|complete|finish|check|cross|tick)(?: off)?|"
    r"(?:i|i've|ive|i have)(?: (?:just|already|now))? "
    r"(?:finished|completed|did|took|called|paid|got|bought|wrapped up|handled|sent|submitted|checked off|"
    r"crossed off|ticked off)|"
    r"(?:i'm|im|i am) (?:now )?done with|(?:i )?just (?:finished|completed|did))"
    rf"{_ARTICLE}")
_COMPLETE_AFTER_WORDS = {"today", "tonight", "now", "already", "just", "this", "morning", "afternoon",
                         "evening", "for", "please", "too", "off", "as", "done", "complete", "completed",
                         "finished", "is", "are", "task", "routine", "habit", "it"}
_REMOVE_BEFORE = re.compile(
    rf"{_OPENER}(?:cancel|stop(?: tracking| doing)?|archive|delete|remove|skip|drop|scrap|quit|"
    r"forget(?: about)?|get rid of|i (?:no longer|don't|dont|do not) (?:need|want)(?: to)?(?: do)?)"
    rf"{_ARTICLE}")
_REMOVE_AFTER_WORDS = {"today", "tonight", "this", "morning", "afternoon", "evening", "for", "good",
                       "forever", "permanently", "anymore", "any", "more", "no", "longer", "from", "now",
                       "on", "not", "but", "please", "task", "routine", "habit", "just", "only", "it"}


def _around_target(utterance, target):
    """(words before, words after) the target in what was said, or None."""
    heard, wanted = _words(utterance), _words(target)
    found = re.search(rf"(?<![a-z0-9']){re.escape(wanted)}(?![a-z0-9'])", heard) if wanted else None
    return (heard[:found.start()].strip(), heard[found.end():].strip()) if found else None


def _says_action(action, utterance, target):
    """Whether the words are a plain command or report of `action` on the target."""
    around = _around_target(utterance, target)
    if around is None:
        return False
    before, after = around
    after_words = after.split()
    if action == "task_complete":
        if before:
            return bool(_COMPLETE_BEFORE.fullmatch(before)) and set(after_words) <= _COMPLETE_AFTER_WORDS
        # "reading is done": no verb in front, so the words after must say it.
        return bool(re.fullmatch(r"(?:is |are |was )?(?:now |already |just )?(?:done|complete|completed|finished)"
                                 r"(?: (?:today|now|already))?", after))
    return bool(_REMOVE_BEFORE.fullmatch(before)) and set(after_words) <= _REMOVE_AFTER_WORDS


_TODAY_ONLY = re.compile(
    r"^(?:today|tonight|right now|now|this (?:morning|afternoon|evening)|for good|forever|permanently|"
    r"from now on|anymore|any more|for today|for tonight)$", re.I)
_OTHER_DAY = re.compile(
    r"\b(?:yesterday|tomorrow|last (?:night|week|month|year|\w+day)|next \w+|day after|day before|"
    r"(?:days?|weeks?|months?) ago|in \w+ (?:days?|weeks?|months?)|"
    r"(?:mon|tues?|wednes|thurs?|fri|satur|sun)day|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december) \d|"
    r"\d{1,2}[/-]\d{1,2}|the \d{1,2}(?:st|nd|rd|th)|this (?:week|weekend)|the day before)\b", re.I)


def _only_today(intent, utterance, title):
    """True when nothing says a day other than today. The task title is left
    out first, so a task called "Monday review" is not a day. The timing
    words he gave, if any, must be today words too."""
    if intent.timing and not _TODAY_ONLY.match(" ".join(intent.timing.split()).strip(" .!,")):
        return False
    rest = re.sub(re.escape(title), " ", utterance, flags=re.I) if title else utterance
    rest = re.sub(re.escape(intent.target), " ", rest, flags=re.I) if intent.target else rest
    return not _OTHER_DAY.search(rest)


def _without_start(text):
    """The request with its first day clause cut out, for the title check."""
    marker = task_scheduling._START.search(text)
    if not marker:
        return text
    daily = task_scheduling._DAILY.search(text, marker.end())
    end = daily.start() if daily else len(text)
    return " ".join(f"{text[:marker.start()]} {text[end:]}".split())


def validate(intent, utterance, today=None, choice=None, tier="hokage"):
    """Turn a recorded Intent into a Verdict. Every failure is a refusal or a
    question with no write; only a verdict of "act" carries a gate."""
    # A question or a read is never a request to change something, whatever the
    # model recorded, so it is never answered with "Nothing was changed".
    if _asks_a_question(utterance):
        return Verdict("none")
    if intent is None:
        return _refuse("I could not tell what you wanted to change. Please say it again.")
    if intent.more or _COMPOUND.search(utterance):
        return _refuse("Please ask for one change at a time.")
    if intent.action == "no_change":
        return Verdict("none")
    if intent.action not in ACTION_TOOL:
        return _refuse("I could not tell what you wanted to change. Please say it again.")
    # The permission floor comes before any lookup, so a speaker below the
    # tool's tier learns nothing about which tasks exist or what they are called.
    floor = ACTION_TOOL[intent.action]
    if floor in registry and not permits(registry.get(floor), tier):
        return _refuse("That needs more clearance than you have.")
    today = today or date.fromisoformat(database.task_day(datetime.now()))
    heard = _words(utterance)
    # An answer to a pending skip or stop question names no task, so the model
    # fills the task in from the question. That is the one case where the target
    # need not appear in the words, and it must be exactly the pending task.
    answering = (choice is not None and intent.action in ("task_archive", "task_skip_today")
                 and _words(intent.target) == _words(choice[1]))
    if (intent.action not in ("timer", "reminder") and intent.target and not answering
            and _words(intent.target) not in heard):
        return _refuse("I could not match that to what you said. Please say it again.")

    if intent.action == "timer":
        seconds = spoken_seconds(intent.timing)
        if not seconds or _words(intent.timing) not in heard:
            return _refuse("How long should the timer run? Say it like 'ten minutes'.")
        return Verdict("act", gate=IntentGate("set_timer", "timer", _timer_args(seconds)))
    if intent.action == "reminder":
        if not intent.target or _words(intent.target) not in heard:
            return _refuse("What should I remind you about?")
        if intent.timing and _words(intent.timing) not in heard:
            return _refuse("I could not match the time to what you said. Please say it again.")
        return Verdict("act", gate=IntentGate("set_reminder", "reminder", _reminder_args(intent, utterance)))
    if intent.action in ("list_add", "list_check"):
        if not intent.target:
            return _refuse("Which item do you mean?")
        return Verdict("act", gate=IntentGate(ACTION_TOOL[intent.action], intent.action,
                                              _list_args(intent, utterance)))
    if intent.action == "task_add":
        return _task_add(intent, utterance, today)

    # complete, archive and skip all name an existing task
    if not intent.target:
        return _refuse("Which task do you mean?")
    if answering:
        task = database.get_task(choice[0])
        if task is None or task["archived_at"] is not None:
            return _refuse(f"{choice[1]} is no longer active.")
        row, problem = (task["id"], task["text"], task["recurrence"]), None
    else:
        row, problem = _one_task(intent.target)
    if problem:
        return _refuse(problem)
    task_id, title, recurrence = row
    check = _task_resolves_to(task_id)
    plan = dict(day=today.isoformat())
    # Only today can be written by voice, so another day is refused, never moved to today.
    if not _only_today(intent, utterance, title):
        return _refuse("I can only do that for today by voice. Use the Tasks tab for another day.")
    if intent.action == "task_complete":
        if not _affirmative(utterance):
            return _refuse(f"I was not sure you finished {title}. Say it as a plain statement, "
                           f"like 'I finished {title}'.")
        if not _says_action("task_complete", utterance, intent.target):
            return _refuse(f"I was not sure you finished {title}. Say it like 'I finished {title}' "
                           f"or 'mark {title} done'.")
        return Verdict("act", gate=IntentGate("task_complete", "task_complete", check,
                                              target_id=task_id, plan=plan))
    if not answering and not _says_action(intent.action, utterance, intent.target):
        return _refuse(f"I was not sure you wanted to stop or skip {title}. Say it like "
                       f"'skip {title} today' or 'stop {title} for good'.")
    scope = _cancel_scope(utterance, recurrence, title)
    if scope == "ask":
        return Verdict("ask", f"Nothing was changed. {title} is a daily task. Skip today or stop it for good?",
                       choice=(task_id, title))
    if scope == "skip":
        return Verdict("act", gate=IntentGate("task_skip_today", "task_skip_today", check,
                                              target_id=task_id, plan=plan))
    return Verdict("act", gate=IntentGate("archive_task", "task_archive", check,
                                              target_id=task_id, plan=plan))
