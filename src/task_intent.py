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
from tools import Permission, registry

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

ACTIONS = ("task_add", "task_complete", "task_archive", "task_skip_today",
           "list_add", "list_check", "timer", "reminder", "other", "none")

# The one registered tool each action may call. "other" names its own.
ACTION_TOOL = {
    "task_add": "task_add", "task_complete": "task_complete", "task_archive": "archive_task",
    "task_skip_today": "task_skip_today", "list_add": "list_add", "list_check": "list_check",
    "timer": "set_timer", "reminder": "set_reminder",
}
TASK_ACTIONS = {"task_add", "task_complete", "task_archive", "task_skip_today"}
# Actions whose outcome is spoken by code from the tool result, not by the model.
AUTHORITATIVE = TASK_ACTIONS | {"list_add", "list_check"}

INTENT_TOOL = {
    "name": "record_intent",
    "description": "Record the one change the user's message asks for. Nothing runs from this.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS),
                       "description": "task_add, task_complete, task_archive (stop tracking or cancel a "
                                      "task), task_skip_today, list_add, list_check, timer, reminder, "
                                      "other (a different change, name it in tool), or none when the "
                                      "message asks for no change, only talks or asks a question."},
            "target": {"type": "string",
                       "description": "The task, list item or thing named, in the user's words, without "
                                      "timing words. Empty when none."},
            "timing": {"type": "string",
                       "description": "The exact words that say when or how often, copied from the "
                                      "message, like 'every day' or 'in four days'. Empty when none."},
            "tool": {"type": "string",
                     "description": "Only for action other: the exact name of the one tool needed."},
            "more_than_one_change": {"type": "boolean",
                                     "description": "True when the message asks for more than one change."},
        },
        "required": ["action", "target", "timing", "more_than_one_change"],
    },
}

_SYSTEM = (
    "You read one message from the user and record what change, if any, it asks his assistant to make. "
    "Call record_intent. You never carry anything out. Use none for chat, questions and requests to "
    "look something up. Pick one action only. If the message asks for two changes, set "
    "more_than_one_change. Copy the timing words exactly as said and never invent any. Changes that "
    "are not a task, list, timer or reminder use action other and name one of these tools in tool: ")


@dataclass(frozen=True)
class Intent:
    action: str
    target: str = ""
    timing: str = ""
    tool: str = ""
    more: bool = False


def _mutating_tool_summaries():
    names = [name for name in registry.names()
             if registry.get(name).permission in (Permission.WRITE, Permission.EXTERNAL_WRITE)
             and name not in ACTION_TOOL.values()]
    return "; ".join(f"{name} ({registry.get(name).summary})" for name in names)


def parse_intent(message):
    """The Intent in a forced tool call reply, or None when it is missing or malformed."""
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_intent":
            data = block.input
            if (isinstance(data, dict) and data.get("action") in ACTIONS
                    and isinstance(data.get("target", ""), str)
                    and isinstance(data.get("timing", ""), str)
                    and isinstance(data.get("tool", ""), str)
                    and type(data.get("more_than_one_change", False)) is bool):
                return Intent(data["action"], data.get("target", "").strip(), data.get("timing", "").strip(),
                              data.get("tool", "").strip(), data.get("more_than_one_change", False))
    return None


async def request_intent(client, utterance, pending_title=None):
    """One forced schema call with no mutating tool offered. Its time is recorded
    in the turn's timing record as intent_ms, so the cost on task turns shows up.

    Returns None when the call fails or answers outside the schema. The caller
    treats that as a refusal, because nothing may run on a guess."""
    system = _SYSTEM + _mutating_tool_summaries()
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
_REMOVAL = re.compile(r"\b(?:cancel|cancell?ed|stop|archive|delete|remove|skip|drop|scrap|quit|forget|"
                      r"get rid|no longer|anymore|don't need|do not need|for good)\b", re.I)
_FINISHED = re.compile(r"\b(?:finish\w*|done|complet\w*|mark\w*|check\w*|cross\w*|tick\w*|took|did|"
                       r"called|paid|got|bought|had|wrapped|handled|sent|submitted)\b", re.I)


@dataclass
class Verdict:
    kind: str                  # "none", "act", "refuse" or "ask"
    text: str = ""
    gate: "IntentGate | None" = None
    choice: tuple | None = None   # (task id, title) when the question waits for an answer


@dataclass
class IntentGate:
    """What one validated request lets the turn run. Used by TurnSafety.preflight
    on every round, so a later round cannot widen it."""
    tool: str | None = None
    action: str = "none"
    args_ok: object = None        # callable(args) -> bool, for tools whose arguments matter
    confirm: bool = False         # a pending approval may be answered this turn
    target_id: int | None = None
    plan: dict = field(default_factory=dict)

    @property
    def authoritative(self):
        return self.action in AUTHORITATIVE

    @property
    def requires_action(self):
        return self.tool is not None or self.confirm

    def offers(self, name):
        return name == self.tool or (self.confirm and name == "confirm_pending_action")

    def permits(self, name, args):
        if not self.offers(name):
            return False
        return self.args_ok is None or name != self.tool or bool(self.args_ok(args))


def closed_gate(confirm=False):
    """No mutating tool at all: the gate for a turn that could not change anything."""
    return IntentGate(confirm=confirm)


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


def _cancel_scope(utterance, recurrence):
    """archive, skip or ask, from the words he used and the kind of task.

    A clear stop request archives. Today or tonight skips only today. A bare
    cancel is ambiguous for a daily task, so it asks, and for a one time task
    its only occurrence is the task, so it archives."""
    if _STOP_FOR_GOOD.search(utterance):
        return "archive"
    if _TODAY.search(utterance):
        return "skip" if recurrence else "archive"
    if re.search(r"\b(?:stop|quit|drop|scrap)\b", utterance, re.I):
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
    plan = task_scheduling.resolve_task_schedule(f"add {intro}{rest}", title=title, today=today)
    if plan.outcome != "supported":
        said = intent.timing or "that timing"
        return Verdict("refuse",
                       f"Nothing was changed. I can't schedule '{said}' by voice. {plan.reason} "
                       f"Do you want it added for today with no timing? If so, say 'add {title}' on its own.")
    if plan.start_date is not None:
        return _refuse("I can't set a first day by voice yet. Add it without one, or use the Tasks tab.")
    # The long standing title check: the title is exactly the words after the
    # command, with no timing word left in it, quotes and media titles aside.
    from voice_action_safety import _intent_matches
    if not _intent_matches("task_add", {"task": title, "recurrence": plan.recurrence}, utterance):
        return Verdict("refuse",
                       f"Nothing was changed. I can't tell whether '{title}' includes a time or a date. "
                       f"Do you want it added for today with no timing? If so, say 'add' and just the task.")
    wanted = _normalized(title)

    def args_ok(args):
        return (_normalized(str(args.get("task", ""))) == wanted
                and args.get("recurrence") == plan.recurrence)
    return Verdict("act", gate=IntentGate("task_add", "task_add", args_ok, plan={"recurrence": plan.recurrence}))


def validate(intent, utterance, today=None, choice=None):
    """Turn a recorded Intent into a Verdict. Every failure is a refusal or a
    question with no write; only a verdict of "act" carries a gate."""
    if intent is None:
        return _refuse("I could not tell what you wanted to change. Please say it again.")
    if intent.more or _COMPOUND.search(utterance):
        return _refuse("Please ask for one change at a time.")
    if intent.action == "none":
        return Verdict("none")
    today = today or date.fromisoformat(database.task_day(datetime.now()))
    heard = _words(utterance)
    # An answer to a pending skip or stop question names no task, so the model
    # fills the task in from the question. That is the one case where the target
    # need not appear in the words, and it must be exactly the pending task.
    answering = (choice is not None and intent.action in ("task_archive", "task_skip_today")
                 and _words(intent.target) == _words(choice[1]))
    if (intent.action not in ("timer", "reminder", "other") and intent.target and not answering
            and _words(intent.target) not in heard):
        return _refuse("I could not match that to what you said. Please say it again.")

    if intent.action in ("timer", "reminder"):
        return Verdict("act", gate=IntentGate(ACTION_TOOL[intent.action], intent.action))
    if intent.action in ("list_add", "list_check"):
        if not intent.target:
            return _refuse("Which item do you mean?")
        return Verdict("act", gate=IntentGate(ACTION_TOOL[intent.action], intent.action))
    if intent.action == "other":
        name = intent.tool
        if (name not in registry or name in ACTION_TOOL.values() or name == "confirm_pending_action"
                or registry.get(name).permission not in (Permission.WRITE, Permission.EXTERNAL_WRITE)):
            return _refuse("I could not tell which change you meant. Please say it again.")
        return Verdict("act", gate=IntentGate(name, "other"))

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
    if intent.action == "task_complete":
        if not _FINISHED.search(utterance):
            return _refuse(f"I was not sure you finished {title}. Please say it again.")
        return Verdict("act", gate=IntentGate("task_complete", "task_complete", check, target_id=task_id))
    if not answering and not _REMOVAL.search(utterance):
        return _refuse(f"I was not sure you wanted to stop or skip {title}. Please say it again.")
    scope = _cancel_scope(utterance, recurrence)
    if scope == "ask":
        return Verdict("ask", f"Nothing was changed. {title} is a daily task. Skip today or stop it for good?",
                       choice=(task_id, title))
    if scope == "skip":
        return Verdict("act", gate=IntentGate("task_skip_today", "task_skip_today", check, target_id=task_id))
    return Verdict("act", gate=IntentGate("archive_task", "task_archive", check, target_id=task_id))


# A reply that claims a change when none ran. Applied only when no mutating tool
# succeeded, so it can only replace words that were never true.
_CLAIM = re.compile(
    r"^\s*(?:done|all set|okay,? done|ok,? done)\b|"
    r"\b(?:i(?:'ve| have| just)?|i'll go ahead and) (?:added|archived|marked|completed|removed|saved|"
    r"scheduled|created|cancell?ed|stopped tracking|checked off|crossed off|skipped|set)\b", re.I)


def claims_a_change(text):
    return bool(_CLAIM.search(text or ""))
