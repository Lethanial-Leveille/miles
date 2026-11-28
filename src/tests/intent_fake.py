"""An offline stand in for the intent model call.

It reads an utterance with a few patterns the way a sensible model would and
returns a task_intent.Intent. Tests that need a particular answer, including a
wrong one, replace it with their own function."""

import re

from task_intent import Intent

_TIMING = re.compile(
    r"\s*\b((?:every|each) day|daily|starting \w+(?: \w+)?|in (?:\w+) (?:days?|weeks?|months?|hours?)|"
    r"at (?:\w+(?: \w+)?)|on \w+days?|by \w+|tomorrow|next \w+|tonight)\s*$", re.I)


def _add(text):
    match = re.match(r"(?:please )?(?:(?:can|could|would|will) you (?:please )?)?"
                     r"(?:add|create|i need to)\s+(?:a daily task (?:to |for )?|a task (?:to |for )?)?(.+)$",
                     text, re.I)
    if not match:
        return None
    rest = re.sub(r"\s+(?:to|on) (?:my|the) (?:tasks|to do list)$", "", match[1], flags=re.I).strip()
    timing = ""
    found = _TIMING.search(rest)
    while found:
        timing = f"{found[1]} {timing}".strip()
        rest = rest[:found.start()].strip()
        found = _TIMING.search(rest)
    return Intent("task_add", rest.strip(" \"“”"), timing)


def fake_intent(text):
    lowered = text.lower().strip(" .!?")
    if lowered.startswith("remember"):
        return Intent("other", tool="remember")
    if "calendar" in lowered:
        return Intent("other", tool="create_calendar_event")
    if re.search(r"\btimer\b", lowered):
        return Intent("timer", "timer")
    if re.match(r"remind me", lowered):
        return Intent("reminder", "reminder")
    on_list = re.match(r"(?:add|put) (.+?) (?:to|on) (?:my |the )?(\w+ )?list$", lowered)
    if on_list:
        return Intent("list_add", on_list[1])
    check = re.match(r"(?:check off|cross off|i (?:got|bought)) (?:the |my )?(.+?)(?: off)?(?: (?:my|the) \w+ list)?$", lowered)
    if check:
        return Intent("list_check", check[1])
    cancel = re.match(r"(?:please )?(?:archive|stop tracking|cancel|stop|delete|skip|remove)(?: the| my)? (.+?)"
                      r"(?: (?:today|tonight|for good))?$", lowered)
    if cancel:
        skipped = lowered.startswith("skip")
        return Intent("task_skip_today" if skipped else "task_archive", cancel[1])
    done = re.match(r"(?:mark|complete|finish|check off|cross off|i (?:finished|completed|did|took|called|paid))"
                    r"(?: the| my)? (.+?)(?: as done| done)?$", lowered)
    if done:
        return Intent("task_complete", done[1])
    added = _add(text.strip(" .!?"))
    if added:
        return added
    return Intent("none")
