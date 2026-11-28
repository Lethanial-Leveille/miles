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


def _fake_followup(followup):
    """What a sensible model makes of an answer to a question about a change:
    the action comes from the earlier request, the target from the answer or
    from the option the answer points at in the question."""
    answer = followup.answer.lower().strip(" .!?")
    earlier = followup.earlier.lower()
    if re.search(r"\bboth\b", answer):
        return Intent("task_archive", "", "", True)
    options = re.search(r"([\w ]+?) or ([\w ]+?)\?", followup.question.lower().replace(",", " "))
    names = [name.split()[-1] for name in options.groups()] if options else []
    if re.search(r"\b(?:first|1st)\b", answer) and names:
        target = names[0]
    elif re.search(r"\b(?:second|2nd|last)\b", answer) and names:
        target = names[-1]
    else:
        target = next((name for name in names if name in answer), "")
    if not target:
        return Intent("no_change")
    if re.search(r"\bskip\b", earlier):
        return Intent("task_skip_today", target)
    if re.search(r"\b(?:finish|complete|done)\b", earlier):
        return Intent("task_complete", target)
    return Intent("task_archive", target)


def fake_intent(text, followup=None):
    if followup is not None:
        return _fake_followup(followup)
    lowered = text.lower().strip(" .!?")
    if lowered.startswith("remember") or "calendar" in lowered:
        # Outside the intent step: the model files these under no_change and the
        # tool's own confirm first flow handles them.
        return Intent("no_change")
    if re.search(r"\btimer\b", lowered):
        length = re.search(r"\b(?:for|of|in)\s+(.+)$", lowered)
        return Intent("timer", "timer", length[1] if length else "")
    remind = re.match(r"remind me (?:to )?(.+?)(?: ((?:at|in|by) .+|tomorrow.*|tonight))?$", lowered)
    if remind:
        return Intent("reminder", remind[1], remind[2] or "")
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
    return Intent("no_change")
