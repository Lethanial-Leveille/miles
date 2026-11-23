"""Lists and tasks as tools: "add eggs to my grocery list", "what do I have to
do today", "I took my vitamins".

The storage is all in database.py. What lives here is the part that is about
speech: turning "my grocery list" into the list's stored name, settling which
row a spoken word means, and the sentence handed back.

Only list_show and tasks_today return to the model, because their whole point
is what comes back. The rest are fire and forget, like set_reminder: Nova's
own words alongside the call are the confirmation. Anything that did not
happen (nothing matched, several did, already on the list) is raised as a
ToolNotice instead of returned. It goes back to her like an error, so she says
it rather than "Done.", and is logged as a notice rather than an error.
"""

from database import (
    active_tasks_matching,
    add_list_item,
    add_task,
    archive_task,
    check_list_item,
    complete_task,
    find_list,
    get_or_create_list,
    list_items,
    tasks_today,
    unchecked_items_matching,
)
from tools import Permission, ToolNotice, tool


def stored_list_name(spoken):
    """The name a list is kept under, from how it was said.

    "My Grocery List", "the grocery list" and "grocery" all become grocery, so
    one list is never split in two by phrasing. Singular versus plural
    ("groceries") is left to the model, which the schema asks for one singular
    word; guessing plurals in code turns "clothes" into "clothe"."""
    name = spoken.strip().lower()
    for prefix in ("my ", "the "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith(" list"):
        name = name[:-len(" list")]
    name = name.strip()
    if not name:
        raise ValueError("a list needs a name, like grocery")
    return name


def _one(matches, what, words):
    """The single row the matches settle on.

    database.py has already applied the rule's first half: when any row
    matches exactly, ignoring case, only the exact rows come back. So one row
    here is either the exact match or the only partial one, and both win
    outright. Several rows means several partial matches and no exact one,
    which is the only case worth a question."""
    if not matches:
        raise ToolNotice(f"No {what} matches '{words}'.")
    if len(matches) > 1:
        names = ", ".join(m[1] for m in matches)
        raise ToolNotice(f"More than one {what} matches '{words}': {names}. "
                          f"Ask which one.")
    return matches[0]


def written_form(words):
    """Words as they should read on a list: trimmed, first letter capitalized.

    "salt" becomes Salt and "take my vitamins" Take my vitamins, so the app
    shows what a person would have written. Only the first letter changes,
    and not at all when the first word already has a capital in it, so
    "iPhone charger" and "AA batteries" stay as said. Duplicates are compared
    ignoring case, so "Eggs" and "eggs" are still one item."""
    words = words.strip()
    first = words.split(" ", 1)[0]
    if not words or any(ch.isupper() for ch in first):
        return words
    return words[0].upper() + words[1:]


# ── Lists ──

_LIST_NAME = {
    "type": "string",
    "description": "The list's name as one singular word, without 'my' or "
                   "'list': 'grocery' for 'my grocery list' or 'groceries'.",
}


@tool(
    name="list_add",
    description=(
        "Add one item to a named list, creating the list the first time. Call "
        "this for 'add eggs to my grocery list' or 'put milk on the list'. Call "
        "it once per item when several are named, even with no 'and' or commas "
        "between them, since speech to text often drops both: 'salt chicken "
        "eggs' is three calls. Keep a name that is one thing together: 'garlic "
        "salt', 'chicken broth' and 'ice cream' are one call each. When no list "
        "is named, use grocery. "
        # Nov 2026: "add jasmine rice salt and pepper" went on the list as
        # "Pepper", which reads as a vegetable in the store. The old schema said
        # "in the words used", so the model copied the word even though the
        # sentence settled the meaning.
        "Write each item the way it should read in the store, settled by what "
        "else he said: 'salt and pepper' means the spice, so it is 'salt' and "
        "'black pepper'; 'peppers and onions for fajitas' is 'bell peppers'. "
        "Only add words the context makes clear. When it truly could be either "
        "and nothing settles it, keep his word as he said it rather than guess."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "item": {"type": "string",
                     "description": "One item, as it should read in the store: "
                                    "his words, made specific when the rest of "
                                    "the sentence makes the meaning clear"},
            "list_name": _LIST_NAME,
        },
        "required": ["item", "list_name"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
)
def list_add_tool(item, list_name):
    name = stored_list_name(list_name)
    item = written_form(item)
    _, added = add_list_item(get_or_create_list(name), item)
    if not added:
        raise ToolNotice(f"{item} is already on the {name} list, so nothing "
                         f"was added. Say it was already there.")
    return f"Added {item} to the {name} list."


@tool(
    name="list_check",
    description=(
        "Check an item off a list because it was bought or done. Call this for "
        "'check off eggs', 'I got the milk' or 'cross bread off the grocery "
        "list'. The item stays in the list's history; nothing is deleted. Omit "
        "list_name when no list is named, and every list is searched."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "item": {"type": "string",
                     "description": "Words from the item, like 'milk'"},
            "list_name": _LIST_NAME,
        },
        "required": ["item"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
)
def list_check_tool(item, list_name=None):
    list_id = None
    if list_name:
        name = stored_list_name(list_name)
        list_id = find_list(name)
        if list_id is None:
            raise ToolNotice(f"There is no {name} list.")
    item_id, text = _one(unchecked_items_matching(list_id, item),
                         "unchecked item", item)
    check_list_item(item_id)
    return f"Checked off {text}."


@tool(
    name="list_show",
    description=(
        "Read what is still on a list, unchecked items only, in the order they "
        "were added. Call this for 'what's on my grocery list' or 'what do I "
        "need from the store'."
    ),
    input_schema={
        "type": "object",
        "properties": {"list_name": _LIST_NAME},
        "required": ["list_name"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    # The same people who can add to a list can hear it. READ alone would open
    # it to a guest.
    min_tier="chunin",
)
def list_show_tool(list_name):
    name = stored_list_name(list_name)
    list_id = find_list(name)
    if list_id is None:
        return f"There is no {name} list."
    items = [row[1] for row in list_items(list_id)]
    if not items:
        return f"The {name} list is empty."
    return f"On the {name} list: " + ", ".join(items) + "."


# ── Tasks ──
# All hokage: what he has to do is private in the way his calendar is.

_TASK_WORDS = {"type": "string",
               "description": "Words from the task, like 'vitamins'"}


@tool(
    name="task_add",
    description=(
        "Add a task to his to do list, once or every day. Call this for 'add "
        "call the bank to my tasks' or 'I need to take my vitamins every day'. "
        "Something to do every day with no time of day is a daily task, not a "
        "reminder, even when he says 'remind me'. Use set_reminder only when "
        "he gives a time to be told at."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The task, in his words"},
            "recurrence": {"type": "string", "enum": ["daily"],
                           "description": "daily when it repeats every day; "
                                          "omit for a one time task"},
        },
        "required": ["task"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def task_add_tool(task, recurrence=None):
    task = written_form(task)
    add_task(task, recurrence)
    if recurrence == "daily":
        return f"Added the daily task: {task}."
    return f"Added the task: {task}."


@tool(
    name="task_complete",
    description=(
        "Mark a task done for today, or for good if it is a one time task. "
        "Call this for 'I took my vitamins' or 'I called the bank'."
    ),
    input_schema={
        "type": "object",
        "properties": {"task": _TASK_WORDS},
        "required": ["task"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def task_complete_tool(task):
    task_id, text, _ = _one(active_tasks_matching(task), "task", task)
    complete_task(task_id)
    return f"Marked done: {text}."


@tool(
    name="archive_task",
    description=(
        "Stop tracking a task, daily or one time, keeping its history. Call "
        "this for 'stop tracking vitamins', 'I don't need to do that anymore' "
        "or 'cancel the bank task'. Not for finishing a task; that is "
        "task_complete."
    ),
    input_schema={
        "type": "object",
        "properties": {"task": _TASK_WORDS},
        "required": ["task"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def archive_task_tool(task):
    task_id, text, _ = _one(active_tasks_matching(task), "task", task)
    archive_task(task_id)
    return f"Stopped tracking: {text}."


@tool(
    name="tasks_today",
    description=(
        "Read what is left to do today: daily tasks not done yet today and one "
        "time tasks not done yet. Call this for 'what do I have to do today' "
        "or 'what's left on my list today'."
    ),
    input_schema={"type": "object", "properties": {}},
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def tasks_today_tool():
    rows = tasks_today()
    if not rows:
        return "Nothing left to do today."
    return "Still to do today: " + ", ".join(row[1] for row in rows) + "."
