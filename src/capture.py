"""Quick capture: one line of text from outside the app becomes a task or a
list item, with a one line answer for a notification.

Who may do it is a capture key, not a login. A key can do exactly what
capture() does: add one task, or add items to one list. It is checked only by
POST /capture, and every other route reads a login JWT, which a key is not, so
a leaked key cannot read a memory, the calendar or anything else. It runs at
CAPTURE_TIER, chunin, never hokage.

That is why capture does not call the voice tools task_add and list_add through
the permission gate. task_add is raised to hokage so a guest's voice in the
room cannot add tasks; a key is not a voice in the room, it is a capability
with two operations, and the narrow route is what enforces it.

A captured task has no day, so it goes to the backlog (migration 44), not onto
today's list; the nightly review suggests from it.

Version 1 decides with rules, not a model: a capture is a list item only when
it says which list ("eggs to the grocery list", "grocery: eggs, milk" for a
list that exists). Everything else is a task. A wrong guess costs one tap to
fix, and no capture ever waits on, or pays for, a model call. Classifying into
task, errand and question comes later, and capture_log keeps what v1 did with
real text so that classifier can be judged against it.
"""

import hashlib
import re
import secrets

from database import (add_list_item, add_task, capture_key_id, find_list,
                      get_or_create_list, log_capture)

CAPTURE_TIER = "chunin"
KEY_PREFIX = "mcap_"
MAX_ITEMS = 20

# "add eggs to my grocery list", "milk on the grocery list". The word list is
# required: "talk to mom" must stay a task, not start a list named mom.
_TO_A_LIST = re.compile(
    r"^(?:add\s+|put\s+)?(?P<items>.+?)\s+(?:to|on)\s+(?:my\s+|the\s+)?"
    r"(?P<list>[a-z][a-z ]{0,30}?)\s+list\.?$", re.IGNORECASE)
# "grocery: eggs, milk" or "grocery list: eggs". Only for a list that exists,
# so "Note: call the bank" stays a task.
_LIST_COLON = re.compile(r"^(?P<list>[a-z][a-z ]{0,30}?)(?:\s+list)?\s*:\s*(?P<items>.+)$",
                         re.IGNORECASE)


def new_key():
    """A fresh key and its hash. The key is shown once; only the hash is kept."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key)


def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def key_id_for(key):
    """The active key's id, or None for anything else, a login JWT included."""
    if not key or not key.startswith(KEY_PREFIX):
        return None
    return capture_key_id(hash_key(key))


def _items(text):
    parts = re.split(r"\s*,\s*|\s+and\s+", text.strip())
    return [p.strip().rstrip(".") for p in parts if p.strip().rstrip(".")]


def classify(text, list_exists=find_list):
    """('list_item', list name, [items]) or ('task', None, [text]). Pure apart
    from list_exists, which tests replace."""
    text = " ".join(text.split())
    match = _TO_A_LIST.match(text)
    if match:
        return "list_item", match["list"].strip().lower(), _items(match["items"])
    match = _LIST_COLON.match(text)
    if match and list_exists(match["list"].strip().lower()) is not None:
        return "list_item", match["list"].strip().lower(), _items(match["items"])
    return "task", None, [text.rstrip(".")]


def capture(text, source, key_id=None, now=None):
    """Add it and log it. Returns (kind, message); message is the one line for
    the notification."""
    kind, list_name, items = classify(text)
    if kind == "task":
        title = items[0][:1].upper() + items[0][1:]
        task_id = add_task(title, now=now)
        message = f"Added to backlog: {title}"
        log_capture(key_id, source, text, "task", message, ref_id=task_id, now=now)
        return "task", message

    items = items[:MAX_ITEMS]
    list_id = get_or_create_list(list_name, now=now)
    added, already = [], []
    for item in items:
        item_id, new = add_list_item(list_id, item, now=now)
        (added if new else already).append(item)
        log_capture(key_id, source, text, "list_item" if new else "duplicate",
                    f"{item} {'added to' if new else 'already on'} {list_name}",
                    list_name=list_name, ref_id=item_id, now=now)
    if added and already:
        message = f"Added {', '.join(added)} to {list_name}; {', '.join(already)} already there"
    elif added:
        message = f"Added {', '.join(added)} to {list_name}"
    else:
        message = f"{', '.join(already)} already on {list_name}"
    return ("list_item" if added else "duplicate"), message
