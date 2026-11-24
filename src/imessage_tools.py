"""send_imessage: Nova's way to text a student or parent. It never sends.

It stores a request in approvals.py and pushes it to the phone. The text leaves
the Mac only after Lethanial taps Approve on that request in Nova, within two
hours, and the Mac sends it. This is an EXTERNAL_WRITE like the calendar and
Evallo tools, so only hokage may call it.
"""

import hashlib
from datetime import datetime

import approvals
from tools import Permission, ToolNotice, tool


@tool(
    name="send_imessage",
    description=(
        "Draft a text message to a tutoring student's iMessage chat, usually a group "
        "with their parent. Call it when he asks to text, message or remind a student or "
        "their parent. It does not send anything: it puts the exact text on "
        "Lethanial's phone with Approve and Skip, and the Mac sends it only if he "
        "taps Approve within two hours. Keep it short, polite and only about "
        "scheduling; the readers are students and parents, many of them minors. "
        "Give the student's first name; the chat comes from his saved mapping and "
        "is never guessed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "student_first_name": {"type": "string", "description": "e.g. Taylor."},
            "text": {"type": "string", "description": "The exact message, under 400 characters."},
        },
        "required": ["student_first_name", "text"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
)
def send_imessage(student_first_name, text):
    name = student_first_name.strip()
    chat = approvals.chat_name_for(name)
    if chat is None:
        raise ToolNotice(f"{name} has no iMessage chat mapped, so nothing was drafted. "
                         "He needs to add them to contacts.json on the Mac.")
    try:
        text = approvals.check_text(text)
    except ValueError as exc:
        raise ToolNotice(str(exc))
    key = "voice:" + hashlib.sha256(f"{name.lower()}|{text}|{datetime.now():%Y-%m-%d}".encode()).hexdigest()[:20]
    row, created = approvals.propose(
        "imessage", f"Text {name}", text, chat,
        payload={"first_name": name, "source": "send_imessage"}, dedupe_key=key)
    if created:
        approvals.notify_new(row)
    return (f"Nothing has been sent. The text to {chat} is waiting on his phone for "
            f"Approve or Skip and expires in two hours.")
