import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from datetime import date, datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field
from jose import JWTError

from auth import (verify_password, create_token, decode_token, DEVICE_SCOPE,
                  DEVICE_TOKEN_EXPIRE_DAYS, AGENT_SCOPE, AGENT_TOKEN_EXPIRE_DAYS)
from brain import ask_nova
from database import (
    init_db, get_active_memories, get_pending_memories,
    approve_memory, delete_memory, get_history,
    save_memory, supersede_memory, get_memory_chain, memory_status, memory_content,
    open_reminders, cancel_reminder_by_id,
    dismiss_memory_pair, dismissed_memory_pairs, merge_memories,
    all_lists, list_items, check_list_item, uncheck_list_item, list_item_exists,
    task_day, tasks_today, complete_task,
    save_push_device, remove_push_device, save_health_sync,
    log_phone_presence, latest_phone_presence,
)
from config import MEMORY_SIMILAR_LIST
from system_state import get_system_state
import embeddings
import memory_merge
import calendar_tools
import push
import agent_watchdog
import approvals
import tts
import database
import sessions
import capture
import tasks_view
# Imported for its side effect: registering the nightly_review session type.
# Without it POST /sessions answers 400 for it and the timer has nothing to
# open. Kept explicit, with this comment, because an unused looking import is
# exactly what a later cleanup deletes.
import nightly_review  # noqa: F401

app = FastAPI(title="M.I.L.E.S. API", version="0.7")
init_db()

# ── Auth dependency ──

_bearer = HTTPBearer()

def _unauthorized():
    # fail closed: any bad or expired token gets a 401, no details leaked
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    """A full login. A token carrying any scope is refused here: scoped tokens
    are narrower than a login by design, and this is every route but the few
    that name get_device_user."""
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise _unauthorized()
    if payload.get("scope"):
        raise _unauthorized()
    return payload["sub"]


def get_device_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    """A full login or the phone's device token. Only for the writes the phone
    makes in the background: where it is, and its Health data."""
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise _unauthorized()
    if payload.get("scope") not in (None, DEVICE_SCOPE):
        raise _unauthorized()
    return payload["sub"]


# ── Request / response models ──

class LoginRequest(BaseModel):
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatRequest(BaseModel):
    message: str
    # How the answer will be rendered, not which client sent it. "voice" means
    # the response is spoken, so the prompt drops markdown and the TTS path runs
    # pronunciation normalization. Absent means voice, because the app existed
    # before this field and its callers should not have to be updated at once.
    channel: Literal["voice", "text"] = "voice"

class ChatResponse(BaseModel):
    response: str

class SpeakRequest(BaseModel):
    text: str

class MemoryText(BaseModel):
    content: str

class MemoryPair(BaseModel):
    a_id: int
    b_id: int

class MemoryMerge(BaseModel):
    keep_id: int
    drop_id: int
    content: str

# An APNs device token is hex, 32 bytes today. Apple says its length may change,
# so the bound is loose; what matters is that nothing else reaches a URL path.
class PushToken(BaseModel):
    token: str = Field(pattern=r"^[0-9a-fA-F]{32,200}$")

class PushRegistration(PushToken):
    # Which APNs host this token belongs to: a build run from Xcode is sandbox,
    # one installed from TestFlight or the App Store is production.
    environment: Literal["sandbox", "production"]
    device_name: Optional[str] = Field(default=None, max_length=100)

# What the Nova app reads from HealthKit and sends when it opens. Every number is
# bounded, because these are quoted back to him by voice: one corrupt row
# ("weight 18300") would be read out as fact.
class HealthDay(BaseModel):
    day: date
    steps: Optional[int] = Field(default=None, ge=0, le=200_000)
    active_kcal: Optional[float] = Field(default=None, ge=0, le=20_000)
    weight_lb: Optional[float] = Field(default=None, gt=0, le=1_000)

class HealthWorkout(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1, max_length=40)
    started_at: datetime
    minutes: float = Field(ge=0, le=1_440)
    distance_mi: Optional[float] = Field(default=None, ge=0, le=500)
    kcal: Optional[float] = Field(default=None, ge=0, le=20_000)

class HealthHour(BaseModel):
    start: datetime
    steps: int = Field(ge=0, le=50_000)

class HealthSync(BaseModel):
    days: list[HealthDay] = Field(default_factory=list, max_length=60)
    workouts: list[HealthWorkout] = Field(default_factory=list, max_length=100)
    # Hourly steps for the last two days, from app builds of Nov 21 2026 on
    hours: list[HealthHour] = Field(default_factory=list, max_length=72)

class CaptureRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    # Where it came from, for the log: mac_hotkey, action_button, ...
    source: str = Field(pattern=r"^[a-z0-9_]{1,32}$")

class PhonePresence(BaseModel):
    state: Literal["home", "away"]
    # When the phone saw the crossing, not when it arrived here
    at: datetime

class EventChange(BaseModel):
    title: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None


# ── Auth endpoints ──

@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    if not verify_password(body.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password")
    return TokenResponse(access_token=create_token("Lethanial"))


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh(user: str = Depends(get_current_user)):
    return TokenResponse(access_token=create_token(user))


# ── Core endpoints ──

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user: str = Depends(get_current_user)):
    response = ask_nova(body.message, device="app", channel=body.channel).text
    return ChatResponse(response=response)


# How long the stream waits with nothing to send before writing a comment
# line. Cloudflare closes a connection that goes quiet, and a turn that calls a
# slow tool can say nothing for several seconds.
_KEEPALIVE_S = 15


def _sse(kind: str, text: str) -> str:
    """One Server Sent Event. The blank line at the end is what ends the event,
    and json keeps a reply containing newlines from ending it early."""
    return f"event: {kind}\ndata: {json.dumps({'text': text})}\n\n"


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, user: str = Depends(get_current_user)):
    """The same turn as /chat, sent as it is written rather than at the end.

    ask_nova runs its own event loop, so it runs on a worker thread and hands
    each piece back here with call_soon_threadsafe, which is the only safe way
    into this loop from another thread. The turn finishes whatever happens to
    this response: it is already writing to the database and, on voice, to the
    speaker."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_text(kind: str, text: str):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, text))

    def run_turn():
        try:
            result = ask_nova(body.message, device="app", channel=body.channel,
                              on_text=on_text)
            on_text("done", result.text)
        except Exception as exc:
            on_text("error", str(exc))

    threading.Thread(target=run_turn, daemon=True).start()

    async def events():
        while True:
            try:
                kind, text = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield _sse(kind, text)
            if kind in ("done", "error"):
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# A reply Nova actually writes runs a few hundred characters. This is not a
# safety limit, it is a bill: ElevenLabs charges per character, and an app bug
# that posted a whole screen of text in a loop would spend real money before
# anyone noticed. Raise it if a legitimate reply ever hits it.
_SPEAK_MAX_CHARS = 2000


@app.post("/speak")
def speak(body: SpeakRequest, user: str = Depends(get_current_user)):
    """Nova's voice for a client that has no speaker of ours: audio in, audio out.

    Deliberately separate from /chat. The app already has the reply as text by
    the time it wants to hear it, and a turn that had to finish speaking before
    it returned is the thing /chat/stream was built to stop doing. This way the
    words appear as they are written and the audio is asked for once, as a whole.

    Asked for as a whole on purpose. eleven_v3 voices each request on its own, so
    a reply synthesized a sentence at a time comes back sounding like it changed
    speakers partway through. See tts.stream_audio.

    Nothing here touches the room speaker. It is the same voice and the same
    pronunciation table, and no part of the path can take speak_lock."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Nothing to speak")
    if len(text) > _SPEAK_MAX_CHARS:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail=f"Text is longer than {_SPEAK_MAX_CHARS} characters")

    # The first chunk is pulled before the response begins, so a request that
    # fails outright is still a status code the app can act on. Once a byte has
    # been sent the status is spent, and a later failure can only end the audio
    # early, which is why stream_audio raises only when nothing arrived.
    try:
        chunks = tts.stream_audio(text)
        first = next(chunks) if chunks is not None else None
    except StopIteration:
        first = None
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"Voice synthesis failed: {exc}")
    if first is None:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Voice synthesis returned no audio")

    def audio():
        yield first
        yield from chunks

    # no-store because this is a recording of him being spoken to by name.
    return StreamingResponse(audio(), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


@app.get("/memories")
def list_memories(limit: int = 50, offset: int = 0, user: str = Depends(get_current_user)):
    rows, total = get_active_memories(limit=limit, offset=offset)
    return {
        "total": total,
        "memories": [
            {"id": r[0], "content": r[1], "category": r[2], "source": r[3], "confidence": r[4], "status": r[5]}
            for r in rows
        ],
    }


@app.get("/memories/pending")
def list_pending_memories(user: str = Depends(get_current_user)):
    """Each pending memory, with the stored one it may repeat or update as
    similar_to, or null. Added Sep 28 2026; an app that predates it ignores the
    field. A failed comparison is null rather than an error, because the queue
    still has to be reviewable when the embedding model is not."""
    rows = get_pending_memories(limit=100)
    try:
        close = embeddings.closest_stored([r[0] for r in rows], MEMORY_SIMILAR_LIST,
                                          dismissed=dismissed_memory_pairs())
    except Exception as exc:
        print(f"Pending similarity failed: {exc}", flush=True)
        close = {}
    return [
        {"id": r[0], "content": r[1], "category": r[2], "source": r[3], "confidence": r[4],
         "similar_to": ({"id": close[r[0]][0], "content": close[r[0]][1],
                         "similarity": round(close[r[0]][2], 2)} if r[0] in close else None)}
        for r in rows
    ]


def _current_memory(memory_id):
    """The text of an active or pending memory, or None. A replaced memory's
    text still exists, and merging or dismissing it would act on history."""
    if memory_status(memory_id) not in ("active", "pending"):
        return None
    return memory_content(memory_id)


# Declared before the /memories/{memory_id} routes. FastAPI tries routes in
# order, and "similar" or "merge" in the id's place fails validation there
# instead of falling through to these.

@app.get("/memories/similar")
def list_similar_memories(user: str = Depends(get_current_user)):
    """Pairs of active memories that may say the same thing, closest first,
    without the ones he has said are different. a is the older of each pair.
    Listed only: which pairs are duplicates is his call (docs/BRAIN.md)."""
    pairs = embeddings.similar_pairs(MEMORY_SIMILAR_LIST, dismissed=dismissed_memory_pairs())
    return {"pairs": [
        {"similarity": round(score, 2),
         "a": {"id": a[0], "content": a[1]}, "b": {"id": b[0], "content": b[1]}}
        for score, a, b in pairs
    ]}


@app.post("/memories/similar/dismiss")
def dismiss_similar_memories(body: MemoryPair, user: str = Depends(get_current_user)):
    """Keep both: these are different facts, so stop listing them together."""
    if body.a_id == body.b_id or _current_memory(body.a_id) is None \
            or _current_memory(body.b_id) is None:
        raise HTTPException(status_code=404, detail="Both memories must exist")
    dismiss_memory_pair(body.a_id, body.b_id)
    return {"dismissed": True}


@app.post("/memories/merge/draft")
def draft_memory_merge(body: MemoryPair, user: str = Depends(get_current_user)):
    """One sentence holding both memories, written by the model for him to edit.
    Stores nothing. drafted is false when the model could not be reached and the
    content is just the longer memory."""
    older, newer = sorted((body.a_id, body.b_id))
    older_text, newer_text = _current_memory(older), _current_memory(newer)
    if older == newer or older_text is None or newer_text is None:
        raise HTTPException(status_code=404, detail="Both memories must exist")
    content, drafted = memory_merge.draft(older_text, newer_text)
    return {"content": content, "drafted": drafted}


@app.post("/memories/merge")
def merge_two_memories(body: MemoryMerge, user: str = Depends(get_current_user)):
    """keep_id becomes content, keeping its history; drop_id is deleted. One
    request, one transaction. drop_id may be pending, which is how a pending
    fact replaces the stored one it updates."""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="A memory needs some text")
    new_id = merge_memories(body.keep_id, body.drop_id, content)
    if new_id is None:
        raise HTTPException(status_code=404, detail="Both must be current, different memories")
    return {"id": new_id, "replaced": body.keep_id, "deleted": body.drop_id}


@app.post("/memories/{memory_id}/approve")
def approve_pending_memory(memory_id: int, user: str = Depends(get_current_user)):
    if not approve_memory(memory_id):
        raise HTTPException(status_code=404, detail="Pending memory not found")
    return {"approved": True}


@app.post("/memories", status_code=201)
def add_memory(body: MemoryText, user: str = Depends(get_current_user)):
    """A memory he typed himself: explicit and active, like one he asked Nova to
    keep, since there is nothing for him to review in his own words."""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="A memory needs some text")
    new_id = save_memory(content, source="explicit", status="active")
    if not new_id:
        raise HTTPException(status_code=409, detail="That memory is already stored")
    return {"id": new_id}


@app.patch("/memories/{memory_id}")
def edit_memory(memory_id: int, body: MemoryText, user: str = Depends(get_current_user)):
    """Change what a memory says by superseding it, not by rewriting the row.

    The old wording is kept and linked, which is how every correction in this
    repo works: an exam that moved is different information from one that was
    always on the new day. The replacement is explicit and active, because he
    wrote it, so editing a pending memory also approves it.

    Only active and pending rows can be edited. Superseding a row that was
    already replaced would fork its history into two current answers."""
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="A memory needs some text")
    if memory_status(memory_id) not in ("active", "pending"):
        raise HTTPException(status_code=404, detail="No current memory with that id")
    if memory_content(memory_id) == content:
        return {"id": memory_id, "changed": False}
    new_id = supersede_memory(memory_id, content, source="explicit")
    return {"id": new_id, "replaced": memory_id, "changed": True}


@app.get("/memories/{memory_id}/history")
def memory_history(memory_id: int, user: str = Depends(get_current_user)):
    """What this memory says now and everything it replaced, newest first."""
    chain = get_memory_chain(memory_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Memory not found")
    return [
        {"id": r[0], "content": r[1], "status": r[2], "created_at": r[3],
         "superseded_by": r[4], "superseded_at": r[5]}
        for r in chain
    ]


@app.delete("/memories/{memory_id}")
def remove_memory(memory_id: int, user: str = Depends(get_current_user)):
    if not delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@app.get("/history")
def history(limit: int = 50, offset: int = 0, user: str = Depends(get_current_user)):
    return get_history(limit=limit, offset=offset)


@app.get("/reminders")
def list_reminders(user: str = Depends(get_current_user)):
    return [{"id": r[0], "content": r[1], "due_at": r[2], "created_at": r[3],
             "kind": r[4]}
            for r in open_reminders()]


@app.delete("/reminders/{reminder_id}")
def cancel_reminder(reminder_id: int, user: str = Depends(get_current_user)):
    if not cancel_reminder_by_id(reminder_id):
        raise HTTPException(status_code=404, detail="No outstanding reminder with that id")
    return {"cancelled": True}


# ── Lists and tasks ──
# What the app reads, and the taps it makes: checking an item off, putting it
# back, and finishing a task. Adding stays voice only for now, as asked. Everything goes
# through the same database functions as the tools, so a tap and "check off
# eggs" leave exactly the same row behind.

@app.get("/lists")
def get_lists(user: str = Depends(get_current_user)):
    return [{"id": l[0], "name": l[1], "unchecked": l[2]} for l in all_lists()]


@app.get("/lists/{list_id}/items")
def get_list_items(list_id: int, include_checked: bool = False,
                   user: str = Depends(get_current_user)):
    """Unchecked items first, in the order added; with include_checked, the
    purchase history after them, most recent first."""
    if list_id not in {l[0] for l in all_lists()}:
        raise HTTPException(status_code=404, detail="No list with that id")
    return [{"id": i[0], "text": i[1], "checked": bool(i[2]),
             "checked_at": i[3], "created_at": i[4]}
            for i in list_items(list_id, include_checked=include_checked)]


@app.post("/lists/items/{item_id}/check")
def check_item(item_id: int, user: str = Depends(get_current_user)):
    """Idempotent: an item that is already checked answers 200 and keeps its
    first checked_at, so a retry from the phone does not read as an error.
    Only a missing item is 404."""
    if not check_list_item(item_id) and not list_item_exists(item_id):
        raise HTTPException(status_code=404, detail="No item with that id")
    return {"checked": True}


@app.post("/lists/items/{item_id}/uncheck")
def uncheck_item(item_id: int, user: str = Depends(get_current_user)):
    """Undo a tap: the item goes back on the list with checked_at cleared.
    Idempotent like check; only a missing item is 404."""
    if not uncheck_list_item(item_id):
        raise HTTPException(status_code=404, detail="No item with that id")
    return {"checked": False}


@app.get("/tasks/today")
def get_tasks_today(user: str = Depends(get_current_user)):
    """day is the 4am day the list is for. At 1am it is still yesterday's
    date, and the app should say so rather than leave him wondering why
    today's vitamins show as done."""
    return {"day": task_day(),
            "tasks": [{"id": t[0], "text": t[1], "recurrence": t[2]}
                      for t in tasks_today()]}


# ── Quick capture ──

def get_capture_key(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> int:
    """A capture key, and only a capture key: a login JWT is refused here just
    as a capture key is refused everywhere else. See capture.py."""
    key_id = capture.key_id_for(credentials.credentials)
    if key_id is None:
        raise _unauthorized()
    return key_id


@app.post("/capture")
def quick_capture(body: CaptureRequest, key_id: int = Depends(get_capture_key)):
    """One line in, a task or list item out, and a line to show in a
    notification. Never reads anything back."""
    kind, message = capture.capture(body.text, body.source, key_id)
    return {"ok": True, "kind": kind, "message": message}


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    where: Literal["today", "backlog"]
    due_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


def _task_http(exc: tasks_view.TaskError):
    return HTTPException(status_code=exc.status, detail=exc.detail)


@app.get("/tasks/overview")
def tasks_overview(days: int = Query(3, ge=1, le=7), user: str = Depends(get_current_user)):
    """The Tasks tab: today, the next days and the backlog. See docs/TASKS_CONTRACT.md."""
    return tasks_view.overview(days)


@app.post("/tasks", status_code=201)
def create_task(body: TaskCreate, user: str = Depends(get_current_user)):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="The title cannot be blank")
    if body.due_date:
        try:
            date.fromisoformat(body.due_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="due_date is not a real date")
    return {"task": tasks_view.create(title, body.where, body.due_date)}


@app.post("/tasks/{task_id}/complete")
def finish_task(task_id: int, user: str = Depends(get_current_user)):
    try:
        task = tasks_view.complete(task_id)
    except tasks_view.TaskError as exc:
        raise _task_http(exc)
    return {"completed": True, "day": task_day(tasks_view.now()), "task": task}


@app.post("/tasks/{task_id}/reopen")
def reopen_task(task_id: int, user: str = Depends(get_current_user)):
    try:
        return {"task": tasks_view.reopen(task_id)}
    except tasks_view.TaskError as exc:
        raise _task_http(exc)


@app.post("/tasks/{task_id}/backlog")
def task_to_backlog(task_id: int, user: str = Depends(get_current_user)):
    """Clears the day and must do, and removes the MILES block through the
    guarded MILES only delete (nightly_review._remove_block_of)."""
    try:
        return {"task": tasks_view.to_backlog(task_id, nightly_review._remove_block_of)}
    except tasks_view.TaskError as exc:
        raise _task_http(exc)


# ── Sessions ──
# Dedicated chats with their own purpose, prompt and tools; the shapes are in
# docs/SESSIONS_CONTRACT.md and the engine is sessions.py.

class SessionStart(BaseModel):
    type: str

class SessionTurn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def _session_http(exc: sessions.SessionError):
    return HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/sessions")
def list_sessions(status: Literal["open", "closed", "missed", "all"] = "open",
                  user: str = Depends(get_current_user)):
    return {"sessions": sessions.listing(status)}


@app.get("/sessions/{session_id}")
def get_session(session_id: int, user: str = Depends(get_current_user)):
    """Opening it here is what stops the reminder push."""
    try:
        return sessions.view(session_id)
    except sessions.SessionError as exc:
        raise _session_http(exc)


@app.post("/sessions", status_code=201)
def start_session(body: SessionStart, response: Response,
                  user: str = Depends(get_current_user)):
    """201 with a new one; 200 with the one already there for a once a day type.
    Never pushes: he is the one who started it."""
    try:
        session, created = sessions.create(body.type)
    except sessions.SessionError as exc:
        raise _session_http(exc)
    if not created:
        response.status_code = 200
        return sessions.body(session)
    # He started it, so the timer must never announce it: mark both pushes as
    # done. Without this, a review started by hand just after 21:30 and not yet
    # fetched by the app could buzz his phone with a session already on screen.
    database.mark_session_pushed(session["id"], "push")
    database.mark_session_pushed(session["id"], "reminder")
    return sessions.body(sessions.get(session["id"]))


@app.post("/sessions/{session_id}/turns")
async def session_turn(session_id: int, body: SessionTurn,
                       user: str = Depends(get_current_user)):
    """One turn, streamed like /chat/stream so the app reuses its reader.

    404 and 409 are answered before the stream starts, while a status code can
    still be sent. The turn runs on a worker thread and finishes whatever
    happens to this response, since it is storing every message as it goes."""
    try:
        session = sessions.get(session_id)
    except sessions.SessionError as exc:
        raise _session_http(exc)
    if session["status"] != "open":
        raise HTTPException(status_code=409, detail="That session is not open")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_text(kind: str, text: str):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, text))

    def run_turn():
        try:
            on_text("done", sessions.run_turn(session_id, body.message, on_text=on_text))
        except Exception as exc:
            on_text("error", str(exc))

    threading.Thread(target=run_turn, daemon=True).start()

    async def events():
        while True:
            try:
                kind, text = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield _sse(kind, text)
            if kind in ("done", "error"):
                return

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/sessions/{session_id}/close")
def close_session(session_id: int, user: str = Depends(get_current_user)):
    try:
        return {"session": sessions.public(sessions.close(session_id))}
    except sessions.SessionError as exc:
        raise _session_http(exc)


# ── Push notifications ──

@app.post("/push/register")
def register_push(body: PushRegistration, user: str = Depends(get_current_user)):
    # Called on every launch, so this is an upsert and a repeat is not an error.
    save_push_device(body.token.lower(), body.environment, body.device_name)
    return {"registered": True}


@app.post("/push/unregister")
def unregister_push(body: PushToken, user: str = Depends(get_current_user)):
    # Signing out should stop Nova's messages reaching that handset. Answering
    # the same whether or not the token was known keeps this safe to retry.
    remove_push_device(body.token.lower())
    return {"unregistered": True}


@app.post("/push/test")
def test_push(user: str = Depends(get_current_user)):
    """Send a notification now, so the whole chain can be checked without
    waiting for a reminder: the key, the team, the bundle id, and whether the
    phone registered against the right APNs host."""
    gaps = push.missing()
    if gaps:
        raise HTTPException(status_code=503,
                            detail="Push is not set up on the Pi yet: " + ", ".join(gaps))
    sent = push.notify("Nova", "This is a test notification.", kind="test")
    if sent == 0:
        raise HTTPException(status_code=502,
                            detail="Apple did not accept it. Check journalctl -u miles-server.")
    return {"sent": sent}


# ── Approvals: tapped in Nova, carried out by the Mac ──

def get_agent_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    """A full login or the Mac's agent token, for the agent's own routes only."""
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise _unauthorized()
    if payload.get("scope") not in (None, AGENT_SCOPE):
        raise _unauthorized()
    return payload["sub"]


class ApprovalProposal(BaseModel):
    kind: Literal["imessage", "evallo_reschedule"]
    summary: str = Field(min_length=1, max_length=500)
    text: Optional[str] = Field(default=None, max_length=approvals.MAX_TEXT_CHARS)
    chat_name: Optional[str] = Field(default=None, max_length=200)
    payload: dict = Field(default_factory=dict)
    dedupe_key: Optional[str] = Field(default=None, max_length=200)


class ApprovalResult(BaseModel):
    ok: bool
    result: str = Field(default="", max_length=2000)


class AgentNote(BaseModel):
    title: str = Field(default="MILES", max_length=60)
    body: str = Field(min_length=1, max_length=500)
    # The whole message, for the app's error screen (the lock screen shows body).
    detail: Optional[str] = Field(default=None, max_length=1800)
    approval_id: Optional[int] = None
    error: bool = False


class ContactNames(BaseModel):
    contacts: dict[str, str]


def _approval_http(exc):
    return HTTPException(status_code=exc.code, detail=str(exc))


@app.post("/auth/agent", response_model=TokenResponse)
def issue_agent_token(user: str = Depends(get_current_user)):
    """The Mac's long lived token. Needs a full login to get."""
    return TokenResponse(access_token=create_token(
        user, expires_minutes=AGENT_TOKEN_EXPIRE_DAYS * 24 * 60, scope=AGENT_SCOPE))


@app.get("/approvals")
def pending_approvals(user: str = Depends(get_current_user)):
    return {"approvals": approvals.list_pending()}


@app.get("/approvals/{approval_id}")
def one_approval(approval_id: int, user: str = Depends(get_current_user)):
    try:
        return approvals.get(approval_id)
    except approvals.ApprovalError as exc:
        raise _approval_http(exc)


@app.post("/approvals/{approval_id}/approve")
def approve_request(approval_id: int, user: str = Depends(get_current_user)):
    # get_current_user refuses scoped tokens: the Mac can never approve.
    try:
        return approvals.decide(approval_id, True)
    except approvals.ApprovalError as exc:
        raise _approval_http(exc)


@app.post("/approvals/{approval_id}/skip")
def skip_request(approval_id: int, user: str = Depends(get_current_user)):
    try:
        return approvals.decide(approval_id, False)
    except approvals.ApprovalError as exc:
        raise _approval_http(exc)


@app.post("/agent/approvals", status_code=201)
def agent_propose(body: ApprovalProposal, user: str = Depends(get_agent_user)):
    try:
        row, created = approvals.propose(body.kind, body.summary, body.text, body.chat_name,
                                         body.payload, body.dedupe_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    pushed = approvals.notify_new(row) if created else 0
    return {"approval": row, "created": created, "pushed": pushed}


@app.post("/agent/approvals/claim")
def agent_claim(kinds: Optional[str] = None, user: str = Depends(get_agent_user)):
    wanted = set(kinds.split(",")) if kinds else None
    return {"approvals": approvals.claim_approved(wanted)}


@app.post("/agent/approvals/{approval_id}/result")
def agent_result(approval_id: int, body: ApprovalResult, user: str = Depends(get_agent_user)):
    try:
        return {"status": approvals.report(approval_id, body.ok, body.result)}
    except approvals.ApprovalError as exc:
        raise _approval_http(exc)


@app.post("/agent/notify")
def agent_notify(body: AgentNote, user: str = Depends(get_agent_user)):
    # Rides beside "aps": a tap opens the error screen (error) or the approvals
    # sheet on approval_id.
    data = {"kind": "agent", "level": "error" if body.error else "info",
            "detail": body.detail or body.body}
    if body.approval_id is not None:
        data["approval_id"] = body.approval_id
    return {"sent": push.notify(body.title, body.body, kind="agent", data=data)}


class Heartbeat(BaseModel):
    source: Literal["poll", "draft", "summaries", "login_check", "retry"]


class JobReport(BaseModel):
    job: Literal["draft", "summaries"]
    day: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    ok: bool
    detail: str = Field(default="", max_length=2000)


class LoginExpiredNote(BaseModel):
    service: Literal["zoom", "google", "evallo"]
    job: str = Field(min_length=1, max_length=200)


@app.post("/agent/heartbeat")
def agent_heartbeat(body: Heartbeat, user: str = Depends(get_agent_user)):
    """The Mac checking in. The answer lists missed jobs he asked to retry."""
    return {"retry": agent_watchdog.heartbeat(body.source)}


@app.post("/agent/jobs/report")
def agent_job_report(body: JobReport, user: str = Depends(get_agent_user)):
    agent_watchdog.job_report(body.job, body.day, body.ok, body.detail)
    return {"recorded": True}


@app.post("/agent/login-expired")
def agent_login_expired(body: LoginExpiredNote, user: str = Depends(get_agent_user)):
    """Once per service per day; the text and the command are fixed here."""
    return {"pushed": agent_watchdog.login_expired(body.service, body.job)}


@app.get("/watchdog")
def watchdog_alerts(user: str = Depends(get_current_user)):
    return {"alerts": agent_watchdog.open_alerts()}


@app.post("/watchdog/{alert_id}/{decision}")
def watchdog_decide(alert_id: int, decision: Literal["retry", "remind", "skip"],
                    user: str = Depends(get_current_user)):
    try:
        return agent_watchdog.decide(alert_id, decision)
    except agent_watchdog.AlertError as exc:
        raise HTTPException(status_code=exc.code, detail=str(exc))


@app.put("/agent/contacts")
def agent_contacts(body: ContactNames, user: str = Depends(get_agent_user)):
    approvals.set_contacts(body.contacts)
    return {"saved": len(body.contacts)}


# ── Device token and phone presence ──

@app.post("/auth/device", response_model=TokenResponse)
def issue_device_token(user: str = Depends(get_current_user)):
    """The phone's long lived token for background writes. Needs a full login,
    so it can only be had by someone who could already do everything it can."""
    return TokenResponse(access_token=create_token(
        user, expires_minutes=DEVICE_TOKEN_EXPIRE_DAYS * 24 * 60, scope=DEVICE_SCOPE))


@app.post("/presence/phone")
def report_phone_presence(body: PhonePresence, user: str = Depends(get_device_user)):
    """A geofence crossing. Logged only: nothing acts on it until a few days of
    these show how promptly and reliably iOS reports them."""
    at = body.at.astimezone().replace(tzinfo=None) if body.at.tzinfo else body.at
    log_phone_presence(body.state, at.isoformat())
    return {"logged": True}


@app.get("/presence/phone")
def phone_presence(user: str = Depends(get_current_user)):
    latest = latest_phone_presence()
    if latest is None:
        return {"state": None, "at": None}
    return {"state": latest[0], "at": latest[1]}


# ── Health ──

@app.post("/health/sync")
def sync_health(body: HealthSync, user: str = Depends(get_device_user)):
    """The app's copy of its Health data. Each day replaces the stored day whole,
    so a reading deleted in Health disappears here on the next sync."""
    days = [{"day": d.day.isoformat(), "steps": d.steps,
             "active_kcal": d.active_kcal, "weight_lb": d.weight_lb}
            for d in body.days]
    workouts = []
    for w in body.workouts:
        # The Pi reads times as its own local time everywhere else. The app sends
        # an instant, so convert it, or a 7 AM run would be stored as noon.
        started = w.started_at.astimezone().replace(tzinfo=None) if w.started_at.tzinfo else w.started_at
        workouts.append({"id": w.id, "kind": w.kind, "started_at": started.isoformat(),
                         "minutes": w.minutes, "distance_mi": w.distance_mi, "kcal": w.kcal})
    # Same conversion: the hour that started at 8 AM here is stored as 08:00
    hours = [{"start": (h.start.astimezone().replace(tzinfo=None) if h.start.tzinfo
                        else h.start).isoformat(), "steps": h.steps}
             for h in body.hours]
    save_health_sync(days, workouts, hours)
    return {"days": len(days), "workouts": len(workouts)}


def _calendar_call(fn, *args, **kwargs):
    """Run a calendar function, turning its refusals into HTTP answers. A lookup
    miss is 404, a bad time is 400, and anything from Google itself is 502 with
    its message, because an expired token has to read as that and not as an
    empty week."""
    try:
        return fn(*args, **kwargs)
    except calendar_tools.EventLookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except calendar_tools.WhenError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Google Calendar: {exc}")


@app.get("/calendar/events")
def calendar_events(days: int = 7, user: str = Depends(get_current_user)):
    """From the start of today, so this morning's class still shows."""
    days = max(1, min(days, 31))
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return _calendar_call(calendar_tools.events_for_app, start, start + timedelta(days=days))


@app.patch("/calendar/events/{event_id}")
def change_calendar_event(event_id: str, body: EventChange, user: str = Depends(get_current_user)):
    said = _calendar_call(calendar_tools.change_event_for_app, event_id,
                          title=body.title, start=body.start, end=body.end)
    return {"result": said}


@app.delete("/calendar/events/{event_id}")
def delete_calendar_event(event_id: str, user: str = Depends(get_current_user)):
    return {"result": _calendar_call(calendar_tools.delete_event_for_app, event_id)}


_SERVICES = ("miles-voice", "miles-server", "miles-tunnel")


def _services_active():
    """Whether each long running service is up. Kept out of get_system_state,
    whose output Nova reads aloud, so the app view cannot change what she says."""
    result = subprocess.run(["systemctl", "is-active", *_SERVICES],
                            capture_output=True, text=True, timeout=5)
    states = result.stdout.split()
    return {name: state == "active" for name, state in zip(_SERVICES, states)}


@app.get("/status/details")
def status_details(user: str = Depends(get_current_user)):
    """What get_system_state tells Nova, plus which services are running."""
    try:
        services = _services_active()
    except Exception:
        # An unknown is not a failure of the page; the other facts still stand.
        services = {name: None for name in _SERVICES}
    return {**get_system_state(), "services": services}


@app.get("/status")
def status_check(user: str = Depends(get_current_user)):
    return {"status": "online", "version": "0.7"}


# ── WebSocket ──

@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()

    # First message must carry the auth token
    try:
        auth_msg = await ws.receive_json()
        token = auth_msg.get("token", "")
        payload = decode_token(token)
        _ = payload["sub"]
    except (JWTError, KeyError, Exception):
        await ws.send_json({"error": "Unauthorized"})
        await ws.close(code=1008)
        return

    await ws.send_json({"status": "authenticated"})

    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "").strip()
            if not message:
                continue
            # Same default as the REST endpoint: a socket frame without a
            # channel is treated as voice.
            channel = data.get("channel", "voice")
            if channel not in ("voice", "text"):
                channel = "voice"
            response = ask_nova(message, device="app", channel=channel).text
            await ws.send_json({"response": response})
    except WebSocketDisconnect:
        pass
