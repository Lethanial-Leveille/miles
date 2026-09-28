import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel
from jose import JWTError

from auth import verify_password, create_token, decode_token
from brain import ask_nova
from database import (
    init_db, get_active_memories, get_pending_memories,
    approve_memory, delete_memory, get_history,
    save_memory, supersede_memory, get_memory_chain, memory_status, memory_content,
    open_reminders, cancel_reminder_by_id,
    dismiss_memory_pair, dismissed_memory_pairs, merge_memories,
)
from config import MEMORY_SIMILAR_LIST
from system_state import get_system_state
import embeddings
import memory_merge
import calendar_tools
import tts

app = FastAPI(title="M.I.L.E.S. API", version="0.7")
init_db()

# ── Auth dependency ──

_bearer = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    try:
        payload = decode_token(credentials.credentials)
        return payload["sub"]
    except JWTError:
        # fail closed: any bad or expired token gets a 401, no details leaked
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


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
