import asyncio
import json
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from typing import Literal

from pydantic import BaseModel
from jose import JWTError

from auth import verify_password, create_token, decode_token
from brain import ask_nova
from database import (
    init_db, get_active_memories, get_pending_memories,
    approve_memory, delete_memory, get_history,
)

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
    rows = get_pending_memories(limit=100)
    return [
        {"id": r[0], "content": r[1], "category": r[2], "source": r[3], "confidence": r[4]}
        for r in rows
    ]


@app.post("/memories/{memory_id}/approve")
def approve_pending_memory(memory_id: int, user: str = Depends(get_current_user)):
    if not approve_memory(memory_id):
        raise HTTPException(status_code=404, detail="Pending memory not found")
    return {"approved": True}


@app.delete("/memories/{memory_id}")
def remove_memory(memory_id: int, user: str = Depends(get_current_user)):
    if not delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@app.get("/history")
def history(limit: int = 50, offset: int = 0, user: str = Depends(get_current_user)):
    return get_history(limit=limit, offset=offset)


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
