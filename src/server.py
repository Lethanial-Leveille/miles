import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError

from auth import verify_password, create_token, decode_token
from brain import ask_nova
from database import (
    init_db, get_memories, delete_memory,
    get_history,
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
    response = ask_nova(body.message, device="app")
    return ChatResponse(response=response)


@app.get("/memories")
def list_memories(user: str = Depends(get_current_user)):
    rows = get_memories(limit=100)
    return [{"id": row[0], "content": row[1]} for row in rows]


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
            response = ask_nova(message, device="app")
            await ws.send_json({"response": response})
    except WebSocketDisconnect:
        pass
