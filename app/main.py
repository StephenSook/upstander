"""Upstander web app: group chat + stage view + mounted MCP server. Run: uvicorn app.main:app --port 8000"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import agent, detect  # noqa: E402
from .mcp_tools import mcp  # noqa: E402
from .state import MEMBERS, ROOM, add_message, broadcast, message_dict, reset, subscribe, unsubscribe, verify_chain  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/mcp", mcp_app)


class Post(BaseModel):
    sender: str
    text: str


class Consent(BaseModel):
    target: str
    answer: str  # yes | no


@app.get("/")
def stage():
    return FileResponse(STATIC / "stage.html")


@app.get("/api/state")
def state():
    return {"room": ROOM.name, "members": MEMBERS, "messages": [message_dict(m) for m in ROOM.messages],
            "incidents": ROOM.incidents, "calls": ROOM.calls, "chain": verify_chain()}


@app.post("/api/message")
async def post_message(p: Post):
    m = add_message(p.sender, p.text)
    await asyncio.to_thread(detect.analyze, m)
    detect.pile_on()  # attaches target attribution to every message in window
    broadcast({"type": "message", "data": message_dict(m)})
    if m.safety.get("hostile"):
        asyncio.create_task(agent.wake(
            f"New message #{m.id} from {m.sender} in the group chat flagged by signals "
            f"{m.safety.get('lexicon')} and Content Safety max severity {m.safety.get('max_severity')}. "
            "Investigate and act per your decision process."))
    return message_dict(m)


@app.post("/api/consent")
async def consent(c: Consent):
    ROOM.consent[c.target] = c.answer
    broadcast({"type": "consent", "target": c.target, "answer": c.answer})
    last = ROOM.incidents[-1]["incident"] if ROOM.incidents else None
    verb = "said YES, please tell my trusted adult" if c.answer == "yes" else "said NOT NOW"
    asyncio.create_task(agent.wake(f"{c.target} {verb}. Latest incident id: {last}. Act per your decision process."))
    return {"ok": True}


@app.post("/api/reset")
def do_reset():
    reset()
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q = subscribe()
    try:
        while True:
            ev = await q.get()
            await sock.send_text(json.dumps(ev, default=str))
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe(q)
