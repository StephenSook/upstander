"""In-memory room state shared by the chat app and the MCP tools, plus the hash-chained evidence log."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
EVIDENCE_FILE = DATA / "evidence.jsonl"

MEMBERS = ["Maya", "Jay", "Tre", "Kai", "Ava"]


@dataclass
class Message:
    id: int
    sender: str
    text: str
    ts: float
    safety: dict = field(default_factory=dict)  # Content Safety severities + lexicon hits


@dataclass
class Room:
    name: str = "8th grade gc"
    messages: list[Message] = field(default_factory=list)
    consent: dict[str, str] = field(default_factory=dict)  # target -> "yes" | "no"
    incidents: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    nudged: set[str] = field(default_factory=set)
    asked: set[str] = field(default_factory=set)


ROOM = Room()
_next_id = 1
_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def broadcast(event: dict) -> None:
    event.setdefault("ts", time.time())
    for q in list(_subscribers):
        q.put_nowait(event)


def add_message(sender: str, text: str) -> Message:
    global _next_id
    m = Message(id=_next_id, sender=sender, text=text, ts=time.time())
    _next_id += 1
    ROOM.messages.append(m)
    return m


def reset() -> None:
    global _next_id
    ROOM.messages.clear()
    ROOM.consent.clear()
    ROOM.incidents.clear()
    ROOM.calls.clear()
    ROOM.nudged.clear()
    ROOM.asked.clear()
    _next_id = 1
    broadcast({"type": "reset"})


def message_dict(m: Message) -> dict:
    return asdict(m)


# ---------- evidence log: append-only, SHA-256 hash chain ----------

def _last_hash() -> str:
    if not EVIDENCE_FILE.exists():
        return "0" * 64
    last = "0" * 64
    for line in EVIDENCE_FILE.read_text().splitlines():
        if line.strip():
            last = json.loads(line)["hash"]
    return last


def append_evidence(record: dict) -> dict:
    prev = _last_hash()
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((prev + body).encode()).hexdigest()
    entry = {**record, "prev_hash": prev, "hash": digest}
    with EVIDENCE_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def verify_chain() -> dict:
    """Recompute every hash; any edited line breaks the chain from that point on."""
    if not EVIDENCE_FILE.exists():
        return {"entries": 0, "valid": True}
    prev = "0" * 64
    n = 0
    for line in EVIDENCE_FILE.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        rec = {k: v for k, v in e.items() if k not in ("prev_hash", "hash")}
        body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        if e["prev_hash"] != prev or hashlib.sha256((prev + body).encode()).hexdigest() != e["hash"]:
            return {"entries": n, "valid": False, "broken_at": n}
        prev = e["hash"]
        n += 1
    return {"entries": n, "valid": True}
