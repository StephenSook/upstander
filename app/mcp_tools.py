"""Upstander MCP server: the only way the agent can see or touch the room.

Every tool takes a `reason` argument, so each step of the agent's decision is logged with its
own justification and shown on the stage. Safety rules live HERE, in code, not in the prompt:
the agent cannot call a trusted adult without the target's consent unless Content Safety flags
imminent risk, and it can never post into the group chat.
"""
from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

from . import detect, notify
from .state import MEMBERS, ROOM, append_evidence, broadcast, verify_chain

mcp = FastMCP("upstander", stateless_http=True, json_response=True, streamable_http_path="/")

IMMINENT_SEVERITY = 4  # Content Safety self-harm or violence at or above this = imminent-risk override

KNOWLEDGE = {
    "response_ladder": (
        "Upstander response ladder (least intrusive first): "
        "1) observe and keep evidence; 2) private nudge to each sender, never public shaming; "
        "3) private check-in with the target that names what happened and offers choices; "
        "4) with the target's consent, notify the ONE trusted adult the target chose; "
        "5) without consent ONLY for imminent risk (self-harm push, credible threat), and tell the target it happened."
    ),
    "privacy": (
        "Store only the messages in the incident, not the whole chat. Evidence is hash-chained so a "
        "school can verify it was not edited. Never reveal a private check-in to other members. "
        "Never post in the group chat; the agent speaks only in private messages."
    ),
    "school_reporting": (
        "Georgia requires every school district to have a bullying policy that covers electronic "
        "communication and a way to notify parents or guardians (O.C.G.A. 20-2-751.4). A dated, "
        "verifiable incident record is what a school administrator needs to act."
    ),
    "why_it_matters": (
        "In the 2022 NCES School Crime Supplement, an adult was notified in 44.2 percent of bullying "
        "cases, so most bullied students never tell an adult. CDC YRBS 2023: 16 percent of high school "
        "students were electronically bullied in the past year."
    ),
    "exclusion": (
        "Exclusion (being pushed out of a group chat, 'nobody invited you', 'leave the gc') is bullying "
        "even when no single message contains a slur. Judge the pattern: many senders, one target, short window."
    ),
}


def _trace(tool: str, reason: str, result: dict | str) -> None:
    broadcast({"type": "trace", "tool": tool, "reason": reason, "result": result})


@mcp.tool()
def get_recent_messages(reason: str, limit: int = 20) -> dict:
    """Read the most recent group-chat messages with their Content Safety severities and lexicon signals."""
    msgs = [{"id": m.id, "sender": m.sender, "text": m.text,
             "content_safety": m.safety.get("content_safety"), "signals": m.safety.get("lexicon"),
             "max_severity": m.safety.get("max_severity")} for m in ROOM.messages[-limit:]]
    out = {"room": ROOM.name, "members": MEMBERS, "messages": msgs}
    _trace("get_recent_messages", reason, f"{len(msgs)} messages")
    return out


@mcp.tool()
def analyze_text_safety(reason: str, text: str) -> dict:
    """Run Azure AI Content Safety on a piece of text. Returns severity 0-7 for hate, sexual, violence, self_harm."""
    res = detect.content_safety(text)
    _trace("analyze_text_safety", reason, res)
    return res


@mcp.tool()
def score_pile_on(reason: str) -> dict:
    """Deterministic pile-on detector: counts DISTINCT hostile senders aimed at one target in a 10 minute window.
    level = pile_on when 3+ distinct senders target one person, forming at 2."""
    res = detect.pile_on()
    _trace("score_pile_on", reason, {t["target"]: f'{t["level"]} ({t["count"]} senders)' for t in res["targets"]} or "no target")
    broadcast({"type": "pile_on", "data": res})
    return res


@mcp.tool()
def lookup_guidance(reason: str, topic: str) -> dict:
    """Knowledge base. topic is one of: response_ladder, privacy, school_reporting, why_it_matters, exclusion."""
    text = KNOWLEDGE.get(topic, "unknown topic; options: " + ", ".join(KNOWLEDGE))
    _trace("lookup_guidance", reason, topic)
    return {"topic": topic, "guidance": text}


@mcp.tool()
def log_evidence(reason: str, target: str, message_ids: list[int], summary: str) -> dict:
    """Append an incident to the tamper-evident evidence log (SHA-256 hash chain). Stores only the listed messages."""
    msgs = [{"id": m.id, "sender": m.sender, "text": m.text, "ts": round(m.ts, 3)}
            for m in ROOM.messages if m.id in set(message_ids)]
    rec = {"incident": len(ROOM.incidents) + 1, "room": ROOM.name, "target": target,
           "summary": summary, "messages": msgs, "logged_at": round(time.time(), 3)}
    entry = append_evidence(rec)
    ROOM.incidents.append(entry)
    out = {"incident_id": entry["incident"], "hash": entry["hash"], "prev_hash": entry["prev_hash"],
           "messages_stored": len(msgs), "chain": verify_chain()}
    _trace("log_evidence", reason, {"incident": entry["incident"], "hash": entry["hash"][:16] + "..."})
    broadcast({"type": "evidence", "data": out})
    return out


@mcp.tool()
def send_private_nudge(reason: str, to: str, text: str) -> dict:
    """Send a PRIVATE message to someone who sent a hostile message. Never public. Keep it short, specific, non-shaming."""
    if to not in MEMBERS:
        return {"error": f"unknown member {to}"}
    if to in ROOM.nudged:
        res = {"delivered": False, "why": f"{to} was already nudged; one nudge per sender avoids piling on the piler"}
        _trace("send_private_nudge", reason, res)
        return res
    ROOM.nudged.add(to)
    broadcast({"type": "dm", "to": to, "kind": "nudge", "text": text})
    _trace("send_private_nudge", reason, f"to {to}")
    return {"delivered": True, "to": to}


@mcp.tool()
def check_in_with_target(reason: str, to: str, text: str) -> dict:
    """Privately check in with the person being targeted and ASK FOR CONSENT to tell their trusted adult.
    Shows them buttons: 'Yes, tell <adult>' / 'Not now'. Their answer comes back to you as a new event."""
    if to not in MEMBERS:
        return {"error": f"unknown member {to}"}
    adult = notify.trusted_adult_name()
    if to in ROOM.asked:
        answered = ROOM.consent.get(to)
        why = (f"{to} already answered '{answered}'; do not ask again" if answered
               else f"{to} already has an open check-in; wait for their answer")
        res = {"delivered": False, "already_asked": True, "answer": answered, "why": why}
        _trace("check_in_with_target", reason, res)
        return res
    ROOM.asked.add(to)
    broadcast({"type": "dm", "to": to, "kind": "checkin", "text": text, "adult": adult})
    _trace("check_in_with_target", reason, f"asked {to} for consent to tell {adult}")
    return {"delivered": True, "to": to, "awaiting_consent": True, "trusted_adult": adult}


@mcp.tool()
def call_trusted_adult(reason: str, target: str, incident_id: int, spoken_summary: str) -> dict:
    """Place a REAL phone call to the target's trusted adult and read a short spoken summary.
    Allowed only if the target said yes, or Content Safety shows imminent risk (self-harm or violence >= 4).
    The spoken summary must not include slurs or the full messages; name the pattern and the next step."""
    recent = next((c for c in reversed(ROOM.calls)
                   if c["target"] == target and c.get("placed") and time.time() - c.get("at", 0) < 600), None)
    if recent:
        res = {"allowed": False, "why": f"{target}'s trusted adult was already called {int(time.time() - recent['at'])}s ago; "
                                        "one call per incident so the adult is not flooded"}
        _trace("call_trusted_adult", reason, res)
        return res
    consent = ROOM.consent.get(target)
    incident = next((i for i in ROOM.incidents if i["incident"] == incident_id), None)
    if incident is None:
        res = {"allowed": False, "why": f"incident {incident_id} not in evidence log; call log_evidence first"}
        _trace("call_trusted_adult", reason, res)
        return res
    ids = {x["id"] for x in incident["messages"]}
    imminent = any(detect.is_imminent(m) for m in ROOM.messages if m.id in ids)
    worst = "imminent" if imminent else "below imminent threshold"
    if consent == "no" and not imminent:
        res = {"allowed": False, "why": f"{target} declined; respect it and keep offering support"}
    elif consent != "yes" and not imminent:
        res = {"allowed": False, "why": f"no consent from {target} yet and no imminent risk ({worst})"}
    else:
        basis = "consent" if consent == "yes" else "imminent_risk_override"
        call = notify.place_call(spoken_summary, incident_hash=incident["hash"])
        rec = {"target": target, "incident": incident_id, "basis": basis, "at": time.time(), **call}
        ROOM.calls.append(rec)
        broadcast({"type": "call", "data": rec})
        if basis == "imminent_risk_override":
            broadcast({"type": "dm", "to": target, "kind": "info",
                       "text": f"I was worried you might be in danger, so I let {notify.trusted_adult_name()} know. You're not in trouble."})
        res = {"allowed": True, "basis": basis, **call}
    _trace("call_trusted_adult", reason, res)
    return res
