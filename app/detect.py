"""Signals the agent reasons over.

1. Azure AI Content Safety (Foundry) severities per message: hate, sexual, violence, self-harm.
2. An exclusion / mockery lexicon, because Content Safety is built for overt harm and the most
   common form teens report (being pushed out of a group chat) is usually polite-looking text.
3. A deterministic pile-on score: how many DISTINCT people are aiming hostile messages at ONE
   person inside a sliding window. Single-message classifiers cannot see this; it only exists
   across senders and time.
"""
from __future__ import annotations

import os
import re
import time

from .state import MEMBERS, ROOM, Message

WINDOW_SECONDS = 600
PILE_ON_SENDERS = 3  # distinct hostile senders at one target = pile-on

# Exclusion and mockery patterns that overt-harm classifiers under-score.
LEXICON = {
    "exclusion": [r"\bleave (the )?(gc|chat|group)\b", r"\bnobody (invited|asked|wants|likes)\b",
                  r"\bno one (invited|asked|wants|likes)\b", r"\bkick (her|him|them)\b",
                  r"\bnot invited\b", r"\bremove (her|him|them)\b", r"\bget out\b"],
    "mockery": [r"💀", r"😂😂", r"\blmao+\b", r"\blook at (her|his|their|\w+'s)\b", r"\bcringe\b",
                r"\bweirdo\b", r"\bloser\b", r"\bugly\b", r"\bfat\b"],
    "self_harm_push": [r"\bkys\b", r"\bkill (your|ur)self\b", r"\bunalive\b", r"\bnobody would miss\b"],
    "threat": [r"\bjump (her|him|you|u)\b", r"\bafter school\b.*\b(fight|catch)\b", r"\bpull up\b"],
}

_cs_client = None


def _content_safety():
    global _cs_client
    if _cs_client is None:
        endpoint = os.getenv("CONTENT_SAFETY_ENDPOINT")
        if not endpoint:
            return None
        from azure.ai.contentsafety import ContentSafetyClient
        key = os.getenv("CONTENT_SAFETY_KEY")
        if key:
            from azure.core.credentials import AzureKeyCredential
            cred = AzureKeyCredential(key)
        else:
            from azure.identity import AzureCliCredential
            cred = AzureCliCredential()
        _cs_client = ContentSafetyClient(endpoint, cred)
    return _cs_client


def content_safety(text: str) -> dict:
    """Returns {category: severity 0..7} from Azure AI Content Safety, or an explicit error."""
    client = _content_safety()
    if client is None:
        return {"error": "CONTENT_SAFETY_ENDPOINT not configured"}
    from azure.ai.contentsafety.models import AnalyzeTextOptions
    try:
        res = client.analyze_text(AnalyzeTextOptions(text=text[:1000]))
        return {c.category.lower().replace("selfharm", "self_harm"): c.severity for c in res.categories_analysis}
    except Exception as e:  # surfaced to the agent and the stage, never swallowed
        return {"error": f"content_safety_failed: {type(e).__name__}: {str(e)[:120]}"}


def lexicon_hits(text: str) -> list[str]:
    t = text.lower()
    return sorted({cat for cat, pats in LEXICON.items() for p in pats if re.search(p, t)})


def analyze(m: Message) -> dict:
    cs = content_safety(m.text)
    hits = lexicon_hits(m.text)
    sev = max((v for k, v in cs.items() if k != "error" and isinstance(v, int)), default=0)
    m.safety = {"content_safety": cs, "lexicon": hits, "max_severity": sev,
                "hostile": sev >= 2 or bool(hits)}
    return m.safety


IMMINENT_SEVERITY = 4


def is_imminent(m: Message) -> bool:
    """Self-harm push or threat: the cases that justify telling an adult without waiting for consent."""
    cs = m.safety.get("content_safety", {})
    sev = max(cs.get("self_harm", 0) or 0, cs.get("violence", 0) or 0)
    return sev >= IMMINENT_SEVERITY or bool({"self_harm_push", "threat"} & set(m.safety.get("lexicon", [])))


def _target_of(m: Message, prev: list[Message]) -> str | None:
    """Who a message is aimed at: an explicit name, else second person right after someone spoke."""
    t = m.text.lower()
    for name in MEMBERS:
        if name != m.sender and re.search(rf"\b{name.lower()}('s)?\b", t):
            return name
    if re.search(r"\b(you|u|ur|your|urself)\b", t):
        for p in reversed(prev[-4:]):
            if p.sender != m.sender and not p.safety.get("hostile"):
                return p.sender
    # joining an ongoing attack ("fr", "💀") inherits the current target
    for p in reversed(prev[-3:]):
        if p.safety.get("hostile") and p.safety.get("target") and p.sender != m.sender:
            return p.safety["target"]
    return None


def pile_on(now: float | None = None) -> dict:
    now = now or time.time()
    msgs = [m for m in ROOM.messages if now - m.ts <= WINDOW_SECONDS]
    by_target: dict[str, dict] = {}
    for i, m in enumerate(msgs):
        if not m.safety:
            continue
        tgt = _target_of(m, msgs[:i])
        m.safety["target"] = tgt
        if not (m.safety.get("hostile") and tgt):
            continue
        d = by_target.setdefault(tgt, {"senders": set(), "message_ids": [], "max_severity": 0, "categories": set(),
                                       "imminent": False})
        d["senders"].add(m.sender)
        d["message_ids"].append(m.id)
        d["imminent"] = d["imminent"] or is_imminent(m)
        d["max_severity"] = max(d["max_severity"], m.safety.get("max_severity", 0))
        d["categories"].update(m.safety.get("lexicon", []))
    out = []
    for tgt, d in by_target.items():
        k = len(d["senders"])
        level = "pile_on" if k >= PILE_ON_SENDERS else ("forming" if k == 2 else "single")
        row = {"target": tgt, "distinct_hostile_senders": sorted(d["senders"]), "count": k,
               "level": level, "message_ids": d["message_ids"], "max_severity": d["max_severity"],
               "signals": sorted(d["categories"]), "imminent_risk": d["imminent"]}
        if d["imminent"]:
            row["required_action"] = ("IMMINENT RISK (self-harm push or threat). Call log_evidence, then "
                                      "call_trusted_adult NOW without waiting for consent, then tell the target privately.")
        out.append(row)
    out.sort(key=lambda x: -x["count"])
    return {"window_seconds": WINDOW_SECONDS, "threshold_distinct_senders": PILE_ON_SENDERS, "targets": out}
