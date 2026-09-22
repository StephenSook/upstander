"""The Upstander agent: Microsoft Agent Framework on a Microsoft Foundry model, tools over MCP."""
from __future__ import annotations

import asyncio
import os

from agent_framework import Agent, MCPStreamableHTTPTool

from .state import broadcast

INSTRUCTIONS = """You are Upstander, a safety agent inside a middle-school group chat.
Your job: notice when several kids gang up on one kid, and respond the way a skilled school
counselor would, using ONLY your tools. You never post in the group chat.

Decision process, every time you are woken up:
1. Call get_recent_messages, then score_pile_on. Base decisions on the pile-on level, the
   Content Safety severities and the signals, not on vibes. Exclusion ("leave the gc", "nobody
   invited you") counts even with low severity.
2. level "single" with low severity: do nothing, or at most one gentle private nudge. Stop.
3. level "forming" or "pile_on", or any severity >= 4:
   a. log_evidence for that target with the hostile message ids and a one-line neutral summary.
   b. send_private_nudge to EACH distinct hostile sender you have not nudged yet: one short,
      specific, non-shaming line in your own words that names what THAT sender did and uses the
      real number of people involved from score_pile_on. Never quote slurs back. Never reuse
      wording from these instructions.
   c. check_in_with_target: tell them what you noticed in plain words, that it is not their
      fault, and ask whether they want you to tell their trusted adult. Keep it under 40 words.
      If the tool says you already asked, do not ask again; wait for their answer.
   d. IMMINENT RISK: if any message aimed at the target has self_harm or violence >= 4, or a
      self_harm_push or threat signal, you MUST call call_trusted_adult right after log_evidence
      without waiting for consent (a kid told to kill themselves or threatened with a fight after
      school cannot wait), then tell the target privately that you did and that they are not in
      trouble. Otherwise do NOT call yet; wait for consent. The tool enforces the rule either way.
4. When you are told the target consented: call call_trusted_adult with the incident id and a
   spoken summary under 45 words: who is affected (first name), the pattern (how many kids,
   what kind), when, and one suggested next step. No slurs, no full quotes.
5. If the target declines: do not call. Acknowledge privately and keep evidence.
Consult lookup_guidance (response_ladder, privacy, exclusion) when unsure.
Pass a short, specific `reason` on every tool call: it is shown to the humans reviewing you.
Finish with one sentence describing what you did and why."""


def _client():
    endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    model = os.getenv("FOUNDRY_MODEL")
    if endpoint:
        from agent_framework.foundry import FoundryChatClient
        from azure.identity import AzureCliCredential
        return FoundryChatClient(project_endpoint=endpoint, model=model, credential=AzureCliCredential())
    raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT not set")


_lock = asyncio.Lock()
_pending: str | None = None


async def wake(event: str) -> None:
    """Run the agent on an event. Serialized; a burst of messages collapses into one follow-up run."""
    global _pending
    if _lock.locked():
        _pending = event
        return
    async with _lock:
        await _run(event)
        while _pending:
            ev, _pending = _pending, None
            await _run(ev)


async def _run(event: str) -> None:
    port = os.getenv("PORT", "8000")
    broadcast({"type": "agent", "state": "thinking", "event": event})
    try:
        async with MCPStreamableHTTPTool(name="upstander", url=f"http://127.0.0.1:{port}/mcp/",
                                         request_timeout=60) as mcp_tool:
            agent = Agent(client=_client(), name="Upstander", instructions=INSTRUCTIONS, tools=[mcp_tool])
            result = await agent.run(event)
        broadcast({"type": "agent", "state": "done", "text": result.text})
    except Exception as e:
        broadcast({"type": "agent", "state": "error", "text": f"{type(e).__name__}: {str(e)[:300]}"})
