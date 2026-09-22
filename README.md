# Upstander

An AI agent that notices when a group chat turns on one kid, steps in privately, and, with that kid's permission, gets a trusted adult on the phone.

Built in one day for the Code for Change Hackathon (Blacks in Technology Atlanta, Microsoft Foundry track), September 22, 2026.

## The problem

- In the 2022 School Crime Supplement, an adult was notified in only 44.2% of bullying cases (NCES 2024-109, Table 2.7).
- 16% of U.S. high school students were electronically bullied in the past year (CDC YRBS 2023).
- The hardest form to catch is a pile-on in a private group chat: several kids aiming mild-looking messages ("nobody invited u", "just leave the gc") at one kid. Each message scores low on its own; the pattern is the harm.

## What it does

1. Every chat message is scored by Azure AI Content Safety plus our own signals for exclusion, mockery, self-harm pushes and threats.
2. A flagged message wakes the Upstander agent. Nobody prompts it.
3. The agent reads the chat, scores the pile-on (distinct hostile senders aimed at one person in a 10-minute window), and saves tamper-evident evidence.
4. It privately nudges each sender and privately checks in with the targeted kid: "It's not your fault. Want me to tell your trusted adult?"
5. If the kid taps Yes, the agent places a real phone call and reads the adult a short summary with an evidence reference code.

The agent never posts in the group chat. Everything it says is a private message.

## Architecture

| Piece | Implementation |
|---|---|
| Model | Microsoft Foundry project, `gpt-4.1-mini` deployment |
| Agent runtime | Microsoft Agent Framework (Python), `FoundryChatClient` |
| Tools | Our own MCP server (FastMCP, streamable HTTP) mounted at `/mcp`, 8 tools |
| Harm scoring | Azure AI Content Safety (hate, sexual, violence, self-harm) |
| Pattern detection | Deterministic pile-on scorer in `app/detect.py` |
| Evidence | Append-only log, each record SHA-256 chained to the previous one |
| Phone call | Vonage Voice API, text-to-speech |
| UI | FastAPI + WebSocket: a stage view (`/`) and a phone chat (`/chat`) |

MCP tools: `get_recent_messages`, `analyze_text_safety`, `score_pile_on`, `lookup_guidance`, `log_evidence`, `send_private_nudge`, `check_in_with_target`, `call_trusted_adult`. Every tool takes a `reason`, which is shown live so a reviewer can see why the agent acted.

## Safety design

- **Consent is enforced in code, not in the prompt.** `call_trusted_adult` refuses unless the targeted kid said yes, or the incident contains an imminent-risk signal (Content Safety self-harm or violence at 4+, a self-harm push, or a threat). In the override case the kid is told privately that their adult was informed.
- **Deterministic safety net.** If an agent run ends with an imminent-risk target and no call placed, the system places the call. In testing, the model sometimes skipped this rule, so it does not rely on the model.
- **One call per kid per 10 minutes**, so an adult is not flooded.
- **Privacy.** Evidence stores only the messages in the incident. The agent has no tool that posts publicly.
- **Bounded runs.** Each agent run times out after 45 seconds so a hung model call cannot silence the agent.

## Limitations

- Demo only: there is no login, so anyone with the link can pick any name, and private messages are not access-controlled on the server. A real deployment would authenticate each member.
- English only; the slang signal list needs ongoing updates; target attribution uses names and second-person pronouns and can guess wrong.
- Not yet measured against a labeled dataset. The demo chat is a scripted scenario, not real students' messages.
- A bridge to a trusted adult, not a replacement for a counselor.

## Run it

```
uv venv .venv
uv pip install --python .venv/bin/python agent-framework fastapi "uvicorn[standard]" azure-identity \
  azure-ai-contentsafety pyjwt cryptography requests python-dotenv mcp
cp .env.example .env   # fill in Foundry, Content Safety and Vonage values
az login               # the agent authenticates to Foundry with your Azure CLI login
./scripts/restart.sh   # starts exactly one server on http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000`, then press Play scenario. To let phones join, expose port 8000 (for example `cloudflared tunnel --url http://localhost:8000`) and open `/chat`.

## License

MIT
