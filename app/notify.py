"""Real phone call to the trusted adult via the Vonage Voice API (text-to-speech)."""
from __future__ import annotations

import os
import time
import uuid

import jwt
import requests


def trusted_adult_name() -> str:
    return os.getenv("TRUSTED_ADULT_NAME", "your trusted adult")


def _token() -> str:
    key = open(os.getenv("VONAGE_PRIVATE_KEY_PATH", "./secrets/vonage_private.key"), "rb").read()
    now = int(time.time())
    claims = {"application_id": os.environ["VONAGE_APPLICATION_ID"], "iat": now, "exp": now + 300,
              "jti": str(uuid.uuid4())}
    return jwt.encode(claims, key, algorithm="RS256")


def place_call(spoken_summary: str, incident_hash: str) -> dict:
    to = os.getenv("TRUSTED_ADULT_NUMBER")
    frm = os.getenv("VONAGE_FROM_NUMBER")
    if not (to and frm):
        return {"placed": False, "error": "TRUSTED_ADULT_NUMBER or VONAGE_FROM_NUMBER not set"}
    text = (f"Hello {trusted_adult_name()}. This is Upstander, a safety assistant. {spoken_summary} "
            f"The evidence is saved with reference code {incident_hash[:6]}. "
            f"I'll repeat that. {spoken_summary}")
    ncco = [{"action": "talk", "text": text, "language": "en-US", "style": 2, "premium": True}]
    try:
        r = requests.post("https://api.nexmo.com/v1/calls",
                          headers={"Authorization": f"Bearer {_token()}"},
                          json={"to": [{"type": "phone", "number": to}],
                                "from": {"type": "phone", "number": frm}, "ncco": ncco},
                          timeout=15)
        body = r.json() if r.content else {}
        ok = r.status_code in (200, 201)
        return {"placed": ok, "status_code": r.status_code, "call_uuid": body.get("uuid"),
                "to_last4": to[-4:], **({} if ok else {"error": str(body)[:200]})}
    except Exception as e:
        return {"placed": False, "error": f"{type(e).__name__}: {e}"[:200]}
