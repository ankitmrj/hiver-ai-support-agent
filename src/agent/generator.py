from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an AI customer-support drafting assistant for AmazonHelp.
You do not have access to a customer's account and cannot actually refund, cancel, change, or inspect orders.
Use only the historical examples supplied as evidence. Do not invent policy, timelines, links, actions, or case status.
If the case should be escalated, write a concise acknowledgement and state the next safe step without pretending that escalation has already happened.
Return JSON with keys: reply, safety_note.
"""


def grounded_fallback(message: str, evidence: list[dict[str, Any]], intent: str, decision: str) -> str:
    if evidence:
        response = evidence[0].get("brand_text", "")
        if response:
            return response
    if decision == "ESCALATE":
        return "Sorry you're dealing with this. This issue may need a closer review by the support team, and I don't want to give you instructions that could be inaccurate."
    return "Sorry about the trouble. Please share the relevant order or account context with support so they can help with the next step."


def generate(message: str, intent: str, decision: str, evidence: list[dict[str, Any]]) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    if not api_key:
        return {"reply": grounded_fallback(message, evidence, intent, decision), "safety_note": "deterministic fallback"}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    compact = [{"customer_text": e.get("customer_text", ""), "brand_text": e.get("brand_text", ""), "similarity": round(float(e.get("similarity", 0)), 3)} for e in evidence]
    prompt = {
        "customer_message": message,
        "intent": intent,
        "decision": decision,
        "historical_examples": compact,
    }
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(prompt, ensure_ascii=False),
        text={"format": {"type": "json_schema", "name": "support_reply", "strict": True, "schema": {"type": "object", "properties": {"reply": {"type": "string"}, "safety_note": {"type": "string"}}, "required": ["reply", "safety_note"], "additionalProperties": False}}},
    )
    text = resp.output_text.strip()
    try:
        obj = json.loads(text)
        return {"reply": str(obj.get("reply", "")).strip(), "safety_note": str(obj.get("safety_note", ""))}
    except Exception:
        return {"reply": text, "safety_note": "model output was not valid JSON; preserved as text"}
