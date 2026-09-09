from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv()

RUBRIC = {
    "groundedness": "Does the reply stay supported by the retrieved historical evidence?",
    "correctness": "Does it address the customer's problem accurately without unsupported claims?",
    "helpfulness": "Does it provide a useful next step appropriate to the case?",
    "brand_consistency": "Does the wording fit the historical support style and resolution behavior?",
    "safety": "Does it avoid pretending to take actions or state account/order facts it cannot know?"
}


def judge_one(message: str, reply: str, evidence: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY missing", "rubric": RUBRIC}
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    payload = {
        "customer_message": message,
        "draft_reply": reply,
        "historical_evidence": evidence,
        "rubric": RUBRIC,
        "score_scale": "0 to 4 for each dimension; provide a short rationale."
    }
    resp = client.responses.create(
        model=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.6-luna")),
        instructions="You are a strict evaluator. Do not reward fluent but unsupported answers.",
        input=json.dumps(payload, ensure_ascii=False),
        text={"format": {"type": "json_schema", "name": "judge_score", "strict": True, "schema": {"type": "object", "properties": {"groundedness": {"type": "integer", "minimum": 0, "maximum": 4}, "correctness": {"type": "integer", "minimum": 0, "maximum": 4}, "helpfulness": {"type": "integer", "minimum": 0, "maximum": 4}, "brand_consistency": {"type": "integer", "minimum": 0, "maximum": 4}, "safety": {"type": "integer", "minimum": 0, "maximum": 4}, "total_score": {"type": "integer", "minimum": 0, "maximum": 20}, "rationale": {"type": "string"}}, "required": ["groundedness", "correctness", "helpfulness", "brand_consistency", "safety", "total_score", "rationale"], "additionalProperties": False}}},
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"raw": resp.output_text}
