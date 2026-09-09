from __future__ import annotations

ESCALATE_INTENTS = {"missing_delivery", "payment_issue", "account_issue"}

def decide(intent: str, confidence: float, top_similarity: float) -> tuple[str, str]:
    if confidence < 0.55:
        return "ESCALATE", "Low intent confidence; the system lacks enough evidence to safely automate the case."
    if top_similarity < 0.20:
        return "ESCALATE", "No sufficiently similar historical support case was retrieved."
    if intent in ESCALATE_INTENTS:
        return "ESCALATE", "This issue may require account/order-level investigation that the automated agent cannot perform."
    return "AUTO-HANDLE", "The intent is confident and a similar historical resolution is available."
