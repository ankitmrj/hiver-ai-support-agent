from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
import json


def load_taxonomy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

KEYWORDS = {
    "missing_delivery": [r"marked delivered", r"says delivered", r"not received", r"never received", r"didn't get", r"did not get"],
    "refund": [r"refund", r"money back", r"refunded"],
    "return": [r"return", r"send.*back", r"return label"],
    "cancellation": [r"cancel", r"cancellation", r"stop.*order"],
    "payment_issue": [r"payment", r"charged", r"charge", r"card", r"billing", r"checkout"],
    "account_issue": [r"login", r"log in", r"password", r"account", r"sign in"],
    "subscription_prime": [r"prime", r"membership", r"subscription", r"renewal"],
    "delivery_delay": [r"late", r"delayed", r"delay", r"overdue", r"not arrived"],
    "order_status": [r"where.*order", r"tracking", r"track.*order", r"eta", r"status.*order"],
    "product_issue": [r"wrong item", r"damaged", r"broken", r"defective", r"missing item"],
}

def heuristic_intent(text: str) -> tuple[str, float]:
    t = text.lower()
    scores = Counter()
    for intent, patterns in KEYWORDS.items():
        for p in patterns:
            if re.search(p, t):
                scores[intent] += 1
    if not scores:
        return "other", 0.20
    intent, score = scores.most_common(1)[0]
    confidence = min(0.95, 0.45 + 0.12 * score)
    return intent, confidence
