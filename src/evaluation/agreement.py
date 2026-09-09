from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def agreement(human_csv: str, llm_jsonl: str) -> dict:
    human = pd.read_csv(human_csv)
    llm = pd.read_json(llm_jsonl, lines=True)
    merged = human.merge(llm, on="example_id", suffixes=("_human", "_llm"))
    dims = ["groundedness", "correctness", "helpfulness", "brand_consistency", "safety"]
    out = {"n": int(len(merged)), "dimensions": {}}
    for d in dims:
        h = pd.to_numeric(merged[f"{d}_human"], errors="coerce")
        l = pd.to_numeric(merged[f"{d}_llm"], errors="coerce")
        mask = h.notna() & l.notna()
        if mask.sum() < 3:
            continue
        corr, _ = spearmanr(h[mask], l[mask])
        within1 = (abs(h[mask] - l[mask]) <= 1).mean()
        exact = (h[mask] == l[mask]).mean()
        out["dimensions"][d] = {"spearman": float(corr), "within_1": float(within1), "exact": float(exact)}
    return out
