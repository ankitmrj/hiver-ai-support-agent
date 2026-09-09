import argparse
from pathlib import Path
import pandas as pd
from src.intents.rules import heuristic_intent

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    df = pd.read_csv(a.pairs)
    df["candidate_intent"] = df["customer_text"].fillna("").map(lambda x: heuristic_intent(x)[0])
    # deterministic stratified candidate selection; labels remain empty for human review
    groups = []
    per = max(1, a.n // max(1, df["candidate_intent"].nunique()))
    for label, g in df.groupby("candidate_intent", sort=True):
        groups.append(g.sample(min(per, len(g)), random_state=42))
    out = pd.concat(groups, ignore_index=True).head(a.n).copy()
    out["gold_intent"] = ""
    out["gold_escalation"] = ""
    out["label_source"] = "human_required"
    out["label_notes"] = ""
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"Created {len(out)} human-review rows: {a.output}")
