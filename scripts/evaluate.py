import argparse, json
from pathlib import Path
import pandas as pd
from src.retrieval.classifier import IntentClassifier
from src.retrieval.tfidf_store import TfidfStore
from src.evaluation.metrics import classification_metrics, binary_metrics
from src.agent.policy import decide

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--gold", required=True); p.add_argument("--index-dir", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); df = pd.read_csv(a.gold)
    if (df["gold_intent"].fillna("").str.strip() == "").any() or (df["gold_escalation"].fillna("").str.strip() == "").any():
        raise SystemExit("Golden set contains unlabeled rows. Complete human labels before evaluation.")
    clf = IntentClassifier.load(Path(a.index_dir) / "intent.joblib")
    store = TfidfStore.load(Path(a.index_dir) / "retrieval.joblib")
    pred_i=[]; pred_e=[]; sims=[]
    for text in df["customer_text"].fillna(""):
        intent, conf = clf.predict_one(text)
        ev = store.search(text, 1)
        sim = ev[0]["similarity"] if ev else 0.0
        decision, _ = decide(intent, conf, sim)
        pred_i.append(intent); pred_e.append(decision); sims.append(sim)
    result={"intent": classification_metrics(df["gold_intent"], pred_i), "escalation": binary_metrics(df["gold_escalation"].eq("ESCALATE"), [x == "ESCALATE" for x in pred_e]), "mean_top_similarity": sum(sims)/max(1,len(sims))}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
