import argparse, json
from pathlib import Path
from src.retrieval.classifier import IntentClassifier
from src.agent.llm_classifier import LLMIntentClassifier
from src.retrieval.tfidf_store import TfidfStore
from src.agent.policy import decide
from src.agent.generator import generate

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--message", required=True); p.add_argument("--index-dir", required=True); p.add_argument("--output")
    a=p.parse_args(); clf=IntentClassifier.load(Path(a.index_dir)/"intent.joblib"); store=TfidfStore.load(Path(a.index_dir)/"retrieval.joblib")
    intent, conf=clf.predict_one(a.message); evidence=store.search(a.message,5); top=evidence[0]["similarity"] if evidence else 0
    decision, reason=decide(intent,conf,top); gen=generate(a.message,intent,decision,evidence)
    obj={"intent":intent,"confidence":round(conf,4),"reply":gen["reply"],"decision":decision,"reason":reason,"evidence":evidence}
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    if a.output: Path(a.output).write_text(json.dumps(obj,ensure_ascii=False)+"\n",encoding="utf-8")
