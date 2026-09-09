import argparse,json
from pathlib import Path
import pandas as pd
from src.retrieval.classifier import IntentClassifier
from src.retrieval.tfidf_store import TfidfStore
from src.agent.policy import decide
from src.agent.generator import generate

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gold',required=True); p.add_argument('--index-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    df=pd.read_csv(a.gold)
    clf=IntentClassifier.load(Path(a.index_dir)/'intent.joblib'); store=TfidfStore.load(Path(a.index_dir)/'retrieval.joblib')
    with open(a.output,'w',encoding='utf-8') as f:
        for i,row in df.iterrows():
            msg=str(row['customer_text']); intent,conf=clf.predict_one(msg); evidence=store.search(msg,5); sim=evidence[0]['similarity'] if evidence else 0
            decision,reason=decide(intent,conf,sim); g=generate(msg,intent,decision,evidence)
            out={'example_id':int(i+1),'message':msg,'gold_intent':row.get('gold_intent',''),'gold_escalation':row.get('gold_escalation',''),'intent':intent,'confidence':conf,'reply':g['reply'],'decision':decision,'reason':reason,'evidence':evidence}
            f.write(json.dumps(out,ensure_ascii=False)+'\n')
    print(f'Wrote {len(df)} replies to {a.output}')
