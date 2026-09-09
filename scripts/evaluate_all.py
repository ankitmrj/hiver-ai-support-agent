import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from src.retrieval.classifier import IntentClassifier
from src.retrieval.tfidf_store import TfidfStore
from src.agent.policy import decide

def binmetrics(y_true,y_pred):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    return {'precision':float(p),'recall':float(r),'f1':float(f)}

def clsmetrics(y_true,y_pred):
    return {'accuracy':float(accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted',zero_division=0))}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--gold',required=True); p.add_argument('--index-dir',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    tr=pd.read_csv(a.train); g=pd.read_csv(a.gold)
    if (g['gold_intent'].fillna('').str.strip()=='').any(): raise SystemExit('gold_intent incomplete')
    if (g['gold_escalation'].fillna('').str.strip()=='').any(): raise SystemExit('gold_escalation incomplete')
    majority=str(tr['bootstrap_intent'].value_counts().idxmax()) if 'bootstrap_intent' in tr else 'other'
    majority_pred=[majority]*len(g)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=75000,sublinear_tf=True)
    X=vec.fit_transform(tr['customer_text'].fillna('')); y=tr['bootstrap_intent']; model=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X,y)
    simple_pred=model.predict(vec.transform(g['customer_text'].fillna('')))
    clf=IntentClassifier.load(Path(a.index_dir)/'intent.joblib'); store=TfidfStore.load(Path(a.index_dir)/'retrieval.joblib')
    prop_i=[]; prop_e=[]
    for text in g['customer_text'].fillna(''):
        it,cf=clf.predict_one(text); ev=store.search(text,1); sim=ev[0]['similarity'] if ev else 0; dec,_=decide(it,cf,sim); prop_i.append(it); prop_e.append(dec)
    true_e=g['gold_escalation'].eq('ESCALATE')
    # Majority and simple escalation baselines are intentionally conservative only where they have explicit lexical signals.
    simple_e=[]
    for text in g['customer_text'].fillna('').str.lower():
        simple_e.append('ESCALATE' if any(k in text for k in ['account','password','login','charged','payment','marked delivered']) else 'AUTO-HANDLE')
    result={
      'majority':{'intent':clsmetrics(g['gold_intent'],majority_pred)},
      'tfidf':{'intent':clsmetrics(g['gold_intent'],simple_pred),'escalation':binmetrics(true_e,[x=='ESCALATE' for x in simple_e])},
      'proposed':{'intent':clsmetrics(g['gold_intent'],prop_i),'escalation':binmetrics(true_e,[x=='ESCALATE' for x in prop_e])}
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
