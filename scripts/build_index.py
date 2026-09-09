import argparse
import pandas as pd
from src.retrieval.tfidf_store import TfidfStore
from src.retrieval.classifier import IntentClassifier

def weak_bootstrap_label(text: str) -> str:
    from src.intents.rules import heuristic_intent
    return heuristic_intent(text)[0]

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pairs',required=True)
    p.add_argument('--index-dir',required=True)
    a=p.parse_args()
    df=pd.read_csv(a.pairs)
    df['bootstrap_intent']=df['customer_text'].fillna('').map(weak_bootstrap_label)
    store=TfidfStore.fit(df)
    store.save(a.index_dir+'/retrieval.joblib')
    clf_df=df[df['bootstrap_intent']!='other'].copy()
    if clf_df['bootstrap_intent'].nunique() >= 2:
        clf=IntentClassifier.fit(clf_df['customer_text'],clf_df['bootstrap_intent'])
        clf.save(a.index_dir+'/intent.joblib')
        print('index + classifier built')
    else:
        raise SystemExit('Need at least 2 bootstrap classes to train classifier')
