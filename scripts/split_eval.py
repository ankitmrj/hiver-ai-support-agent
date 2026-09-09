import argparse
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pairs',required=True)
    p.add_argument('--gold',required=True)
    p.add_argument('--train-output',required=True)
    a=p.parse_args()
    pairs=pd.read_csv(a.pairs); gold=pd.read_csv(a.gold)
    held=set(gold['customer_tweet_id'].astype(str))
    train=pairs[~pairs['customer_tweet_id'].astype(str).isin(held)].copy()
    Path(a.train_output).parent.mkdir(parents=True,exist_ok=True)
    train.to_csv(a.train_output,index=False)
    print(f'train={len(train)} heldout={len(held)}')
