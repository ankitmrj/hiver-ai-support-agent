import argparse
import json
import pandas as pd

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--generated",required=True); p.add_argument("--output",required=True); p.add_argument("--n",type=int,default=50); a=p.parse_args()
    rows=[json.loads(x) for x in open(a.generated,encoding="utf-8")]
    df=pd.DataFrame(rows).head(a.n)
    for d in ["groundedness","correctness","helpfulness","brand_consistency","safety"]: df[d+"_human"]=""
    keep=["example_id"]+[d+"_human" for d in ["groundedness","correctness","helpfulness","brand_consistency","safety"]]
    df[keep].to_csv(a.output,index=False)
