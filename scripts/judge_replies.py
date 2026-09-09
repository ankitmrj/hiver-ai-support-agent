import argparse, json
from pathlib import Path
from src.evaluation.judge import judge_one

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    out=[]
    with open(a.input,encoding="utf-8") as f:
        for line in f:
            row=json.loads(line); score=judge_one(row["message"],row["reply"],row.get("evidence",[])); score["example_id"]=row["example_id"]; out.append(score)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for x in out: f.write(json.dumps(x,ensure_ascii=False)+"\n")
