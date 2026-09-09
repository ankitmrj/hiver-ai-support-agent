import argparse, json
from pathlib import Path
from src.evaluation.agreement import agreement

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--human",required=True); p.add_argument("--llm",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    out=agreement(a.human,a.llm); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
