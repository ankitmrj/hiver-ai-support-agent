# Submission checklist

Before submitting to Hiver, make sure all of these are true:

- [ ] `data/golden/golden_set.csv` contains 150–250 examples with **actual human-reviewed** `gold_intent` and `gold_escalation` values.
- [ ] No row in the final golden file has `label_source=synthetic_bootstrap`.
- [ ] `data/processed/train_pairs.csv` excludes the final golden examples.
- [ ] `results/evaluation_all.json` contains measured results for majority, TF-IDF, and proposed systems.
- [ ] `results/generated_replies.jsonl` contains replies for the same evaluation set.
- [ ] 50 of those replies have human rubric scores in `data/golden/judge_human_review.csv`.
- [ ] `results/judge_agreement.json` is generated from those actual human scores.
- [ ] Five failure modes include real examples from the evaluation run.
- [ ] The report replaces every `TBD` with a measured result and includes the mandatory misleading-headline-number section.
- [ ] README commands work from a clean virtual environment.
- [ ] The raw Kaggle dataset is not committed to the repository.
- [ ] All borrowed ideas/code are cited.

## Final 15-minute smoke test

```bash
python -m pytest -q
python scripts/build_index.py --pairs data/processed/train_pairs.csv --index-dir data/processed/index
python scripts/evaluate_all.py --train data/processed/train_pairs.csv --gold data/golden/golden_set.csv --index-dir data/processed/index --output results/evaluation_all.json
python scripts/make_eval_replies.py --gold data/golden/golden_set.csv --index-dir data/processed/index --output results/generated_replies.jsonl
```

The LLM judge and LLM intent classifier are optional API-dependent steps and should be run only after the deterministic evaluation is passing.
