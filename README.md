# Hiver SDE Intern — AI Customer Support Agent

A reproducible support-agent pipeline built for the **Customer Support on Twitter** dataset. The implementation focuses on the three requirements in the take-home:

1. intent classification,
2. historically grounded reply drafting,
3. safe auto-handle vs human escalation.

The reference brand is **AmazonHelp**. The source dataset contains ~2.8M tweets, with `inbound`, `text`, `response_tweet_id`, and `in_response_to_tweet_id` fields that allow conversations to be reconstructed. The official Kaggle page describes the dataset as a CSV of tweets/replies and documents these fields. [Kaggle dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

> **Important submission integrity note:** the repo includes a 200-row bootstrap evaluation set and the tooling to create the required 150–250-row golden set from the real data. The bootstrap labels are explicitly marked as `synthetic_bootstrap` and must not be presented as human labels. The report also refuses to claim human/LLM judge agreement until that review has actually been completed.

## Project structure

```text
hiver_sde_intern_takehome/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                 # put twcs.csv here (not committed)
│   ├── processed/
│   ├── golden/
│   └── demo/
├── src/
│   ├── preprocessing/
│   ├── intents/
│   ├── retrieval/
│   ├── agent/
│   └── evaluation/
├── scripts/
├── report/
├── results/
└── tests/
```

## 1. Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download `twcs.csv` from Kaggle and place it at:

```text
 data/raw/twcs.csv
```

The official dataset is licensed CC BY-NC-SA 4.0; check the dataset terms before redistribution or commercial use.

## 3. Build a reproducible AmazonHelp subset

This reads the large CSV in chunks, filters to AmazonHelp conversations, reconstructs direct customer→brand pairs, and writes a compact working corpus.

```bash
python scripts/prepare_data.py \
  --input data/raw/twcs.csv \
  --output data/processed/amazonhelp_pairs.csv \
  --brand AmazonHelp \
  --max-pairs 20000
```

Typical output columns:

```text
conversation_root_id,customer_tweet_id,customer_text,brand_tweet_id,brand_text,created_at
```

## 4. Define taxonomy

`src/intents/taxonomy.json` contains the initial 10-intent taxonomy. Before final submission, inspect a stratified sample and edit the definitions; the taxonomy is deliberately small and brand-specific.

## 5. Build the retrieval index and classifier

No paid LLM is required for the baseline evaluation.

```bash
python scripts/build_index.py \
  --pairs data/processed/amazonhelp_pairs.csv \
  --index-dir data/processed/index
```

The simple model is TF-IDF + linear classifier. Retrieval uses TF-IDF cosine similarity. An optional SentenceTransformer index can be enabled later, but it is not required for the headline baseline comparison.

## 6. Build the golden evaluation queue

```bash
python scripts/build_golden_queue.py \
  --pairs data/processed/amazonhelp_pairs.csv \
  --output data/golden/golden_review_queue.csv \
  --n 200
```

The queue is stratified across intents using deterministic candidate rules. You must manually review/label these rows. The file contains `label_source=human_required` and cannot be mistaken for a finished evaluation set.

## 7. Run automated evaluation

After you have filled `gold_intent` and `gold_escalation` for the review queue:

```bash
python scripts/evaluate.py \
  --gold data/golden/golden_review_queue.csv \
  --index-dir data/processed/index \
  --output results/evaluation.json
```

This reports:

- accuracy and macro-F1 for intent classification,
- escalation precision/recall/F1,
- simple retrieval quality diagnostics,
- confusion matrix.

## 8. Run the grounded agent

Without an API key the command produces a deterministic retrieval-based draft. With an OpenAI key it uses the Responses API to draft the final answer from retrieved evidence.

```bash
copy .env.example .env
# set OPENAI_API_KEY=...
python scripts/run_agent.py \
  --message "My package says delivered but I never received it" \
  --index-dir data/processed/index
```

The agent returns JSON like:

```json
{
  "intent": "missing_delivery",
  "confidence": 0.86,
  "reply": "...",
  "decision": "ESCALATE",
  "reason": "The issue may require order-level investigation and the response should not claim an action the assistant cannot perform.",
  "evidence": []
}
```

## 9. Judge replies

The judge uses a rubric covering:

- groundedness,
- correctness,
- helpfulness,
- brand-consistency,
- unsupported claims/safety.

```bash
python scripts/judge_replies.py \
  --input results/generated_replies.jsonl \
  --output results/judge_scores.jsonl
```

### Human agreement requirement

Create a human review file from the same 50 replies:

```bash
python scripts/make_judge_review.py \
  --generated results/generated_replies.jsonl \
  --output data/golden/judge_human_review.csv \
  --n 50
```

Have a human score it, then run:

```bash
python scripts/judge_agreement.py \
  --human data/golden/judge_human_review.csv \
  --llm results/judge_scores.jsonl \
  --output results/judge_agreement.json
```

The script reports exact agreement, within-1 agreement, and Spearman correlation where enough observations exist.

## 10. Demo mode

The repo includes a tiny synthetic demo corpus so reviewers can verify the software path without downloading the full dataset:

```bash
python scripts/demo.py
```

This is **not** a substitute for the Kaggle evaluation set and must not be used as the headline result.

## Evaluation design

### Baseline A — majority class

Always predict the most frequent intent in the training split.

### Baseline B — TF-IDF linear model + nearest response

Classify with TF-IDF + Logistic Regression and draft the reply by selecting the top retrieved historical response. This is intentionally simple and non-generative.

### Proposed agent

1. classify intent,
2. retrieve top historical cases,
3. apply escalation policy,
4. generate a constrained response grounded in those cases.

The proposed system is intentionally conservative: low confidence, weak evidence, or account/order-sensitive cases are more likely to escalate.

## What counts as "good"

For AmazonHelp, a good automated response should correctly identify the customer's issue, remain faithful to historical support behavior, avoid fabricating actions/policies, and escalate when account-specific investigation is necessary. The target is not maximum automation; it is **useful automation with controlled risk**.

## What is intentionally not built

- Twitter API integration
- autonomous refunds/orders/account changes
- a production customer-facing UI
- multi-brand routing
- model fine-tuning
- long-term user memory

Those features add engineering surface area without improving the core proof that the agent is reliable on this dataset.

## Reproduction target

The intended final submission should keep a fixed processed subset and golden set under version control (excluding the original raw dataset). The headline evaluation operates on the small golden set and should complete in under 15 minutes on a normal laptop, subject to API latency if the LLM judge is enabled.

## Sources

- Thought Vector, *Customer Support on Twitter*, Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Kechtel et al. reference repository for preprocessing/conversation reconstruction examples: https://github.com/kechtel/ericsson
- OpenAI API documentation: https://platform.openai.com/docs/

## Strict evaluation order (important)

To avoid training/evaluation leakage, create the review queue first, then hold those rows out before building the classifier/index:

```bash
python scripts/build_golden_queue.py --pairs data/processed/amazonhelp_pairs.csv --output data/golden/golden_review_queue.csv --n 200
python scripts/split_eval.py --pairs data/processed/amazonhelp_pairs.csv --gold data/golden/golden_review_queue.csv --train-output data/processed/train_pairs.csv
python scripts/build_index.py --pairs data/processed/train_pairs.csv --index-dir data/processed/index
```

Then fill the human labels and run:

```bash
python scripts/evaluate_all.py --train data/processed/train_pairs.csv --gold data/golden/golden_set.csv --index-dir data/processed/index --output results/evaluation_all.json
python scripts/make_eval_replies.py --gold data/golden/golden_set.csv --index-dir data/processed/index --output results/generated_replies.jsonl
```

This makes the holdout relationship explicit: no golden example is used to fit the baseline/proposed classifier or retrieval index.

### Proposed intent classifier

When `OPENAI_API_KEY` is present, the proposed agent uses the LLM classifier in `src/agent/llm_classifier.py` with the explicit 11-intent taxonomy and structured JSON output. Without an API key, it falls back to the locally trained TF-IDF classifier so the full software path remains testable.
