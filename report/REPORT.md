# Hiver SDE Intern — Take-Home Report

## 1. Problem framing

### Goal

I built a support agent for **AmazonHelp** using the Customer Support on Twitter dataset. The agent has three responsibilities:

1. **Intent classification** — map an incoming customer message to a small, brand-specific intent taxonomy.
2. **Grounded reply drafting** — retrieve historically similar AmazonHelp customer-support interactions and draft a response using those examples as evidence.
3. **Safe routing** — decide whether to auto-handle the message or escalate it to a human, with an explicit reason.

### What “good” means

For AmazonHelp, a good automated interaction is not simply one that produces a fluent answer. I define success as:

- the customer's issue is classified correctly;
- the drafted response addresses the actual issue;
- the response is grounded in observed historical support behavior rather than invented policy;
- the agent does not claim to have performed an action it cannot perform;
- cases that require account/order investigation, contain insufficient evidence, or have low-confidence intent predictions are routed to a human.

The system is therefore optimized for **useful automation with controlled risk**, not maximum automation rate.

### What I chose not to build

I intentionally did not build Twitter API integration, autonomous refunds/orders/account changes, a production customer-facing UI, multi-brand routing, model fine-tuning, or long-term user memory. Those features increase production surface area but do not materially improve the central question of this assignment: whether the agent can classify, ground, and route support requests reliably on historical data.

---

## 2. Data and preprocessing

The source dataset is the Customer Support on Twitter corpus. I work with a documented subsample rather than the full ~3M-tweet corpus. Raw data stays outside version control.

The preprocessing pipeline:

1. streams the source CSV in chunks;
2. filters to `AmazonHelp` conversations;
3. uses `inbound`, `response_tweet_id`, and `in_response_to_tweet_id` to reconstruct direct customer-to-brand response pairs;
4. removes empty/obviously unusable rows;
5. creates a compact evidence corpus of `customer_text → historical_brand_response` pairs;
6. creates a held-out golden-review queue before building the evaluation index to reduce leakage risk.

The repository keeps the raw dataset out of Git and makes the processed subset reproducible from the preparation script.

---

## 3. Intent taxonomy

The working AmazonHelp taxonomy contains 11 intents:

| Intent | Meaning |
|---|---|
| `order_status` | Questions about current order status or tracking |
| `delivery_delay` | Order is late or delivery date has slipped |
| `missing_delivery` | Package is marked delivered or expected but is not found |
| `refund` | Refund requested, pending, missing, or questioned |
| `return` | Return eligibility/process/questions |
| `cancellation` | Request or question about cancelling an order |
| `payment_issue` | Payment failure, charge, or payment-method problem |
| `account_issue` | Login/account access/profile-related support |
| `subscription_prime` | Prime/subscription-related issue |
| `product_issue` | Product quality, defect, compatibility, or product-specific problem |
| `other` | Insufficient evidence or issue outside the defined taxonomy |

The taxonomy deliberately stays small. A very fine-grained label space would increase annotation ambiguity and make the 150–250-example evaluation set less reliable.

Before final submission, the taxonomy should be frozen after manual inspection of the sampled AmazonHelp data. `src/intents/taxonomy.json` is the single source of truth.

---

## 4. System design

```text
Incoming customer message
          |
          v
   Intent classification
          |
          +----------------+
          |                |
          v                v
   Intent/confidence   Historical retrieval
          |                |
          +--------+-------+
                   v
           Escalation policy
              /         \
             /           \
      AUTO-HANDLE       ESCALATE
             |
             v
      Grounded response
```

The proposed agent separates **classification**, **retrieval**, and **generation** so each component can be evaluated independently.

The generated response is given retrieved historical examples rather than a blank prompt. The escalation policy uses confidence, evidence strength, and conservative rules for cases that may require private account/order access.

---

## 5. Baselines

### Baseline A — majority class

The trivial baseline always predicts the most frequent intent in the training split. Its purpose is to establish the performance floor under class imbalance.

### Baseline B — TF-IDF classifier + nearest historical response

The simple baseline uses TF-IDF features with a linear classifier for intent prediction and lexical cosine similarity to select a historical brand response. It does not use generative AI.

### Proposed agent

The proposed system combines:

- intent classification;
- top-k historical retrieval;
- conservative escalation policy;
- constrained LLM reply drafting using retrieved evidence.

This comparison isolates whether the extra complexity of retrieval + generation + routing provides value over simple, transparent methods.

---

## 6. Results

**Important integrity note:** the required headline numbers must be generated from the final, manually labelled golden set. They must not be estimated from the bootstrap/demo data. The repository therefore leaves the values below as placeholders until the real evaluation is run.

| System | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 | Reply Judge Score |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| TF-IDF baseline | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Proposed agent | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

### How to fill the table

Run the evaluation after the 150–250 examples have been hand-labelled:

```bash
python scripts/evaluate_all.py \
  --train data/processed/train_pairs.csv \
  --gold data/golden/golden_set.csv \
  --index-dir data/processed/index \
  --output results/evaluation_all.json
```

The final README should quote the exact output file and report the dataset size, label distribution, and random seed alongside the headline metrics.

---

## 7. Failure analysis

The final report must use **real examples produced by the evaluation run**. The table below is the required structure; example categories are hypotheses, not claimed measured failures.

| Failure mode | Real example | Prediction | Gold | Hypothesis |
|---|---|---|---|---|
| Ambiguous short follow-up | `TBD — copy from evaluation` | `TBD` | `TBD` | A message like “still waiting” can be under-specified without previous turns. |
| Multi-intent message | `TBD — copy from evaluation` | `TBD` | `TBD` | Single-label classification may force one intent when the user asks about multiple issues. |
| Missing conversation context | `TBD — copy from evaluation` | `TBD` | `TBD` | The latest tweet may depend on earlier turns that were not passed into the classifier. |
| Retrieval false positive | `TBD — copy from evaluation` | `TBD` | `TBD` | TF-IDF similarity can overweight shared words while missing semantic differences. |
| Rare intent / distribution shift | `TBD — copy from evaluation` | `TBD` | `TBD` | Intents with few examples provide weak supervision and may overlap with `other`. |

### Failure-analysis procedure

After running evaluation, sort errors by frequency and inspect at least one concrete example from each of the five largest categories. For every failure, record:

- original customer text;
- retrieved evidence;
- predicted intent and confidence;
- predicted escalation decision;
- gold intent/action;
- why the error happened;
- a specific engineering change that could reduce it.

This prevents the failure section from becoming a generic list of possible LLM weaknesses.

---

## 8. What is misleading about my headline number?

A single overall intent-accuracy number is useful but incomplete, and it can be actively misleading in this setting.

First, the customer-support distribution is not uniform. A model can obtain a strong accuracy score by performing well on common intents while performing poorly on rare but operationally important cases. This is why the report also uses **macro-F1** and per-intent results.

Second, intent accuracy says nothing about whether the drafted answer is grounded. A system can classify `refund` correctly and still hallucinate a refund policy, promise an action it cannot execute, or give an unhelpful response. Reply-quality and unsupported-claim scores therefore need to be reported separately.

Third, an accurate classifier can still be an unsafe automation system. If the escalation decision is wrong, the more serious failure may be an inappropriate **auto-handle** rather than an incorrect label. Escalation precision/recall/F1 and qualitative review of false auto-handles therefore matter independently.

Finally, the golden set is only a sample. Its size, stratification strategy, and human labelling quality influence the reported result. The headline number should therefore be read as **performance on this sampled evaluation set**, not as a guarantee over all future AmazonHelp messages.

The final README should put the strongest evidence together in one compact summary such as:

```text
Intent accuracy: X%
Macro-F1: Y%
Escalation F1: Z%
Average judged reply quality: W/4
Golden set: N manually labelled examples
```

---

## 9. LLM-as-judge and human agreement

The reply judge scores five dimensions:

1. groundedness;
2. correctness;
3. helpfulness;
4. brand consistency;
5. safety / unsupported claims.

The judge should not be treated as ground truth. A 50-example subset of generated replies is exported for human review, and the same examples are scored by the LLM judge.

The repository computes:

- exact score agreement;
- agreement within one point;
- Spearman correlation when there are enough valid observations.

**Do not report an agreement value until the human review file has actually been completed.** The final report should state both the number of human-reviewed examples and the observed agreement.

---

## 10. What I would do with one more week

### 1. Make the agent conversation-aware

Instead of classifying only the latest tweet, pass the most recent few turns into both classification and retrieval. This should directly target context-dependent failures.

### 2. Improve retrieval quality

Move from lexical retrieval to dense embeddings with reranking, while keeping the original TF-IDF system as an interpretable baseline.

### 3. Calibrate escalation thresholds

Tune thresholds against the cost of false auto-handles versus unnecessary escalations. The preferred operating point should be chosen from measured error costs, not an arbitrary confidence value.

### 4. Add active learning

Use the highest-uncertainty and highest-disagreement cases to expand the golden set and refine taxonomy definitions.

### 5. Measure cost and latency

Compare candidate models on response latency, token cost, and quality so that the best system is not merely the most accurate but also practical to operate.

---

## 11. Reproducibility and submission integrity

The final submission should contain:

- the exact processed subset or deterministic preparation instructions;
- the final 150–250-example human-labelled golden set;
- evaluation outputs generated from that set;
- at least two baselines evaluated on the same examples;
- 5 real failure examples;
- human-vs-LLM judge agreement evidence;
- this report and the decision log.

The included bootstrap/demo data is for software validation only and must not be described as the final human-labelled evaluation evidence.

## Sources

- Thought Vector, *Customer Support on Twitter*, Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Kechtel et al., *ERICSSON — Event log constRuctIon from CuStomer Service cOnversatioNs*: https://github.com/kechtel/ericsson
- OpenAI Platform documentation: https://platform.openai.com/docs/
