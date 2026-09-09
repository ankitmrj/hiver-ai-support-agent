# Hiver SDE Intern — AI Customer Support Agent

## 1. Problem Framing

### Goal

I built an AI support agent for **AmazonHelp** using the Customer Support on Twitter dataset. The system is designed around three decisions for every incoming customer message:

1. **What is the customer's intent?**
2. **What response would be appropriate given how AmazonHelp historically handled similar cases?**
3. **Can this case be safely auto-handled, or should it be escalated to a human?**

The key design principle is that a useful support agent should not optimize for automation rate alone. A good system should automate repetitive, low-risk requests while avoiding unsupported claims and escalating cases that require account-level investigation, sensitive information, or additional context.

### What "good" means for AmazonHelp

For this project, a response is considered good when it:

- identifies the customer's actual issue correctly;
- produces a helpful and actionable response;
- remains grounded in historically observed AmazonHelp support behavior;
- does not invent policies, refunds, actions, or outcomes;
- makes a conservative escalation decision when evidence is insufficient.

I therefore optimize for **trustworthy automation**, rather than maximum automation.

### What I chose not to build

I intentionally did not build:

- Twitter API integration;
- autonomous order/refund/account actions;
- a production customer-facing UI;
- multi-brand routing;
- model fine-tuning;
- long-term customer memory.

These features increase implementation complexity but do not materially improve the core question being evaluated: whether the system can classify, retrieve useful historical evidence, draft a grounded response, and make a safe escalation decision.

---

## 2. Data and Pipeline

The primary dataset is the **Customer Support on Twitter** dataset. It contains customer/support tweets and response relationships represented through fields including `inbound`, `response_tweet_id`, and `in_response_to_tweet_id`.

I selected AmazonHelp because the dataset contains a large number of linked AmazonHelp support responses, providing enough repeated cases to evaluate retrieval and response grounding.

The pipeline is:

```text
Raw Twitter dataset
        ↓
AmazonHelp filtering
        ↓
Conversation / response-link reconstruction
        ↓
Customer → AmazonHelp response pairs
        ↓
Golden-set creation and holdout
        ↓
Intent classification
        ↓
Historical-case retrieval
        ↓
Escalation policy
        ↓
Grounded reply generation
        ↓
Evaluation
```

The working corpus is capped at a fixed subset for reproducibility rather than processing all source records during every experiment.

---

## 3. Intent Taxonomy

I use a small AmazonHelp-specific intent taxonomy rather than directly importing a large generic intent dataset.

The initial taxonomy contains:

| Intent |
|---|
| `order_status` |
| `delivery_delay` |
| `missing_delivery` |
| `refund` |
| `return` |
| `cancellation` |
| `payment_issue` |
| `account_issue` |
| `prime_subscription` |
| `product_issue` |
| `other` |

The taxonomy is based on observed support patterns and is deliberately compact. A smaller taxonomy reduces ambiguity between closely related classes and makes the escalation decision more actionable.

Examples:

- "Where is my order?" → `order_status`
- "My package is a week late." → `delivery_delay`
- "Tracking says delivered but I never got it." → `missing_delivery`
- "My refund still hasn't arrived." → `refund`
- "I want to cancel this order." → `cancellation`

---

## 4. System Design

### Intent classification

The proposed pipeline uses an explicit intent taxonomy and structured model output.

The system records:

```json
{
  "intent": "...",
  "confidence": 0.0
}
```

A local TF-IDF classifier is also retained so that the evaluation pipeline remains reproducible without requiring a paid model API.

### Historical grounding

For each incoming message, the system retrieves similar historical customer/support cases.

The retrieved examples provide evidence for the response generator rather than asking the language model to answer from general knowledge.

Conceptually:

```text
Customer message
       ↓
Retrieve similar historical cases
       ↓
Top-k customer/support examples
       ↓
Generate response using evidence
```

The generation prompt explicitly forbids unsupported policies and actions.

### Escalation

The escalation policy combines:

- intent confidence;
- retrieval/evidence strength;
- whether the issue is account/order specific;
- whether the response would require an action the agent cannot perform;
- whether the message is ambiguous or insufficiently contextualized.

Typical escalation cases include:

- missing/delivered-but-not-received orders;
- account/security concerns;
- unresolved or repeated complaints;
- cases requiring account-specific investigation;
- low-confidence classifications;
- insufficient historical evidence.

The system returns both a decision and a reason:

```json
{
  "decision": "ESCALATE",
  "reason": "The issue may require order-level investigation and the available evidence is insufficient for a safe automated resolution."
}
```

---

# 5. Evaluation Methodology

## Golden evaluation set

The evaluation set contains **200 examples** sampled from the real AmazonHelp corpus.

The intended sampling strategy is stratified so that the evaluation contains examples across the defined intents and includes difficult cases rather than only easy, repetitive messages.

Each example is manually reviewed for:

- customer intent;
- whether the case should be auto-handled or escalated;
- optional labeling notes.

The golden examples are held out before building the classifier/retrieval index to avoid evaluation leakage.

## Reply evaluation

Generated replies are evaluated on:

- groundedness;
- correctness;
- helpfulness;
- brand consistency;
- unsupported claims/safety.

An LLM judge is used as an automated evaluator, but its reliability is measured against a human-scored subset rather than assumed.

---

# 6. Baselines

I compare the proposed system against two baselines.

### Baseline 1 — Majority-class classifier

The trivial baseline always predicts the most frequent training-set intent.

This establishes the performance floor and exposes class-imbalance effects.

### Baseline 2 — TF-IDF retrieval/classification

The simple baseline uses TF-IDF representations and a linear classifier for intent prediction, together with nearest historical responses for reply drafting.

It contains no generative reasoning and serves as a useful test of whether the proposed LLM/retrieval pipeline actually provides additional value.

### Proposed system

The proposed system combines:

```text
intent classification
+
historical retrieval
+
conservative escalation policy
+
grounded response generation
```

All systems are evaluated on the same held-out golden examples.

---

# 7. Results

The final submission reports the following metrics after the golden set has been completely human-labelled and the evaluation pipeline has been run.

| System | Intent Accuracy | Macro-F1 | Escalation F1 |
|---|---:|---:|---:|
| Majority baseline | **TBD** | **TBD** | **TBD** |
| TF-IDF baseline | **TBD** | **TBD** | **TBD** |
| Proposed agent | **TBD** | **TBD** | **TBD** |

Reply-quality results:

| Metric | Proposed Agent |
|---|---:|
| Groundedness | **TBD / 4** |
| Correctness | **TBD / 4** |
| Helpfulness | **TBD / 4** |
| Brand consistency | **TBD / 4** |
| Safety / unsupported claims | **TBD / 4** |

LLM-judge agreement with human ratings:

| Measure | Result |
|---|---:|
| Exact agreement | **TBD** |
| Within ±1 point | **TBD** |
| Spearman correlation | **TBD** |

These values should only be filled from the actual evaluation run.

---

# 8. Failure Analysis

The final report uses five failures taken directly from the held-out evaluation set.

## Failure mode 1 — Ambiguous short messages

**Example**

> "Still waiting."

**Observed failure**

The message may correspond to several intents such as delivery, refund, or account issues.

**Hypothesis**

A single isolated tweet does not always contain enough information. The model needs prior conversation context.

**Improvement**

Use the previous one to three conversation turns during classification and retrieval.

---

## Failure mode 2 — Multi-intent messages

**Example**

> "My order is late and I also want the delivery fee refunded."

**Observed failure**

The system may choose one intent and suppress the second issue.

**Hypothesis**

The current taxonomy is single-label, while real support messages can contain multiple requests.

**Improvement**

Add multi-intent detection followed by intent prioritization.

---

## Failure mode 3 — Superficially similar retrieval

**Example**

A customer asking about a missing delivery retrieves a historically similar delivery-delay case.

**Observed failure**

The retrieved examples share vocabulary such as "delivery", "package", and "late" but imply different resolutions.

**Hypothesis**

Lexical similarity alone does not always represent semantic support relevance.

**Improvement**

Use stronger embedding retrieval and reranking, while keeping the retrieved evidence visible for auditability.

---

## Failure mode 4 — Rare intents

**Example**

A low-frequency issue is classified as `other` even though a more specific intent exists.

**Hypothesis**

The training distribution is naturally imbalanced because real customer support traffic is not evenly distributed.

**Improvement**

Increase sampling for rare intents and introduce confidence-aware fallback behavior.

---

## Failure mode 5 — Missing context for account-specific cases

**Example**

A customer provides only:

> "Details sent. Please check."

**Observed failure**

The message does not describe the original problem.

**Hypothesis**

The meaningful intent exists in an earlier turn and cannot be recovered from the isolated tweet.

**Improvement**

Represent the conversation as a short rolling context instead of classifying individual tweets independently.

---

# 9. What Is Misleading About My Headline Number?

The most tempting headline number is **intent accuracy**.

For example, if the system achieved high overall accuracy, that would not by itself demonstrate that it is safe to deploy.

Accuracy can be misleading because:

1. common intents dominate the dataset;
2. rare intents may have much lower recall;
3. the hardest customer messages are often ambiguous or context-dependent;
4. intent accuracy does not measure reply quality;
5. intent accuracy does not measure whether escalation decisions are safe;
6. a correct intent paired with a hallucinated reply is still a bad support interaction.

For this reason, I report **macro-F1, escalation F1, reply quality, and human-vs-LLM judge agreement** alongside accuracy.

The stronger interpretation is not:

> "The classifier is X% accurate."

It is:

> "On a held-out, human-reviewed set, the system achieves X macro-F1, Y escalation F1, and Z reply-quality score, with the judge agreeing with human review at A."

Even that should be interpreted cautiously because the golden set is only 200 examples and may not represent all future support traffic.

---

# 10. What I Would Do With One More Week

With one additional week, I would prioritize reliability rather than adding a UI.

### 1. Conversation-aware classification

Feed the model a short conversation window instead of a single tweet. This directly addresses ambiguous and context-dependent messages.

### 2. Better retrieval

Replace the initial lexical retrieval path with embedding retrieval plus reranking, and evaluate retrieval recall separately.

### 3. Multi-intent support

Detect messages containing multiple independent customer requests and prioritize the most important one for automation.

### 4. Confidence calibration

Use a held-out calibration set to map model scores to actual error probability. Escalation thresholds could then be selected based on an explicit risk target.

### 5. Human-in-the-loop evaluation

Expand the human review set and independently label difficult/rare cases to reduce uncertainty around the headline results.

---

# 11. Conclusion

The main outcome of this project is not a chatbot demo. It is an evaluated support-agent pipeline with explicit evidence, baselines, failure analysis, and escalation controls.

The design deliberately prefers a conservative failure mode:

> **When the system lacks sufficient evidence, it should escalate rather than confidently invent a resolution.**

That trade-off is appropriate for customer support because the cost of a wrong automated answer can be higher than the cost of sending a difficult case to a human.

## Sources

- Thought Vector, *Customer Support on Twitter*, Kaggle  
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- OpenAI API documentation  
  https://platform.openai.com/docs/
- Kechtel et al. reference preprocessing/conversation reconstruction repository  
  https://github.com/kechtel/ericsson