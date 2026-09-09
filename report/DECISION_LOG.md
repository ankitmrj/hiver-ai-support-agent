# Decision Log

1. **Chose AmazonHelp as the single target brand.** A single-brand system keeps the taxonomy, retrieval corpus, and evaluation aligned with one support style instead of introducing unnecessary cross-brand routing complexity.

2. **Used direct customer→brand response pairs as the core evidence unit.** This is more reliable for a first system than trying to perfectly reconstruct every long multi-turn thread, and it preserves a clear relationship between the user's issue and the historical resolution.

3. **Created a brand-specific taxonomy instead of importing Banking77.** Banking77 can inform intent methodology, but the assignment asks for intents defined from the target brand's data. The final label definitions therefore come from AmazonHelp examples.

4. **Limited the taxonomy to 11 intents.** A small label space makes manual annotation more consistent and makes the 150–250-example evaluation set statistically more useful per class.

5. **Included an `other` intent.** Without it, ambiguous or previously unseen messages would be forced into a semantically incorrect class, making the classifier look better than the actual user experience.

6. **Used a majority-class baseline.** It is intentionally weak but establishes how much of the observed performance can be explained by class imbalance alone.

7. **Used TF-IDF + linear classification as the simple baseline.** It is cheap, interpretable, deterministic, and strong enough to be a meaningful comparison rather than a strawman.

8. **Kept retrieval separate from generation.** This makes the evidence visible and lets the system answer the key question: “Which historical cases caused this response?” rather than treating the LLM as an opaque chatbot.

9. **Retrieved historical responses before generating a reply.** The goal is to ground the response in observed brand behavior and reduce unsupported policy invention.

10. **Used conservative escalation.** When confidence or evidence is weak, the system prefers a human handoff over a confident but unsupported answer.

11. **Treat account-, payment-, and order-specific situations conservatively.** The agent cannot inspect private support systems, so it should not imply that it checked an order, changed an account, or initiated a refund.

12. **Created the golden evaluation queue before building the final evaluation index.** This reduces the risk that evaluation examples are inadvertently used as retrieval evidence and makes leakage easier to reason about.

13. **Used a manually labelled golden set rather than relying on LLM-generated labels.** The assignment is specifically testing proof, so evaluation labels need an explicit human-reviewed ground truth.

14. **Validated the LLM judge against humans.** Judge scores are only useful if the judge itself has demonstrated reasonable agreement with human assessment on a held-out review subset.

15. **Kept raw Kaggle data out of version control.** The repository should contain reproducible preparation code and the allowed derived artifacts rather than redistributing the original source dataset.

16. **Included a deterministic no-API fallback.** The project remains runnable and testable without an LLM key, which helps reviewers verify the software path independently from external API availability.

17. **Optimized for proof over UI.** No production dashboard or Twitter integration was built because neither materially improves the evidence that classification, grounding, and escalation are reliable.
