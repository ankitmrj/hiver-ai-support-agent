# Golden set protocol

The assignment requires 150–250 **hand-labelled** examples. This folder intentionally does not pretend that generated labels are human labels.

## Procedure

1. Run `scripts/build_golden_queue.py` against the real AmazonHelp pairs.
2. Review 200 rows, stratified by candidate intent, plus hard/ambiguous examples.
3. Fill `gold_intent` with one taxonomy label.
4. Fill `gold_escalation` with `AUTO-HANDLE` or `ESCALATE`.
5. Add a short `label_notes` when the case is ambiguous or multi-intent.
6. Save the reviewed file as `golden_set.csv`.
7. Keep the original review queue separately so the audit trail is preserved.

Do not call synthetic bootstrap labels “human labels” in the report.
