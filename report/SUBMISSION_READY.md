# Hiver Submission Checklist

## Required deliverables

- [ ] Public/private repository link with reviewer access.
- [ ] README with reproduction commands that run the headline evaluation in under 15 minutes.
- [ ] Final 150–250-example **human-labelled** golden set.
- [ ] Short note documenting golden-set sampling and labelling procedure.
- [ ] Automated evaluation harness.
- [ ] Trivial baseline results.
- [ ] Simple baseline results.
- [ ] Proposed-agent results.
- [ ] LLM-as-judge rubric.
- [ ] 50-example human review of judge scores.
- [ ] Human-vs-LLM judge agreement statistics.
- [ ] Five real failure examples from the final evaluation.
- [ ] Six-page-or-less report (or equivalent README section).
- [ ] 10–15 item decision log.

## Final integrity checks

Before submitting, verify that:

1. `data/golden/golden_set.csv` contains real human labels, not bootstrap labels.
2. The evaluation examples are held out from the training/retrieval corpus.
3. Every number in the report can be regenerated from a checked-in command and output file.
4. Every failure-analysis example comes from an actual evaluation run.
5. Human-vs-LLM judge agreement is based on actual human scoring.
6. The report does not present demo/bootstrap results as benchmark results.

## Submission form

Submit through the Hiver-provided Notion form:

https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105

Include the repository link and the final report as requested by the form. Do not email the submission.
