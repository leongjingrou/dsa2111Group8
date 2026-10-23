# Laya decision visualizations

These PNGs are static, human-readable exports of the Laya turn-by-turn analysis
on the same 850-row synthetic holdout used in `SCENARIO_RESULTS.md`.

- `conversation-only-*`: conversation-only Laya input.
- `prioritized-evidence-*`: alternate input with prioritised clinical evidence.
- `*-condition-pathways.png`: diagnosis decisions after each patient answer.
- `*-escalation-pathways.png`: routine/urgent decisions after each patient answer.
- `*-confidence-intervals.png`: diagnosis, scope and escalation accuracy with
  Wilson 95% confidence intervals.

Pathway diagrams retain transitions followed by at least 20 conversations. This
keeps the static images readable; the underlying JSON report retains all paths.
The data are synthetic and these results are not clinical validation.
