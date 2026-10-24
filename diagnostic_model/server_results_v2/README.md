# Server results: conversational stopping v2

These compact artifacts come from the complete matched EBM/Laya run on the
project's NVIDIA T4 server. The run used Git revision `9871157`, started at
2026-10-24 07:09 UTC, and completed at 08:23 UTC.

- `comparison/` contains the direct model table, JSON, and chart.
- `ebm/` and `laya/` contain per-split metrics, validation threshold sweeps,
  confidence-narrowing summaries, and aggregate/individual-case figures.
- `laya/laya_head_finetune.json` records the 10-epoch head-training summary.
- `run.log` records the revision, CUDA check, training progress, metrics, paths,
  and successful completion timestamp.

The raw decision logs and the 808 MB Laya checkpoint are intentionally retained
only in persistent server storage. The outputs use synthetic generator data and
must not be interpreted as clinical validation.
