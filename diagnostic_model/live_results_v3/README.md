# Server live-conversation results

The live artifacts were generated on the project server from the trained v2
EBM and Laya models using branch revision `caeff4b`; the citation-bearing
DDXPlus conversion used `5f70691`. All patient text is synthetic; there is no
personal or clinical record data in this folder.

## Matched live replays

| Model | Conversation | Answers used | Final diagnosis | Diagnosis confidence | Escalation | Escalation confidence | Stop reason |
|---|---|---:|---|---:|---|---:|---|
| EBM | routine respiratory | 5 | influenza | 72.97% | routine | 67.51% | maximum questions |
| Laya | routine respiratory | 5 | out_of_scope | 64.92% | routine | 79.66% | maximum questions |
| EBM | chest pain + breathing difficulty | 1 | out_of_scope | 65.89% | urgent | 96.34% | explicitly enabled `sg_moh` safety profile |
| Laya | chest pain + breathing difficulty | 1 | out_of_scope | 79.14% | urgent | 96.72% | explicitly enabled `sg_moh` safety profile |

With the default `none` profile, the same urgent conversation behaved as
follows:

| Model | Answers used | Final diagnosis | Escalation | Raw learned-stop result |
|---|---:|---|---|---|
| EBM | 5 | out_of_scope (80.76%) | urgent (97.20%) | stopped at the five-answer cap; final stop probability 93.44% |
| Laya | 2 | out_of_scope (73.98%) | urgent (98.90%) | learned stop at answer 2; stop probability 78.15% |

When `sg_moh` is truthy, both stop at answer 1. Crucially, the logs show
`model_stop: false` at that checkpoint and `safety_override_applied: true`, so
the jurisdictional rule is never misreported as a model decision.

The two routine files use identical patient replies and expose a useful model
difference: EBM commits to influenza while Laya abstains from the supported
condition list. Neither has ground truth because the conversation was written
as a free-form live probe. The safety profile is disabled by default; it was
explicitly enabled for the urgent replay. The output retains the raw learned
stop decision, then records whether the profile changed the final action.

- `live-*-*.json` contains every raw probability map and stopping checkpoint.
- `live-*-*.txt` is the corresponding readable doctor/patient transcript.
- `raw-ebm-samples.json` and `raw-laya-samples.json` each contain four selected
  held-out examples: a correct learned stop, a correct forced-final stop, a
  diagnosis miss, and an escalation miss where available. In those files,
  `source_ground_truth` is the synthetic generator label and is kept separate
  from `final_model_response`.
- `ddxplus-grounded-test-samples.jsonl` contains 12 records converted from the
  checksum-verified official English DDXPlus test split. These preserve the
  source pathology, differential distribution, evidence tokens, and split.
  Their dialogue is marked as reconstructed rather than observed.

The saved output is a research trace, not a clinical diagnosis or validation.
