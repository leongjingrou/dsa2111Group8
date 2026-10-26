# Scenario model results

## Conversational v0.3 transfer check

Generator v0.3 renders each structured scenario as five alternating
doctor/patient exchanges and Laya records a prediction after every patient
answer. The existing selected v0.2 head was evaluated without retraining on the
same 425-row holdout, so these numbers measure transfer to the new conversation
format rather than a newly adapted model.

| Model | Condition accuracy | Condition macro-F1 | Scope accuracy | Escalation accuracy | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| EBM (structured v0.3 fields) | 0.9600 | 0.9468 | 0.9600 | 0.9835 | not measured |
| Laya v0.2 head on v0.3 conversations | 0.6824 | 0.3777 | 0.8188 | 0.7271 | 61.6 ms/row |

The turn-level trace shows that more dialogue does not monotonically improve
the old text model. Condition accuracy is 0.6329 after the chief complaint,
drops to 0.5224 after duration/severity/progression, peaks at 0.7012 after the
negative risk screen, and ends at 0.6824 after demographics and temperature.
Escalation accuracy starts at 0.5647, peaks at 0.7859 after associated symptoms,
and ends at 0.7271. This is evidence of a prompt/data-format distribution shift:
the v0.2 head should not be treated as calibrated for chained consultations.
A v0.3-specific fine-tune and an independently reviewed validation set are the
next model-development steps. Escalation remains a near-term triage-risk proxy,
not a clinical or long-term prognosis.

The persistent server outputs are in
`/home/ubuntu/dsa2111-persistent/results/conversation-v0.3/`. They include 2,125
intermediate confidence states across the 425 held-out consultations.

## Conversational v0.3 fine-tune on a stratified half split

A follow-up experiment used a deterministic 50/50 split stratified by condition
and escalation: 850 consultations for training and 850 for testing. The Laya
encoder remained frozen and its decision head was fine-tuned for ten epochs.
The EBM was retrained on the identical training half for a matched comparison.

| Model | Condition accuracy | Condition macro-F1 | Scope accuracy | Escalation accuracy | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| EBM, 50% conversational dataset | 0.9506 | 0.9338 | 0.9518 | 0.9788 | not measured |
| Laya v0.3 head, 50% conversational dataset | 0.7282 | 0.3905 | 0.8106 | 0.8235 | 61.6 ms/row |

Laya's own decision-level evaluation improved from 0.4459 to 0.7765 accuracy;
expected calibration error fell from 0.1754 to 0.0509. End-to-end escalation
accuracy improved materially over the v0.2-to-v0.3 transfer check, although the
different holdout size means the two runs are not a paired comparison.

The final-turn scope decision is conservative: precision is 0.9484 but recall
is 0.4900. The model therefore rarely labels unsupported cases as supported,
but misses about half of the supported cases. Condition accuracy rises from
0.4871 after the chief complaint to 0.7282 after the fifth patient answer, and
escalation accuracy rises from 0.5565 to 0.8235. Unlike the unadapted transfer
run, the best end-to-end results now occur after the complete consultation.

Persistent outputs are under
`/home/ubuntu/dsa2111-persistent/results/conversation-v0.3-half/`, and the final
model is under `/home/ubuntu/dsa2111-persistent/models/laya-head-v0.3-half/`.
The redundant training checkpoint copy was removed after verification; the
loadable final model and all result artifacts remain.

## Prioritized evidence injection A/B

The v0.3 fine-tuned head was evaluated again on the identical 850-case holdout
with a label-free `priority_evidence` field added to each conversation. Findings
were ordered P1-P5 from present red flags through ordinary negative findings,
and each intermediate trace contained only evidence revealed by that turn.

| Laya input | Condition accuracy | Condition macro-F1 | Scope accuracy | Escalation accuracy | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conversation only | 0.7282 | 0.3905 | 0.8106 | 0.8235 | 61.6 ms/row |
| Conversation + prioritized evidence | 0.5400 | 0.3356 | 0.7165 | 0.8447 | 94.3 ms/row |

Priority injection improved escalation by 2.12 percentage points and increased
urgent recall from 0.7388 to 0.8329. However, condition accuracy fell by 18.82
points, scope accuracy fell by 9.41 points, and latency increased by about 53%.
The existing head was not trained on the extra evidence representation, and a
single risk-first summary biases both decisions even though diagnosis and
triage need different evidence orderings.

The result argues against enabling this combined path as the default. A better
next experiment is question-specific evidence: differential-diagnosis findings
for condition/scope, red flags for escalation, followed by fine-tuning on that
same representation. The generated `pathway_map` records decision nodes and
transitions for visual analysis. Persistent outputs are under
`/home/ubuntu/dsa2111-persistent/results/conversation-v0.3-half-prioritized/`.

Server evaluation used the fixed v0.2.0 synthetic scenario dataset, a matched
condition-plus-escalation split, and 425 untouched holdout rows. Laya ran on an
NVIDIA Tesla T4. These results measure interpolation within the rule-generated
dataset only and are not clinical validation.

## Selected models

| Model | Condition accuracy | Condition macro-F1 | Scope accuracy | Escalation accuracy | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| EBM | 0.9600 | 0.9468 | 0.9600 | 0.9835 | not measured |
| Laya base checkpoint | 0.0682 | 0.0450 | 0.3529 | 0.7718 | 43.6 ms/row |
| Laya head fine-tune | 0.7106 | 0.3581 | 0.8706 | 0.8871 | 44.9 ms/row |

The selected Laya model freezes the encoder and fine-tunes the task head for ten
epochs. Laya's independent held-out decision report improved from 0.4200 to
0.7988 accuracy, while expected calibration error fell from 0.1831 to 0.0330.

## Result-driven design check

The first fine-tune exposed class imbalance in the combined condition choice:
275 of 425 holdout cases are out of scope. A second experiment therefore split
Laya into binary scope, six-way diagnosis, and binary escalation decisions,
training diagnosis only on in-scope rows. It did not improve the matched
end-to-end result:

| Laya formulation | Condition accuracy | Condition macro-F1 | Scope accuracy | Escalation accuracy | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined condition plus `out_of_scope` | 0.7106 | 0.3581 | 0.8706 | 0.8871 | 44.9 ms/row |
| Decomposed experiment | 0.6894 | 0.2838 | 0.8235 | 0.8494 | 61.1 ms/row |

The combined formulation is retained because it is empirically better on every
end-to-end metric and is faster. The decomposition experiment remains useful
evidence that changing task shape alone does not solve supported-condition
confusion. A future experiment should balance condition-head examples without
duplicating escalation examples, or fine-tune the encoder with an external,
clinically reviewed validation set.

## Interpretation

- EBM is the stronger model for the present structured, rule-generated data.
  Its advantage is expected because the generator rules map directly to the
  structured symptom features.
- The base Laya checkpoint never predicted `out_of_scope`; task-head adaptation
  fixed most of that failure and substantially improved escalation.
- The adapted Laya model still confuses supported diagnoses. Overall condition
  accuracy is inflated by the out-of-scope majority, so macro-F1 must be reported
  alongside accuracy.
- Neither model result supports a clinical performance claim. Evaluation on a
  different generator version and reviewed real-world cases is required before
  selecting a deployable approach.

## Server layout

The working checkout, virtual environment, model cache, logs, and selected model
are under `/home/ubuntu`, which is backed by the persistent `/dev/sda1` EBS root
volume. The unmounted instance-storage NVMe device is not used.

- checkout: `/home/ubuntu/dsa2111Group8-current`
- selected Laya model: `/home/ubuntu/dsa2111-persistent/models/laya-current`
- Hugging Face cache: `/home/ubuntu/dsa2111-persistent/cache/huggingface`
- logs: `/home/ubuntu/dsa2111-persistent/logs`

