# Scenario model results

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

