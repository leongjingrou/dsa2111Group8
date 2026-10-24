# V2 conversational diagnosis and learned stopping

## Protocol

Every synthetic consultation contains five fixed doctor/patient Q&A pairs. The
train/validation/test split is stratified by diagnosis and escalation and uses
1,020 / 340 / 340 consultations. Model fitting uses train only, the stopping
threshold is selected on validation, and test is retained for final reporting.

The stopping controller sees only information available at inference time:
diagnosis and escalation confidence, probability margins, prediction stability,
confidence changes, and answer number. It never sees the true diagnosis or
escalation label at inference. The fifth answer is a hard fallback stop.

The validation objective is:

`joint accuracy - 0.03 × question fraction - 0.5 × urgent miss rate`

This makes the accuracy/efficiency/safety trade-off explicit rather than using
an arbitrary confidence threshold.

## Sequential EBM result

The selected threshold was 0.90.

| Split | Diagnosis accuracy | Escalation accuracy | Joint accuracy | Urgent recall | Mean answers | Stopped before answer 5 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 96.96% | 98.04% | 95.10% | 97.06% | 3.159 | 90.88% |
| Validation | 95.59% | 97.06% | 92.65% | 96.47% | 3.221 | 90.88% |
| Test | 93.53% | 95.59% | 89.12% | 94.12% | 3.194 | 88.53% |

On test, always asking all five questions produced 94.71% diagnosis accuracy,
95.88% escalation accuracy, 90.59% joint accuracy, and 94.71% urgent recall.
The learned policy therefore saved 1.806 patient answers on average while losing
1.47 percentage points of joint accuracy and 0.59 percentage points of urgent
recall. This is a real trade-off and should not be described as a free efficiency
gain.

Turn-level test joint accuracy was 41.76%, 62.94%, 87.65%, 88.53%, and 90.59%
after answers 1 through 5. Most diagnostic information arrives with the third
answer (associated symptoms); question 5 still adds measurable value.

## Laya status

The v2 Laya data are exported as 5,100 train states, 1,700 validation states,
and 1,700 test states. The fine-tune entry point now uses these partial states by
default, evaluates all three splits, fits a Laya-specific stopping controller,
and writes the same metrics and decision-map schema as EBM.

A new Laya checkpoint was not trained on this workstation. The official typed
decision checkpoint is approximately 843 MB and this environment has no CUDA
device. The repository's existing Laya artifact contains final-answer decisions
only, not the five confidence states needed to fit or audit a stop policy. It
would be methodologically invalid to fabricate those missing states or relabel
the old holdout as a new train/validation/test run.

Run `diagnostic_model.laya_finetune` on the existing GPU server, then pass its
v2 metrics and decisions to `diagnostic_model.export_stopping_visuals` to create
the matching Laya overview and individual maps.

All figures and results use rule-generated synthetic data. They validate the
pipeline and its audit trail, not clinical safety, calibration, or effectiveness.
