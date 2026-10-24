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

The confidence-narrowing audit reaches the same conclusion from the full class
distribution. On test, mean probability assigned to the true diagnosis rises
from 53.39% after answer 1 to 88.48% after answer 3 and 91.43% after answer 5.
The mean effective number of plausible diagnoses falls from 2.665 to 1.308 and
then 1.226 over the same checkpoints. Influenza remains the principal weakness:
final test recall is 55.0% and mean probability on the true influenza class is
55.37%, compared with at least 85.06% mean true-class probability for every
other displayed diagnosis group.

The plots use Beta(1,1) posteriors for correctness and equal-tailed 95% credible
intervals. Confidence, margin and entropy-derived concentration are not treated
as Bayesian posteriors; their bands are explicitly labelled Gaussian 95%
intervals for the sample mean. This distinction prevents the model's class
probability from being misrepresented as epistemic clinical certainty.

## Laya status

The v2 Laya data are exported as 5,100 train states, 1,700 validation states,
and 1,700 test states. The fine-tune entry point now uses these partial states by
default, evaluates all three splits, fits a Laya-specific stopping controller,
and writes the same metrics and decision-map schema as EBM.

A new Laya checkpoint must be trained on the existing CUDA server. The local
Codex executor does not hold an SSH key or AWS profile for that server; the two
known NUS hosts currently reject it with `Permission denied (publickey)`. The
repository's existing Laya artifact contains final-answer decisions only, not
the five confidence states needed to fit or audit a stop policy. It would be
methodologically invalid to fabricate those missing states or relabel the old
holdout as a new train/validation/test run.

On the authenticated GPU server, run `scripts/run_laya_v2_server.sh`. It trains
the v2 Laya head, evaluates all splits, fits the stop controller, and creates the
matching Laya overview and individual maps under
`/home/ubuntu/dsa2111-persistent/results/conversation-v0.3-v2/`.

All figures and results use rule-generated synthetic data. They validate the
pipeline and its audit trail, not clinical safety, calibration, or effectiveness.
