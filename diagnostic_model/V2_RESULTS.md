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
| Train | 96.96% | 97.94% | 95.00% | 97.06% | 3.154 | 90.88% |
| Validation | 95.59% | 97.35% | 92.94% | 97.06% | 3.221 | 90.29% |
| Test | 93.53% | 95.59% | 89.12% | 94.12% | 3.212 | 88.24% |

On test, always asking all five questions produced 94.41% diagnosis accuracy,
95.88% escalation accuracy, 90.29% joint accuracy, and 94.71% urgent recall.
The learned policy therefore saved 1.788 patient answers on average while losing
1.17 percentage points of joint accuracy and 0.59 percentage points of urgent
recall. This is a real trade-off and should not be described as a free efficiency
gain.

Turn-level test joint accuracy was 41.76%, 62.94%, 87.65%, 88.53%, and 90.29%
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

## Laya server result

The v2 Laya data contain 5,100 train states, 1,700 validation states, and 1,700
test states. A 10-epoch head fine-tune was run on the authenticated NVIDIA T4
server at revision `9871157`. The selected stopping threshold was 0.75.

| Split | Diagnosis accuracy | Escalation accuracy | Joint accuracy | Urgent recall | Mean answers | Stopped before answer 5 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 80.78% | 90.78% | 73.43% | 92.55% | 3.913 | 53.43% |
| Validation | 78.53% | 89.41% | 69.71% | 92.35% | 3.924 | 53.53% |
| Test | 78.24% | 84.41% | 66.18% | 82.35% | 3.938 | 54.12% |

On test, always asking all five questions produced 78.82% diagnosis accuracy,
83.53% escalation accuracy, 65.88% joint accuracy, and 84.71% urgent recall.
The learned Laya policy saved 1.062 answers and improved joint accuracy by 0.30
percentage points, but lost 2.36 percentage points of urgent recall. That urgent
recall loss is safety-relevant and means the current policy is not suitable for
clinical deployment.

Laya's test mean probability on the true diagnosis rose from 49.15% after the
first answer to 71.34% after the third, then slipped to 70.47% after the fifth.
The mean effective number of plausible diagnoses narrowed from 3.223 to 2.342
and ended at 2.393. This non-monotonic result is visible in the confidence plots
and shows that later answers do not automatically make this fitted head more
certain or more correct.

## Matched comparison and artifacts

The server reran EBM from the same revision and split before training Laya. On
test, EBM exceeded Laya by 15.29 percentage points of diagnosis accuracy, 11.18
points of escalation accuracy, 22.94 points of joint accuracy, and 11.77 points
of urgent recall, while using 0.726 fewer answers on average. The result is a
model comparison on synthetic data, not evidence of clinical performance.

Compact reproducibility artifacts are committed under `server_results_v2/`:
per-split metrics, validation threshold sweeps, the Laya fine-tune summary, the
run log, the matched comparison, and both models' aggregate and individual-case
figures. The 808 MB Laya checkpoint and large raw decision logs remain in the
persistent server paths documented by the run log and are deliberately not
stored in Git.

Rerunning `scripts/run_laya_v2_server.sh` reproduces the full workflow under
`/home/ubuntu/dsa2111-persistent/results/conversation-v0.3-v2/` and keeps the
checkpoint under `/home/ubuntu/dsa2111-persistent/models/laya-head-v0.3-v2/`.

All figures and results use rule-generated synthetic data. They validate the
pipeline and its audit trail, not clinical safety, calibration, or effectiveness.
