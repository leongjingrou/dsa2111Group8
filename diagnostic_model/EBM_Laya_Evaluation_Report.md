# Evaluation of the Explainable Boosting Machine and Laya as an Alternative Diagnostic Model

## Executive conclusion

The Explainable Boosting Machine (EBM) should remain the primary diagnostic model for the current prototype. On the matched 45-case synthetic holdout, it achieved 97.78% accuracy, a macro F1 score of 0.9774, a multiclass Brier score of 0.0295, and an amortized inference latency of 0.0353 ms per case. The fully fine-tuned Laya decision model achieved 86.67% accuracy, a macro F1 score of 0.8661, a Brier score of 0.2057, and 16.18 ms per case.

Laya nevertheless showed substantial task adaptation: full fine-tuning raised its accuracy from 24.44% to 86.67% and reduced its log loss by approximately 74.4%. These results justify retaining Laya as an experimental, text-native alternative or secondary decision layer. They do not support replacing the EBM, because Laya remained less accurate, produced weaker probability estimates, and was approximately 458 times slower in the present batched benchmark.

All results were obtained from a small synthetic dataset. They demonstrate that the software and evaluation architecture work, but they do not establish clinical safety, effectiveness, or generalisability.

## 1. Relationship to the project proposal

The group proposal describes an artificial intelligence (AI)-assisted telemedicine system that organises patient information, estimates compatible conditions, identifies cases requiring escalation, and retains clinicians as the final decision-makers (DSA2111 Group 8, 2026). It makes three design choices that are directly relevant to this evaluation:

1. **Diagnostic estimation is separated from safety and escalation.** The diagnostic model estimates which in-scope condition is compatible with the available information. A separate rule-based safety layer handles red flags, insufficient information, out-of-scope cases, and urgent escalation. Diagnostic confidence must therefore never be treated as a substitute for safety screening.
2. **The diagnostic component should be interpretable.** The proposal identifies a Generalized Additive Model or EBM as a suitable approach because feature-level contributions can be reviewed. This aligns with the additive structure of EBMs, in which learned feature functions contribute to the final prediction ([InterpretML, 2023](https://interpret.ml/docs/ebm.html)). Clinical research on generalized additive models also indicates that interpretability depends on the complexity of the learned risk functions rather than on the model label alone ([Hegselmann et al., 2020](https://proceedings.mlr.press/v126/hegselmann20a.html)).
3. **A structured decision record should precede patient-facing explanation.** The language layer should communicate a recorded assessment rather than invent unsupported clinical reasoning. The EBM contributes feature-level scores, while Laya contributes a typed choice, probability distribution, confidence value, and JSON Lines (JSONL) decision record. These are complementary audit records, although a Laya probability log is not equivalent to an explanation of clinical reasoning.

The Part 3 diagnostic-model requirements are therefore: a trained and tuned EBM, feature-contribution plots, calibration analysis, and comparison against a simple baseline such as logistic regression. The implementation satisfies these prototype deliverables. It produces a tuned EBM artifact, global feature-importance and calibration plots, and a matched logistic-regression comparison. Laya extends the evaluation beyond the required deliverables without changing the proposal's safety boundaries.

## 2. Models and evaluation protocol

### 2.1 Dataset

The prototype dataset contains 180 synthetic vignettes, 29 structured features, and six in-scope condition classes:

- allergic rhinitis;
- common cold;
- gastroenteritis;
- influenza;
- migraine; and
- urinary tract infection.

A stratified 25% holdout produced 45 evaluation cases. The EBM and Laya comparison used the same held-out rows. The EBM consumed structured variables, whereas Laya consumed vignette text derived from those variables. The comparison is therefore matched by patient scenario but not by input representation.

### 2.2 EBM and logistic-regression baseline

The low-level EBM disables interaction terms to retain a simple additive architecture. Its learning rate and bin count were selected through three-fold grid search on the training data using macro F1. The selected configuration used a learning rate of 0.03 and 16 bins. A standardized logistic-regression pipeline served as the simple baseline.

This structure is consistent with the proposal's preference for reviewable feature contributions. EBM explanations can show how individual features affect a score, while the logistic model provides a useful check on whether the synthetic task is already linearly separable.

### 2.3 Laya alternative

Laya is a non-autoregressive decision model that accepts text or structured state together with typed questions and returns typed answers and probabilities ([Convai Innovations, 2026](https://huggingface.co/convaiinnovations/laya)). Three settings were evaluated:

- the base model without task-specific tuning;
- head-only tuning; and
- full encoder and decision-head tuning.

Full tuning used 115 training cases, 20 calibration cases, and the untouched 45-case holdout. The model was trained for four epochs, followed by temperature calibration on the 20-case calibration subset. The calibration subset is too small for strong claims about probability reliability, but it provides an initial implementation of the required calibration workflow.

## 3. Meaning and measurement of the metrics

### Accuracy

Accuracy is the fraction of holdout cases for which the highest-probability class equals the ground-truth diagnosis. It is easy to interpret but can conceal weak performance on less frequent conditions. Scikit-learn defines normalized accuracy as the fraction of correctly classified samples ([scikit-learn, 2026a](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)).

### Macro F1

F1 is the harmonic mean of precision and recall. Macro F1 calculates an F1 score separately for each diagnosis and takes their unweighted mean. Each condition therefore contributes equally, even if class frequencies differ ([scikit-learn, 2026b](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)).

### Log loss

Log loss evaluates the complete probability distribution. It penalises confident errors more severely than uncertain errors, so it distinguishes models that have the same accuracy but assign different probabilities. Zero is ideal and the upper range is unbounded. It is the negative log-likelihood, also known as cross-entropy loss ([scikit-learn, 2026c](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)).

### Multiclass Brier score

The project uses the unscaled multiclass Brier score:

$$
\text{Brier}=\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}(p_{ic}-y_{ic})^2.
$$

Here, \(p_{ic}\) is the predicted probability for class \(c\), and \(y_{ic}\) is 1 for the true class and 0 otherwise. A perfect score is 0; this multiclass form ranges from 0 to 2. Lower values indicate that probability mass is placed closer to the correct outcomes ([scikit-learn, 2026d](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)).

The Brier score reflects both calibration and discrimination. It should not be described as a pure calibration statistic. A separate reliability diagram or calibration-error measure is required to identify systematic overconfidence or underconfidence.

### Expected calibration error

Expected calibration error (ECE) groups predictions into confidence bins and measures the weighted gap between mean confidence and observed accuracy. The fully tuned Laya ECE of 0.0845 represents an estimated average confidence-accuracy gap of 8.45 percentage points under the chosen binning method. It does not mean that every probability is accurate within 8.45 percentage points.

ECE is sensitive to sample size, bin definitions, and whether probabilities are examined globally or by class. Conventional ECE can therefore produce unstable conclusions, particularly with only 45 evaluation cases ([Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html); [Nixon et al., 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf)). ECE was not recorded for the EBM, so the present experiment does not support a direct ECE comparison between the two architectures.

### Latency

Latency was measured with Python's high-resolution `perf_counter` around the prediction call in [`laya_model.py`](https://github.com/leongjingrou/dsa2111Group8/blob/feature/laya-decision-layer/diagnostic_model/laya_model.py). Both models first completed an untimed warm-up prediction.

- Laya predicted all 45 cases in batches of 16. Five complete runs were timed.
- The EBM called `predict_proba` on all 45 cases. Twenty complete runs were timed.
- The median total duration was divided by 45 to obtain latency per row.

This is a warm, batched, amortized inference measurement. It excludes model loading, comma-separated values (CSV) file loading, feature or vignette construction, log writing, network transfer, and application overhead. It is best interpreted as a throughput proxy on the test server, not as the response time a patient would observe for one request. A production evaluation should report end-to-end p50, p95, and p99 latency, cold-start time, throughput under concurrency, memory consumption, and hardware configuration.

## 4. Results

| Model | Accuracy | Macro F1 | Log loss | Brier score | ECE | Latency per row |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.9778 | 0.9774 | 0.1051 | 0.0392 | Not measured | Not measured |
| EBM | 0.9778 | 0.9774 | 0.0596 | 0.0295 | Not measured | 0.0353 ms |
| Laya zero-shot | 0.2444 | 0.1628 | 1.5823 | 0.7724 | 0.1751 | 15.84 ms |
| Laya head-only tuning | 0.2444 | 0.1628 | 1.4507 | 0.7300 | 0.1589 | 15.94 ms |
| Laya full tuning | 0.8667 | 0.8661 | 0.4057 | 0.2057 | 0.0845 | 16.18 ms |

### 4.1 EBM versus logistic regression

The EBM and logistic regression classified the same proportion of holdout cases correctly: approximately 44 of 45. The EBM nevertheless achieved a 43.3% lower log loss and a 24.7% lower Brier score. On this dataset, the evidence therefore supports improved probability quality rather than improved hard-label accuracy.

The identical accuracy also indicates that the current synthetic task may be relatively easy or largely linearly separable. A larger and more varied evaluation is needed to determine whether the EBM's nonlinear feature functions produce a meaningful diagnostic advantage. Its present advantages are the probability scores and the ability to visualise feature-level contributions.

### 4.2 Laya adaptation

Head-only tuning reduced log loss and Brier score but did not improve accuracy or macro F1. Full tuning was necessary to adapt the representation to the diagnostic task. Relative to zero-shot Laya, full tuning produced:

- a 62.23 percentage-point increase in accuracy;
- a 0.7033 increase in macro F1;
- an approximately 74.4% reduction in log loss;
- an approximately 73.3% reduction in Brier score; and
- a reduction in ECE from 0.1751 to 0.0845.

This result is consistent with the Laya model card's statement that domain fine-tuning can materially improve decision accuracy. The model card is vendor-provided evidence, however, and does not independently validate Laya for clinical diagnosis.

### 4.3 EBM versus fully tuned Laya

The EBM exceeded fully tuned Laya by 11.11 percentage points in accuracy and by 0.1113 in macro F1. It also obtained substantially lower log loss and Brier score. In count terms, the EBM produced approximately five fewer errors on the 45-case holdout.

Laya required 16.18 ms per row compared with 0.0353 ms for the EBM, yielding the reported 458.36-fold latency ratio. Because this is a small batched benchmark and the architectures use different execution paths, the ratio should not be generalised beyond the present server and implementation.

## 5. Interpretation within the proposed system

### EBM as the primary diagnostic estimator

The EBM is the better fit for the current Part 3 deliverable. It provides high prototype accuracy, strong probability scores, very low inference latency, and direct feature-contribution plots. This matches the proposal's intention to expose how symptoms such as fever, breathing difficulty, duration, and progression affect the preliminary diagnostic estimate.

The feature contributions remain model explanations rather than clinical proof. They show how the model used its encoded variables; they do not demonstrate causality or guarantee that the learned relationship is medically appropriate. Clinician review and external validation remain necessary.

### Laya as an alternative or supplementary layer

Laya offers two useful capabilities. First, it can operate directly on vignette text rather than requiring a complete structured feature vector. Second, it produces a typed decision and probability record that can be stored for audit and model-disagreement analysis.

These capabilities do not yet outweigh the measured performance gap. Laya should therefore be treated as one of the following:

1. an experimental alternative evaluated alongside the EBM;
2. a secondary text-native decision whose disagreement with the EBM triggers review; or
3. a bridge between the language-based intake layer and the structured diagnostic model.

Laya should not control safety escalation. Red flags, insufficient information, contraindications, pregnancy or age limits, and out-of-scope conditions must remain within the independent safety layer described by the proposal. This separation is important because symptom-checker diagnostic and triage performance can differ, and failures can be clinically consequential ([Semigran et al., 2015](https://doi.org/10.1136/bmj.h3480)).

## 6. Limitations

1. **Synthetic data:** The vignettes were generated for prototype development and may contain cleaner class boundaries than real patient descriptions.
2. **Small holdout:** Forty-five cases are insufficient for precise overall estimates or reliable per-diagnosis and subgroup analysis. Confidence intervals were not included in the current artifacts.
3. **No external validation:** Training, calibration, and evaluation data originate from the same synthetic generation process.
4. **Different representations:** The EBM uses structured features while Laya uses derived text. The experiment compares complete model pipelines rather than isolating architecture alone.
5. **Partial calibration comparison:** Brier score is available for all models, but ECE was not recorded for the EBM or logistic baseline.
6. **Narrow latency benchmark:** The reported latency excludes most end-to-end application costs and does not include tail latency or concurrency.
7. **No safety-layer evaluation:** The results measure diagnostic classification. They do not measure escalation sensitivity, false escalation rate, contraindication detection, or out-of-scope recognition.

These limitations are material under the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis plus AI (TRIPOD+AI) statement, which emphasises transparent reporting of intended use, data, evaluation procedures, findings, limitations, and relevant subgroups ([Collins et al., 2024](https://doi.org/10.1136/bmj-2023-078378)). This report follows those principles as guidance but does not claim formal TRIPOD+AI compliance.

## 7. Recommended next steps

### Immediate architecture

- Retain the EBM as the primary diagnostic estimator.
- Retain logistic regression as the simple baseline.
- Keep Laya on an experimental branch as a secondary text-native decision model.
- Log EBM feature contributions, Laya probabilities, model disagreement, missing information, and safety-rule results in the structured decision record.
- Escalate disagreements and low-confidence cases rather than automatically selecting one model.

### Evaluation expansion

1. Build the proposal's clinician-reviewed set of 100-200 handwritten vignettes, covering all in-scope diagnoses and dangerous mimics.
2. Reserve a genuinely external test set and add cases representing new diagnoses, ambiguous presentations, contradictory histories, and out-of-scope conditions.
3. Report per-class sensitivity, specificity, precision, recall, macro F1, confusion matrices, log loss, Brier score, calibration plots, and confidence intervals.
4. Evaluate the safety layer separately using escalation sensitivity, missed-urgent-case rate, false escalation rate, and time to escalation.
5. Add ECE or classwise calibration error for every model using identical binning, while retaining Brier score and reliability plots.
6. Use paired bootstrap intervals and an appropriate paired test, such as McNemar's test for classification errors, before claiming that one model is statistically superior.
7. Measure cold and warm p50, p95, and p99 latency under realistic request concurrency.
8. Test a hybrid policy in which agreement permits routine processing while disagreement, abstention, or safety triggers require human review.
9. Preserve a replay set of previous diagnoses during future tuning to detect catastrophic forgetting.
10. Add new cases to training only after clinician review and dataset versioning; do not learn automatically from raw patient interactions or model-generated labels.

## 8. Final assessment

The prototype results support the EBM as the main diagnostic component envisioned in the proposal. It meets the immediate need for a low-complexity, inspectable model and performs at least as well as logistic regression on hard classifications while providing better probability scores. The current data do not yet demonstrate that its nonlinear structure improves diagnostic accuracy over the simple baseline.

Laya demonstrates that a text-native System 1 decision model can be adapted to the same diagnostic labels. Full fine-tuning produced a large improvement over its zero-shot behaviour, but its remaining accuracy, probability-quality, and latency disadvantages prevent it from being recommended as a complete replacement. Its most defensible role is as an alternative experimental model or complementary decision-tracking layer within an EBM-led, safety-separated architecture.

## References

Collins, G. S., Moons, K. G. M., Dhiman, P., et al. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. https://doi.org/10.1136/bmj-2023-078378

Convai Innovations. (2026). *Laya model card*. Hugging Face. https://huggingface.co/convaiinnovations/laya

DSA2111 Group 8. (2026). *AI-Assisted Telemedicine Routing and Escalation in Singapore* [Unpublished group project proposal].

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of Machine Learning Research, 70*, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html

Hegselmann, S., Volkert, T., Ohlenburg, H., Gottschalk, A., Dugas, M., & Ertmer, C. (2020). An evaluation of the doctor-interpretability of generalized additive models with interactions. *Proceedings of Machine Learning Research, 126*, 46-79. https://proceedings.mlr.press/v126/hegselmann20a.html

InterpretML Contributors. (2023). *Explainable Boosting Machine*. https://interpret.ml/docs/ebm.html

Nixon, J., Dusenberry, M., Jerfel, G., Nguyen, T., Liu, J., Zhang, L., & Tran, D. (2019). Measuring calibration in deep learning. *CVPR Workshops*. https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf

Scikit-learn Developers. (2026a). *accuracy_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

Scikit-learn Developers. (2026b). *f1_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

Scikit-learn Developers. (2026c). *log_loss*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

Scikit-learn Developers. (2026d). *brier_score_loss*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

Semigran, H. L., Linder, J. A., Gidengil, C., & Mehrotra, A. (2015). Evaluation of symptom checkers for self diagnosis and triage: Audit study. *BMJ, 351*, h3480. https://doi.org/10.1136/bmj.h3480
