"""Learn when a fixed-length diagnostic conversation may stop safely.

The controller never sees labels at inference time.  It consumes only the
current model's probabilities, probability margins, prediction stability, and
the current answer number.  Training labels mean that both diagnosis and
escalation are correct at that checkpoint.  The fifth answer is always a hard
stop, regardless of the controller score.
"""

from collections import Counter

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score


MAX_PATIENT_ANSWERS = 5
STOP_FEATURES = [
    "answer_fraction",
    "condition_confidence",
    "condition_margin",
    "escalation_confidence",
    "escalation_margin",
    "condition_stable",
    "escalation_stable",
    "joint_stable",
    "condition_confidence_delta",
    "escalation_confidence_delta",
]


def _probability_summary(answer):
    values = sorted((float(value) for value in answer["probabilities"].values()), reverse=True)
    best = values[0] if values else 0.0
    second = values[1] if len(values) > 1 else 0.0
    return best, best - second


def trace_feature_rows(traces):
    """Flatten confidence traces into one leakage-free row per checkpoint."""
    rows = []
    for consultation, trace in enumerate(traces):
        previous = None
        for answer_number, checkpoint in enumerate(trace, 1):
            answers = checkpoint["answers"]
            condition = answers["condition"]
            escalation = answers["escalation"]
            condition_confidence, condition_margin = _probability_summary(condition)
            escalation_confidence, escalation_margin = _probability_summary(escalation)
            condition_stable = float(
                previous is not None
                and previous["condition"]["choice"] == condition["choice"]
            )
            escalation_stable = float(
                previous is not None
                and previous["escalation"]["choice"] == escalation["choice"]
            )
            rows.append({
                "consultation": consultation,
                "answer_number": answer_number,
                "answer_fraction": answer_number / MAX_PATIENT_ANSWERS,
                "condition_confidence": condition_confidence,
                "condition_margin": condition_margin,
                "escalation_confidence": escalation_confidence,
                "escalation_margin": escalation_margin,
                "condition_stable": condition_stable,
                "escalation_stable": escalation_stable,
                "joint_stable": condition_stable * escalation_stable,
                "condition_confidence_delta": (
                    0.0 if previous is None
                    else condition_confidence - _probability_summary(previous["condition"])[0]
                ),
                "escalation_confidence_delta": (
                    0.0 if previous is None
                    else escalation_confidence - _probability_summary(previous["escalation"])[0]
                ),
            })
            previous = answers
    return pd.DataFrame(rows)


def stop_targets(traces, truths):
    """Label checkpoints where both live decisions are correct."""
    labels = []
    for trace, truth in zip(traces, truths):
        for checkpoint in trace:
            answers = checkpoint["answers"]
            labels.append(int(
                answers["condition"]["choice"] == truth["condition"]
                and answers["escalation"]["choice"] == truth["escalation"]
            ))
    return np.asarray(labels, dtype=int)


def fit_stop_controller(traces, truths, family="ebm"):
    """Fit a small stop/continue head for an EBM or Laya confidence trace."""
    features = trace_feature_rows(traces)
    target = stop_targets(traces, truths)
    if len(np.unique(target)) < 2:
        raise ValueError("Stopping controller requires both safe and unsafe checkpoints")
    if family == "ebm":
        model = ExplainableBoostingClassifier(
            interactions=0,
            max_rounds=150,
            outer_bags=2,
            validation_size=0.15,
            random_state=2111,
            n_jobs=1,
        )
    elif family == "laya":
        # Laya supplies the semantic decisions and probabilities.  This compact
        # calibrated head turns its trace into an auditable stop decision.
        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=2111
        )
    else:
        raise ValueError(f"Unknown stopping family: {family}")
    model.fit(features[STOP_FEATURES], target)
    return model


def stopping_probabilities(model, traces):
    features = trace_feature_rows(traces)
    positive = list(model.classes_).index(1)
    probabilities = model.predict_proba(features[STOP_FEATURES])[:, positive]
    result = []
    offset = 0
    for trace in traces:
        result.append(probabilities[offset:offset + len(trace)].tolist())
        offset += len(trace)
    return result


def apply_stop_policy(traces, stop_probabilities, threshold, min_answers=2):
    selected = []
    for trace, probabilities in zip(traces, stop_probabilities):
        chosen = len(trace) - 1
        for index, probability in enumerate(probabilities):
            answer_number = index + 1
            if answer_number >= min_answers and probability >= threshold:
                chosen = index
                break
        selected.append(chosen)
    return selected


def policy_metrics(traces, truths, selected):
    condition_truth = [truth["condition"] for truth in truths]
    escalation_truth = [truth["escalation"] for truth in truths]
    condition_prediction = []
    escalation_prediction = []
    answer_counts = []
    for trace, index in zip(traces, selected):
        answers = trace[index]["answers"]
        condition_prediction.append(answers["condition"]["choice"])
        escalation_prediction.append(answers["escalation"]["choice"])
        answer_counts.append(index + 1)
    condition_correct = np.asarray(condition_prediction) == np.asarray(condition_truth)
    escalation_correct = np.asarray(escalation_prediction) == np.asarray(escalation_truth)
    urgent_truth = np.asarray(escalation_truth) == "urgent"
    urgent_prediction = np.asarray(escalation_prediction) == "urgent"
    return {
        "rows": len(traces),
        "condition_accuracy": round(float(np.mean(condition_correct)), 4),
        "escalation_accuracy": round(float(np.mean(escalation_correct)), 4),
        "joint_accuracy": round(float(np.mean(condition_correct & escalation_correct)), 4),
        "urgent_recall": round(float(recall_score(
            urgent_truth, urgent_prediction, zero_division=0
        )), 4),
        "mean_patient_answers": round(float(np.mean(answer_counts)), 3),
        "median_patient_answers": float(np.median(answer_counts)),
        "stopped_before_final": round(float(np.mean(np.asarray(answer_counts) < MAX_PATIENT_ANSWERS)), 4),
        "stop_distribution": {
            str(answer): count for answer, count in sorted(Counter(answer_counts).items())
        },
    }


def select_threshold(model, traces, truths, question_cost=0.03, urgent_miss_cost=0.5):
    """Select a validation threshold with explicit efficiency/safety costs."""
    probabilities = stopping_probabilities(model, traces)
    candidates = np.linspace(0.2, 0.9, 29)
    rows = []
    for threshold in candidates:
        selected = apply_stop_policy(traces, probabilities, float(threshold))
        metrics = policy_metrics(traces, truths, selected)
        urgent_miss_rate = 1 - metrics["urgent_recall"]
        question_fraction = (metrics["mean_patient_answers"] - 1) / (MAX_PATIENT_ANSWERS - 1)
        utility = (
            metrics["joint_accuracy"]
            - question_cost * question_fraction
            - urgent_miss_cost * urgent_miss_rate
        )
        rows.append({"threshold": round(float(threshold), 3), "utility": utility, **metrics})
    best = max(rows, key=lambda item: (item["utility"], -item["mean_patient_answers"]))
    return best["threshold"], rows


def annotate_records(records, model, threshold):
    traces = [record["confidence_trace"] for record in records]
    probabilities = stopping_probabilities(model, traces)
    selected = apply_stop_policy(traces, probabilities, threshold)
    annotated = []
    for record, row_probabilities, selected_index in zip(records, probabilities, selected):
        copied = dict(record)
        copied_trace = []
        for index, (checkpoint, probability) in enumerate(
            zip(record["confidence_trace"], row_probabilities)
        ):
            copied_checkpoint = dict(checkpoint)
            copied_checkpoint["stop_probability"] = round(float(probability), 4)
            copied_checkpoint["policy_stop"] = index == selected_index
            copied_trace.append(copied_checkpoint)
        copied["confidence_trace"] = copied_trace
        copied["selected_answer"] = selected_index + 1
        copied["selected_answers"] = copied_trace[selected_index]["answers"]
        annotated.append(copied)
    return annotated
