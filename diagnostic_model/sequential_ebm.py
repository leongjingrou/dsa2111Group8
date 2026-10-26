"""Train EBM diagnosis, escalation, and learned stopping on five Q&A states."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from diagnostic_model.laya_model import OUT_OF_SCOPE, conversation_turns
from diagnostic_model.scenario_data import (
    knowledge_base_condition_ids,
    load_scenarios,
    scenario_features,
    scenario_targets,
    scenario_train_validation_test_split,
)
from diagnostic_model.stopping import (
    annotate_records,
    apply_stop_policy,
    fit_stop_controller,
    policy_metrics,
    select_threshold,
    stopping_probabilities,
)


MAX_PATIENT_ANSWERS = 5


def progressive_scenario_features(data, patient_answers, symptom_ids=None):
    """Expose only evidence heard by a given patient-answer checkpoint."""
    if patient_answers not in range(1, MAX_PATIENT_ANSWERS + 1):
        raise ValueError("patient_answers must be from 1 to 5")
    full, symptom_ids = scenario_features(data, symptom_ids=symptom_ids)
    partial = pd.DataFrame(0.0, index=data.index, columns=full.columns)
    parsed = [json.loads(value) for value in data["symptoms"].astype(str)]

    for row_index, symptoms in zip(data.index, parsed):
        present = [name for name, value in symptoms.items() if bool(value)]
        absent = [name for name, value in symptoms.items() if not bool(value)]
        assessed = []
        if patient_answers >= 1 and present:
            assessed.append(present[0])
        if patient_answers >= 3:
            assessed.extend(present[1:])
        if patient_answers >= 4:
            assessed.extend(absent[:5])
        for symptom_id in assessed:
            partial.loc[row_index, f"symptom_{symptom_id}_asked"] = 1.0
            partial.loc[row_index, f"symptom_{symptom_id}_present"] = float(
                bool(symptoms[symptom_id])
            )

    if patient_answers >= 2:
        history = [
            "duration_days", "severity_mild", "severity_normal", "severity_severe",
            "progression_improving", "progression_stable", "progression_worsening",
        ]
        partial.loc[:, history] = full.loc[:, history]
    if patient_answers >= 5:
        demographics = ["age", "temperature_c", "sex_female", "pregnant"]
        partial.loc[:, demographics] = full.loc[:, demographics]

    partial["history_observed"] = float(patient_answers >= 2)
    partial["demographics_observed"] = float(patient_answers >= 5)
    partial["patient_answer"] = patient_answers / MAX_PATIENT_ANSWERS
    return partial, symptom_ids


def _new_model():
    return ExplainableBoostingClassifier(
        interactions=0,
        max_rounds=200,
        outer_bags=2,
        validation_size=0.15,
        random_state=2111,
        n_jobs=1,
    )


def _expanded_features(rows, symptom_ids=None):
    frames = []
    for patient_answers in range(1, MAX_PATIENT_ANSWERS + 1):
        frame, symptom_ids = progressive_scenario_features(
            rows, patient_answers, symptom_ids=symptom_ids
        )
        frame = frame.copy()
        frame["source_index"] = rows.index
        frame["answer_number"] = patient_answers
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), symptom_ids


def _answer(choice, labels, probabilities):
    mapped = {label: float(probability) for label, probability in zip(labels, probabilities)}
    return {
        "choice": str(choice),
        "probabilities": mapped,
        "answer_confidence": float(max(mapped.values())),
    }


def ebm_confidence_traces(rows, bundles):
    traces = [[] for _ in range(len(rows))]
    condition_labels = bundles["condition_labels"]
    for answer_number in range(1, MAX_PATIENT_ANSWERS + 1):
        x, _ = progressive_scenario_features(
            rows, answer_number, symptom_ids=bundles["symptom_ids"]
        )
        scope_model = bundles["scope"]
        diagnosis_model = bundles["diagnosis"]
        escalation_model = bundles["escalation"]
        scope_probability = scope_model.predict_proba(x)[:, list(scope_model.classes_).index(1)]
        diagnosis_probability = diagnosis_model.predict_proba(x)
        escalation_probability = escalation_model.predict_proba(x)
        combined = np.zeros((len(rows), len(condition_labels)))
        combined[:, condition_labels.index(OUT_OF_SCOPE)] = 1 - scope_probability
        for column, label in enumerate(diagnosis_model.classes_):
            combined[:, condition_labels.index(label)] = (
                scope_probability * diagnosis_probability[:, column]
            )
        condition_choice = np.asarray(condition_labels)[combined.argmax(axis=1)]
        escalation_labels = ["routine", "urgent"]
        escalation_choice = np.where(escalation_model.predict(x).astype(int), "urgent", "routine")
        for index in range(len(rows)):
            patient_turns = [
                turn for turn in conversation_turns(rows.iloc[index]) if turn["role"] == "patient"
            ]
            traces[index].append({
                "patient_answer": answer_number,
                "latest_response": patient_turns[answer_number - 1]["content"],
                "answers": {
                    "condition": _answer(
                        condition_choice[index], condition_labels, combined[index]
                    ),
                    "escalation": _answer(
                        escalation_choice[index], escalation_labels, escalation_probability[index]
                    ),
                },
            })
    return traces


def _truths(rows):
    escalation = scenario_targets(rows)["escalation"]
    return [
        {
            "condition": (
                row["condition_id"] if row["scope_status"] == "in_scope" else OUT_OF_SCOPE
            ),
            "escalation": "urgent" if bool(urgent) else "routine",
        }
        for (_, row), urgent in zip(rows.iterrows(), escalation)
    ]


def _records(rows, traces, split):
    truths = _truths(rows)
    return [
        {
            "scenario_id": row["scenario_id"],
            "split": split,
            "conversation": conversation_turns(row),
            "truth": truth,
            "confidence_trace": trace,
            "dataset_version": row["dataset_version"],
        }
        for (_, row), truth, trace in zip(rows.iterrows(), truths, traces)
    ]


def train_sequential_ebm(data, output_dir, validation_size=0.2, test_size=0.2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_indices = scenario_train_validation_test_split(
        data, validation_size=validation_size, test_size=test_size
    )
    train_rows = data.loc[split_indices["train"]]
    expanded, symptom_ids = _expanded_features(train_rows)
    feature_columns = [
        column for column in expanded.columns if column not in {"source_index", "answer_number"}
    ]
    # _expanded_features is ordered by turn, so tile rather than repeat targets.
    targets = scenario_targets(train_rows)
    scope_target = np.tile(targets["scope"].to_numpy(), MAX_PATIENT_ANSWERS)
    escalation_target = np.tile(targets["escalation"].to_numpy(), MAX_PATIENT_ANSWERS)
    diagnosis_target = np.tile(targets["diagnosis"].to_numpy(), MAX_PATIENT_ANSWERS)
    in_scope = np.tile(targets["scope"].to_numpy().astype(bool), MAX_PATIENT_ANSWERS)

    scope_model = _new_model().fit(expanded[feature_columns], scope_target)
    escalation_model = _new_model().fit(expanded[feature_columns], escalation_target)
    diagnosis_model = _new_model().fit(
        expanded.loc[in_scope, feature_columns], diagnosis_target[in_scope]
    )
    bundles = {
        "scope": scope_model,
        "diagnosis": diagnosis_model,
        "escalation": escalation_model,
        "symptom_ids": symptom_ids,
        "features": feature_columns,
        "condition_labels": sorted(knowledge_base_condition_ids() + [OUT_OF_SCOPE]),
    }

    traces = {}
    truths = {}
    records = {}
    for split, indices in split_indices.items():
        rows = data.loc[indices].reset_index(drop=True)
        traces[split] = ebm_confidence_traces(rows, bundles)
        truths[split] = _truths(rows)
        records[split] = _records(rows, traces[split], split)

    stop_model = fit_stop_controller(traces["train"], truths["train"], family="ebm")
    threshold, threshold_search = select_threshold(
        stop_model, traces["validation"], truths["validation"]
    )
    report = {
        "model": "sequential EBM with learned EBM stopping controller",
        "fixed_patient_answers": MAX_PATIENT_ANSWERS,
        "split": {name: len(indices) for name, indices in split_indices.items()},
        "selected_stop_threshold": threshold,
        "threshold_objective": (
            "joint_accuracy - 0.03*question_fraction - 0.5*urgent_miss_rate"
        ),
        "splits": {},
        "warning": "Synthetic generator data; not clinical validation.",
    }
    with (output_dir / "ebm_sequential_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        for split in ("train", "validation", "test"):
            probabilities = stopping_probabilities(stop_model, traces[split])
            selected = apply_stop_policy(traces[split], probabilities, threshold)
            split_metrics = policy_metrics(traces[split], truths[split], selected)
            split_metrics["by_patient_answer"] = [
                {
                    "patient_answer": answer,
                    **policy_metrics(
                        traces[split], truths[split], [answer - 1] * len(traces[split])
                    ),
                }
                for answer in range(1, MAX_PATIENT_ANSWERS + 1)
            ]
            split_metrics["answer_5_baseline"] = split_metrics["by_patient_answer"][-1]
            report["splits"][split] = split_metrics
            for record in annotate_records(records[split], stop_model, threshold):
                log.write(json.dumps(record) + "\n")

    joblib.dump(bundles, output_dir / "ebm_sequential_models.joblib")
    joblib.dump(
        {"model": stop_model, "threshold": threshold},
        output_dir / "ebm_stopping_controller.joblib",
    )
    (output_dir / "ebm_sequential_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "ebm_threshold_search.json").write_text(
        json.dumps(threshold_search, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    report = train_sequential_ebm(
        load_scenarios(args.data), args.output_dir, args.validation_size, args.test_size
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
