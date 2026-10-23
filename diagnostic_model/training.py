import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnostic_model.scenario_data import (
    load_scenarios,
    regenerate_scenarios,
    scenario_features,
    scenario_split,
    scenario_targets,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SEED = 2111


def split_features(data):
    x = data.drop(columns=["condition_id", "vignette_id", "vignette_text"], errors="ignore")
    return x, data["condition_id"].astype(str)


def new_ebm():
    return ExplainableBoostingClassifier(
        interactions=0,
        max_rounds=300,
        outer_bags=4,
        validation_size=0.2,
        random_state=SEED,
        n_jobs=1,
    )


def get_scores(model, x, y):
    prediction = model.predict(x)
    probability = model.predict_proba(x)
    truth = pd.get_dummies(y).reindex(columns=model.classes_, fill_value=0).to_numpy()
    return {
        "accuracy": round(accuracy_score(y, prediction), 4),
        "f1_macro": round(f1_score(y, prediction, average="macro"), 4),
        "log_loss": round(log_loss(y, probability, labels=model.classes_), 4),
        "brier": round(np.mean(np.sum((probability - truth) ** 2, axis=1)), 4),
    }


def get_binary_scores(model, x, y):
    prediction = model.predict(x).astype(int)
    probability = model.predict_proba(x)[:, list(model.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y, prediction), 4),
        "f1": round(f1_score(y, prediction, zero_division=0), 4),
        "precision": round(precision_score(y, prediction, zero_division=0), 4),
        "recall": round(recall_score(y, prediction, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, probability), 4),
        "average_precision": round(average_precision_score(y, probability), 4),
        "brier": round(float(np.mean((probability - np.asarray(y)) ** 2)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_plots(model, x, y, folder):
    overall = model.explain_global().data()
    order = np.argsort(overall["scores"])[-15:]
    plt.figure(figsize=(9, 6))
    plt.barh(np.asarray(overall["names"])[order], np.asarray(overall["scores"])[order])
    plt.xlabel("Mean absolute contribution")
    plt.title("EBM global feature importance")
    plt.tight_layout()
    plt.savefig(folder / "feature_importance.png", dpi=150)
    plt.close()

    probability = model.predict_proba(x)
    plt.figure(figsize=(7, 7))
    for i, label in enumerate(model.classes_):
        observed, predicted = calibration_curve((y == label).astype(int), probability[:, i], n_bins=5)
        plt.plot(predicted, observed, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("One-vs-rest calibration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(folder / "calibration.png", dpi=150)
    plt.close()


def train_and_evaluate(data, output_dir, tune=True):
    x, y = split_features(data)
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )

    if tune:
        search = GridSearchCV(
            new_ebm(),
            {"learning_rate": [0.03, 0.05], "max_bins": [16, 32]},
            scoring="f1_macro",
            cv=3,
        ).fit(train_x, train_y)
        ebm, best = search.best_estimator_, search.best_params_
    else:
        ebm = new_ebm().fit(train_x, train_y)
        best = {"learning_rate": ebm.learning_rate, "max_bins": ebm.max_bins}

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    baseline.fit(train_x, train_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": ebm, "features": list(x.columns)}, output_dir / "ebm_model.joblib")
    save_plots(ebm, test_x, test_y, output_dir)

    report = {
        "rows": len(data),
        "features": x.shape[1],
        "classes": sorted(y.unique()),
        "ebm": get_scores(ebm, test_x, test_y),
        "logistic_regression": get_scores(baseline, test_x, test_y),
        "best_parameters": best,
        "warning": "Synthetic demo data is not clinical validation.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def _fit_ebm(train_x, train_y, tune=False, scoring="f1_macro"):
    if tune:
        search = GridSearchCV(
            new_ebm(),
            {"learning_rate": [0.03, 0.05], "max_bins": [16, 32]},
            scoring=scoring,
            cv=3,
        ).fit(train_x, train_y)
        return search.best_estimator_, search.best_params_
    model = new_ebm().fit(train_x, train_y)
    return model, {"learning_rate": model.learning_rate, "max_bins": model.max_bins}


def train_scenario_models(data, output_dir, tune=False):
    """Train diagnosis, scope, and escalation EBMs on the v0.2 scenario schema."""
    x, symptom_ids = scenario_features(data)
    targets = scenario_targets(data)
    train_index, test_index = scenario_split(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for task in ("scope", "escalation"):
        model, best = _fit_ebm(
            x.loc[train_index], targets[task].loc[train_index], tune=tune, scoring="f1"
        )
        joblib.dump(
            {
                "model": model,
                "features": list(x.columns),
                "symptom_ids": symptom_ids,
                "task": task,
                "dataset_version": sorted(data["dataset_version"].astype(str).unique()),
            },
            output_dir / f"{task}_model.joblib",
        )
        tasks[task] = {
            "metrics": get_binary_scores(model, x.loc[test_index], targets[task].loc[test_index]),
            "best_parameters": best,
        }

    in_scope = targets["scope"].eq(1)
    diagnosis_train = np.intersect1d(train_index, data.index[in_scope])
    diagnosis_test = np.intersect1d(test_index, data.index[in_scope])
    diagnosis_model, diagnosis_best = _fit_ebm(
        x.loc[diagnosis_train],
        targets["diagnosis"].loc[diagnosis_train],
        tune=tune,
        scoring="f1_macro",
    )
    joblib.dump(
        {
            "model": diagnosis_model,
            "features": list(x.columns),
            "symptom_ids": symptom_ids,
            "task": "diagnosis",
            "dataset_version": sorted(data["dataset_version"].astype(str).unique()),
        },
        output_dir / "diagnosis_model.joblib",
    )
    save_plots(
        diagnosis_model,
        x.loc[diagnosis_test],
        targets["diagnosis"].loc[diagnosis_test],
        output_dir,
    )
    tasks["diagnosis"] = {
        "metrics": get_scores(
            diagnosis_model,
            x.loc[diagnosis_test],
            targets["diagnosis"].loc[diagnosis_test],
        ),
        "best_parameters": diagnosis_best,
        "classes": list(diagnosis_model.classes_),
    }

    dataset_versions = sorted(data["dataset_version"].astype(str).unique())
    report = {
        "dataset_version": dataset_versions,
        "rows": len(data),
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "in_scope_rows": int(targets["scope"].sum()),
        "out_of_scope_rows": int((1 - targets["scope"]).sum()),
        "features": x.shape[1],
        "split": "matched condition_id + escalation stratification",
        "tasks": tasks,
        "warning": (
            "All results use rule-generated synthetic data. Random holdout performance measures "
            f"interpolation within generator version {', '.join(dataset_versions)}, "
            "not clinical validity."
        ),
    }
    (output_dir / "scenario_metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_model/artifacts"))
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--scenario-schema", action="store_true")
    parser.add_argument(
        "--generate-scenarios",
        action="store_true",
        help="Regenerate the scenario CSV with data/generate.py before training.",
    )
    args = parser.parse_args()
    if args.generate_scenarios and not args.scenario_schema:
        parser.error("--generate-scenarios requires --scenario-schema")
    if args.generate_scenarios:
        data = regenerate_scenarios(args.data)
    elif args.scenario_schema:
        data = load_scenarios(args.data)
    else:
        data = pd.read_csv(args.data)
    result = (
        train_scenario_models(data, args.output_dir, not args.no_tune)
        if args.scenario_schema
        else train_and_evaluate(data, args.output_dir, not args.no_tune)
    )
    print(json.dumps(result, indent=2))
