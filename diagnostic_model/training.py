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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_model/artifacts"))
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()
    result = train_and_evaluate(pd.read_csv(args.data), args.output_dir, not args.no_tune)
    print(json.dumps(result, indent=2))
