import joblib
import numpy as np
import pandas as pd


def predict_with_explanation(model_path, record):
    saved = joblib.load(model_path)
    model, features = saved["model"], saved["features"]
    row = pd.DataFrame([{name: record[name] for name in features}])
    probability = model.predict_proba(row)[0]
    choice = int(np.argmax(probability))
    local = model.explain_local(row).data(0)
    scores = np.asarray(local["scores"])
    if scores.ndim == 2:
        scores = scores[:, choice]
    contributions = sorted(
        zip(local["names"], scores), key=lambda item: abs(item[1]), reverse=True
    )[:10]
    return {
        "condition": model.classes_[choice],
        "probabilities": dict(zip(model.classes_, probability.round(6))),
        "contributions": [(name, round(float(score), 6)) for name, score in contributions],
    }
