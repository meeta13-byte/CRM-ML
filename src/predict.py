"""Single-customer scoring for the lead-conversion page.

`conversion.joblib` holds all four fitted pipelines, so the page can score one
chosen algorithm or compare all four on the same customer.
"""
import json

import joblib
import numpy as np
import pandas as pd

from .config import BANDS, MODELS

STAGE = "conversion"


def load_model(stage=STAGE):
    path = MODELS / f"{stage}.joblib"
    return joblib.load(path) if path.exists() else None


def load_meta(stage=STAGE):
    path = MODELS / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else None


def priority(probability):
    if probability >= BANDS["High"]:
        return "High"
    if probability >= BANDS["Medium"]:
        return "Medium"
    return "Low"


def _frame(payload, specs):
    """One row with the dtypes the pipelines saw during training."""
    row = {}
    for spec in specs:
        value = payload.get(spec["name"])
        if spec["kind"] == "numeric":
            try:
                row[spec["name"]] = float(value)
            except (TypeError, ValueError):
                row[spec["name"]] = np.nan  # imputed by the pipeline
        else:
            row[spec["name"]] = str(value)
    return pd.DataFrame([row])


def _one(model, frame):
    probability = float(model.predict_proba(frame)[0, 1])
    score = (float(model.decision_function(frame)[0])
             if hasattr(model, "decision_function") else None)
    return {
        "probability": round(probability, 4),
        "prediction": int(probability >= 0.5),
        "priority": priority(probability),
        "decision_score": None if score is None else round(score, 4),
        "probability_supported": True,  # all four were fitted with predict_proba
    }


def score(payload, model_name=None, stage=STAGE):
    """Score one algorithm. Raises FileNotFoundError before training."""
    models = load_model(stage)
    if models is None:
        raise FileNotFoundError(
            f"models/{stage}.joblib not found; run `python3 -m src.train_conversion`"
        )
    meta = load_meta(stage)
    frame = _frame(payload, meta["features"])
    if model_name is None:
        model_name = meta["best_model"]
    if model_name not in models:
        raise KeyError(f"{model_name} is not one of {sorted(models)}")
    result = _one(models[model_name], frame)
    result["model"] = model_name
    return result


def score_all(payload, stage=STAGE):
    """Same customer through every algorithm, for the comparison view."""
    models = load_model(stage)
    if models is None:
        raise FileNotFoundError(
            f"models/{stage}.joblib not found; run `python3 -m src.train_conversion`"
        )
    meta = load_meta(stage)
    frame = _frame(payload, meta["features"])
    rows = []
    for name, model in models.items():
        result = _one(model, frame)
        result["model"] = name
        rows.append(result)
    return pd.DataFrame(rows)[
        ["model", "probability", "prediction", "priority", "decision_score"]
    ]
