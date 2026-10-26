"""Single-row scoring used by the dashboard."""
import json

import joblib
import pandas as pd

from .config import BANDS, MODELS


def load_model(stage):
    path = MODELS / f"{stage}.joblib"
    return joblib.load(path) if path.exists() else None


def load_meta(stage):
    path = MODELS / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else None


def priority(probability):
    if probability >= BANDS["High"]:
        return "High"
    if probability >= BANDS["Medium"]:
        return "Medium"
    return "Low"


def score(stage, payload):
    """payload: {feature_name: value} straight from the app inputs."""
    model = load_model(stage)
    if model is None:
        raise FileNotFoundError(f"models/{stage}.joblib not found; run training first")
    probability = float(model.predict_proba(pd.DataFrame([payload]))[0, 1])
    return {
        "probability": round(probability, 4),
        "prediction": int(probability >= 0.5),
        "priority": priority(probability),
    }
