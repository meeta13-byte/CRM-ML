"""Tuning, metrics, confusion matrices and artifact saving."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from .config import CV_FOLDS, FIGURES, SELECTION_METRIC


def save_json(payload, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _save_confusion(cm, name, path_prefix, title=None):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or name)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{path_prefix}_{name.replace(' ', '_')}.png", dpi=140)
    plt.close(fig)


def _clean_params(params):
    return {k.replace("model__", ""): v for k, v in params.items()}


def tune(name, pipeline, grid, X_train, y_train, X_test, y_test, path_prefix,
         final_pipeline=None):
    """GridSearchCV on the training split, then one honest test score.

    Tuning runs without probability estimates (much faster for SVC); the chosen
    hyperparameters are then fitted once with calibration enabled, and the
    reported metrics come from that exact artifact so the app matches the table.
    """
    search = GridSearchCV(
        pipeline, grid, scoring=SELECTION_METRIC, cv=CV_FOLDS, n_jobs=-1
    )
    search.fit(X_train, y_train)

    best = final_pipeline if final_pipeline is not None else search.best_estimator_
    if final_pipeline is not None:
        final_pipeline.set_params(**search.best_params_)
        final_pipeline.fit(X_train, y_train)

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    _save_confusion(cm, name, path_prefix,
                    title=f"{name} ({meta_label(path_prefix)})")

    try:
        auc = round(float(roc_auc_score(y_test, proba)), 4) if proba is not None else None
    except ValueError:
        auc = None  # single-class test fold

    results = search.cv_results_
    order = np.argsort(results["rank_test_score"])[:8]
    grid_top = [
        {
            "params": _clean_params(results["params"][i]),
            "cv_score": round(float(results["mean_test_score"][i]), 4),
            "cv_std": round(float(results["std_test_score"][i]), 4),
        }
        for i in order
    ]

    return {
        "algorithm": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "majority_baseline": round(max(float(np.mean(y_test)), 1 - float(np.mean(y_test))), 4),
        "roc_auc": auc,
        "probability_supported": proba is not None,
        "cv_score": round(float(search.best_score_), 4),
        "best_params": _clean_params(search.best_params_),
        "grid_top": grid_top,
        "confusion_matrix": cm.tolist(),
    }, best


def meta_label(prefix):
    return {"conversion": "lead conversion"}.get(prefix, prefix)
