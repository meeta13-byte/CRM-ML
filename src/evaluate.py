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
    path.write_text(json.dumps(payload, indent=2))


def _save_confusion(cm, name, path_prefix):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(name)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{path_prefix}_{name.replace(' ', '_')}.png", dpi=140)
    plt.close(fig)


def tune(name, pipeline, grid, X_train, y_train, X_test, y_test, path_prefix):
    """GridSearchCV on the training split only, then one honest test score."""
    search = GridSearchCV(
        pipeline, grid, scoring=SELECTION_METRIC, cv=CV_FOLDS, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    _save_confusion(cm, name, path_prefix)

    try:
        auc = round(float(roc_auc_score(y_test, proba)), 4)
    except ValueError:
        auc = None  # single-class test fold

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
        "cv_score": round(float(search.best_score_), 4),
        "best_params": {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        },
        "confusion_matrix": cm.tolist(),
    }, best
