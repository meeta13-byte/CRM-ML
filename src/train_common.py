"""The four syllabus algorithms, one shared training loop.

KNN, Gaussian Naive Bayes, Linear SVM, RBF SVM -- nothing else. Probabilities
are fitted only for the final chosen model (SVC with probability=True runs an
internal 5-fold Platt scaling, which is slow inside a grid search).
"""
from datetime import datetime

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import MODELS, RANDOM_STATE
from .data.preprocess import build_pipeline, column_types, split_data
from .data.schema import assert_no_leakage, feature_specs
from .evaluate import save_json, tune


def model_zoo():
    """Fresh estimator + grid per algorithm. make(probability=True) for the final fit."""
    def knn(probability=False):
        return KNeighborsClassifier()

    def gnb(probability=False):
        return GaussianNB()

    def linear(probability=False):
        return SVC(kernel="linear", probability=probability, random_state=RANDOM_STATE)

    def rbf(probability=False):
        return SVC(kernel="rbf", probability=probability, random_state=RANDOM_STATE)

    return {
        "KNN": (knn, {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        }),
        "GaussianNB": (gnb, {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}),
        "LinearSVM": (linear, {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        }),
        "RBFSVM": (rbf, {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01],
            "model__class_weight": [None, "balanced"],
        }),
    }


def fmt_pct(value):
    return f"{value:.1%}"


def run(stage, X, y, info, forbidden, out_prefix, max_rows=None, test_size=None):
    assert_no_leakage(X.columns, forbidden, stage)
    if y.nunique() < 2:
        raise ValueError(f"{stage}: target has a single class, cannot train")
    if not 0.05 <= y.mean() <= 0.95:
        print(f"note: {stage} is imbalanced ({fmt_pct(y.mean())} positive). "
              "Accuracy alone is misleading; positive-class F1 drives selection and "
              "positive-class precision/recall are reported.")

    numeric, categorical = column_types(X)
    X_train, X_test, y_train, y_test = split_data(X, y, max_rows, test_size)

    results, fitted = [], {}
    for name, (make, grid) in model_zoo().items():
        tune_pipe = build_pipeline(numeric, categorical, make(probability=False))
        final_pipe = build_pipeline(numeric, categorical, make(probability=True))
        metrics, model = tune(
            name, tune_pipe, grid, X_train, y_train, X_test, y_test,
            out_prefix, final_pipeline=final_pipe,
        )
        results.append(metrics)
        fitted[name] = model
        print(f"  {name:11s} acc={metrics['accuracy']:.3f} "
              f"macro_f1={metrics['macro_f1']:.3f} f1={metrics['f1']:.3f} "
              f"recall={metrics['recall']:.3f} auc={metrics['roc_auc']}")

    # Selection rule fixed in advance: positive-class F1, recall breaks ties.
    best = max(results, key=lambda r: (r["f1"], r["recall"]))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, MODELS / f"{out_prefix}.joblib")

    payload = {
        "stage": stage,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": info,
        "target": "y = subscribed a term deposit",
        "features": feature_specs(X),
        "features_excluded": info.get("excluded", []),
        "test_size": float(test_size if test_size else 0.2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "best_model": best["algorithm"],
        "selection_rule": "highest test positive-class F1; recall breaks ties",
        "majority_baseline": best["majority_baseline"],
        "models": results,
    }
    save_json(payload, MODELS / f"{out_prefix}.json")

    print(f"\n{stage}: {len(X)} rows | best = {best['algorithm']} "
          f"(F1 {best['f1']:.3f}, macro-F1 {best['macro_f1']:.3f}, "
          f"recall {best['recall']:.3f})")
    print(f"majority-class baseline accuracy: {best['majority_baseline']:.3f}")
    print(pd.DataFrame(results)[
        ["algorithm", "accuracy", "precision", "recall", "f1",
         "macro_f1", "roc_auc", "cv_score"]
    ].to_string(index=False))
    return payload
