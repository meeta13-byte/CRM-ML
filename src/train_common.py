"""The four syllabus algorithms, one shared training loop."""
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
    """Fresh estimator + grid per algorithm; no model is reused across runs."""
    return {
        "KNN": (
            lambda: KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "GaussianNB": (
            lambda: GaussianNB(),
            {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
        "LinearSVM": (
            lambda: SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "RBFSVM": (
            lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.01],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


def run(stage, X, y, info, forbidden, out_prefix, max_rows=None):
    assert_no_leakage(X.columns, forbidden, stage)
    if y.nunique() < 2:
        raise ValueError(f"{stage}: target has a single class, cannot train")
    if not 0.2 <= y.mean() <= 0.8:
        print(f"warning: {stage} is imbalanced ({y.mean():.1%} positive). "
              "Accuracy will look flattering; trust F1/recall and the "
              "confusion matrix. Adjust the label threshold in src/config.py.")

    numeric, categorical = column_types(X)
    X_train, X_test, y_train, y_test = split_data(X, y, max_rows)

    results, fitted = [], {}
    for name, (make, grid) in model_zoo().items():
        metrics, model = tune(
            name,
            build_pipeline(numeric, categorical, make()),
            grid,
            X_train, y_train, X_test, y_test,
            out_prefix,
        )
        results.append(metrics)
        fitted[name] = model
        print(f"  {name:11s} acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} macro_f1={metrics['macro_f1']:.3f} "
              f"recall={metrics['recall']:.3f}")

    # Selection rule is fixed in advance: macro-F1, recall breaks ties.
    best = max(results, key=lambda r: (r["macro_f1"], r["recall"]))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best["algorithm"]], MODELS / f"{out_prefix}.joblib")

    payload = {
        "stage": stage,
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": info,
        "target": "high_cpa (cost per acquisition above target)" if out_prefix == "stage1"
                  else "converted",
        "features": feature_specs(X),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "best_model": best["algorithm"],
        "selection_rule": "highest test macro-F1; recall breaks ties",
        "majority_baseline": best["majority_baseline"],
        "models": results,
    }
    save_json(payload, MODELS / f"{out_prefix}.json")

    print(f"\n{stage}: {len(X)} rows | best = {best['algorithm']} "
          f"(macro-F1 {best['macro_f1']:.3f}, F1 {best['f1']:.3f}, "
          f"recall {best['recall']:.3f})")
    print(f"majority-class baseline accuracy: {best['majority_baseline']:.3f}")
    print(pd.DataFrame(results)[
        ["algorithm", "accuracy", "precision", "recall", "f1",
         "macro_f1", "roc_auc", "cv_score"]
    ].to_string(index=False))
    return payload
