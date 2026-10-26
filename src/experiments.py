"""Two experiments the report needs: the KNN K sweep and the test-size comparison.

    python3 -m src.experiments

Both use the same 12k stratified bank sample as training (same seed), so the
numbers are comparable with models/conversion.json. Hyperparameters are held
fixed at the values GridSearchCV chose for the 0.2 split, which makes the split
size the only thing that changes in the second experiment.

Outputs: models/experiments.json + reports/figures/k_sweep.png,
reports/figures/test_size_f1.png. The app reads the JSON, never retrains.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .config import (
    CV_FOLDS, FIGURES, MAX_BANK_ROWS, MODELS, RANDOM_STATE, SELECTION_METRIC,
    TEST_SIZES,
)
from .data.loaders import load_bank
from .data.preprocess import build_pipeline, column_types, split_data
from .data.schema import BANK_FORBIDDEN, assert_no_leakage
from .evaluate import save_json
from .train_common import model_zoo

K_RANGE = range(1, 16)


def sample(X, y, max_rows=MAX_BANK_ROWS):
    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE
        )
    return X, y


def k_sweep(X, y, numeric, categorical):
    """5-fold CV for k = 1..15, uniform weights, euclidean distance."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k in K_RANGE:
        pipe = build_pipeline(
            numeric, categorical, KNeighborsClassifier(n_neighbors=k)
        )
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=["accuracy", "f1"], n_jobs=-1
        )
        rows.append({
            "k": int(k),
            "cv_accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "cv_f1": round(float(scores["test_f1"].mean()), 4),
        })
        print(f"  k={k:2d}  accuracy={rows[-1]['cv_accuracy']:.3f}  "
              f"F1={rows[-1]['cv_f1']:.3f}")
    return rows


def test_sizes(X, y, numeric, categorical, best_params, sample_size):
    """One fit per (model, split). Same hyperparameters, different hold-out size."""
    rows = []
    for size in TEST_SIZES:
        X_train, X_test, y_train, y_test = split_data(
            X, y, sample_size, test_size=size
        )
        for name, (make, _grid) in model_zoo().items():
            pipe = build_pipeline(numeric, categorical, make(probability=False))
            pipe.set_params(**{f"model__{k}": v
                               for k, v in best_params[name].items()})
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score, recall_score
            rows.append({
                "test_size": size,
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            })
            print(f"  split {size:.1f}  {name:11s} acc={rows[-1]['accuracy']:.3f} "
                  f"F1={rows[-1]['f1']:.3f} recall={rows[-1]['recall']:.3f}")
    return rows


def figures(k_rows, tests):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ks = [r["k"] for r in k_rows]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, [r["cv_accuracy"] for r in k_rows], marker="o", label="Accuracy")
    ax.plot(ks, [r["cv_f1"] for r in k_rows], marker="s", label="Positive-class F1")
    best = max(k_rows, key=lambda r: (r["cv_f1"], -r["k"]))
    ax.axvline(best["k"], color="grey", ls="--", lw=1)
    ax.annotate(f"best F1 at k={best['k']}", (best["k"], best["cv_f1"]),
                textcoords="offset points", xytext=(8, 10), fontsize=8)
    ax.set(xlabel="n_neighbors (k)", ylabel="5-fold CV score",
           title="KNN K sweep (uniform weights, euclidean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "k_sweep.png", dpi=140)
    plt.close(fig)

    df = pd.DataFrame(tests)
    models = df["model"].unique().tolist()
    x = np.arange(len(TEST_SIZES))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, name in enumerate(models):
        vals = df[df["model"] == name].set_index("test_size").loc[list(TEST_SIZES), "f1"]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(models) - 1) / 2,
                  [f"{s:.1f} test" for s in TEST_SIZES])
    ax.set(ylabel="Test positive-class F1", title="Test-size comparison (fixed hyperparameters)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_size_f1.png", dpi=140)
    plt.close(fig)


def main():
    X, y, info = load_bank()
    assert_no_leakage(X.columns, BANK_FORBIDDEN, "experiments")
    X, y = sample(X, y)
    numeric, categorical = column_types(X)
    print(f"experiments | {len(X)} rows | 5-fold CV | positive rate {y.mean():.1%}")

    meta = MODELS / "conversion.json"
    if not meta.exists():
        raise FileNotFoundError(
            "models/conversion.json missing - run `python3 -m src.train_conversion` first"
        )
    import json
    trained = json.loads(meta.read_text())
    best_params = {m["algorithm"]: m["best_params"] for m in trained["models"]}

    print("K sweep")
    k_rows = k_sweep(X, y, numeric, categorical)
    print("Test-size comparison")
    tests = test_sizes(X, y, numeric, categorical, best_params, MAX_BANK_ROWS)

    figures(k_rows, tests)
    best_k = max(k_rows, key=lambda r: (r["cv_f1"], -r["k"]))
    payload = {
        "dataset": info,
        "rows": int(len(X)),
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "selection_metric": SELECTION_METRIC,
        "k_sweep": k_rows,
        "k_sweep_best": best_k,
        "k_sweep_note": "Uniform weights and euclidean distance fixed; only k varies. "
                        "The main grid search also tunes weights and metric.",
        "test_sizes": tests,
        "test_size_note": "Hyperparameters are fixed at the values chosen on the 0.2 "
                          "split, so the split size is the only variable. A fixed "
                          "seed is used, but the splits are not nested, so small "
                          "differences are noise, not evidence.",
        "figures": ["reports/figures/k_sweep.png", "reports/figures/test_size_f1.png"],
    }
    save_json(payload, MODELS / "experiments.json")
    print(f"\nBest CV F1 in the sweep: k={best_k['k']} ({best_k['cv_f1']:.3f})")
    print("Test-size F1 range per model:")
    print(pd.DataFrame(tests).pivot(index="model", columns="test_size",
                                    values="f1").to_string())


if __name__ == "__main__":
    main()
