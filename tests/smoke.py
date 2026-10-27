"""Fast self-check with a tiny synthetic bank-shaped fixture.

    python3 tests/smoke.py

The fixture exists only to test the code path; the app always trains on the
real UCI file. Covers the loader contract, the leakage guard, preprocessing,
one nearest-neighbour fit, the campaign benchmark and the SQLite history.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.campaign import forecast  # noqa: E402
from src.data.loaders import load_bank  # noqa: E402
from src.data.preprocess import build_pipeline, column_types, split_data  # noqa: E402
from src.data.schema import BANK_FORBIDDEN, assert_no_leakage  # noqa: E402
from src.history import (  # noqa: E402
    get_campaign, get_prediction, list_campaigns, list_predictions,
    save_campaign, save_prediction, set_actual,
)
from src.outliers import iqr_table  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402


def bank_fixture(path):
    """45 rows in the official bank-full.csv layout (semicolon separated)."""
    rng = np.random.default_rng(7)
    n = 45
    pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "job": rng.choice(["admin.", "blue-collar", "management"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": "no",
        "balance": rng.integers(-500, 50000, n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": "no",
        "contact": rng.choice(["cellular", "unknown"], n),
        "day": rng.integers(1, 32, n),
        "month": rng.choice(["may", "jun", "nov"], n),
        "duration": rng.integers(10, 900, n),
        "campaign": rng.integers(1, 6, n),
        "pdays": rng.choice([-1, 30, 200], n),
        "previous": rng.integers(0, 4, n),
        "poutcome": rng.choice(["unknown", "failure", "success"], n),
        "y": ["yes", "no"] * 22 + ["no"],
    }).to_csv(path, index=False, sep=";")


def campaign_fixture():
    """Small ad-level frame with the derived columns forecast() expects."""
    n = 60
    df = pd.DataFrame({
        "age": np.tile(["30-34", "35-39"], 30),
        "gender": np.tile(["M", "F"], 30),
        "interest": np.tile(["15", "16", "18"], 20),
        "campaign": np.tile(["916", "936"], 30),
        "conversions": np.tile([2.0, 0.0, 1.0], 20),
        "conversion_rate": np.tile([0.10, 0.0, 0.05], 20),
        "cpc": 1.5,
        "ctr": 0.02,
        "cpm": 0.3,
        "spent": 100.0,
        "impressions": 10000.0,
        "clicks": 200.0,
    })
    df["cpa"] = np.where(df["conversions"] > 0, 30.0 / df["conversions"], np.inf)
    df["conversions_per_dollar"] = df["conversions"] / 100.0
    return df


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        csv = tmp / "bank-small.csv"
        bank_fixture(csv)

        X, y, info = load_bank(csv)
        assert "duration" not in X.columns, "duration must be dropped"
        assert "y" not in X.columns, "target must not be a feature"
        assert info["positive_rate"] > 0
        assert "previously_contacted" in X.columns
        assert X.loc[X["previously_contacted"] == 0, "pdays"].isna().all()

        # The guard must reject post-outcome columns if a caller adds them back.
        for leaky in ("duration", "y"):
            try:
                assert_no_leakage(list(X.columns) + [leaky], BANK_FORBIDDEN, "test")
            except ValueError:
                pass
            else:
                raise AssertionError(f"leakage guard did not fire for {leaky}")

        numeric, categorical = column_types(X)
        model = build_pipeline(numeric, categorical, KNeighborsClassifier(n_neighbors=3))
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        assert model.predict(X_test).shape == (len(X_test),)
        assert model.predict_proba(X_test).shape == (len(X_test), 2)

        df = campaign_fixture()
        lead = forecast(df, "30-34", "M", "15", "916", 1000, "Lead Generation",
                        leads_per_dollar=0.1)
        assert abs(lead["target_cpa"] - 10.0) < 1e-9, "CPA = 1 / leads per $"
        assert lead["estimated_leads"] is not None
        assert lead["neighbours"] == min(25, len(df))
        traffic = forecast(df, "30-34", "M", "15", "916", 1000, "Traffic")
        assert traffic["estimated_clicks"] is not None
        awareness = forecast(df, "30-34", "M", "15", "916", 1000, "Awareness")
        assert awareness["estimated_impressions"] is not None
        assert "reach" in awareness["reach_note"].lower()

        db = tmp / "history.db"
        pid = save_prediction("fixture", "KNN", {"age": 40}, 1, probability=0.4,
                              priority="Medium", path=db)
        assert list_predictions(path=db).iloc[0]["model"] == "KNN"
        assert get_prediction(pid, path=db)["features"] == {"age": 40}
        set_actual(pid, 1, path=db)
        assert int(get_prediction(pid, path=db)["actual"]) == 1

        rid = save_campaign({"objective": "Awareness"}, {"estimated_cpm": 0.3},
                            "Awareness", lead["method"], path=db)
        assert list_campaigns(path=db).iloc[0]["objective"] == "Awareness"
        assert get_campaign(rid, path=db)["result"]["estimated_cpm"] == 0.3

        # IQR flags: known quartiles for this fixture, and no rows are dropped.
        frame = pd.DataFrame({"x": [10, 12, 14, 16, 18, 20, 22, 100],
                              "label": ["keep"] * 8})
        table = iqr_table(frame, ["x"], {"x": "X"})
        row = table.iloc[0]
        assert row.feature == "X" and row["count"] == 8
        assert abs(row.q1 - 13.5) < 1e-9 and abs(row.q3 - 20.5) < 1e-9
        assert abs(row.lower - 3.0) < 1e-9 and abs(row.upper - 31.0) < 1e-9
        assert row.flagged == 1 and abs(row.share - 1 / 8) < 1e-9
        assert len(frame) == 8, "outlier analysis must not drop rows"
        assert frame.loc[7, "x"] == 100, "outlier analysis must not alter values"

    print("smoke ok: bank loader, leakage guard, preprocessing, KNN, campaign "
          "benchmark, SQLite history")


if __name__ == "__main__":
    main()
