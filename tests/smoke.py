"""Fast self-check with tiny fixtures (not the real datasets).

    python tests/smoke.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_stage1_kaggle, load_stage2
from src.data.preprocess import build_pipeline, column_types, split_data
from src.data.schema import STAGE1_FORBIDDEN, STAGE2_FORBIDDEN, assert_no_leakage


def _stage1_fixture(path):
    rng = np.random.default_rng(0)
    n = 90
    pd.DataFrame({
        "ad_id": np.repeat(np.arange(9), 10),
        "xyz_campaign_id": np.tile([1, 2, 3], 30),
        "fb_campaign_id": np.repeat(np.arange(18), 5),
        "age": np.tile(["30-34", "35-39", "40-44"], 30),
        "gender": np.tile(["M", "F"], 45),
        "interest": np.tile([15, 16, 18], 30),
        "Impressions": rng.integers(1000, 90000, n),
        "Clicks": rng.integers(0, 200, n),
        "Spent": rng.uniform(20, 400, n).round(2),
        "Total_Conversion": rng.integers(0, 10, n),
        "Approved_Conversion": rng.integers(0, 6, n),
    }).to_csv(path, index=False)


def _stage2_fixture(path):
    rng = np.random.default_rng(1)
    n = 90
    pd.DataFrame({
        "Converted": ["Yes", "No"] * 45,
        "Lead Origin": ["Landing Page Submission", "API"] * 45,
        "Lead Source": ["Google", "Direct"] * 45,
        "TotalVisits": rng.integers(0, 20, n),
        "Total Time Spent on Website": rng.integers(0, 2000, n),
        "Page Views Per Visit": rng.uniform(0, 4, n).round(2),
        "Occupation": ["Working Professional", "Student"] * 45,
        "Tags": ["converted", "lost"] * 45,
        "Lead Quality": ["High", "Low"] * 45,
        "Last Notable Activity": ["Email Sent", "SMS Sent"] * 45,
        "Asymmetrique Activity Index": ["02.Medium", "01.Low"] * 45,
        "Last Activity": ["SMS Sent", "Email Opened"] * 45,
    }).to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s1, s2 = tmp / "stage1.csv", tmp / "stage2.csv"
        _stage1_fixture(s1)
        _stage2_fixture(s2)

        X1, y1, info1 = load_stage1_kaggle(s1)
        assert set(y1.unique()) == {0, 1}, "stage 1 needs both classes"
        assert info1["target_cpa"] > 0
        assert not set(X1.columns) & {c.lower() for c in STAGE1_FORBIDDEN}
        numeric, categorical = column_types(X1)
        model = build_pipeline(numeric, categorical, KNeighborsClassifier(n_neighbors=3))
        Xtr, Xte, ytr, yte = split_data(X1, y1)
        model.fit(Xtr, ytr)
        assert model.predict(Xte).shape == (len(Xte),)
        assert model.predict_proba(Xte).shape == (len(Xte), 2)

        X2, y2, info2 = load_stage2(s2)
        for leaky in ["Tags", "Lead Quality", "Last Notable Activity",
                      "Asymmetrique Activity Index", "Last Activity"]:
            assert leaky not in X2.columns, f"{leaky} should be dropped"
        assert "Select" not in X2.astype(str).values
        numeric, categorical = column_types(X2)
        model = build_pipeline(numeric, categorical, GaussianNB())
        Xtr, Xte, ytr, yte = split_data(X2, y2)
        model.fit(Xtr, ytr)
        assert model.predict_proba(Xte).shape == (len(Xte), 2)

        try:
            assert_no_leakage(["spent", "age"], STAGE1_FORBIDDEN, "test")
        except ValueError:
            pass
        else:
            raise AssertionError("leakage guard did not fire")

        assert "converted" in STAGE2_FORBIDDEN

        # The real Kaggle copy ships the target as 0/1, the docs as Yes/No.
        raw = pd.read_csv(s2)
        raw["Converted"] = raw["Converted"].map({"Yes": 1, "No": 0})
        raw.to_csv(s2, index=False)
        _, y2_numeric, _ = load_stage2(s2)
        assert y2_numeric.sum() == y2.sum(), "0/1 target parsed differently to Yes/No"

    print("smoke ok: adapters, leakage guard, preprocessing, KNN, GaussianNB")


if __name__ == "__main__":
    main()
