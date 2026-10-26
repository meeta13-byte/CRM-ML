"""Dataset adapters: raw files -> one canonical frame per stage.

Two public sources are supported for Stage 1 (your own Meta account, or the
Kaggle Facebook-ads export). Both produce the same columns, so the training
code never knows which one it is.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DATA_RAW,
    MIN_SPEND,
    STAGE1_KAGGLE,
    STAGE1_META,
    STAGE2_XEDUCATION,
    TARGET_CPA_OVERRIDE,
    TARGET_CPA_QUANTILE,
)

STAGE1_FEATURES = ["age", "gender", "interest", "campaign"]

META_NUMERIC = [
    "budget", "duration_days", "age_min", "age_max", "gender_count",
    "country_count", "interest_count", "custom_audience_count",
    "start_month", "start_weekday",
]
META_CATEGORICAL = [
    "objective", "optimization_goal", "buying_type", "budget_type",
    "publisher_platforms", "device_platforms",
]
META_FEATURES = META_NUMERIC + META_CATEGORICAL


def resolve_csv(preferred: Path, required_cols) -> Path:
    """Use the configured path, else find any CSV in data/raw with the columns."""
    if preferred.exists():
        return preferred
    required = {c.lower() for c in required_cols}
    for path in sorted(DATA_RAW.rglob("*.csv")):
        try:
            cols = {c.strip().lower() for c in pd.read_csv(path, nrows=0).columns}
        except Exception:
            continue
        if required <= cols:
            return path
    raise FileNotFoundError(
        f"Could not find a CSV with columns {sorted(required)} under {DATA_RAW}. "
        "See README > Data (manual step)."
    )


def _add_stage1_label(df, conversions_col="approved_conversion"):
    """label = 1 when cost per acquisition is above the cohort target.

    Zero conversions means infinite CPA, i.e. a clear miss, so those rows are
    labelled 1 rather than dropped (dropping them would bias the model).
    """
    df = df.copy()
    spend = pd.to_numeric(df["spent"], errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(df[conversions_col], errors="coerce").fillna(0.0)
    df = df[spend >= MIN_SPEND]
    spend = spend.loc[df.index]
    conversions = conversions.loc[df.index]

    df["cpa"] = np.where(conversions > 0, spend / conversions, np.inf)
    finite = np.isfinite(df["cpa"])

    if TARGET_CPA_OVERRIDE is not None:
        target = float(TARGET_CPA_OVERRIDE)
    elif finite.any():
        target = float(np.quantile(df.loc[finite, "cpa"], TARGET_CPA_QUANTILE))
    else:
        target = float("inf")  # degenerate data; every row becomes label 0

    df["high_cpa"] = (df["cpa"] > target).astype(int)
    return df, target


def load_stage1_kaggle(path=None):
    """Kaggle 'Sales Conversion Optimization' (real anonymised Facebook ads)."""
    path = Path(path) if path else resolve_csv(
        STAGE1_KAGGLE, {"approved_conversion", "spent", "interest"}
    )
    raw = pd.read_csv(path)
    df = raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    df = df.dropna(subset=["spent"])
    df, target = _add_stage1_label(df)

    # Planning-time attributes only: audience plan + which campaign it belongs to.
    df["campaign"] = df["xyz_campaign_id"].astype(str)
    df["interest"] = df["interest"].astype(str)
    df["age"] = df["age"].astype(str)
    df["gender"] = df["gender"].astype(str)

    X = df[STAGE1_FEATURES].copy()
    y = df["high_cpa"].astype(int)

    info = {
        "source": "kaggle: loveall/clicks-conversion-tracking",
        "file": str(path),
        "raw_rows": int(len(raw)),
        "rows_after_min_spend": int(len(df)),
        "min_spend": MIN_SPEND,
        "target_cpa": None if not np.isfinite(target) else round(target, 2),
        "positive_rate": round(float(y.mean()), 4),
        "feature_note": "planning fields only; spend/clicks/impressions excluded",
    }
    return X, y, info


def load_stage1_meta(path=None):
    """CSV produced by scripts/sync_meta.py from your own ad account."""
    path = Path(path) if path else resolve_csv(
        STAGE1_META, {"spend", "conversions", "objective"}
    )
    df = pd.read_csv(path)
    df, target = _add_stage1_label(df, conversions_col="conversions")

    for col in META_CATEGORICAL:
        df[col] = df[col].astype(str)
    for col in META_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[META_FEATURES].copy()
    y = df["high_cpa"].astype(int)

    info = {
        "source": "meta: own ad account (Marketing API)",
        "file": str(path),
        "rows_after_min_spend": int(len(df)),
        "min_spend": MIN_SPEND,
        "target_cpa": None if not np.isfinite(target) else round(target, 2),
        "positive_rate": round(float(y.mean()), 4),
        "feature_note": "ad-set configuration at query time; spend/results excluded",
    }
    return X, y, info


def load_stage2(path=None, include_snapshot_fields=False):
    """X Education lead scoring data (real, anonymised education company).

    include_snapshot_fields=True keeps Last Activity and the Asymmetrique
    scores, which are captured at snapshot time. The headline model drops them
    so the prediction only uses what exists at lead capture.
    """
    path = Path(path) if path else resolve_csv(
        STAGE2_XEDUCATION, {"converted", "lead source"}
    )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Converted" not in df.columns:
        raise ValueError(f"{path} has no 'Converted' column")

    duplicates = 0
    if "Prospect ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["Prospect ID"]).reset_index(drop=True)
        duplicates = before - len(df)

    # Public copies ship the target as 0/1, some as Yes/No.
    y = (
        df.pop("Converted").astype(str).str.strip().str.lower()
        .isin({"1", "1.0", "yes", "true"})
    ).astype(int)

    dropped = [
        "Prospect ID", "Lead Number", "Tags", "Lead Quality", "Lead Profile",
        "Last Notable Activity",
    ]
    if not include_snapshot_fields:
        dropped += [c for c in df.columns if c.startswith("Asymmetrique") or c == "Last Activity"]
    dropped_present = [c for c in dropped if c in df.columns]
    df = df.drop(columns=dropped_present)

    # 'Select' is the form's empty placeholder, not a real category.
    df = df.replace({"Select": np.nan, "select": np.nan})
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df = df.dropna(axis=1, how="all").reset_index(drop=True)
    y = y.reset_index(drop=True)

    info = {
        "source": "kaggle: amritachatterjee09/lead-scoring-dataset (X Education)",
        "file": str(path),
        "rows": int(len(df)),
        "duplicates_removed": duplicates,
        "positive_rate": round(float(y.mean()), 4),
        "dropped_columns": dropped_present,
        "include_snapshot_fields": include_snapshot_fields,
    }
    return df, y, info
