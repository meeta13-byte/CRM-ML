"""Dataset adapters: raw files -> one canonical frame per module.

Two real datasets, two jobs:

- UCI Bank Marketing (45,211 rows) -> the four-algorithm conversion comparison
  and the customer segmentation.
- Kaggle Facebook ads (1,143 rows)  -> advertising performance and campaign
  planning (descriptive only; its supervised model was abandoned, see README).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BANK_CSV, CAMPAIGN_CSV, DATA_RAW, MIN_SPEND

# ---------------------------------------------------------------- bank data ---

BANK_TARGET = "y"
# Only known after the call: enrolling it would leak the answer.
BANK_LEAKY = "duration"
BANK_NUMERIC = [
    "age", "balance", "day", "campaign", "pdays", "previous",
    "previously_contacted",
]
BANK_CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]
BANK_FEATURES = BANK_NUMERIC + BANK_CATEGORICAL


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
        "See README > Data."
    )


def load_bank(path=None):
    """UCI Bank Marketing (bank-full.csv): does this customer subscribe?

    `duration` is dropped because it is only known after the call. `pdays` uses
    -1 as "never contacted", which is not a distance, so it is split into a
    flag and a nullable recency value.
    """
    path = Path(path) if path else resolve_csv(BANK_CSV, {"y", "job", "pdays"})
    raw = pd.read_csv(path, sep=";")
    raw.columns = [c.strip() for c in raw.columns]

    if BANK_TARGET not in raw.columns:
        raise ValueError(f"{path} has no '{BANK_TARGET}' column")
    if BANK_LEAKY not in raw.columns:
        raise ValueError(f"{path} has no '{BANK_LEAKY}' column to drop")

    duplicates = int(raw.duplicated().sum())
    y = raw.pop(BANK_TARGET).astype(str).str.strip().str.lower().eq("yes").astype(int)
    raw = raw.drop(columns=[BANK_LEAKY])

    for col in BANK_CATEGORICAL:
        raw[col] = raw[col].astype(str).str.strip()
    raw["previously_contacted"] = (raw["pdays"] != -1).astype(int)
    raw["pdays"] = raw["pdays"].where(raw["pdays"] != -1)  # NaN -> median imputer

    X = raw[BANK_FEATURES].copy()
    info = {
        "name": "UCI Bank Marketing (bank-full.csv)",
        "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "file": str(path),
        "rows": int(len(X)),
        "columns_available": 17,
        "features_used": len(BANK_FEATURES),
        "target": "y (yes = subscribed a term deposit)",
        "positive_rate": round(float(y.mean()), 4),
        "excluded": ["duration (known only after the call -> leakage)", "y"],
        "duplicate_rows": duplicates,
        "derived": "previously_contacted = pdays != -1; pdays -1 -> missing",
        "missing_note": "the file has no blanks; 'unknown' is kept as a real "
                        "category, and the pdays sentinel is the only NaN",
    }
    return X, y, info


# ------------------------------------------------------------ campaign data ---

CAMPAIGN_PLAN_FIELDS = ["age", "gender", "interest", "campaign"]


def load_campaign_ads(path=None):
    """Real anonymised Facebook ads (Kaggle 'clicks-conversion-tracking').

    Returns the ad-level frame plus the efficiency metrics that can be computed
    from it. Nothing here is a model input; this dataset feeds the descriptive
    campaign analysis.
    """
    path = Path(path) if path else resolve_csv(
        CAMPAIGN_CSV, {"approved_conversion", "spent", "interest"}
    )
    raw = pd.read_csv(path)
    df = raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    df = df.dropna(subset=["spent"]).copy()
    df = df[df["spent"] >= MIN_SPEND].copy()

    cid = df["xyz_campaign_id"].astype(str)
    campaign_map = {"916": "1", "936": "2", "1178": "3"}
    df["campaign"] = cid.map(campaign_map).fillna(cid)
    df["interest"] = df["interest"].astype(str)
    approved = pd.to_numeric(df["approved_conversion"], errors="coerce").fillna(0.0)
    clicks = pd.to_numeric(df["clicks"], errors="coerce").fillna(0.0)
    impressions = pd.to_numeric(df["impressions"], errors="coerce").fillna(0.0)
    spend = pd.to_numeric(df["spent"], errors="coerce").fillna(0.0)

    df["conversions"] = approved
    df["cpa"] = np.where(approved > 0, spend / approved, np.inf)
    df["ctr"] = np.where(impressions > 0, clicks / impressions, np.nan)
    df["cpc"] = np.where(clicks > 0, spend / clicks, np.nan)
    df["cpm"] = np.where(impressions > 0, spend / impressions * 1000, np.nan)
    df["conversion_rate"] = np.where(clicks > 0, approved / clicks, np.nan)
    df["conversions_per_dollar"] = np.where(spend > 0, approved / spend, np.nan)

    info = {
        "name": "Kaggle clicks-conversion-tracking (anonymised Facebook ads)",
        "source": "https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking",
        "file": str(path),
        "rows_raw": int(len(raw)),
        "rows_used": int(len(df)),
        "min_spend": MIN_SPEND,
        "columns": 11,
        "note": "one row is one ad, not a campaign; no budget, duration, "
                "location, creative, CTA or revenue column exists",
    }
    return df, info


def bank_benchmark_cpa(path=None):
    """Median CPA of converting campaign ads: the default planning target."""
    df, _ = load_campaign_ads(path)
    finite = df.loc[np.isfinite(df["cpa"]), "cpa"]
    return float(finite.median()) if len(finite) else 0.0
