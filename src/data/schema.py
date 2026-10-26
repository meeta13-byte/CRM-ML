"""Column contracts and the leakage guard.

Every model is scored at a fixed moment in time. The forbidden sets below are
columns that only exist after that moment, so training calls
assert_no_leakage() and fails loudly instead of quietly inflating accuracy.
"""
import pandas as pd

# Known only after the campaign has run.
STAGE1_FORBIDDEN = {
    "ad_id", "impressions", "clicks", "spent", "total_conversion",
    "approved_conversion", "cpa", "ctr", "cpc", "cpm", "reach",
    "frequency", "actions", "results", "cost_per_result",
    "effective_status", "updated_time", "conversions",
}

# Meta account sync: same idea, plus fields edited after launch.
META_FORBIDDEN = STAGE1_FORBIDDEN | {"name", "adset_id", "campaign_id"}

# Written by the sales team after the lead was contacted.
STAGE2_FORBIDDEN = {
    "prospect id", "lead number", "tags", "lead quality",
    "last notable activity", "converted",
}


def assert_no_leakage(features, forbidden, stage):
    bad = sorted({f.strip().lower() for f in features} & forbidden)
    if bad:
        raise ValueError(
            f"{stage}: post-outcome columns reached the feature matrix: {bad}. "
            "Remove them from the adapter or from the feature list."
        )


def feature_specs(df: pd.DataFrame):
    """Describe the raw feature columns so the app can build its inputs."""
    specs = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            specs.append({
                "name": col,
                "kind": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
            })
        else:
            cats = sorted(s.dropna().astype(str).unique().tolist())
            specs.append({"name": col, "kind": "categorical", "categories": cats})
    return specs
