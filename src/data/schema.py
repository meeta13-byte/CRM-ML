"""Column contracts and the leakage guard.

The conversion model is scored before the call, so any column that only exists
after the call must never reach the feature matrix. Training calls
assert_no_leakage() and fails loudly instead of quietly inflating accuracy.
"""
import pandas as pd

# `duration` is the call length; it is known only after the call and separates
# subscribers from non-subscribers extremely well (mean 537s vs 221s).
BANK_FORBIDDEN = {"y", "duration", "target", "converted"}


def assert_no_leakage(features, forbidden, stage):
    bad = sorted({str(f).strip().lower() for f in features} & forbidden)
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
