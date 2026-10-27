"""IQR outlier flags for numeric columns.

Analysis only: this module never mutates, filters or drops rows. A flag means
"outside 1.5 x IQR", which is one convention, not a verdict on data quality.
"""
import numpy as np
import pandas as pd

IQR_MULTIPLIER = 1.5


def iqr_table(df, columns, labels=None):
    """One row per column with original-unit quartiles, fences and flags."""
    labels = labels or {}
    rows = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = (float(v) for v in values.quantile([0.25, 0.75]))
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        flagged = int(((values < lower) | (values > upper)).sum())
        rows.append({
            "feature": labels.get(col, col),
            "count": int(len(values)),
            "missing": int(len(df) - len(values)),
            "minimum": float(values.min()),
            "q1": q1,
            "median": float(values.median()),
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "max": float(values.max()),
            "flagged": flagged,
            "share": flagged / len(values),
        })
    return pd.DataFrame(rows)


def series_for(df, columns, labels=None):
    """Numeric series keyed by display label, missing values dropped."""
    labels = labels or {}
    out = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.empty:
            out[labels.get(col, col)] = values.to_numpy(dtype=float)
    return out


def describe_flags(table):
    """Plain-English summary of which features contain flagged observations."""
    if table.empty:
        return "No numeric columns were available for the IQR check."
    flagged = table[table["flagged"] > 0].sort_values("share", ascending=False)
    if flagged.empty:
        return ("No observations fall outside the 1.5 x IQR fences in this "
                "selection, so the IQR rule flags nothing here.")
    parts = [f"**{row.feature}** {row.flagged:,} ({row.share:.1%})"
             for row in flagged.itertuples()]
    return ("Outside the 1.5 x IQR fences: " + ", ".join(parts) +
            ". These are flagged for review only — genuine high-performing "
            "ads and unusual customers stay in the data and in every model.")
