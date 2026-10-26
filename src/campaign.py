"""Campaign intelligence: descriptive analytics + an honest planning benchmark.

There is no supervised model here on purpose. Measured on this dataset,
conversions per dollar are unlearnable from the four planning fields
(CV R^2 -0.03 KNN / -0.14 SVR), so a "predicted CPA" from a regressor would be
decoration. Instead the planner looks up the most similar historical ads and
reports what actually happened to them, with the sample size stated.

See docs/data-audit.md for the measurements behind this decision.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from .config import MIN_SPEND, NEIGHBOURS

PLAN_FEATURES = ["age", "gender", "interest", "campaign"]
OBJECTIVES = ["Lead Generation", "Sales", "Traffic", "Awareness"]

# Metrics the training data can and cannot speak to.
SUPPORTED = {
    "Lead Generation": ["leads", "cpa", "conversion rate"],
    "Sales": ["conversions", "cpa", "conversion rate"],
    "Traffic": ["clicks", "ctr", "cpc"],
    "Awareness": ["impressions", "cpm"],
}
UNSUPPORTED = {
    "reach": "no reach or audience-size column exists in the ad export",
    "frequency": "needs reach",
    "ROAS": "no revenue column exists",
    "predicted CPA from a regression": "CV R^2 is negative for conversions per dollar",
}


def overview(df):
    """Descriptive numbers used by the Campaign Intelligence page."""
    def ratios(spend, impressions, clicks, conversions):
        return {
            "ctr": clicks / impressions if impressions else np.nan,
            "cpc": spend / clicks if clicks else np.nan,
            "cpm": spend / impressions * 1000 if impressions else np.nan,
            "cpa": spend / conversions if conversions else np.inf,
        }

    totals = {
        "ads": int(len(df)),
        "campaigns": int(df["campaign"].nunique()),
        "spend": float(df["spent"].sum()),
        "impressions": int(df["impressions"].sum()),
        "clicks": int(df["clicks"].sum()),
        "conversions": int(df["conversions"].sum()),
        "zero_conversion_ads": float((df["conversions"] == 0).mean()),
        "benchmark_cpa": float(df.loc[np.isfinite(df["cpa"]), "cpa"].median()),
    }
    totals.update(ratios(totals["spend"], totals["impressions"],
                         totals["clicks"], totals["conversions"]))

    def grouped(keys, name):
        g = df.groupby(keys).agg(
            ads=("spent", "size"), spend=("spent", "sum"),
            impressions=("impressions", "sum"), clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        ).reset_index()
        for col in ["ctr", "cpc", "cpm", "cpa"]:
            g[col] = np.nan
        mask = g["impressions"] > 0
        g.loc[mask, "ctr"] = g.loc[mask, "clicks"] / g.loc[mask, "impressions"]
        mask = g["clicks"] > 0
        g.loc[mask, "cpc"] = g.loc[mask, "spend"] / g.loc[mask, "clicks"]
        mask = g["impressions"] > 0
        g.loc[mask, "cpm"] = g.loc[mask, "spend"] / g.loc[mask, "impressions"] * 1000
        mask = g["conversions"] > 0
        g.loc[mask, "cpa"] = g.loc[mask, "spend"] / g.loc[mask, "conversions"]
        g["cpa_median"] = df.groupby(keys)["cpa"].median().values
        return g.sort_values(name)

    interests = df.groupby("interest").agg(
        ads=("spent", "size"), spend=("spent", "sum"),
        conversions=("conversions", "sum"),
    ).sort_values("spend", ascending=False).head(12).reset_index()

    return {
        "totals": totals,
        "by_campaign": grouped(["campaign"], "campaign"),
        "by_age": grouped(["age"], "age"),
        "by_gender": grouped(["gender"], "gender"),
        "top_interests": interests,
    }


def _encoder(df):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    matrix = enc.fit_transform(df[PLAN_FEATURES].astype(str))
    return enc, matrix


def forecast(df, age, gender, interest, campaign, budget, objective,
             target_cpa=None, leads_per_dollar=None, k=NEIGHBOURS):
    """Look up similar past ads and report what they achieved.

    Returns plain data so the caller can display it or store it in history.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if budget <= 0:
        raise ValueError("budget must be positive")

    target = _target_cpa(target_cpa, leads_per_dollar)
    enc, matrix = _encoder(df)
    query = pd.DataFrame([{
        "age": str(age), "gender": str(gender),
        "interest": str(interest), "campaign": str(campaign),
    }])[PLAN_FEATURES]
    q = enc.transform(query)

    k = min(k, len(df))
    distances, indices = NearestNeighbors(
        n_neighbors=k, metric="euclidean"
    ).fit(matrix).kneighbors(q)
    neighbours = df.iloc[indices[0]]

    cpa = neighbours["cpa"].to_numpy()
    finite = np.isfinite(cpa)
    eff = neighbours["conversions_per_dollar"].to_numpy()  # 0 when no conversion
    p10, p50, p90 = np.percentile(eff, [10, 50, 90])

    notes = [
        f"Based on the {k} historical ads closest to this plan "
        f"(out of {len(df)} ads).",
        "This is a historical benchmark, not a supervised prediction: "
        "conversions per dollar were not learnable from the planning fields.",
        "Budget scales the estimate linearly; real delivery shows diminishing "
        "returns that this dataset cannot model.",
        f"{float((neighbours['conversions'] == 0).mean()):.0%} of these ads "
        "never converted.",
    ]
    if target is not None:
        met = int(np.sum(cpa <= target))
        notes.append(
            f"{met} of {k} similar ads reached a CPA of ${target:,.2f} or less."
        )
        cheapest = float(np.min(df.loc[np.isfinite(df["cpa"]), "cpa"]))
        if target < cheapest:
            notes.append(
                f"Your target is below the cheapest CPA ever observed in this "
                f"dataset (${cheapest:,.2f}), so nothing historical supports it."
            )

    same_campaign = int((neighbours["campaign"] == str(campaign)).sum())
    exact = int((neighbours[PLAN_FEATURES].astype(str)
                 == query.iloc[0].astype(str)).all(axis=1).sum())

    result = {
        "objective": objective,
        "budget": float(budget),
        "target_cpa": None if target is None else round(float(target), 4),
        "target_leads_per_dollar": None if target is None else round(1 / target, 6),
        "neighbours": k,
        "same_campaign_neighbours": same_campaign,
        "exact_feature_matches": exact,
        "zero_conversion_share": round(float((neighbours["conversions"] == 0).mean()), 3),
        "cpa_p10": _r(np.nanpercentile(cpa[finite], 10) if finite.any() else None),
        "cpa_median": _r(np.nanpercentile(cpa[finite], 50) if finite.any() else None),
        "cpa_p90": _r(np.nanpercentile(cpa[finite], 90) if finite.any() else None),
        "efficiency_median": _r(p50),
        "supported_metrics": SUPPORTED[objective],
        "notes": notes,
        "method": "historical nearest-neighbour benchmark",
    }

    if objective in ("Lead Generation", "Sales"):
        leads = budget * float(p50)
        result.update({
            "estimated_leads": round(leads, 1),
            "estimated_leads_low": round(budget * float(p10), 1),
            "estimated_leads_high": round(budget * float(p90), 1),
            "estimated_cpa": _r(1 / p50 if p50 > 0 else None),
            "estimated_conversion_rate": round(
                float(neighbours["conversion_rate"].median(skipna=True)), 4
            ),
        })
        if target is not None:
            result["target_met"] = bool(result["estimated_cpa"] is not None
                                        and result["estimated_cpa"] <= target)
            result["ads_meeting_target"] = int(np.sum(cpa <= target))
    elif objective == "Traffic":
        # Aggregate ratios: the median ad-level CTR is zero for tiny ads with one
        # impression, which would make the benchmark useless.
        clicks = float(neighbours["clicks"].sum())
        impressions = float(neighbours["impressions"].sum())
        spend = float(neighbours["spent"].sum())
        ctr = clicks / impressions if impressions > 0 else None
        cpc = spend / clicks if clicks > 0 else None
        result.update({
            "estimated_cpc": _r(cpc),
            "estimated_ctr": _r(ctr, 6),
            "estimated_clicks": round(budget / cpc, 0) if cpc and cpc > 0 else None,
        })
    else:  # Awareness
        impressions = float(neighbours["impressions"].sum())
        spend = float(neighbours["spent"].sum())
        cpm = spend / impressions * 1000 if impressions > 0 else None
        result.update({
            "estimated_cpm": _r(cpm),
            "estimated_impressions": round(budget / cpm * 1000, 0) if cpm and cpm > 0 else None,
            "reach_note": "reach and frequency are not in the dataset",
        })

    result["status"] = _status(result, objective, target)
    return result


def _target_cpa(target_cpa, leads_per_dollar):
    if target_cpa is not None and target_cpa > 0:
        return float(target_cpa)
    if leads_per_dollar is not None and leads_per_dollar > 0:
        return float(1 / leads_per_dollar)
    return None


def _r(value, digits=2):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _status(result, objective, target):
    """One plain sentence; never claims certainty."""
    if target is None:
        return "No target set - showing the historical benchmark only."
    if objective in ("Lead Generation", "Sales"):
        if result.get("target_met"):
            return (f"Historically, {result.get('ads_meeting_target')} of "
                    f"{result['neighbours']} similar ads met this target; the "
                    "benchmark is inside your goal, but it is not a guarantee.")
        return (f"Historically only {result.get('ads_meeting_target')} of "
                f"{result['neighbours']} similar ads met this target. The "
                "benchmark is above your goal - expect to need better "
                "targeting, creative or a higher budget.")
    return ("Target CPA only applies to conversion objectives; this objective "
            "is estimated from CPM/CTR instead.")


def default_target(df):
    """Cohort benchmark: median CPA of ads that converted."""
    cpa = df.loc[np.isfinite(df["cpa"]), "cpa"]
    return float(cpa.median()) if len(cpa) else 0.0


def coverage_rows():
    """Field-by-field honesty table for the UI."""
    return pd.DataFrame([
        {"Field": "Age range", "Status": "Supported", "Where": "trained feature (4 brackets)"},
        {"Field": "Gender", "Status": "Supported", "Where": "trained feature (M/F)"},
        {"Field": "Interest / audience", "Status": "Supported", "Where": "trained feature (40 anonymised IDs)"},
        {"Field": "Campaign", "Status": "Supported", "Where": "trained feature (3 campaign IDs)"},
        {"Field": "Budget", "Status": "Calculated", "Where": "scales the benchmark linearly"},
        {"Field": "Objective", "Status": "Calculated", "Where": "chooses which metrics are shown"},
        {"Field": "Target CPA / leads per $", "Status": "Calculated", "Where": "CPA = 1 / leads per $"},
        {"Field": "Brand, industry, platform, location, duration, creative, CTA, offer",
         "Status": "Context only", "Where": "saved with the analysis, never seen by a model"},
        {"Field": "Reach, frequency, ROAS", "Status": "Not available", "Where": "no reach or revenue column"},
    ])
