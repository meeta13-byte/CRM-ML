"""CRM marketing intelligence demo — seven tabs, two real datasets.

    streamlit run app/streamlit_app.py

Every number is read from models/*.json, models/*.joblib and reports/figures/*
produced by the training scripts. Nothing is trained while the app runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.campaign import (  # noqa: E402
    OBJECTIVES, coverage_rows, default_target, forecast, overview,
)
from src.config import BANDS, FIGURES, HISTORY_DB, MODELS  # noqa: E402
from src.data.loaders import load_campaign_ads  # noqa: E402
from src.history import (  # noqa: E402
    get_campaign, get_prediction, list_campaigns, list_predictions,
    save_campaign, save_prediction, set_actual,
)
from src.predict import load_meta, load_model, score, score_all  # noqa: E402

st.set_page_config(page_title="CRM-Adv", page_icon="📈", layout="wide")

DISCLAIMER = (
    "Model output is an estimate from historical data, not a guarantee of "
    "customer behaviour."
)
PROBABILITY_NOTE = {
    "KNN": "frequency of subscribers among the nearest training customers "
           "(coarse: the estimate moves in steps of 1/k)",
    "GaussianNB": "posterior from Gaussian Naive Bayes — assumes features are "
                  "independent, so it is often overconfident",
    "LinearSVM": "Platt-scaled decision score (internal 5-fold calibration)",
    "RBFSVM": "Platt-scaled decision score (internal 5-fold calibration)",
}

st.title("CRM-Adv — marketing & sales intelligence")
st.caption("Campaign analytics on real Facebook ads · four-algorithm conversion "
           "comparison and segmentation on UCI Bank Marketing · SQLite history")

tab_dash, tab_campaign, tab_convert, tab_segments, tab_models, tab_data, \
    tab_history = st.tabs([
        "Dashboard", "Campaign Intelligence", "Lead Conversion Prediction",
        "Customer Segmentation", "Model Comparison",
        "Dataset & Training History", "Prediction History",
    ])


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value, pattern="{:,.2f}", dash="not available"):
    if value is None:
        return dash
    try:
        if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
            return dash
    except TypeError:
        pass
    return pattern.format(value)


def _money(value):
    return _fmt(value, "${:,.2f}")


def _pct(value):
    return _fmt(value, "{:.1%}")


def _rate(value):
    """CTR values are fractions of a percent; three decimals keeps them visible."""
    return _fmt(value, "{:.3%}")


@st.cache_data(show_spinner=False)
def _ads():
    return load_campaign_ads()


@st.cache_data(show_spinner=False)
def _overview():
    df, _info = _ads()
    return overview(df)


CONVERSION = load_meta("conversion")
CLUSTERS = _json(MODELS / "clusters.json")
EXPERIMENTS = _json(MODELS / "experiments.json")
try:
    ADS, ADS_INFO = _ads()
except Exception as exc:  # missing CSV: the other tabs still work
    ADS, ADS_INFO = None, {"error": str(exc)}

# ---------------------------------------------------------------- dashboard ---

with tab_dash:
    bank = CONVERSION["dataset"] if CONVERSION else {}
    totals = _overview()["totals"] if ADS is not None else None

    cards = st.columns(4)
    cards[0].metric("Bank customers", f"{bank.get('rows', 0):,}")
    cards[1].metric("Subscribed (y = yes)", _pct(bank.get("positive_rate")))
    if CONVERSION:
        best = next(m for m in CONVERSION["models"]
                    if m["algorithm"] == CONVERSION["best_model"])
        cards[2].metric(f"Best model — {CONVERSION['best_model']}",
                        f"F1 {best['f1']:.3f}")
        cards[2].caption(f"Recall {best['recall']:.1%} · ROC-AUC {best['roc_auc']} "
                         f"· accuracy {best['accuracy']:.1%}")
    else:
        cards[2].metric("Conversion model", "not trained")
    if CLUSTERS:
        cards[3].metric("K-Means segments", f"k = {CLUSTERS['k']}")
        cards[3].caption(f"Silhouette {CLUSTERS['silhouette']} · Ward agreement "
                         f"(ARI) {CLUSTERS['ari_kmeans_vs_hierarchical']}")

    st.subheader("What each module does")
    st.dataframe(pd.DataFrame([
        {"Module": "Campaign Intelligence",
         "Question": "What did similar Facebook ads achieve, and what do the "
                     "planning inputs imply?",
         "Method": "Descriptive metrics + nearest-neighbour historical benchmark",
         "Data": "1,143 anonymised Facebook ads (one row per ad)"},
        {"Module": "Lead Conversion Prediction",
         "Question": "Will this bank customer subscribe to a term deposit?",
         "Method": "KNN, GaussianNB, Linear SVM, RBF SVM (same split/preprocessing)",
         "Data": "UCI Bank Marketing, 45,211 rows, 11.7% positive"},
        {"Module": "Customer Segmentation",
         "Question": "What kinds of customers exist, ignoring the outcome?",
         "Method": "K-Means + Ward hierarchical clustering, silhouette-selected k",
         "Data": "Bank customer profile and relationship history"},
    ]), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Advertising dataset at a glance")
        if totals is None:
            st.info("Campaign dataset not found under `data/raw`; see README > Data.")
        else:
            st.dataframe(pd.DataFrame([
                {"Metric": "Ads analysed", "Value": f"{totals['ads']:,}"},
                {"Metric": "Campaigns", "Value": totals["campaigns"]},
                {"Metric": "Spend", "Value": _money(totals["spend"])},
                {"Metric": "Impressions", "Value": f"{totals['impressions']:,}"},
                {"Metric": "Clicks", "Value": f"{totals['clicks']:,}"},
                {"Metric": "Approved conversions", "Value": totals["conversions"]},
                {"Metric": "CTR", "Value": _rate(totals["ctr"])},
                {"Metric": "CPM", "Value": _money(totals["cpm"])},
                {"Metric": "Benchmark CPA (median, converting ads)",
                 "Value": _money(totals["benchmark_cpa"])},
                {"Metric": "Ads that never converted",
                 "Value": _pct(totals["zero_conversion_ads"])},
            ], dtype=str), hide_index=True, width="stretch")
    with right:
        st.subheader("Conversion models, held-out test split")
        if CONVERSION:
            st.dataframe(pd.DataFrame(CONVERSION["models"])[
                ["algorithm", "accuracy", "precision", "recall", "f1",
                 "macro_f1", "roc_auc", "cv_score"]
            ], hide_index=True, width="stretch")
            st.caption(
                f"Majority-class baseline accuracy is "
                f"{CONVERSION['majority_baseline']:.1%}: predicting \"no\" for "
                "everyone looks accurate but finds zero subscribers. That is why "
                "the model is selected on positive-class F1."
            )
        else:
            st.info("Run `python3 -m src.train_conversion`.")

    st.warning(
        "Honest limits: `duration` is excluded because it is only known after "
        "the call. The ad dataset has no budget, duration, location, creative, "
        "CTA or revenue column, so campaign outputs are historical benchmarks. "
        "Segments are weakly separated (silhouette ≈ "
        f"{CLUSTERS['silhouette'] if CLUSTERS else 'n/a'}) and should be read "
        "as coarse portraits."
    )

# ------------------------------------------------------- campaign intelligence ---

def _campaign_page():
    if ADS is None:
        st.error("Campaign dataset not available. " + ADS_INFO["error"]
                 + " Put KAG_conversion_data.csv in data/raw/ (see README > Data).")
        return
    st.subheader("Facebook advertising performance")
    st.caption(f"{ADS_INFO['name']} · {ADS_INFO['rows_used']} ads after "
               f"MIN_SPEND = {ADS_INFO['min_spend']} · {ADS_INFO['note']}")

    totals = _overview()["totals"]
    row = st.columns(6)
    row[0].metric("Ads", f"{totals['ads']:,}")
    row[1].metric("Spend", _money(totals["spend"]))
    row[2].metric("CTR", _rate(totals["ctr"]))
    row[3].metric("CPC", _money(totals["cpc"]))
    row[4].metric("CPM", _money(totals["cpm"]))
    row[5].metric("Benchmark CPA", _money(totals["benchmark_cpa"]))

    by_campaign = _overview()["by_campaign"]
    c1, c2 = st.columns(2)
    c1.markdown("**Spend and conversions by campaign**")
    c1.bar_chart(by_campaign.set_index("campaign")[["spend"]], height=240)
    c2.markdown("**CPA by campaign** (spend ÷ conversions)")
    cpa_chart = by_campaign.set_index("campaign")["cpa"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    c2.bar_chart(cpa_chart, height=240)

    c3, c4 = st.columns(2)
    c3.markdown("**CTR by age bracket**")
    c3.bar_chart(_overview()["by_age"].set_index("age")["ctr"], height=220)
    c4.markdown("**CTR by gender**")
    c4.bar_chart(_overview()["by_gender"].set_index("gender")["ctr"], height=220)

    with st.expander("Campaign comparison table"):
        st.dataframe(by_campaign, hide_index=True, width="stretch")
        st.caption("Median CPA is infinite for campaigns where most ads never "
                   "converted; blank/missing means no finite conversions.")

    st.divider()
    st.subheader("Plan a campaign")
    st.caption(
        "Supported = used by the nearest-neighbour benchmark · Calculated = "
        "derived from your inputs · Context only = saved for your records and "
        "never used by a model."
    )

    benchmark = default_target(ADS)
    with st.form("campaign_plan"):
        st.markdown("**Campaign** — objective and campaign are supported; the "
                    "rest is context only")
        c1, c2, c3 = st.columns(3)
        brand = c1.text_input("Brand [context only]", "Acme Bank")
        industry = c2.text_input("Industry [context only]", "Financial services")
        objective = c3.selectbox("Objective [calculated]", OBJECTIVES)
        c4, c5, c6 = st.columns(3)
        platform = c4.selectbox("Platform [context only]", ["Facebook", "Instagram"])
        campaign = c5.selectbox("Campaign [supported]",
                                sorted(ADS["campaign"].unique()), index=0)
        duration = c6.number_input("Duration (days) [context only]",
                                   min_value=1, max_value=365, value=30)

        st.markdown("**Audience** — all four fields are supported")
        a1, a2, a3, a4 = st.columns(4)
        age = a1.selectbox("Age bracket [supported]",
                           sorted(ADS["age"].unique()))
        gender = a2.selectbox("Gender [supported]", sorted(ADS["gender"].unique()))
        interest = a3.selectbox("Interest code [supported]",
                                sorted(ADS["interest"].unique(), key=int))
        location = a4.text_input("Location [context only]", "India")

        st.markdown("**Budget and target** — budget is calculated, "
                    "not a trained feature")
        b1, b2, b3 = st.columns(3)
        budget = b1.number_input("Budget ($)", min_value=1.0, max_value=1e6,
                                 value=1000.0, step=100.0)
        target_mode = b2.radio("Target expressed as",
                               ["Target CPA ($)", "Target leads per $1"],
                               horizontal=True)
        default_leads = 1 / benchmark if benchmark else 0.008
        target_value = b3.number_input(
            "Target value", min_value=0.000001, value=round(
                benchmark if target_mode == "Target CPA ($)" else default_leads, 6),
            format="%.6f")

        st.markdown("**Creative and history** — context only; none of these "
                    "columns exist in the training data")
        d1, d2, d3 = st.columns(3)
        creative = d1.selectbox("Creative format [context only]",
                                ["Image", "Video", "Carousel", "Collection"])
        cta = d2.selectbox("Call to action [context only]",
                           ["Learn more", "Sign up", "Apply now", "Contact us"])
        offer = d3.text_input("Offer [context only]", "Fee waiver")
        d4, d5 = st.columns(2)
        prev_ctr = d4.number_input("Previous CTR (%) [context only]",
                                   min_value=0.0, max_value=100.0, value=0.0)
        prev_cpa = d5.number_input("Previous CPA ($) [context only]",
                                   min_value=0.0, value=0.0)
        save_it = st.checkbox("Save this analysis to history", value=True)
        submitted = st.form_submit_button("Run historical benchmark",
                                          type="primary")

    with st.expander("Which fields can the data actually support?"):
        st.dataframe(coverage_rows(), hide_index=True, width="stretch")

    if submitted:
        context = {
            "brand": brand, "industry": industry, "objective": objective,
            "platform": platform, "campaign": campaign, "duration_days": duration,
            "age": age, "gender": gender, "interest": interest,
            "location": location, "budget": budget,
            "target_mode": target_mode, "target_value": target_value,
            "creative_format": creative, "cta": cta, "offer": offer,
            "previous_ctr_pct": prev_ctr, "previous_cpa": prev_cpa,
        }
        try:
            result = forecast(
                ADS, age=age, gender=gender, interest=interest,
                campaign=campaign, budget=budget, objective=objective,
                target_cpa=target_value if target_mode == "Target CPA ($)" else None,
                leads_per_dollar=(target_value
                                  if target_mode == "Target leads per $1" else None),
            )
        except Exception as exc:
            st.error(f"Could not run the benchmark: {exc}")
        else:
            st.session_state["campaign_result"] = result
            st.session_state["campaign_context"] = context
            if save_it:
                record = save_campaign(
                    context, result, objective,
                    result["method"] + ": " + " ".join(result["notes"]),
                )
                st.success(f"Saved to campaign history as record #{record}.")

    result = st.session_state.get("campaign_result")
    if result:
        st.markdown(f"### Estimate for {result['objective']}")
        st.warning("Historical benchmark from the nearest past ads — not a "
                   "trained prediction, and not a promise that the target is "
                   "achievable.")
        if result["objective"] in ("Lead Generation", "Sales"):
            m = st.columns(3)
            m[0].metric("Estimated leads", _fmt(result.get("estimated_leads"),
                                                "{:,.0f}"))
            m[1].metric("Implied CPA", _money(result.get("estimated_cpa")))
            m[2].metric("Leads range (P10–P90)",
                        f"{_fmt(result.get('estimated_leads_low'), '{:,.0f}')} – "
                        f"{_fmt(result.get('estimated_leads_high'), '{:,.0f}')}")
            st.caption(f"Target CPA {_money(result['target_cpa'])} · "
                       f"conversion-rate median "
                       f"{_pct(result.get('estimated_conversion_rate'))} · "
                       f"{result.get('ads_meeting_target', 0)} of "
                       f"{result['neighbours']} similar ads met it.")
        elif result["objective"] == "Traffic":
            m = st.columns(3)
            m[0].metric("Estimated clicks", _fmt(result.get("estimated_clicks"),
                                                 "{:,.0f}"))
            m[1].metric("CPC", _money(result.get("estimated_cpc")))
            m[2].metric("CTR", _rate(result.get("estimated_ctr")))
        else:
            m = st.columns(2)
            m[0].metric("Estimated impressions",
                        _fmt(result.get("estimated_impressions"), "{:,.0f}"))
            m[1].metric("CPM", _money(result.get("estimated_cpm")))
            st.caption("Reach and frequency cannot be estimated: the dataset has "
                       "no reach or audience-size column.")

        st.info(result["status"])
        st.caption(f"Neighbours: {result['neighbours']} · exact feature matches: "
                   f"{result['exact_feature_matches']} · same campaign: "
                   f"{result['same_campaign_neighbours']} · ads that never "
                   f"converted: {_pct(result['zero_conversion_share'])}")
        for note in result["notes"]:
            st.write(f"- {note}")


with tab_campaign:
    _campaign_page()

# ------------------------------------------------------- lead conversion ---

FEATURE_GROUPS = {
    "Customer profile": ["age", "job", "marital", "education", "default",
                         "balance", "housing", "loan"],
    "Last contact": ["contact", "month", "day", "campaign"],
    "Previous campaign": ["poutcome", "pdays", "previous",
                          "previously_contacted"],
}
DEFAULTS = {
    "age": 40, "balance": 1000.0, "day": 15, "campaign": 2, "pdays": 30,
    "previous": 0, "previously_contacted": 0, "job": "management",
    "education": "secondary", "contact": "cellular", "month": "may",
    "poutcome": "unknown", "default": "no", "housing": "no", "loan": "no",
}
LABELS = {
    "age": "Age", "job": "Job", "marital": "Marital status",
    "education": "Education", "default": "Credit in default",
    "balance": "Average yearly balance (€)", "housing": "Housing loan",
    "loan": "Personal loan", "contact": "Contact channel", "month": "Month",
    "day": "Day of month", "campaign": "Contacts this campaign",
    "poutcome": "Previous campaign outcome", "pdays": "Days since previous contact",
    "previous": "Contacts before this campaign",
    "previously_contacted": "Previously contacted",
}

with tab_convert:
    st.subheader("Will this customer subscribe to a term deposit?")
    if not CONVERSION or load_model("conversion") is None:
        st.info("Not trained yet. Run:\n\n"
                "`python3 -m src.train_conversion`")
    else:
        st.caption(f"Trained on {CONVERSION['train_rows']:,} customers "
                   f"({CONVERSION['positive_rate']:.1%} subscribed), tested on "
                   f"{CONVERSION['test_rows']:,}. `duration` is excluded because "
                   "it is only known after the call.")
        choices = ["Compare all four"] + [
            f"{m['algorithm']} (test F1 {m['f1']:.3f})"
            for m in CONVERSION["models"]
        ]
        selection = st.selectbox("Algorithm", choices)
        chosen = None if selection == "Compare all four" else selection.split(" (")[0]

        specs = {s["name"]: s for s in CONVERSION["features"]}
        payload = {}
        for group, names in FEATURE_GROUPS.items():
            st.markdown(f"**{group}**")
            cols = st.columns(4)
            for i, name in enumerate(names):
                spec = specs[name]
                key = f"conversion_{name}"
                with cols[i % 4]:
                    if spec["kind"] == "categorical":
                        options = spec["categories"]
                        default = DEFAULTS.get(name, options[0])
                        payload[name] = st.selectbox(
                            LABELS.get(name, name), options,
                            index=options.index(default) if default in options else 0,
                            key=key,
                        )
                    else:
                        payload[name] = st.number_input(
                            LABELS.get(name, name),
                            value=float(DEFAULTS.get(name, 0.0)),
                            min_value=float(min(0, spec["min"])),
                            max_value=float(spec["max"]),
                            key=key,
                        )
        st.caption("`pdays` is days since the previous contact; it is missing "
                   "for customers who were never contacted before "
                   "(`previously_contacted = 0`). The pipeline imputes it.")

        if st.button("Predict conversion", type="primary"):
            dataset = (f"{CONVERSION['dataset']['name']} "
                       f"(trained {CONVERSION['created']})")
            try:
                if chosen:
                    result = score(payload, chosen)
                    st.session_state["conversion_result"] = result
                    record = save_prediction(
                        dataset, chosen, payload, result["prediction"],
                        probability=result["probability"],
                        probability_supported=result["probability_supported"],
                        decision_score=result["decision_score"],
                        priority=result["priority"],
                    )
                    st.success(f"Saved to prediction history as record #{record}.")
                else:
                    table = score_all(payload)
                    st.session_state["conversion_compare"] = table
                    ids = [
                        save_prediction(
                            dataset, row.model, payload, int(row.prediction),
                            probability=float(row.probability),
                            probability_supported=True,
                            decision_score=(None if pd.isna(row.decision_score)
                                            else float(row.decision_score)),
                            priority=row.priority,
                        )
                        for row in table.itertuples()
                    ]
                    st.success(f"Saved four predictions: records {ids}.")
            except Exception as exc:
                st.error(f"Could not score this customer: {exc}")

        result = st.session_state.get("conversion_result")
        if result and (chosen is None or result["model"] == chosen):
            st.markdown(f"### {result['model']}")
            m = st.columns(3)
            m[0].metric("Predicted class",
                        "Will subscribe" if result["prediction"] else "Will not subscribe")
            m[1].metric("Probability of subscribing",
                        f"{result['probability']:.1%}")
            m[2].metric("Priority", result["priority"])
            st.progress(min(max(result["probability"], 0.0), 1.0))
            if result.get("decision_score") is not None:
                st.caption(f"Decision score: {result['decision_score']:+.3f} "
                           "(positive leans towards subscribing).")
            st.caption(f"Probability source: {PROBABILITY_NOTE.get(result['model'], 'model estimate')}.")
            st.caption(f"Priority bands: High ≥ {BANDS['High']:.0%}, "
                       f"Medium ≥ {BANDS['Medium']:.0%} (cohort rate "
                       f"{CONVERSION['positive_rate']:.1%}).")
            st.info(DISCLAIMER)

        compare = st.session_state.get("conversion_compare")
        if compare is not None and chosen is None:
            st.markdown("### Four algorithms, same customer")
            st.dataframe(compare, hide_index=True, width="stretch")
            st.caption("Models disagree because they fit different decision "
                       "boundaries; disagreement is information, not an error. "
                       + " ".join(f"{k}: {v}." for k, v in PROBABILITY_NOTE.items()))
            st.info(DISCLAIMER)

# ------------------------------------------------------------ segmentation ---

with tab_segments:
    st.subheader("Customer segments (unsupervised)")
    if not CLUSTERS:
        st.info("No segments yet. Run `python3 -m src.clustering`.")
    else:
        m = st.columns(4)
        m[0].metric("Chosen k", CLUSTERS["k"])
        m[1].metric("Silhouette", CLUSTERS["silhouette"])
        m[2].metric("Ward agreement (ARI)", CLUSTERS["ari_kmeans_vs_hierarchical"])
        m[3].metric("Customers", f"{CLUSTERS['rows']:,}")
        st.caption(
            "k is the highest silhouette in 2–10. The whole curve is flat and "
            "the score is low, so these are coarse portraits, not well-separated "
            "groups. Clusters are formed without the target; the conversion rate "
            "is only used afterwards to describe them."
        )

        profiles = pd.DataFrame(CLUSTERS["profiles"])
        left, right = st.columns([1, 1])
        left.markdown("**Conversion rate by segment**")
        left.bar_chart(profiles.set_index("cluster")["conversion_rate"], height=240)
        right.markdown("**Customers by segment**")
        right.bar_chart(profiles.set_index("cluster")["customers"], height=240)

        st.dataframe(profiles, hide_index=True, width="stretch")
        st.caption(
            f"Tier is relative to the {CLUSTERS['base_conversion_rate']:.1%} "
            "cohort rate: hot ≥ 2×, warm ≥ 1×, cold below. Action: hot — call "
            "within 24h; warm — nurture and re-check; cold — automated email only."
        )
        for row in CLUSTERS["profiles"]:
            st.markdown(
                f"- **Segment {row['cluster']}** — {row['customers']:,} customers "
                f"({row['share']:.1%}); observed subscription rate "
                f"{row['conversion_rate']:.1%} vs "
                f"{CLUSTERS['base_conversion_rate']:.1%} overall → "
                f"**{row['tier']}**."
            )

        c1, c2 = st.columns(2)
        for col, name, caption in [
            (c1, "elbow.png", "Elbow method — WCSS by k"),
            (c2, "silhouette.png", "Silhouette by k"),
        ]:
            path = FIGURES / name
            if path.exists():
                col.image(str(path), caption=caption)
        c1, c2 = st.columns(2)
        for col, name, caption in [
            (c1, "kmeans_pca.png", "K-Means segments (PCA view)"),
            (c2, "hierarchical_pca.png", "Ward segments (PCA view)"),
        ]:
            path = FIGURES / name
            if path.exists():
                col.image(str(path), caption=caption)
        path = FIGURES / "dendrogram.png"
        if path.exists():
            st.image(str(path), caption="Dendrogram (Ward, 300-customer sample)")
        st.info("Segments describe customers, not conversion probability; use the "
                "Lead Conversion Prediction tab when you need a per-customer score.")

# --------------------------------------------------------- model comparison ---

with tab_models:
    st.subheader("Four algorithms, one target, one split, one preprocessing")
    if not CONVERSION:
        st.info("Run `python3 -m src.train_conversion` first.")
    else:
        table = pd.DataFrame(CONVERSION["models"])[
            ["algorithm", "accuracy", "precision", "recall", "f1", "macro_f1",
             "roc_auc", "cv_score", "majority_baseline"]
        ].sort_values("f1", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Selection rule: {CONVERSION['selection_rule']}. "
                   f"Majority baseline accuracy: "
                   f"{CONVERSION['majority_baseline']:.1%}. CV score is the mean "
                   f"of {5}-fold positive-class F1 on the training split, used by "
                   "GridSearchCV.")
        st.bar_chart(
            table.set_index("algorithm")[["accuracy", "precision", "recall", "f1"]],
            height=300,
        )
        st.warning(
            "Why accuracy is not enough: always predicting \"no\" scores "
            f"{CONVERSION['majority_baseline']:.1%} accuracy and finds 0 "
            "subscribers (precision and recall undefined/zero). Positive-class "
            "F1 balances precision against recall, so it is the selection metric."
        )
        st.markdown("**Confusion matrices** (rows = actual, columns = predicted)")
        cols = st.columns(4)
        for col, model in zip(cols, CONVERSION["models"]):
            path = FIGURES / f"conversion_{model['algorithm'].replace(' ', '_')}.png"
            if path.exists():
                col.image(str(path), caption=model["algorithm"])
            else:
                col.caption(f"{model['algorithm']}: no figure — retrain to draw it")

        with st.expander("GridSearchCV results and chosen hyperparameters"):
            for model in CONVERSION["models"]:
                st.markdown(f"**{model['algorithm']}** — chosen: "
                            f"`{model['best_params']}`")
                grid = pd.DataFrame(model["grid_top"])
                grid["params"] = grid["params"].astype(str)
                st.dataframe(grid, hide_index=True, width="stretch")

        st.divider()
        st.subheader("Experiments")
        if not EXPERIMENTS:
            st.info("Run `python3 -m src.experiments`.")
        else:
            best_k = EXPERIMENTS["k_sweep_best"]
            st.markdown(
                f"**KNN K sweep** — best mean CV F1 at k = {best_k['k']} "
                f"({best_k['cv_f1']:.3f}). The curve is flat: the gap to the next "
                "k is well inside fold-to-fold noise, so no single k is "
                "statistically special. The main grid search (which also tunes "
                "weights and metric) chose "
                f"`{next(m['best_params'] for m in CONVERSION['models'] if m['algorithm'] == 'KNN')}` "
                "on its 3–15 range, where k = 3 is the most stable neighbour count."
            )
            path = FIGURES / "k_sweep.png"
            if path.exists():
                st.image(str(path))
            st.caption(EXPERIMENTS["k_sweep_note"])

            st.markdown("**Test-size comparison (0.3 / 0.2 / 0.1)**")
            path = FIGURES / "test_size_f1.png"
            if path.exists():
                st.image(str(path))
            sizes = pd.DataFrame(EXPERIMENTS["test_sizes"]).pivot(
                index="model", columns="test_size", values="f1")
            st.dataframe(sizes, width="stretch")
            st.caption(EXPERIMENTS["test_size_note"])

# ------------------------------------------------------- dataset & training ---

with tab_data:
    st.subheader("Datasets")
    if CONVERSION:
        info = CONVERSION["dataset"]
        st.dataframe(pd.DataFrame([
            {"Item": "Conversion dataset", "Value": info["name"]},
            {"Item": "Source", "Value": info["source"]},
            {"Item": "Rows × columns", "Value": f"{info['rows']:,} × {info['columns_available']}"},
            {"Item": "Positive rate", "Value": _pct(info["positive_rate"])},
            {"Item": "Duplicate rows", "Value": info["duplicate_rows"]},
            {"Item": "Derived", "Value": info["derived"]},
            {"Item": "Missing values", "Value": info["missing_note"]},
            {"Item": "Features used", "Value": info["features_used"]},
            {"Item": "Excluded", "Value": "; ".join(info["excluded"])},
            {"Item": "Target", "Value": CONVERSION["target"]},
        ], dtype=str), hide_index=True, width="stretch")

    if ADS is None:
        st.info("Campaign dataset not found: " + ADS_INFO["error"])
    else:
        st.dataframe(pd.DataFrame([
            {"Item": "Campaign dataset", "Value": ADS_INFO["name"]},
            {"Item": "Source", "Value": ADS_INFO["source"]},
            {"Item": "Rows", "Value": f"{ADS_INFO['rows_raw']} raw, "
                                      f"{ADS_INFO['rows_used']} used"},
            {"Item": "Note", "Value": ADS_INFO["note"]},
        ], dtype=str), hide_index=True, width="stretch")

    if CONVERSION:
        st.subheader("Training run")
        left, right = st.columns(2)
        with left:
            st.markdown("**Preprocessing**")
            st.markdown(
                "- One stratified sample of 12,000 rows (seed 42) shared by all "
                "four models; RBF SVC is roughly O(n²) on 45,211 rows\n"
                "- Numeric: median imputation + StandardScaler, fitted on the "
                "training split only\n"
                "- Categorical: most-frequent imputation + one-hot encoding "
                "(`handle_unknown='ignore'`)\n"
                "- `pdays = -1` → missing flag + nullable recency; "
                "`previously_contacted` derived\n"
                "- Stratified hold-out, test size "
                f"{CONVERSION['test_size']:.0%}, seed 42\n"
                "- Class imbalance: `class_weight='balanced'` in the SVM grids; "
                "imbalance documented, not resampled"
            )
        with right:
            st.markdown("**Run details**")
            st.markdown(
                f"- Trained: `{CONVERSION['created']}`\n"
                f"- Train rows: {CONVERSION['train_rows']:,} · "
                f"test rows: {CONVERSION['test_rows']:,}\n"
                f"- Best model: **{CONVERSION['best_model']}**\n"
                f"- Selection: {CONVERSION['selection_rule']}\n"
                f"- Artifacts: `models/conversion.joblib`, "
                f"`models/conversion.json`\n"
                f"- SQLite history: `{HISTORY_DB.relative_to(ROOT)}`"
            )
        st.markdown("**Hyperparameters chosen by GridSearchCV**")
        st.dataframe(pd.DataFrame([
            {"Algorithm": m["algorithm"], "Parameters": str(m["best_params"]),
             "CV F1": m["cv_score"]}
            for m in CONVERSION["models"]
        ]), hide_index=True, width="stretch")

        if EXPERIMENTS:
            st.markdown("**Experiment results**")
            st.dataframe(pd.DataFrame(EXPERIMENTS["k_sweep"]), hide_index=True,
                         width="stretch")
            st.dataframe(pd.DataFrame(EXPERIMENTS["test_sizes"]), hide_index=True,
                         width="stretch")

    st.caption("The app never retrains on startup: it reads the JSON, joblib and "
               "PNG files above. Delete them and the pages show the command to run.")

# ------------------------------------------------------------- prediction history ---

with tab_history:
    st.subheader("Saved predictions")
    history = list_predictions()
    if history.empty:
        st.info("No predictions saved yet. Score a customer in the Lead "
                "Conversion Prediction tab.")
    else:
        st.caption("Edit the `actual` column when the real outcome is known, then "
                   "save. Actual outcomes make future calibration checks possible.")
        view = history.copy()
        view["actual"] = view["actual"].map({1: "yes", 0: "no"}).fillna("")
        view = view[["id", "created", "model", "prediction", "probability",
                     "priority", "actual"]]
        edited = st.data_editor(
            view, hide_index=True, width="stretch",
            column_config={
                "id": st.column_config.NumberColumn(disabled=True),
                "created": st.column_config.TextColumn(disabled=True),
                "model": st.column_config.TextColumn(disabled=True),
                "prediction": st.column_config.NumberColumn(disabled=True),
                "probability": st.column_config.NumberColumn(
                    disabled=True, format="%.4f"),
                "priority": st.column_config.TextColumn(disabled=True),
                "actual": st.column_config.SelectboxColumn(options=["", "yes", "no"]),
            },
            key="actuals",
        )
        if st.button("Save actual outcomes"):
            current = dict(zip(view["id"], view["actual"]))
            changed = 0
            for row in edited.itertuples():
                new = "" if pd.isna(row.actual) else row.actual
                if current.get(row.id, "") != new:
                    set_actual(int(row.id),
                               None if new == "" else int(new == "yes"))
                    changed += 1
            st.success(f"Updated {changed} record(s).")
            st.rerun()

        record_id = st.selectbox("Inspect one prediction",
                                 history["id"].tolist())
        record = get_prediction(int(record_id))
        if record:
            st.json({k: record[k] for k in
                     ("created", "dataset", "model", "prediction", "probability",
                      "decision_score", "priority", "actual", "features")})

    st.divider()
    st.subheader("Saved campaign analyses")
    campaigns = list_campaigns()
    if campaigns.empty:
        st.info("No campaign plans saved yet.")
    else:
        st.dataframe(campaigns[["id", "created", "objective"]],
                     hide_index=True, width="stretch")
        pick = st.selectbox("Inspect one analysis", campaigns["id"].tolist())
        record = get_campaign(int(pick))
        if record:
            st.json({"created": record["created"],
                     "objective": record["objective"],
                     "context": record["context"],
                     "result": {
                         "method": record["result"].get("method"),
                         "estimated metrics": {
                             k: v for k, v in record["result"].items()
                             if k.startswith("estimated")
                         },
                     },
                     "methodology": record["methodology"]})

st.divider()
st.caption(
    "Prediction time: before the call, using only pre-contact features. "
    "`duration` is excluded by a hard leakage check. Campaign estimates are "
    "historical benchmarks, not supervised predictions. "
    f"Priority bands: High ≥ {BANDS['High']:.0%}, Medium ≥ {BANDS['Medium']:.0%}."
)
