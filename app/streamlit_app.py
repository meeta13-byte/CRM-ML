"""MarketMind AI — CRM marketing intelligence.

    streamlit run app/streamlit_app.py

Pages read the artifacts written by src/train_conversion.py,
src/experiments.py and src/clustering.py. Nothing is retrained on startup.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme as ui  # noqa: E402
from src.campaign import (  # noqa: E402
    OBJECTIVES, coverage_rows, default_target, forecast, overview,
)
from src.config import BANDS, FIGURES, HISTORY_DB, MODELS  # noqa: E402
from src.data.loaders import load_campaign_ads  # noqa: E402
from src.history import (  # noqa: E402
    get_campaign, get_prediction, list_campaigns, list_predictions,
    save_campaign, save_prediction, set_actual,
)
from src.predict import load_meta, score, score_all  # noqa: E402

st.set_page_config(
    page_title="MarketMind AI",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject()

DISCLAIMER = ("Model output is an estimate from historical data, not a guarantee "
              "of customer behaviour.")
PRETTY = {"KNN": "KNN", "GaussianNB": "Gaussian NB",
          "LinearSVM": "Linear SVM", "RBFSVM": "RBF SVM"}
PROBABILITY_NOTE = {
    "KNN": "Share of subscribers among the nearest training customers (coarse: "
           "moves in steps of 1/k).",
    "GaussianNB": "Gaussian Naive Bayes posterior — assumes features are "
                  "independent, so it is often overconfident.",
    "LinearSVM": "Platt-scaled decision score from an internal 5-fold calibration.",
    "RBFSVM": "Platt-scaled decision score from an internal 5-fold calibration.",
}
FEATURE_GROUPS = {
    "Customer profile": ["age", "job", "marital", "education", "default",
                         "balance", "housing", "loan"],
    "Last contact": ["contact", "month", "day", "campaign"],
    "Previous campaign": ["poutcome", "pdays", "previous",
                          "previously_contacted"],
}
DEFAULTS = {
    "age": 40.0, "balance": 1000.0, "day": 15.0, "campaign": 2.0,
    "pdays": 30.0, "previous": 0.0, "previously_contacted": 0.0,
    "job": "management", "education": "secondary", "contact": "cellular",
    "month": "may", "poutcome": "unknown", "default": "no", "housing": "no",
    "loan": "no",
}
LABELS = {
    "age": "Age", "job": "Job", "marital": "Marital status",
    "education": "Education", "default": "Credit in default",
    "balance": "Average yearly balance (€)", "housing": "Housing loan",
    "loan": "Personal loan", "contact": "Contact channel", "month": "Month",
    "day": "Day of month", "campaign": "Contacts this campaign",
    "poutcome": "Previous campaign outcome", "pdays": "Days since last contact",
    "previous": "Earlier contacts", "previously_contacted": "Contacted before",
}


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value, pattern="{:,.2f}", dash="—"):
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
    return _fmt(value, "{:.3%}")


def _pretty(name):
    return PRETTY.get(name, name)


@st.cache_data(show_spinner=False)
def campaign_data():
    return load_campaign_ads()


@st.cache_data(show_spinner=False)
def campaign_overview():
    df, _ = campaign_data()
    return overview(df)


def load_ads():
    try:
        return campaign_data()
    except Exception as exc:  # missing CSV must not break the other pages
        return None, {"error": str(exc)}


CONVERSION = load_meta("conversion")
CLUSTERS = _json(MODELS / "clusters.json")
EXPERIMENTS = _json(MODELS / "experiments.json")
ADS, ADS_INFO = load_ads()
BANK = CONVERSION["dataset"] if CONVERSION else {}
MODEL_ROWS = CONVERSION["models"] if CONVERSION else []

# ---------------------------------------------------------------- navigation ---

NAV = [
    ("Overview", [
        ("Dashboard", ":material/space_dashboard:"),
        ("Campaign Intelligence", ":material/campaign:"),
        ("Lead Conversion Prediction", ":material/person_search:"),
        ("Customer Segmentation", ":material/groups:"),
    ]),
    ("Machine Learning", [
        ("Model Comparison", ":material/compare_arrows:"),
        ("Dataset & Training History", ":material/database:"),
        ("Prediction History", ":material/history:"),
    ]),
    ("Workspace", [
        ("Settings", ":material/settings:"),
        ("About", ":material/info:"),
    ]),
]


def _go(page):
    st.session_state.page = page


st.session_state.setdefault("page", "Dashboard")

with st.sidebar:
    st.markdown(
        f'<div class="mm-brand"><div class="mm-brand-mark">{ui.icon("chart", 19, 2)}</div>'
        '<div><div class="mm-brand-name">MarketMind AI</div>'
        '<div class="mm-brand-sub">CRM Marketing Intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    for section, items in NAV:
        st.markdown(f'<div class="mm-nav-section">{section}</div>',
                    unsafe_allow_html=True)
        for label, icon_name in items:
            st.button(
                label, key=f"nav-{label}", icon=icon_name, width="stretch",
                type="primary" if st.session_state.page == label else "secondary",
                on_click=_go, args=(label,),
            )
    st.markdown(
        '<div class="mm-profile"><div class="mm-avatar">BA</div>'
        '<div><div class="mm-profile-name">Business Analyst</div>'
        '<div class="mm-profile-sub">Local workspace · demo</div></div></div>',
        unsafe_allow_html=True,
    )


def _best(metric="f1"):
    if not MODEL_ROWS:
        return None
    return max(MODEL_ROWS, key=lambda m: m.get(metric) or 0)


def _history_counts():
    preds = list_predictions(limit=2000)
    camps = list_campaigns(limit=2000)
    return preds, camps


# ------------------------------------------------------------------ dashboard ---

def page_dashboard():
    ui.page_header(
        "Dashboard",
        "Monitor campaign performance and machine learning results.",
    )
    preds, _camps = _history_counts()
    best = _best()

    cards = st.columns(4, gap="medium")
    with cards[0]:
        ui.kpi("Dataset Records", f"{BANK.get('rows', 0):,}",
               "UCI Bank Marketing · 16 features", "database", "lavender")
    with cards[1]:
        ui.kpi("Best Model F1-score",
               f"{best['f1']:.3f}" if best else "—",
               (f"{_pretty(best['algorithm'])} · held-out test split"
                if best else "Run the training script"), "target", "green")
    with cards[2]:
        ui.kpi("Saved Predictions", f"{len(preds):,}",
               "SQLite history · customer scores", "history", "blue")
    with cards[3]:
        ui.kpi("Actual Conversion Rate", _pct(BANK.get("positive_rate")),
               "Observed y = yes — not a model prediction", "chart", "orange")

    st.write("")
    left, right = st.columns([2.05, 1], gap="large")

    with left:
        with st.container(border=True):
            ui.section("Model performance",
                       "Held-out test split · same target, split and preprocessing")
            if CONVERSION:
                fig = ui.grouped_series(
                    [_pretty(m["algorithm"]) for m in MODEL_ROWS],
                    {
                        "F1": [m["f1"] for m in MODEL_ROWS],
                        "Recall": [m["recall"] for m in MODEL_ROWS],
                        "Precision": [m["precision"] for m in MODEL_ROWS],
                        "Accuracy": [m["accuracy"] for m in MODEL_ROWS],
                    },
                    colors=[ui.C["purple_deep"], ui.C["orange"],
                            ui.C["blue"], ui.C["lavender"]],
                    width=6.4, height=3.0,
                )
                ui.show(fig)
                ui.note(
                    f"Selected on positive-class F1: <b>{_pretty(best['algorithm'])}</b> "
                    f"at {best['f1']:.3f}. Majority-class accuracy is "
                    f"{CONVERSION['majority_baseline']:.1%}, so accuracy alone "
                    "would reward a model that never predicts a subscriber."
                )
            else:
                ui.empty("No trained model",
                         "Run <code>python3 -m src.train_conversion</code>.")

    with right:
        with st.container(border=True):
            ui.section("Actual class distribution",
                       "UCI Bank Marketing · target y")
            if CONVERSION:
                pos = float(BANK.get("positive_rate", 0))
                ui.show(ui.donut(
                    ["No", "Yes"], [1 - pos, pos],
                    [ui.C["lavender"], ui.C["purple_deep"]],
                    _pct(pos), "subscribed", width=3.2, height=2.7,
                ))
                ui.stat_rows([
                    ("Subscribed (yes)", f"{round(pos * BANK['rows']):,}"),
                    ("Not subscribed (no)", f"{BANK['rows'] - round(pos * BANK['rows']):,}"),
                    ("Imbalance", f"{1 - pos:.0%} majority"),
                ])
            else:
                ui.empty("No dataset loaded",
                         "Place <code>bank-full.csv</code> in data/raw.")

    st.write("")
    left, right = st.columns([2.05, 1], gap="large")

    with left:
        with st.container(border=True):
            ui.section("Campaign performance",
                       "Facebook ads · 915 ads after the $1 spend floor")
            if ADS is not None:
                totals = campaign_overview()["totals"]
                mini = st.columns(4)
                with mini[0]:
                    ui.mini("Spend", _money(totals["spend"]))
                with mini[1]:
                    ui.mini("Approved conversions", f"{totals['conversions']:,}")
                with mini[2]:
                    ui.mini("CTR", _rate(totals["ctr"]))
                with mini[3]:
                    ui.mini("Benchmark CPA", _money(totals["benchmark_cpa"]))
                by_campaign = campaign_overview()["by_campaign"]
                ui.show(ui.h_bars(
                    [f"Campaign {c}" for c in by_campaign["campaign"]],
                    by_campaign["conversions"].astype(float).tolist(),
                    color=[ui.C["purple_deep"], ui.C["lavender"], ui.C["blue"]],
                    width=6.2, height=2.25, fmt="{:,.0f}",
                ))
                ui.note("Conversions are summed per campaign. One row in this "
                        "dataset is one ad, not a campaign.")
            else:
                ui.empty("Campaign dataset not found",
                         "Place <code>KAG_conversion_data.csv</code> in data/raw.")

    with right:
        with st.container(border=True):
            ui.section("Recent predictions", "Latest records in SQLite")
            if preds.empty:
                ui.empty("No predictions yet",
                         "Score a customer to start the history.")
            else:
                for row in preds.head(5).itertuples():
                    tone = "green" if int(row.prediction) == 1 else "lavender"
                    label = "subscribe" if int(row.prediction) == 1 else "no subscription"
                    st.markdown(
                        f'<div class="mm-activity">{ui.badge(_pretty(row.model), "lavender")}'
                        f'<span>{_pct(row.probability)} · {label}</span>'
                        f'<span class="mm-activity-time">'
                        f'{str(row.created)[5:16].replace("T", " ")}</span></div>',
                        unsafe_allow_html=True,
                    )

    st.write("")
    left, right = st.columns([2.05, 1], gap="large")

    with left:
        with st.container(border=True):
            ui.section("Latest training run", "Model status and provenance")
            if CONVERSION:
                mini = st.columns(4)
                with mini[0]:
                    ui.mini("Best model", _pretty(CONVERSION["best_model"]))
                with mini[1]:
                    ui.mini("Selection metric", "Positive-class F1")
                with mini[2]:
                    ui.mini("Train / test rows",
                            f"{CONVERSION['train_rows']:,} / {CONVERSION['test_rows']:,}")
                with mini[3]:
                    ui.mini("Trained", str(CONVERSION["created"])[:10])
                ui.note(
                    f"{ui.badge('Completed', 'green')}  "
                    f"{CONVERSION['dataset']['name']} · stratified 12,000-row sample "
                    f"· {CONVERSION['selection_rule']}. `duration` is excluded to "
                    "prevent target leakage."
                )
            else:
                ui.empty("Training pending",
                         "Run <code>python3 -m src.train_conversion</code>.")
        with st.container(border=True):
            ui.section("Quick actions", "Jump straight into a workflow")
            action_cols = st.columns(3)
            with action_cols[0]:
                st.button("Predict a conversion", icon=":material/person_search:",
                          width="stretch", on_click=_go,
                          args=("Lead Conversion Prediction",))
            with action_cols[1]:
                st.button("Compare models", icon=":material/compare_arrows:",
                          width="stretch", on_click=_go,
                          args=("Model Comparison",))
            with action_cols[2]:
                st.button("Review history", icon=":material/history:",
                          width="stretch", on_click=_go,
                          args=("Prediction History",))

    with right:
        with st.container(border=True):
            ui.section("Artifacts", "What this demo is reading")
            ui.stat_rows([
                ("Conversion model", _badge_state((MODELS / "conversion.joblib").exists())),
                ("Experiments", _badge_state(EXPERIMENTS is not None)),
                ("Segmentation", _badge_state(CLUSTERS is not None)),
                ("Campaign data", _badge_state(ADS is not None)),
                ("History database", _badge_state(HISTORY_DB.exists())),
            ])
            ui.note("No retraining happens when the app starts.")


def _badge_state(ok):
    return ui.badge("Ready", "green") if ok else ui.badge("Missing", "orange")


# ------------------------------------------------------------------- campaign ---

def page_campaign():
    ui.page_header(
        "Campaign Intelligence",
        "Analyse real ad performance and benchmark a planned campaign.",
    )
    if ADS is None:
        ui.empty("Campaign dataset not available",
                 f"{ADS_INFO.get('error', '')}<br>Place "
                 "<code>KAG_conversion_data.csv</code> in <code>data/raw/</code>.")
        return

    ov = campaign_overview()
    totals = ov["totals"]
    with st.container(border=True):
        ui.section("Advertising performance",
                   "Descriptive metrics from the Facebook ad export")
        mini = st.columns(5)
        with mini[0]:
            ui.mini("Ads analysed", f"{totals['ads']:,}")
        with mini[1]:
            ui.mini("Spend", _money(totals["spend"]))
        with mini[2]:
            ui.mini("Impressions", f"{totals['impressions']:,}")
        with mini[3]:
            ui.mini("Approved conversions", f"{totals['conversions']:,}")
        with mini[4]:
            ui.mini("Benchmark CPA", _money(totals["benchmark_cpa"]))
        chart_left, chart_right = st.columns(2)
        with chart_left:
            ui.section("Spend by campaign")
            ui.show(ui.h_bars(
                [f"Campaign {c}" for c in ov["by_campaign"]["campaign"]],
                ov["by_campaign"]["spend"].astype(float).tolist(),
                color=[ui.C["purple_deep"], ui.C["lavender"], ui.C["blue"]],
                width=5.4, height=2.2, fmt="${:,.0f}",
            ))
        with chart_right:
            ui.section("CTR by age bracket")
            ui.show(ui.h_bars(
                ov["by_age"]["age"].astype(str).tolist(),
                (ov["by_age"]["ctr"].astype(float) * 100).tolist(),
                color=ui.C["purple"],
                width=5.4, height=2.2, fmt="{:.3f}%",
            ))
        with st.expander("Campaign comparison table"):
            st.dataframe(ov["by_campaign"], hide_index=True, width="stretch")
            ui.note("Median CPA is infinite where most ads never converted. "
                    "One row is one ad, not a campaign; no budget, location, "
                    "creative or revenue columns exist.")

    st.write("")
    benchmark = default_target(ADS)
    with st.container(border=True):
        ui.section(
            "Plan a campaign",
            "Only age, gender, interest and campaign ID are model inputs. "
            "Everything tagged context-only is saved but never used in the estimate.",
        )

        # Outside the form so switching the target unit reveals the matching input.
        t1, t2 = st.columns(2)
        target_mode = t1.radio(
            "Target expressed as · calculated",
            ["Target CPA ($)", "Target leads per $1"], horizontal=True)
        default_leads = 1 / benchmark if benchmark else 0.008
        if target_mode == "Target CPA ($)":
            target_cpa = t2.number_input(
                "Target CPA ($) · calculated", min_value=0.0001,
                value=round(benchmark, 2), format="%.4f")
            target_leads = None
        else:
            target_cpa = None
            target_leads = t2.number_input(
                "Target leads per $1 · calculated", min_value=0.000001,
                value=round(default_leads, 6), format="%.6f",
                help="Target CPA = 1 / target leads per dollar.")

        with st.form("campaign_plan"):
            with st.expander("A · Campaign details", expanded=True):
                a1, a2, a3 = st.columns(3)
                brand = a1.text_input("Brand · context only", "Acme Bank")
                industry = a2.text_input("Industry · context only", "Financial services")
                objective = a3.selectbox("Campaign objective · calculated",
                                         OBJECTIVES)
                a4, a5, a6 = st.columns(3)
                campaign_type = a4.selectbox(
                    "Campaign type · context only",
                    ["Prospecting", "Retargeting", "Brand", "Always-on"])
                platform = a5.selectbox(
                    "Platform · context only",
                    ["Facebook", "Instagram", "Audience Network"])
                campaign_name = a6.text_input("Campaign name · context only",
                                              "Spring term deposit push")
                a7, a8 = st.columns(2)
                campaign = a7.selectbox("Campaign ID · model input",
                                        sorted(ADS["campaign"].unique()))
                location = a8.text_input("Location · context only", "India")

            with st.expander("B · Audience"):
                b1, b2, b3 = st.columns(3)
                target_audience = b1.text_input(
                    "Target audience · context only", "Existing current accounts")
                audience_size = b2.number_input(
                    "Audience size · unsupported", min_value=0, value=100000,
                    step=1000, help="No audience-size column exists in the export.")
                acquisition = b3.selectbox(
                    "Audience type · context only", ["New audience", "Retargeting"])
                b4, b5, b6 = st.columns(3)
                age = b4.selectbox("Age range · model input",
                                   sorted(ADS["age"].unique()))
                gender = b5.selectbox("Gender · model input",
                                      sorted(ADS["gender"].unique()))
                interest = b6.selectbox("Interest code · model input",
                                        sorted(ADS["interest"].unique(), key=int))

            with st.expander("C · Budget and schedule"):
                c1, c2, c3 = st.columns(3)
                budget = c1.number_input("Budget ($) · calculated", min_value=1.0,
                                         max_value=1_000_000.0, value=1000.0,
                                         step=100.0)
                duration = c2.number_input("Duration (days) · context only",
                                           min_value=1, max_value=365, value=30)
                bid_strategy = c3.selectbox(
                    "Bid strategy · context only",
                    ["Lowest cost", "Cost cap", "Bid cap", "Target cost"])
                c4, c5, c6 = st.columns(3)
                start_date = c4.date_input("Start date · context only",
                                           value=date.today())
                end_date = c5.date_input("End date · context only",
                                         value=date.today())
                placement = c6.selectbox(
                    "Placement · context only",
                    ["Automatic", "Feed", "Stories", "Reels"])

            with st.expander("D · Creative and offer"):
                d1, d2, d3, d4 = st.columns(4)
                creative = d1.selectbox("Creative format · context only",
                                        ["Image", "Video", "Carousel", "Collection"])
                cta = d2.selectbox("Call to action · context only",
                                   ["Learn more", "Sign up", "Apply now", "Contact us"])
                offer = d3.text_input("Offer · context only", "Fee waiver")
                creatives = d4.number_input("Number of creatives · context only",
                                            min_value=1, max_value=100, value=3)

            with st.expander("E · Historical performance"):
                e1, e2, e3, e4 = st.columns(4)
                prev_engagement = e1.number_input(
                    "Previous engagement rate (%) · context only", min_value=0.0,
                    max_value=100.0, value=0.0, step=0.1)
                prev_ctr = e2.number_input(
                    "Previous CTR (%) · context only", min_value=0.0,
                    max_value=100.0, value=0.0, step=0.01, format="%.3f")
                prev_cpc = e3.number_input(
                    "Previous CPC ($) · context only", min_value=0.0, value=0.0)
                prev_cpm = e4.number_input(
                    "Previous CPM ($) · context only", min_value=0.0, value=0.0)
                e5, e6, e7, e8 = st.columns(4)
                prev_cvr = e5.number_input(
                    "Previous conversion rate (%) · context only", min_value=0.0,
                    max_value=100.0, value=0.0, step=0.1)
                prev_spend = e6.number_input(
                    "Previous spend ($) · context only", min_value=0.0, value=0.0)
                prev_leads = e7.number_input(
                    "Previous leads · context only", min_value=0.0, value=0.0)
                prev_cpa = e8.number_input(
                    "Previous CPA ($) · context only", min_value=0.0, value=0.0)

            save_it = st.checkbox("Save this analysis to history", value=True)
            submitted = st.form_submit_button("Run historical benchmark",
                                              type="primary")

        with st.expander("Field support matrix"):
            st.dataframe(coverage_rows(), hide_index=True, width="stretch")
            ui.note("Unsupported fields are accepted for record-keeping but have "
                    "no column in the training data, so they cannot influence the "
                    "estimate.")

    if submitted:
        if end_date < start_date:
            st.error("End date must be on or after the start date.")
        else:
            context = {
                "brand": brand, "industry": industry,
                "campaign_objective": objective, "campaign_type": campaign_type,
                "platform": platform, "campaign_name": campaign_name,
                "campaign_id": campaign, "location": location,
                "target_audience": target_audience, "audience_size": int(audience_size),
                "audience_type": acquisition, "age": age, "gender": gender,
                "interest": interest, "budget": float(budget),
                "duration_days": int(duration), "start_date": str(start_date),
                "end_date": str(end_date), "bid_strategy": bid_strategy,
                "placement": placement, "target_mode": target_mode,
                "target_cpa_input": (float(target_cpa)
                                     if target_cpa is not None else None),
                "target_leads_per_dollar_input": (float(target_leads)
                                                  if target_leads is not None
                                                  else None),
                "creative_format": creative, "cta": cta, "offer": offer,
                "creatives": int(creatives),
                "previous_engagement_rate_pct": float(prev_engagement),
                "previous_ctr_pct": float(prev_ctr),
                "previous_cpc": float(prev_cpc), "previous_cpm": float(prev_cpm),
                "previous_conversion_rate_pct": float(prev_cvr),
                "previous_spend": float(prev_spend),
                "previous_leads": float(prev_leads), "previous_cpa": float(prev_cpa),
            }
            try:
                result = forecast(
                    ADS, age=age, gender=gender, interest=interest,
                    campaign=campaign, budget=float(budget), objective=objective,
                    target_cpa=(float(target_cpa)
                                if target_mode == "Target CPA ($)" else None),
                    leads_per_dollar=(float(target_leads)
                                      if target_mode != "Target CPA ($)" else None),
                )
            except Exception as exc:
                st.error(f"Could not run the benchmark: {exc}")
            else:
                record = None
                if save_it:
                    record = save_campaign(
                        context, result, objective,
                        result["method"] + ": " + " ".join(result["notes"]),
                    )
                st.session_state["campaign_result"] = result
                st.session_state["campaign_record"] = record
                st.session_state["campaign_objective"] = objective

    result = st.session_state.get("campaign_result")
    if result:
        st.write("")
        with st.container(border=True):
            ui.section(
                f"Historical benchmark · {st.session_state.get('campaign_objective')}",
                f"{result['neighbours']} nearest historical ads · empirical estimate, "
                "not a trained prediction",
            )
            if result["objective"] in ("Lead Generation", "Sales"):
                cols = st.columns(3)
                with cols[0]:
                    ui.mini("Estimated leads",
                            _fmt(result.get("estimated_leads"), "{:,.0f}"),
                            "Budget × median leads per $1")
                with cols[1]:
                    ui.mini("Implied CPA", _money(result.get("estimated_cpa")),
                            "Median efficiency of similar ads")
                with cols[2]:
                    ui.mini("Lead range (P10–P90)",
                            f"{_fmt(result.get('estimated_leads_low'), '{:,.0f}')} – "
                            f"{_fmt(result.get('estimated_leads_high'), '{:,.0f}')}")
                met = result.get("ads_meeting_target")
                if met is None:
                    tone, text = "grey", "No target set"
                elif result.get("target_met"):
                    tone, text = "green", "Benchmark inside target"
                else:
                    tone, text = "orange", "Benchmark above target"
                target_badge = ui.badge(
                    "Target CPA " + _money(result.get("target_cpa")), "lavender")
                met_badge = ui.badge(
                    f"{met or 0} of {result['neighbours']} ads met it", "grey")
                st.markdown(
                    f"{ui.badge(text, tone)} {target_badge} {met_badge}",
                    unsafe_allow_html=True,
                )
            elif result["objective"] == "Traffic":
                cols = st.columns(3)
                with cols[0]:
                    ui.mini("Estimated clicks", _fmt(result.get("estimated_clicks"),
                                                     "{:,.0f}"), "Budget ÷ benchmark CPC")
                with cols[1]:
                    ui.mini("CPC", _money(result.get("estimated_cpc")))
                with cols[2]:
                    ui.mini("CTR", _rate(result.get("estimated_ctr")))
            else:
                cols = st.columns(2)
                with cols[0]:
                    ui.mini("Estimated impressions",
                            _fmt(result.get("estimated_impressions"), "{:,.0f}"),
                            "Budget ÷ benchmark CPM")
                with cols[1]:
                    ui.mini("CPM", _money(result.get("estimated_cpm")))
                ui.note("Reach and frequency cannot be estimated: the dataset has "
                        "no reach or audience-size column.")
            st.markdown(result["status"])
            st.markdown("\n".join(f"- {n}" for n in result["notes"]))
            ui.stat_rows([
                ("Exact feature matches", result["exact_feature_matches"]),
                ("Same campaign neighbours", result["same_campaign_neighbours"]),
                ("Ads that never converted",
                 _pct(result["zero_conversion_share"])),
                ("Method", result["method"]),
            ])
            if st.session_state.get("campaign_record"):
                ui.note(f"Saved to campaign history as record "
                        f"#{st.session_state['campaign_record']}. Context-only "
                        "fields were stored with it but did not affect the result.")


# ------------------------------------------------------------------- predict ---

def page_convert():
    ui.page_header(
        "Lead Conversion Prediction",
        "Score one customer with the four-algorithm bank conversion model.",
    )
    if not CONVERSION or not (MODELS / "conversion.joblib").exists():
        ui.empty("Model not trained",
                 "Run <code>python3 -m src.train_conversion</code> first.")
        return

    specs = {s["name"]: s for s in CONVERSION["features"]}
    models = [m["algorithm"] for m in MODEL_ROWS]
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(border=True):
            ui.section("Customer features",
                       "Pre-contact inputs only · `duration` is excluded by a "
                       "hard leakage guard")
            chosen = st.selectbox("Algorithm", models,
                                  index=models.index(CONVERSION["best_model"]))
            payload = {}
            for i, (group, names) in enumerate(FEATURE_GROUPS.items()):
                with st.expander(group, expanded=(i == 0)):
                    cols = st.columns(2)
                    for j, name in enumerate(names):
                        spec = specs[name]
                        key = f"conv_{name}"
                        with cols[j % 2]:
                            if spec["kind"] == "categorical":
                                options = spec["categories"]
                                default = DEFAULTS.get(name, options[0])
                                payload[name] = st.selectbox(
                                    LABELS.get(name, name), options,
                                    index=(options.index(default)
                                           if default in options else 0),
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
            ui.note("`pdays` is missing for customers never contacted before "
                    "(`previously_contacted = 0`); the pipeline imputes it.")
            b1, b2 = st.columns(2)
            predict = b1.button("Predict conversion", type="primary",
                                width="stretch", icon=":material/bolt:")
            compare = b2.button("Compare all four", width="stretch",
                                icon=":material/compare_arrows:")

    if predict:
        dataset = (f"{CONVERSION['dataset']['name']} "
                   f"(trained {CONVERSION['created']})")
        try:
            result = score(payload, chosen)
            record = save_prediction(
                dataset, chosen, payload, result["prediction"],
                probability=result["probability"],
                probability_supported=result["probability_supported"],
                decision_score=result["decision_score"],
                priority=result["priority"],
            )
            st.session_state["prediction_result"] = result
            st.session_state["prediction_record"] = record
            st.session_state["prediction_compare"] = None
        except Exception as exc:
            st.error(f"Could not score this customer: {exc}")

    if compare:
        dataset = (f"{CONVERSION['dataset']['name']} "
                   f"(trained {CONVERSION['created']})")
        try:
            table = score_all(payload)
            ids = []
            for row in table.itertuples():
                ids.append(save_prediction(
                    dataset, row.model, payload, int(row.prediction),
                    probability=float(row.probability),
                    probability_supported=True,
                    decision_score=(None if pd.isna(row.decision_score)
                                    else float(row.decision_score)),
                    priority=row.priority,
                ))
            st.session_state["prediction_compare"] = table
            st.session_state["prediction_records"] = ids
            st.session_state["prediction_result"] = None
        except Exception as exc:
            st.error(f"Could not score this customer: {exc}")

    with right:
        with st.container(border=True):
            ui.section("Prediction", "Probability, class and priority band")
            result = st.session_state.get("prediction_result")
            table = st.session_state.get("prediction_compare")
            if result:
                pos = result["prediction"] == 1
                st.markdown(
                    f'{ui.badge("Will subscribe" if pos else "Will not subscribe", "green" if pos else "grey")} '
                    f'{ui.badge(result["priority"] + " priority", {"High": "orange", "Medium": "blue", "Low": "grey"}[result["priority"]])}',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mm-result-label">Probability of subscribing</div>'
                    f'<div class="mm-result-value">{result["probability"]:.1%}</div>'
                    f'<div class="mm-note">{_pretty(result["model"])} · threshold 50%</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(result["probability"], 0.0), 1.0))
                ui.stat_rows([
                    ("Priority band", result["priority"]),
                    ("Decision score",
                     f"{result['decision_score']:+.3f}"
                     if result.get("decision_score") is not None else "Not available"),
                    ("Record", f"#{st.session_state.get('prediction_record')}"),
                ])
                ui.note(PROBABILITY_NOTE.get(result["model"], "Model estimate."))
                ui.note(f"Bands: High ≥ {BANDS['High']:.0%}, Medium ≥ "
                        f"{BANDS['Medium']:.0%} against a "
                        f"{CONVERSION['positive_rate']:.1%} cohort rate.")
                st.info(DISCLAIMER)
            elif table is not None:
                view = table.copy()
                view["probability"] = view["probability"].map(lambda v: f"{v:.1%}")
                view["predicted class"] = view["prediction"].map(
                    {1: "Subscribed", 0: "Not subscribed"})
                view = view.rename(columns={
                    "model": "Algorithm", "priority": "Priority",
                    "decision_score": "Decision score",
                })
                st.dataframe(view[["Algorithm", "probability", "predicted class",
                                   "Priority", "Decision score"]],
                             hide_index=True, width="stretch")
                st.caption("Four predictions saved: records "
                           f"{st.session_state.get('prediction_records')}.")
                st.info(DISCLAIMER)
                ui.note("Models disagree because they fit different decision "
                        "boundaries; disagreement is information, not an error.")
            else:
                ui.empty("No prediction yet",
                         "Choose an algorithm and click Predict conversion, or "
                         "compare all four on the same customer.")


# --------------------------------------------------------------- model results ---

def page_models():
    ui.page_header(
        "Model Comparison",
        "Fair evaluation of KNN, Gaussian NB, Linear SVM and RBF SVM on one split.",
    )
    if not CONVERSION:
        ui.empty("No evaluation available",
                 "Run <code>python3 -m src.train_conversion</code>.")
        return

    metric_labels = {"f1": "F1-score", "recall": "Recall",
                     "precision": "Precision", "accuracy": "Accuracy",
                     "roc_auc": "ROC-AUC", "cv_score": "CV F1"}
    metric = st.selectbox("Rank models by", list(metric_labels),
                          format_func=lambda m: metric_labels[m])

    ranked = sorted(MODEL_ROWS, key=lambda m: m.get(metric) or 0, reverse=True)
    leader = ranked[0]
    cards = st.columns(4, gap="medium")
    with cards[0]:
        ui.kpi("Dataset", "UCI Bank Marketing",
               f"{CONVERSION['dataset']['rows']:,} rows · target y",
               "database", "lavender")
    with cards[1]:
        ui.kpi(f"Best {metric_labels[metric]}", _fmt(leader.get(metric), "{:.3f}"),
               _pretty(leader["algorithm"]), "target", "green")
    with cards[2]:
        ui.kpi("Test-set size", f"{CONVERSION['test_rows']:,}",
               f"Stratified {CONVERSION['test_size']:.0%} hold-out",
               "layers", "blue")
    with cards[3]:
        ui.kpi("Cross-validation", "5 folds",
               f"GridSearchCV · mean CV F1 {leader['cv_score']:.3f}",
               "chart", "orange")

    st.write("")
    left, right = st.columns([1.22, 1], gap="large")
    with left:
        with st.container(border=True):
            ui.section("Comparison table", f"Ranked by {metric_labels[metric]}")
            table = pd.DataFrame(MODEL_ROWS)[
                ["algorithm", "accuracy", "precision", "recall", "f1",
                 "macro_f1", "roc_auc", "cv_score", "majority_baseline"]
            ].copy()
            table["algorithm"] = table["algorithm"].map(_pretty)
            table = table.sort_values(
                "algorithm", key=lambda s: s.map(
                    {_pretty(m["algorithm"]): i
                     for i, m in enumerate(ranked)})
            )
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={
                    "algorithm": "Algorithm", "accuracy": "Accuracy",
                    "precision": "Precision", "recall": "Recall",
                    "f1": "F1", "macro_f1": "Macro-F1",
                    "roc_auc": "ROC-AUC", "cv_score": "CV F1",
                    "majority_baseline": "Baseline acc.",
                },
            )
            ui.note(f"Selection rule: {CONVERSION['selection_rule']}. "
                    f"Majority-class baseline accuracy: "
                    f"{CONVERSION['majority_baseline']:.1%}. CV F1 is the mean of "
                    "5 folds on the training split.")

    with right:
        with st.container(border=True):
            ui.section("Metric comparison", "Test split · higher is better")
            ui.show(ui.grouped_series(
                [_pretty(m["algorithm"]) for m in MODEL_ROWS],
                {
                    "F1": [m["f1"] for m in MODEL_ROWS],
                    "Recall": [m["recall"] for m in MODEL_ROWS],
                    "Precision": [m["precision"] for m in MODEL_ROWS],
                    "Accuracy": [m["accuracy"] for m in MODEL_ROWS],
                },
                colors=[ui.C["purple_deep"], ui.C["orange"],
                        ui.C["blue"], ui.C["lavender"]],
                width=5.4, height=3.1,
            ))

    st.write("")
    ui.section("Confusion matrices", "Rows = actual · columns = predicted")
    grid = st.columns(2)
    for i, model in enumerate(MODEL_ROWS):
        with grid[i % 2], st.container(border=True):
            ui.show(ui.confusion(model["confusion_matrix"],
                                 _pretty(model["algorithm"]),
                                 width=4.4, height=2.5))
            ui.note(f"F1 {model['f1']:.3f} · recall {model['recall']:.3f} · "
                    f"precision {model['precision']:.3f} · "
                    f"accuracy {model['accuracy']:.3f}")

    st.write("")
    best_acc = max(MODEL_ROWS, key=lambda m: m["accuracy"])
    best_rec = max(MODEL_ROWS, key=lambda m: m["recall"])
    knn = next(m for m in MODEL_ROWS if m["algorithm"] == "KNN")
    with st.container(border=True):
        ui.section("Interpretation", "What the numbers actually say")
        st.markdown(
            f"- **{_pretty(leader['algorithm'])}** ranks first on "
            f"{metric_labels[metric]} ({_fmt(leader.get(metric), '{:.3f}')}).\n"
            f"- **{_pretty(best_acc['algorithm'])}** has the highest accuracy "
            f"({best_acc['accuracy']:.3f}) but recall only "
            f"{best_acc['recall']:.3f}; it finds fewer subscribers than "
            f"{_pretty(best_rec['algorithm'])} ({best_rec['recall']:.3f} recall).\n"
            f"- KNN is the weakest at finding subscribers (recall "
            f"{knn['recall']:.3f}, F1 {knn['f1']:.3f}) — high accuracy comes from "
            "the 88.3% majority class.\n"
            f"- Every model is below the {CONVERSION['majority_baseline']:.1%} "
            "majority baseline on accuracy, which is expected when the positive "
            "class is protected by class weighting; F1 and recall are the useful "
            "headline metrics here."
        )
        ui.note("All four models share the same target, stratified split, "
                "preprocessing and sample. Test metrics are reported separately "
                "from cross-validation scores.")

    st.write("")
    if EXPERIMENTS:
        exp_left, exp_right = st.columns(2, gap="large")
        with exp_left:
            with st.container(border=True):
                best_k = EXPERIMENTS["k_sweep_best"]
                ui.section("KNN K sweep", "5-fold CV · uniform weights, euclidean")
                ui.show(ui.lines(
                    [r["k"] for r in EXPERIMENTS["k_sweep"]],
                    {"Accuracy": [r["cv_accuracy"] for r in EXPERIMENTS["k_sweep"]],
                     "F1": [r["cv_f1"] for r in EXPERIMENTS["k_sweep"]]},
                    colors=[ui.C["blue"], ui.C["purple_deep"]],
                    width=5.6, height=2.6,
                    vline=(best_k["k"], f"best F1 k={best_k['k']}"),
                ))
                ui.note("The curve is flat; k = 1 is marginally highest, well "
                        "inside fold noise. The main grid searched 3–15 and chose "
                        f"k = {next(m['best_params']['n_neighbors'] for m in MODEL_ROWS if m['algorithm'] == 'KNN')}. "
                        "No single k is statistically special.")
        with exp_right:
            with st.container(border=True):
                ui.section("Test-size experiment", "Fixed hyperparameters · seed 42")
                sizes = ["0.3", "0.2", "0.1"]
                rows = EXPERIMENTS["test_sizes"]
                series = {}
                for model in ["KNN", "GaussianNB", "LinearSVM", "RBFSVM"]:
                    series[model] = [
                        next(r["f1"] for r in rows
                             if r["model"] == model
                             and str(r["test_size"]) == size)
                        for size in sizes
                    ]
                ui.show(ui.grouped_bars(sizes, series, width=5.6, height=2.6))
                ui.note("A 0.1 split leaves 1,200 test rows, so differences of a "
                        "few thousandths are noise. 0.2 remains the balanced "
                        "default, not the highest score.")
    else:
        ui.empty("Experiments pending",
                 "Run <code>python3 -m src.experiments</code>.")

    with st.expander("GridSearchCV details and chosen hyperparameters"):
        for model in MODEL_ROWS:
            st.markdown(f"**{_pretty(model['algorithm'])}** — chosen "
                        f"`{model['best_params']}`")
            grid = pd.DataFrame(model["grid_top"])
            grid["params"] = grid["params"].astype(str)
            st.dataframe(grid, hide_index=True, width="stretch")


# ---------------------------------------------------------------- segmentation ---

def page_segments():
    ui.page_header(
        "Customer Segmentation",
        "K-Means and Ward hierarchical clustering on bank customer profiles.",
    )
    if not CLUSTERS:
        ui.empty("No segmentation available",
                 "Run <code>python3 -m src.clustering</code>.")
        return

    cards = st.columns(4, gap="medium")
    with cards[0]:
        ui.kpi("Customers", f"{CLUSTERS['rows']:,}",
               "Bank customer profiles", "users", "lavender")
    with cards[1]:
        ui.kpi("Selected clusters", f"k = {CLUSTERS['k']}",
               "Highest silhouette · k = 2…10", "layers", "blue")
    with cards[2]:
        ui.kpi("Silhouette", f"{CLUSTERS['silhouette']:.3f}",
               "Low score → coarse portraits", "chart", "orange")
    with cards[3]:
        ui.kpi("Ward agreement", f"{CLUSTERS['ari_kmeans_vs_hierarchical']:.3f}",
               "Adjusted Rand index (K-Means vs Ward)", "target", "green")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            ui.section("Elbow method", "Within-cluster sum of squares")
            ui.show(ui.lines(
                [r["k"] for r in CLUSTERS["elbow"]],
                {"WCSS": [r["inertia"] for r in CLUSTERS["elbow"]]},
                colors=[ui.C["purple_deep"]], width=5.6, height=2.6,
                vline=(CLUSTERS["k"], f"chosen k={CLUSTERS['k']}"),
            ))
    with right:
        with st.container(border=True):
            ui.section("Silhouette by k", "Higher is better · k chosen here")
            ui.show(ui.lines(
                [r["k"] for r in CLUSTERS["silhouette_by_k"]],
                {"Silhouette": [r["score"] for r in CLUSTERS["silhouette_by_k"]]},
                colors=[ui.C["purple"]], width=5.6, height=2.6,
                vline=(CLUSTERS["k"], f"chosen k={CLUSTERS['k']}"),
            ))

    st.write("")
    left, right = st.columns(2, gap="large")
    profiles = pd.DataFrame(CLUSTERS["profiles"])
    with left:
        with st.container(border=True):
            ui.section("Cluster sizes", "Customers per segment")
            ui.show(ui.h_bars(
                [f"Segment {r.cluster}" for r in profiles.itertuples()],
                profiles["customers"].astype(float).tolist(),
                color=[ui.C["purple_deep"], ui.C["purple"], ui.C["lavender"],
                       ui.C["blue"], ui.C["pink"]],
                width=5.4, height=2.4, fmt="{:,.0f}",
            ))
    with right:
        with st.container(border=True):
            ui.section("Dendrogram", "Ward linkage · 300-customer sample")
            path = FIGURES / "dendrogram.png"
            if path.exists():
                st.image(path, width="stretch")
            else:
                ui.empty("Figure missing", "Re-run <code>python3 -m src.clustering</code>.")

    st.write("")
    left, right = st.columns(2, gap="large")
    for col, name, title, sub in [
        (left, "kmeans_pca.png", "K-Means segments", "PCA view · full sample"),
        (right, "hierarchical_pca.png", "Hierarchical segments",
         "PCA view · Ward sample"),
    ]:
        with col:
            with st.container(border=True):
                ui.section(title, sub)
                path = FIGURES / name
                if path.exists():
                    st.image(path, width="stretch")
                else:
                    ui.empty("Figure missing",
                             "Re-run <code>python3 -m src.clustering</code>.")

    st.write("")
    with st.container(border=True):
        ui.section("Cluster profiles", "Feature summaries and observed outcome")
        st.dataframe(profiles, hide_index=True, width="stretch")
        ui.note(f"Tier is relative to the {CLUSTERS['base_conversion_rate']:.1%} "
                "cohort subscription rate: hot ≥ 2×, warm ≥ 1×, cold below. "
                "The target was used only to describe clusters, never to form them.")
        for row in CLUSTERS["profiles"]:
            st.markdown(
                f"- **Segment {row['cluster']}** — {row['customers']:,} customers "
                f"({row['share']:.1%}), observed subscription "
                f"{row['conversion_rate']:.1%} vs "
                f"{CLUSTERS['base_conversion_rate']:.1%} overall → "
                f"**{row['tier']}**. "
                f"Most common job: {row.get('job', 'n/a')}; education: "
                f"{row.get('education', 'n/a')}; previous outcome: "
                f"{row.get('poutcome', 'n/a')}."
            )
        ui.note("The silhouette is low and the curve is flat: treat these as "
                "coarse portraits, not well-separated groups.")


# --------------------------------------------------------------- dataset page ---

def page_dataset():
    ui.page_header(
        "Dataset & Training History",
        "Sources, preprocessing, hyperparameters and experiment results.",
    )
    if not CONVERSION:
        ui.empty("No training history",
                 "Run <code>python3 -m src.train_conversion</code>.")
        return

    cards = st.columns(4, gap="medium")
    with cards[0]:
        ui.kpi("Dataset", "Bank Marketing",
               CONVERSION["dataset"]["file"].split("/")[-1], "database", "lavender")
    with cards[1]:
        ui.kpi("Rows × columns",
               f"{CONVERSION['dataset']['rows']:,} × "
               f"{CONVERSION['dataset']['columns_available']}",
               "Semicolon-separated official UCI file", "layers", "blue")
    with cards[2]:
        ui.kpi("Positive class", _pct(CONVERSION["positive_rate"]),
               "y = yes · imbalanced target", "target", "green")
    with cards[3]:
        ui.kpi("Training run", str(CONVERSION["created"])[:10],
               f"Best: {_pretty(CONVERSION['best_model'])}", "clock", "orange")

    st.write("")
    tab_data, tab_prep, tab_train, tab_exp = st.tabs(
        ["Datasets", "Preprocessing", "Training & metrics", "Experiments"]
    )

    with tab_data:
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            ui.section("Conversion dataset", "Used for the four-algorithm study")
            ui.stat_rows([
                ("Name", CONVERSION["dataset"]["name"]),
                ("Source", CONVERSION["dataset"]["source"]),
                ("Rows", f"{CONVERSION['dataset']['rows']:,}"),
                ("Columns", CONVERSION["dataset"]["columns_available"]),
                ("Target", CONVERSION["target"]),
                ("Duplicate rows", CONVERSION["dataset"]["duplicate_rows"]),
                ("Missing values", "None in the file; `unknown` is a category"),
            ])
            st.markdown(f"[Open the official dataset]"
                        f"({CONVERSION['dataset']['source']})")
        with right, st.container(border=True):
            ui.section("Class distribution", "Target y")
            pos = float(CONVERSION["positive_rate"])
            ui.show(ui.donut(
                ["No", "Yes"], [1 - pos, pos],
                [ui.C["lavender"], ui.C["purple_deep"]],
                _pct(pos), "subscribed", width=4.2, height=2.6,
            ))
            ui.note("The imbalance is why the model is selected on positive-class "
                    "F1 and why accuracy alone is reported next to a majority "
                    "baseline.")

        with st.container(border=True):
            ui.section("Campaign dataset", "Used for descriptive campaign analytics")
            if ADS is not None:
                ui.stat_rows([
                    ("Name", ADS_INFO["name"]),
                    ("Source", ADS_INFO["source"]),
                    ("Rows raw / used",
                     f"{ADS_INFO['rows_raw']} / {ADS_INFO['rows_used']}"),
                    ("Limitation", ADS_INFO["note"]),
                ])
                st.markdown(f"[Open the Kaggle dataset]({ADS_INFO['source']})")
            else:
                ui.empty("Campaign dataset not found", ADS_INFO.get("error", ""))

    with tab_prep:
        with st.container(border=True):
            ui.section("Preprocessing pipeline", "Fitted on the training split only")
            st.markdown(
                "- One stratified sample of **12,000 rows** (seed 42) shared by "
                "all four algorithms; RBF SVC is roughly O(n²) on 45,211 rows.\n"
                "- Numeric: median imputation + `StandardScaler`.\n"
                "- Categorical: most-frequent imputation + one-hot encoding "
                "(`handle_unknown='ignore'`).\n"
                "- `pdays = -1` is split into a contact flag and a nullable "
                "recency value.\n"
                "- Stratified hold-out: **20% test**, seed 42.\n"
                "- Class imbalance documented (11.7% positive) and handled with "
                "`class_weight='balanced'` in the SVM grids.\n"
                "- `duration` is excluded because it is known only after the "
                "call; a hard guard fails training if it reaches the features."
            )
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            ui.section("Features used", f"{len(CONVERSION['features'])} columns")
            st.markdown(", ".join(s["name"] for s in CONVERSION["features"]))
        with right, st.container(border=True):
            ui.section("Features excluded", "Leakage and target")
            st.markdown("\n".join(f"- {f}" for f in CONVERSION["features_excluded"]))
            ui.note("`duration` separates subscribers almost perfectly but is "
                    "unavailable before the call.")

    with tab_train:
        left, right = st.columns([1.2, 1], gap="large")
        with left, st.container(border=True):
            ui.section("Chosen hyperparameters", "GridSearchCV · 5-fold CV")
            st.dataframe(
                pd.DataFrame([
                    {"Algorithm": _pretty(m["algorithm"]),
                     "Chosen parameters": str(m["best_params"]),
                     "CV F1": m["cv_score"]}
                    for m in MODEL_ROWS
                ]),
                hide_index=True, width="stretch",
            )
        with right, st.container(border=True):
            ui.section("Run details", "Provenance")
            ui.stat_rows([
                ("Trained", CONVERSION["created"]),
                ("Train rows", f"{CONVERSION['train_rows']:,}"),
                ("Test rows", f"{CONVERSION['test_rows']:,}"),
                ("Best model", _pretty(CONVERSION["best_model"])),
                ("Selection", CONVERSION["selection_rule"]),
                ("Artifact", "models/conversion.joblib"),
            ])
            st.markdown(f"{ui.badge('Completed', 'green')} "
                        f"{ui.badge('Stratified', 'lavender')} "
                        f"{ui.badge('Leakage-checked', 'blue')}",
                        unsafe_allow_html=True)

        with st.container(border=True):
            ui.section("Test metrics", "Held-out split · separate from CV")
            st.dataframe(
                pd.DataFrame(MODEL_ROWS)[
                    ["algorithm", "accuracy", "precision", "recall", "f1",
                     "macro_f1", "roc_auc", "cv_score", "majority_baseline"]
                ].assign(algorithm=lambda d: d["algorithm"].map(_pretty)),
                hide_index=True, width="stretch",
            )

        with st.expander("GridSearchCV top candidates"):
            for model in MODEL_ROWS:
                st.markdown(f"**{_pretty(model['algorithm'])}**")
                grid = pd.DataFrame(model["grid_top"])
                grid["params"] = grid["params"].astype(str)
                st.dataframe(grid, hide_index=True, width="stretch")

    with tab_exp:
        if not EXPERIMENTS:
            ui.empty("Experiments pending",
                     "Run <code>python3 -m src.experiments</code>.")
        else:
            folds_badge = ui.badge(
                f"{EXPERIMENTS['cv_folds']}-fold CV", "lavender")
            st.markdown(f"{ui.badge('Completed', 'green')}  "
                        f"{folds_badge}  "
                        f"{ui.badge('Seed 42', 'blue')}",
                        unsafe_allow_html=True)
            left, right = st.columns(2, gap="large")
            with left, st.container(border=True):
                ui.section("KNN K sweep", "K = 1…15")
                ui.show(ui.lines(
                    [r["k"] for r in EXPERIMENTS["k_sweep"]],
                    {"Accuracy": [r["cv_accuracy"] for r in EXPERIMENTS["k_sweep"]],
                     "F1": [r["cv_f1"] for r in EXPERIMENTS["k_sweep"]]},
                    colors=[ui.C["blue"], ui.C["purple_deep"]],
                    width=5.6, height=2.6,
                    vline=(EXPERIMENTS["k_sweep_best"]["k"],
                           f"best k={EXPERIMENTS['k_sweep_best']['k']}"),
                ))
                st.dataframe(pd.DataFrame(EXPERIMENTS["k_sweep"]),
                             hide_index=True, width="stretch", height=210)
            with right, st.container(border=True):
                ui.section("Test-size experiment", "0.3 / 0.2 / 0.1")
                ui.show(ui.grouped_bars(
                    ["0.3", "0.2", "0.1"],
                    {m: [next(r["f1"] for r in EXPERIMENTS["test_sizes"]
                              if r["model"] == m
                              and str(r["test_size"]) == s)
                         for s in ("0.3", "0.2", "0.1")]
                     for m in ["KNN", "GaussianNB", "LinearSVM", "RBFSVM"]},
                    width=5.6, height=2.6,
                ))
                st.dataframe(pd.DataFrame(EXPERIMENTS["test_sizes"]),
                             hide_index=True, width="stretch", height=210)
            ui.note(EXPERIMENTS["test_size_note"])


# ------------------------------------------------------------------ history ---

def page_history():
    ui.page_header(
        "Prediction History",
        "Search saved customer scores and campaign analyses; export either table.",
    )
    preds, campaigns = _history_counts()
    tab_pred, tab_camp = st.tabs(["Customer predictions", "Campaign analyses"])

    with tab_pred:
        if preds.empty:
            ui.empty("No customer predictions",
                     "Score a customer in Lead Conversion Prediction; every "
                     "prediction is saved automatically.")
        else:
            preds = preds.copy()
            preds["created_dt"] = pd.to_datetime(preds["created"], errors="coerce")
            f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.3])
            query = f1.text_input("Search", placeholder="model, priority, timestamp",
                                  key="pred_search")
            model_filter = f2.selectbox(
                "Algorithm", ["All"] + sorted(preds["model"].unique()),
                key="pred_model")
            class_filter = f3.selectbox(
                "Predicted class",
                ["All", "Subscribed", "Not subscribed"], key="pred_class")
            dates = preds["created_dt"].dropna()
            if len(dates):
                picked = f4.date_input(
                    "Date range",
                    value=(dates.min().date(), dates.max().date()),
                    key="pred_dates")
            else:
                picked = None

            view = preds.copy()
            view["predicted class"] = view["prediction"].map(
                {1: "Subscribed", 0: "Not subscribed"})
            if query:
                view = view[view.apply(
                    lambda r: query.lower() in " ".join(
                        str(r[c]) for c in ("model", "priority", "created",
                                            "predicted class")).lower(), axis=1)]
            if model_filter != "All":
                view = view[view["model"] == model_filter]
            if class_filter != "All":
                view = view[view["predicted class"] == class_filter]
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
                view = view[(view["created_dt"] >= start)
                            & (view["created_dt"] <= end + pd.Timedelta(days=1))]

            st.download_button(
                "Export CSV", view.drop(columns=["created_dt"]).to_csv(
                    index=False).encode("utf-8"),
                file_name="customer_predictions.csv", mime="text/csv",
                key="download_preds",
            )
            st.dataframe(
                view[["id", "created", "model", "predicted class",
                      "probability", "priority", "actual"]].rename(columns={
                          "id": "ID", "created": "Timestamp", "model": "Algorithm",
                          "probability": "Probability", "priority": "Priority",
                          "actual": "Actual"}),
                hide_index=True, width="stretch",
                column_config={
                    "Probability": st.column_config.NumberColumn(format="%.4f"),
                    "Actual": st.column_config.NumberColumn(format="%d"),
                },
            )
            ui.note("Edit actual outcomes below when they become known.")
            editor = view[["id", "actual"]].copy()
            editor["actual"] = editor["actual"].map({1: "yes", 0: "no"}).fillna("")
            edited = st.data_editor(
                editor, hide_index=True, width="stretch", key="actuals_editor",
                column_config={
                    "id": st.column_config.NumberColumn(disabled=True),
                    "actual": st.column_config.SelectboxColumn(
                        "Actual outcome", options=["", "yes", "no"]),
                },
            )
            if st.button("Save actual outcomes"):
                before = dict(zip(editor["id"], editor["actual"]))
                changed = 0
                for row in edited.itertuples():
                    new = "" if pd.isna(row.actual) else row.actual
                    if before.get(row.id, "") != new:
                        set_actual(int(row.id),
                                   None if new == "" else int(new == "yes"))
                        changed += 1
                st.success(f"Updated {changed} record(s).")
                st.rerun()

            if view.empty:
                ui.empty("No matching predictions",
                         "Adjust the search, algorithm, class or date filters.")
            else:
                record_id = st.selectbox("Inspect a record",
                                         view["id"].tolist(), key="pred_inspect")
                record = get_prediction(int(record_id))
                if record:
                    st.json({k: record[k] for k in
                             ("created", "dataset", "model", "prediction",
                              "probability", "decision_score", "priority",
                              "actual", "features")})

    with tab_camp:
        if campaigns.empty:
            ui.empty("No campaign analyses",
                     "Run a campaign benchmark and tick Save this analysis.")
        else:
            summary = []
            for row in campaigns.itertuples():
                ctx = json.loads(row.context)
                res = json.loads(row.result)
                estimate = (res.get("estimated_cpa") or res.get("estimated_cpc")
                            or res.get("estimated_cpm"))
                summary.append({
                    "id": row.id,
                    "created": row.created,
                    "objective": row.objective,
                    "brand": ctx.get("brand", ""),
                    "campaign": ctx.get("campaign_name", ""),
                    "budget": ctx.get("budget"),
                    "target_mode": ctx.get("target_mode", ""),
                    "estimate": estimate,
                })
            cdf = pd.DataFrame(summary)
            f1, f2 = st.columns([1.6, 1])
            query = f1.text_input("Search", placeholder="objective, brand, campaign",
                                  key="camp_search")
            objective = f2.selectbox(
                "Objective", ["All"] + sorted(cdf["objective"].unique()),
                key="camp_objective")
            view = cdf.copy()
            if query:
                view = view[view.apply(
                    lambda r: query.lower() in
                    f"{r.objective} {r.brand} {r.campaign}".lower(), axis=1)]
            if objective != "All":
                view = view[view["objective"] == objective]
            st.download_button(
                "Export CSV", view.to_csv(index=False).encode("utf-8"),
                file_name="campaign_analyses.csv", mime="text/csv",
                key="download_camps",
            )
            st.dataframe(view, hide_index=True, width="stretch")
            if view.empty:
                ui.empty("No matching analyses",
                         "Adjust the search or objective filter.")
            else:
                pick = st.selectbox("Inspect an analysis", view["id"].tolist(),
                                    key="camp_inspect")
                record = get_campaign(int(pick))
                if record:
                    st.json({
                        "created": record["created"],
                        "objective": record["objective"],
                        "context": record["context"],
                        "result": record["result"],
                        "methodology": record["methodology"],
                    })


# ---------------------------------------------------------------- settings/about ---

def page_settings():
    ui.page_header(
        "Settings",
        "Local configuration, design tokens and storage diagnostics.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            ui.section("System status", "Files read by this demo")
            ui.stat_rows([
                ("Conversion model", _badge_state((MODELS / "conversion.joblib").exists())),
                ("Training metadata", _badge_state((MODELS / "conversion.json").exists())),
                ("Experiments", _badge_state(EXPERIMENTS is not None)),
                ("Segmentation", _badge_state(CLUSTERS is not None)),
                ("Campaign dataset", _badge_state(ADS is not None)),
                ("History database", _badge_state(HISTORY_DB.exists())),
            ])

        with st.container(border=True):
            ui.section("Storage", "Relative paths inside the project")
            ui.stat_rows([
                ("Models", "models/"),
                ("Figures", "reports/figures/"),
                ("History DB", "data/history.db"),
                ("Raw data", "data/raw/ (gitignored)"),
            ])
            if HISTORY_DB.exists():
                st.download_button(
                    "Download history database",
                    HISTORY_DB.read_bytes(),
                    file_name="history.db",
                    mime="application/x-sqlite3",
                )
            st.button("Clear cached data", on_click=st.cache_data.clear)

    with right:
        with st.container(border=True):
            ui.section("Design tokens", "Soft lavender SaaS palette")
            swatches = [
                ("Deep purple", ui.C["purple_deep"]),
                ("Lavender", ui.C["purple"]),
                ("Soft lavender", ui.C["lavender"]),
                ("Orange", ui.C["orange"]),
                ("Green", ui.C["green"]),
                ("Blue", ui.C["blue"]),
                ("Pink", ui.C["pink"]),
                ("Border", ui.C["border"]),
            ]
            for row in [swatches[:4], swatches[4:]]:
                cols = st.columns(4)
                for col, (name, hex_value) in zip(cols, row):
                    col.markdown(
                        f'<div class="mm-swatch" style="background:{hex_value};'
                        f'border:1px solid {ui.C["border"]}"></div>'
                        f'<div class="mm-note" style="margin-top:4px">'
                        f'{name}<br>{hex_value}</div>',
                        unsafe_allow_html=True,
                    )
            ui.note("The same values are used by the matplotlib charts, so pages "
                    "and figures stay consistent.")

        with st.container(border=True):
            ui.section("Environment", "No secrets or API keys are required")
            ui.stat_rows([
                ("Runtime", "Streamlit · local"),
                ("Charts", "Matplotlib"),
                ("Database", "SQLite (stdlib)"),
                ("Random seed", "42"),
            ])
            ui.note("Meta credentials are not used; the demo runs on the two "
                    "committed datasets.")


def page_about():
    ui.page_header(
        "About",
        "What MarketMind AI does, which data it uses and what it cannot do.",
    )
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            ui.section("Project", "Machine-learning CRM marketing intelligence")
            st.markdown(
                "MarketMind AI combines three modules in one Streamlit app:\n\n"
                "1. **Lead conversion prediction** — KNN, Gaussian Naive Bayes, "
                "Linear SVM and RBF SVM on the UCI Bank Marketing dataset, with "
                "one shared target, split and preprocessing pipeline.\n"
                "2. **Campaign intelligence** — descriptive analysis of real "
                "Facebook ads plus a nearest-neighbour historical benchmark for "
                "planning. No supervised CPA model is claimed because the measured "
                "regression R² is negative.\n"
                "3. **Customer segmentation** — K-Means and Ward hierarchical "
                "clustering, with k chosen by the highest silhouette score."
            )
        with st.container(border=True):
            ui.section("Datasets", "Both are real; they are never joined")
            st.markdown(
                "- **UCI Bank Marketing** (`bank-full.csv`, 45,211 × 17): target "
                "`y`; `duration` excluded because it is only known after the call. "
                "[Official source](https://archive.ics.uci.edu/dataset/222/bank+marketing)\n"
                "- **Kaggle clicks-conversion-tracking**: anonymised Facebook ads, "
                "1,143 rows × 11 columns, one row per ad. It has no budget, "
                "duration, location, creative, CTA or revenue columns. "
                "[Dataset page](https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking)"
            )

    with right:
        with st.container(border=True):
            ui.section("Method notes", "What is validated")
            st.markdown(
                "- Model selection uses **positive-class F1**; accuracy alone is "
                "misleading at an 11.7% positive rate.\n"
                "- SVC probabilities are Platt-scaled; KNN probabilities are "
                "neighbour frequencies; Gaussian NB is often overconfident.\n"
                "- Campaign outputs are empirical benchmarks from the 25 nearest "
                "historical ads, not model predictions.\n"
                "- Clusters are formed without the target; the target only "
                "describes them afterwards."
            )
        with st.container(border=True):
            ui.section("Limitations", "Read before quoting a number")
            st.markdown(
                "- RBF SVC is O(n²), so the study uses a stratified 12k sample.\n"
                "- Best positive-class F1 is 0.42: a screening aid, not an "
                "automated decision.\n"
                "- The ad dataset is one advertiser and one platform; CPA is not "
                "reliably predictable from its planning fields.\n"
                "- Segmentation is weakly separated (silhouette ≈ 0.38).\n"
                "- Actual outcomes exist only when a user enters them."
            )
        ui.note("Built for a college ML demonstration. Every number on every page "
                "comes from models/, reports/figures/ or the SQLite history.")


# -------------------------------------------------------------------- router ---

PAGES = {
    "Dashboard": page_dashboard,
    "Campaign Intelligence": page_campaign,
    "Lead Conversion Prediction": page_convert,
    "Customer Segmentation": page_segments,
    "Model Comparison": page_models,
    "Dataset & Training History": page_dataset,
    "Prediction History": page_history,
    "Settings": page_settings,
    "About": page_about,
}

PAGES.get(st.session_state.page, page_dashboard)()

st.markdown('<div class="mm-divider"></div>', unsafe_allow_html=True)
st.caption("MarketMind AI · CRM Marketing Intelligence · estimates, not "
           "guarantees · data/raw files stay local")
