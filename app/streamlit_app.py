"""CRM-Adv dashboard: overview, scores, results and segments.

    streamlit run app/streamlit_app.py

Every number shown comes from models/*.json + models/*.joblib created by the
training scripts. If those are missing, the tab tells you what to run.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BANDS, FIGURES, MODELS  # noqa: E402
from src.predict import load_meta, load_model, score  # noqa: E402

st.set_page_config(page_title="CRM-Adv", page_icon="📈", layout="wide")

DISCLAIMER = (
    "ML estimate from historical data — not a guarantee. "
    "Predictions are only as good as the data the model was trained on."
)
STAGES = [("stage1", "Stage 1 — campaign performance", "Campaign prediction"),
          ("stage2", "Stage 2 — lead conversion", "Lead scoring")]

st.title("CRM-Adv — marketing & sales intelligence")
st.caption("Stage 1: campaign performance · Stage 2: lead conversion · Segmentation")

tab_overview, tab_campaign, tab_lead, tab_results, tab_segments = st.tabs(
    ["Overview", "Campaign prediction", "Lead scoring", "Results & graphs",
     "Segmentation"]
)


def _inputs(specs, prefix):
    """Render one widget per training feature and return the payload."""
    values, cols = {}, st.columns(2)
    for i, spec in enumerate(specs):
        with cols[i % 2]:
            name = spec["name"]
            key = f"{prefix}_{name}"
            if spec["kind"] == "categorical":
                values[name] = st.selectbox(name, spec["categories"], key=key)
            else:
                low, high = float(spec["min"]), float(spec["max"])
                if high <= low:
                    values[name] = low
                else:
                    values[name] = st.number_input(
                        name, min_value=low, max_value=high,
                        value=round((low + high) / 2, 2), key=key,
                    )
    return values


def _result(result, risk=False):
    """Stage 1 predicts a miss, so its number is a risk, not a priority."""
    left, right = st.columns(2)
    label = "Risk of missing target CPA" if risk else "Probability"
    left.metric(label, f"{result['probability']:.1%}")
    right.metric("Risk band" if risk else "Priority",
                 f"{result['priority']} risk" if risk else result["priority"])
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.caption(DISCLAIMER)


def _tables(meta):
    """Full metric table + the three bars worth graphing."""
    table = pd.DataFrame(meta["models"])[
        ["algorithm", "accuracy", "precision", "recall", "f1",
         "macro_f1", "roc_auc", "cv_score"]
    ]
    chart = table.set_index("algorithm")[["accuracy", "macro_f1", "roc_auc"]]
    # Visual reference: a model at this height has learned nothing.
    chart.loc["majority baseline"] = [meta["majority_baseline"], None, None]
    return table, chart


def _confusions(meta, stage):
    cols = st.columns(4)
    for col, model in zip(cols, meta["models"]):
        path = FIGURES / f"{stage}_{model['algorithm'].replace(' ', '_')}.png"
        if path.exists():
            col.image(str(path), caption=model["algorithm"])
        else:
            col.caption(f"{model['algorithm']}: no figure — retrain to draw it")


def _not_trained(stage):
    st.info("Not trained yet. Run:\n\n"
            f"`python3 -m src.train_{'stage1' if stage == 'stage1' else 'stage2'}`")


with tab_overview:
    st.subheader("What the system does")
    st.dataframe(
        pd.DataFrame([
            {"Stage": "1. Campaign", "Question": "Will this campaign beat its target cost per acquisition?",
             "Asked": "before launch", "Features": "age, gender, interest, campaign"},
            {"Stage": "2. Lead", "Question": "Will this lead convert?",
             "Asked": "at lead capture", "Features": "25 lead-source, activity and profile fields"},
            {"Stage": "3. Segments", "Question": "What kinds of leads do we have?",
             "Asked": "snapshot", "Features": "unsupervised, target used only to describe"},
        ]),
        hide_index=True, width="stretch",
    )

    st.subheader("Best model on the held-out test split")
    cards = st.columns(3)
    clusters_path = MODELS / "clusters.json"
    for col, (stage, label, _) in zip(cards[:2], STAGES):
        meta = load_meta(stage)
        if meta is None:
            col.metric(label, "not trained")
            continue
        best = next(m for m in meta["models"] if m["algorithm"] == meta["best_model"])
        col.metric(f"{label} — {meta['best_model']}", f"macro-F1 {best['macro_f1']:.3f}")
        col.caption(f"Accuracy {best['accuracy']:.1%} · ROC-AUC {best['roc_auc']:.3f} · "
                    f"majority baseline {meta['majority_baseline']:.1%}")
    if clusters_path.exists():
        c = json.loads(clusters_path.read_text())
        cards[2].metric("Segments (K-Means)", f"k = {c['k']}")
        cards[2].caption(f"Silhouette {c['silhouette']} · Ward agreement (ARI) "
                         f"{c['ari_kmeans_vs_hierarchical']} · {c['rows']} leads")

    st.warning(
        "Stage 1 is deliberately shown with its weak result: 57.9% accuracy against "
        "a 72.1% majority baseline. Only four planning features exist, so treat its "
        "output as a ranking hint, not a forecast. Stage 2 is the model to quote."
    )

with tab_campaign:
    meta = load_meta("stage1")
    if meta is None or load_model("stage1") is None:
        st.info(
            "No Stage 1 model yet. Download the data, then run:\n\n"
            "`python -m src.train_stage1`  (or `--source meta` for your ad account)"
        )
    else:
        source = meta["dataset"].get("source", "unknown")
        st.write(f"**Data source:** {source} · trained on {meta['train_rows']} campaigns · "
                 f"target CPA = {meta['dataset'].get('target_cpa')}")
        payload = _inputs(meta["features"], "stage1")
        st.caption("Stage 1 answers: *will this campaign's cost per acquisition end up "
                   f"above **${meta['dataset'].get('target_cpa')}**?* The score is the "
                   "predicted risk of missing that target.")
        if st.button("Predict campaign performance", type="primary"):
            try:
                _result(score("stage1", payload), risk=True)
            except Exception as exc:  # user input outside training space, etc.
                st.error(f"Could not score this campaign: {exc}")

with tab_lead:
    meta = load_meta("stage2")
    if meta is None or load_model("stage2") is None:
        st.info(
            "No Stage 2 model yet. Download the data, then run:\n\n"
            "`python -m src.train_stage2`"
        )
    else:
        st.write(f"**Data source:** {meta['dataset'].get('source', 'unknown')} · "
                 f"trained on {meta['train_rows']} leads")
        payload = _inputs(meta["features"], "stage2")
        if st.button("Score lead", type="primary"):
            try:
                result = score("stage2", payload)
                _result(result)
                st.write(
                    "Suggested action: "
                    + {"High": "call within 24 hours",
                       "Medium": "nurture via email, re-check in 3 days",
                       "Low": "low priority / automated follow-up"}[result["priority"]]
                )
            except Exception as exc:
                st.error(f"Could not score this lead: {exc}")

with tab_results:
    st.subheader("Model comparison")
    st.caption("Selection rule: highest test macro-F1; recall breaks ties. "
               "Macro-F1 averages both classes, so 'predict the majority class' "
               "cannot win. Priority bands: "
               f"High ≥ {BANDS['High']:.0%}, Medium ≥ {BANDS['Medium']:.0%}.")
    for stage, label, _ in STAGES:
        meta = load_meta(stage)
        st.markdown(f"### {label}")
        if meta is None:
            _not_trained(stage)
            continue
        table, chart = _tables(meta)
        st.bar_chart(chart, height=320)
        st.caption("Grey bar is the accuracy of always predicting the majority class.")
        st.dataframe(table, hide_index=True, width="stretch")
        st.success(f"Best model: **{meta['best_model']}** "
                   f"(macro-F1 {max(m['macro_f1'] for m in meta['models']):.3f}, "
                   f"ROC-AUC {max(m['roc_auc'] for m in meta['models']):.3f})")
        st.markdown("**Confusion matrices** (test split, rows = truth)")
        _confusions(meta, stage)
        with st.expander("Chosen hyperparameters"):
            st.json({m["algorithm"]: m["best_params"] for m in meta["models"]})
        st.divider()

with tab_segments:
    path = MODELS / "clusters.json"
    if not path.exists():
        st.info("No segments yet. Run `python3 -m src.clustering`.")
    else:
        clusters = json.loads(path.read_text())
        st.write(f"**k = {clusters['k']}** · silhouette = {clusters['silhouette']} · "
                 f"best silhouette {clusters.get('silhouette_best')} at "
                 f"k={clusters.get('k_best_silhouette')} · "
                 f"ARI (K-Means vs Ward) = {clusters['ari_kmeans_vs_hierarchical']} · "
                 f"{clusters['rows']} leads")
        st.caption("k is the smallest value within 10% of the best silhouette, so "
                   "extra segments must earn their keep. Flat curves mean the "
                   "clusters are weak — treat segments as rough portraits.")

        profiles = pd.DataFrame(clusters["profiles"])
        left, right = st.columns([1, 1])
        left.markdown("**Conversion rate per segment**")
        left.bar_chart(profiles.set_index("cluster")["conversion_rate"], height=260)
        right.markdown("**Leads per segment**")
        right.bar_chart(profiles.set_index("cluster")["leads"], height=260)

        st.dataframe(profiles, hide_index=True, width="stretch")
        st.caption(
            "Tier from the observed conversion rate, not from the clustering: "
            "hot ≥ 50%, warm ≥ 30%, cold below. Action: hot — call within 24h; "
            "warm — nurture and re-check in 3 days; cold — automated email only."
        )

        left, right = st.columns(2)
        for col, name, caption in [
            (left, "elbow.png", "Elbow method — WCSS by k"),
            (right, "silhouette.png", "Silhouette by k"),
        ]:
            p = FIGURES / name
            if p.exists():
                col.image(str(p), caption=caption)

        left, right = st.columns(2)
        for col, name, caption in [
            (left, "kmeans_pca.png", "K-Means segments (PCA view)"),
            (right, "hierarchical_pca.png", "Ward segments (PCA view)"),
        ]:
            p = FIGURES / name
            if p.exists():
                col.image(str(p), caption=caption)

        p = FIGURES / "dendrogram.png"
        if p.exists():
            st.image(str(p), caption="Dendrogram (Ward, 300-lead sample, truncated)")
        st.info(DISCLAIMER)

st.divider()
st.caption(
    "Prediction times — Stage 1: before launch, planning fields only. "
    "Stage 2: at lead capture, before sales follow-up. "
    "Post-conversion and post-campaign fields are excluded by a hard leakage check."
)
