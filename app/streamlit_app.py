"""CRM-Adv dashboard: campaign score, lead score, model comparison, segments.

    streamlit run app/streamlit_app.py

Every number shown comes from models/*.json + models/*.joblib created by the
training scripts. If those are missing, the tab tells you what to run.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BANDS, FIGURES, MODELS  # noqa: E402
from src.predict import load_meta, load_model, priority, score  # noqa: E402

st.set_page_config(page_title="CRM-Adv", page_icon="📈", layout="wide")

DISCLAIMER = (
    "ML estimate from historical data — not a guarantee. "
    "Predictions are only as good as the data the model was trained on."
)

st.title("CRM-Adv — marketing & sales intelligence")
st.caption("Stage 1: campaign performance · Stage 2: lead conversion · Segmentation")

tab_campaign, tab_lead, tab_models, tab_segments = st.tabs(
    ["Campaign prediction", "Lead scoring", "Model comparison", "Segmentation"]
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


def _result(result):
    left, right = st.columns(2)
    left.metric("Probability", f"{result['probability']:.1%}")
    right.metric("Priority", result["priority"])
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.caption(DISCLAIMER)


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
        if st.button("Predict campaign performance", type="primary"):
            try:
                _result(score("stage1", payload))
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

with tab_models:
    st.subheader("Model comparison")
    st.caption("Selection rule: highest test macro-F1; recall breaks ties. "
               f"Priority bands: High ≥ {BANDS['High']:.0%}, Medium ≥ {BANDS['Medium']:.0%}.")
    for stage, label in [("stage1", "Stage 1 — campaign performance"),
                         ("stage2", "Stage 2 — lead conversion")]:
        meta = load_meta(stage)
        st.markdown(f"### {label}")
        if meta is None:
            st.info(f"Not trained yet. Run the Stage {'1' if stage == 'stage1' else '2'} training script.")
            continue
        table = pd.DataFrame(meta["models"])[
            ["algorithm", "accuracy", "precision", "recall", "f1",
             "macro_f1", "roc_auc", "cv_score"]
        ]
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(f"Majority-class baseline accuracy: {meta['majority_baseline']:.1%} — "
                   "any accuracy close to this means the model has learned nothing.")
        st.success(f"Best model: **{meta['best_model']}**  "
                   f"(macro-F1 {max(m['macro_f1'] for m in meta['models']):.3f})")
        with st.expander("Confusion matrices"):
            cols = st.columns(4)
            for col, model in zip(cols, meta["models"]):
                path = FIGURES / f"{stage}_{model['algorithm'].replace(' ', '_')}.png"
                col.image(str(path), caption=model["algorithm"]) if path.exists() else None
        with st.expander("Chosen hyperparameters"):
            st.json({m["algorithm"]: m["best_params"] for m in meta["models"]})

with tab_segments:
    path = MODELS / "clusters.json"
    if not path.exists():
        st.info("No segments yet. Run `python -m src.clustering`.")
    else:
        import json
        clusters = json.loads(path.read_text())
        st.write(f"**k = {clusters['k']}** · silhouette = {clusters['silhouette']} · "
                 f"best silhouette {clusters.get('silhouette_best')} at "
                 f"k={clusters.get('k_best_silhouette')} · "
                 f"ARI (K-Means vs Ward) = {clusters['ari_kmeans_vs_hierarchical']} · "
                 f"{clusters['rows']} leads")
        st.caption("k is the smallest value within 10% of the best silhouette, so "
                   "extra segments must earn their keep. Flat curves mean the "
                   "clusters are weak — treat segments as rough portraits.")
        st.info(DISCLAIMER)

        st.dataframe(pd.DataFrame(clusters["profiles"]), hide_index=True,
                     use_container_width=True)
        st.caption(
            "Tier from the observed conversion rate, not from the clustering: "
            "hot ≥ 50%, warm ≥ 30%, cold below. Action: hot — call within 24h; "
            "warm — nurture and re-check in 3 days; cold — automated email only."
        )

        left, right = st.columns(2)
        for col, name, caption in [
            (left, "elbow.png", "Elbow method"),
            (right, "silhouette.png", "Silhouette by k"),
        ]:
            p = FIGURES / name
            if p.exists():
                col.image(str(p), caption=caption)

        left, right = st.columns(2)
        for col, name, caption in [
            (left, "kmeans_pca.png", "K-Means segments"),
            (right, "hierarchical_pca.png", "Hierarchical (Ward) segments"),
        ]:
            p = FIGURES / name
            if p.exists():
                col.image(str(p), caption=caption)

        p = FIGURES / "dendrogram.png"
        if p.exists():
            st.image(str(p), caption="Dendrogram (Ward, truncated)")

st.divider()
st.caption(
    "Prediction times — Stage 1: before launch, planning fields only. "
    "Stage 2: at lead capture, before sales follow-up. "
    "Post-conversion and post-campaign fields are excluded by a hard leakage check."
)
