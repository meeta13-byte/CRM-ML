# CRM-Adv — ML-based marketing & sales intelligence

A simplified AI-CRM for a college ML project. Two supervised models plus one
clustering stage, built only from the syllabus algorithms:

| Stage | Question | Timing | Algorithm set |
|---|---|---|---|
| 1. Campaign | Will this campaign beat its target cost per acquisition? | **before launch** | KNN, GaussianNB, Linear SVM, RBF SVM |
| 2. Lead | Will this lead convert? | **at lead capture** | same four |
| 3. Segmentation | What kinds of leads do we have? | snapshot | K-Means, Ward hierarchical |

`streamlit run app/streamlit_app.py` turns the three into one dashboard.

---

## 1. Install

```bash
cd ~/Documents/CRM-Adv
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## 2. Data (manual step — 5 minutes)

Two **real** public datasets are used. They are never joined; each stage has its
own model and its own dataset.

### 2.1 Kaggle API token

1. Log in to <https://www.kaggle.com> → click your avatar → **Settings** →
   **Create New Token**. This downloads `kaggle.json`.
2. Install it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

### 2.2 Download both datasets

Open each dataset page once and click **I Understand and Accept** if prompted.

```bash
# Stage 2 — X Education lead scoring (real, 9,240 x 37, target: Converted)
kaggle datasets download -d amritachatterjee09/lead-scoring-dataset -p data/raw --unzip

# Stage 1 — real anonymised Facebook ads (1,143 ads x 11 cols, spend + conversions)
kaggle datasets download -d loveall/clicks-conversion-tracking -p data/raw --unzip
```

Expected files: `data/raw/Lead Scoring.csv` and
`data/raw/KAG_conversion_data.csv`. If the names differ, no problem: the
loaders scan `data/raw` for a CSV that has the required columns (`Converted` +
`Lead Source`, or `spent` + `Approved_Conversion`).

> Stage 1 alternative: your own Meta ad account (§3) instead of the Kaggle file.

## 3. Meta Marketing API (optional, for live data)

Use this only if you have (or can borrow) an ad account with real spend. Reading
your **own** ad accounts needs no App Review.

1. <https://developers.facebook.com> → **My Apps** → **Create App** → type
   **Business**.
2. On the app dashboard, add the **Marketing API** product. The default
   **Limited access** tier is enough for your own account.
3. Make sure you are admin/analyst on the ad account **and** admin on the Page.
4. Business settings → **Users → System users** → create one → **Add assets**
   (ad account + Page) → **Generate token** with:
   `ads_read`, `ads_management`, `leads_retrieval`, `pages_show_list`,
   `pages_read_engagement`, `pages_manage_ads`, `pages_manage_metadata`.
5. Copy `.env.example` to `.env` and fill in:

```bash
META_ACCESS_TOKEN=EAAG...
META_AD_ACCOUNT_ID=act_1234567890
```

6. Sync and train:

```bash
set -a; source .env; set +a
python scripts/sync_meta.py --list-accounts
python scripts/sync_meta.py --since 2024-01-01 --until 2026-09-30
python -m src.train_stage1 --source meta
```

Notes:

- Insights reach back **37 months**; `reach`/`frequency` with breakdowns only
  **13 months**. Lead data is retained ~**90 days**.
- App Review + Business Verification are only needed when you read **someone
  else's** assets, or want Full access (500 calls / 15 days). Not needed here.
- Fewer than ~100 ad sets with spend → the 4-model comparison is unstable. Stay
  on the Kaggle source for the report and use live data for the demo.
- Meta returns **lead features but not the lead's conversion outcome**. That
  label lives in a CRM, so live leads can be scored but not used for training.
  The Stage 2 model is trained on X Education instead.
- Token = secret. `.env` is git-ignored.

## 4. Train

```bash
python3 -m src.train_stage1         # campaign model  (~3 min: SVC grids)
python3 -m src.train_stage2         # lead model     (~25 min: SVC probability=True)
python3 -m src.clustering           # elbow, K-Means, Ward, dendrogram (~1 min)
python3 tests/smoke.py              # 2-second self-check, no real data needed
```

Artifacts:

```
models/stage1.joblib, stage1.json   # best pipeline + every model's metrics
models/stage2.joblib, stage2.json
models/clusters.json
reports/figures/*.png                # confusion matrices, elbow, PCA, dendrogram
```

Trained models are committed, so the dashboard works straight after cloning;
only the datasets are local.

### Measured results (held-out test split)

| Stage | Best model | Accuracy | Macro-F1 | ROC-AUC | Majority baseline |
|---|---|---|---|---|---|
| 1. Campaign (`high_cpa`) | Linear SVM | 0.579 | 0.539 | 0.586 | **0.721** |
| 2. Lead (`Converted`) | RBF SVM | 0.788 | 0.778 | 0.826 | 0.615 |

Segmentation: k = 3, silhouette 0.225, ARI(K-Means vs Ward) 0.554, segment
conversion rates 70% / 34% / 19%.

Stage 2 is the model worth presenting. Stage 1 is below its own majority
baseline — see limitation 1.

## 5. Dashboard

```bash
streamlit run app/streamlit_app.py
```

Four tabs: campaign prediction, lead scoring, model comparison, segmentation.
If a model is missing the tab tells you which command to run.

---

## 6. Stage 1 target — why it is not arbitrary

```
CPA   = spend / primary_results          (0 results => infinite CPA)
label = 1 if CPA > target CPA else 0
```

- **CPA is the metric media buyers actually optimise** and Meta reports directly.
- The **target CPA is a business KPI**: set `TARGET_CPA_OVERRIDE` in
  `src/config.py`, or the default is the cohort median
  (`TARGET_CPA_QUANTILE`), reported in every run's JSON.
- **Cohorts matter**: CPA is compared within the same objective, because lead
  campaigns and traffic campaigns are not comparable.
- **Zero-conversion rows are failures, not dropped.** Dropping them would bias
  the model towards campaigns that already worked.
- **Label sensitivity (measured, not promised).** Five-fold CV ROC-AUC on the
  same 4 planning features, for four candidate labels. The chosen label is not
  lucky; every definition is around 0.6:

  | Candidate label | Positive rate | CV ROC-AUC |
  |---|---|---|
  | `CPA > cohort median` (chosen) | 0.72 | 0.646 |
  | produced any conversion | 0.55 | 0.641 |
  | conversion rate > median | 0.28 | 0.645 |
  | conversions per impression > median | 0.28 | 0.609 |

Rejected targets: CTR/CPC (proxy metrics), raw median split (arbitrary unless
framed as "vs baseline"), and any label derived from the input features.

## 7. Segmentation — how k is chosen

1. Scale + one-hot the lead features (the `Converted` target is never an input).
2. K-Means for k = 2..10, recording WCSS (elbow) and silhouette.
3. Silhouette creeps upward with k, so the raw maximum landed on k = 10 with a
   3-lead cluster. Rule used instead: **the smallest k within 10% of the best
   silhouette** (`SILHOUETTE_TOLERANCE`). On this data that is k = 3.
4. Ward hierarchical on a 2,000-lead sample, same k, compared with
   `adjusted_rand_score` (0.554 = the two methods broadly agree).
5. Profiles show each segment's size, conversion rate and defining fields. Only
   after clustering is the target used, and only to describe segments.

Reported honestly: silhouette ≈ 0.23 is weak-to-moderate, so the segments are
rough portraits, not law. The dendrogram is drawn on a 300-lead sample because
Ward's distance matrix is O(n²).

## 8. Leakage policy

| Stage | Forbidden during training | Why |
|---|---|---|
| 1 | `spend`, `impressions`, `clicks`, `ctr`, `cpc`, `cpm`, `reach`, `actions`, `results`, `cost_per_result`, `conversions`, `cpa` | only exist after launch |
| 1 (Meta) | plus `updated_time`, `effective_status`, `name` | edited/assigned after results are seen |
| 2 | `Tags`, `Lead Quality`, `Last Notable Activity`, IDs, `Converted` | written by sales after follow-up |
| 2 (strict) | `Last Activity`, `Asymmetrique *` | captured at snapshot time, not at form fill |

`src/data/schema.py::assert_no_leakage` fails the run if any of these reach the
feature matrix. Preprocessing is fitted **inside** the pipeline, after the
train/test split, so scaling never sees test data.

For the live Meta path there is one unavoidable caveat: the API returns the
**current** budget/targeting of an ad set, not its config at launch. `duration`,
`start_month`, demographics and objective are stable; budget and targeting may
have been edited. The report must say so.

## 9. Limitations (state these in the viva)

1. **Stage 1 is weak, and this is the honest headline.** With only four
   planning fields (age, gender, interest, campaign) the best model reaches test
   macro-F1 0.539 / ROC-AUC 0.586, and its accuracy (57.9%) is **below the
   72.1% majority-class baseline**. It ranks campaigns slightly better than
   chance; it cannot replace human judgement. Say this in the viva before an
   examiner finds it. Stage 2 (25 lead fields) is the strong model.
2. Stage 1 uses a single-anonymised-advertiser export (3 campaigns, 1,143 ads),
   so it learns audience effects, not full planning strategy. A real ad account
   (`--source meta`) adds budget, duration, objective and placements.
3. The Stage 1 row cap (`MAX_STAGE1_ROWS`, 12,000) never binds here; it exists
   because SVC with `probability=True` is O(n²).
3. Attribution and iOS/ATT under-report conversions → label noise.
4. SVM probabilities are Platt-scaled, KNN probabilities are coarse; "72%" is a
   ranking signal, not a calibrated promise.
5. K-Means and Ward use one-hot + scaling on mixed data (no Gower distance in
   the syllabus), which distorts distances somewhat.
6. No causal claim: a high score does not mean the campaign *caused* a sale.
7. The two datasets come from different domains; the link is the CRM workflow,
   not a synthetic join.

## 10. Structure

```
data/raw/            downloaded datasets (git-ignored)
models/              trained pipelines + metrics.json
src/
  config.py          thresholds, paths, seeds
  data/schema.py     feature contracts + leakage guard
  data/loaders.py    Kaggle / X Education / Meta adapters
  data/preprocess.py split, imputers, scaler, one-hot
  evaluate.py        GridSearchCV, metrics, confusion matrices
  train_common.py    the four algorithms + shared training loop
  train_stage1.py    campaign model
  train_stage2.py    lead model
  clustering.py      elbow, K-Means, Ward, dendrogram, profiles
  predict.py         single-row scoring for the app
  connectors/meta.py read-only Graph API client
scripts/sync_meta.py one-shot Meta pull -> data/raw/stage1_meta.csv
app/streamlit_app.py dashboard
tests/smoke.py       tiny fixture test of the whole plumbing
```
