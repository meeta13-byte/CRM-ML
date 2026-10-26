# CRM-Adv — ML-based CRM marketing intelligence

A college ML project that compares **four classification algorithms on one real
customer-conversion task**, adds honest campaign analytics on a second real
dataset, and segments customers without using the target.

| Module | Question | Method | Data |
|---|---|---|---|
| Lead conversion | Will this customer subscribe to a term deposit? | KNN, Gaussian Naive Bayes, Linear SVM, RBF SVM | UCI Bank Marketing (`bank-full.csv`) |
| Campaign intelligence | What did similar Facebook ads achieve? | descriptive metrics + nearest-neighbour historical benchmark | Kaggle anonymised Facebook ads |
| Customer segmentation | What kinds of customers exist? | K-Means + Ward hierarchical, silhouette-selected k | Bank customer profiles |

Both datasets are real. They are never joined and no rows are fabricated.

```bash
python3 -m src.train_conversion     # four-model comparison -> models/
python3 -m src.experiments          # K sweep + test-size comparison -> models/
python3 -m src.clustering           # K-Means + hierarchical -> models/
streamlit run app/streamlit_app.py  # MarketMind AI dashboard
python3 tests/smoke.py              # fast self-check, no real data needed
```

---

## 1. Setup

```bash
cd ~/Documents/CRM-Adv
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.9+. No database server: history uses the standard-library `sqlite3`.

### Data

**Bank Marketing (main task)** — official UCI source:
<https://archive.ics.uci.edu/dataset/222/bank+marketing>

```bash
curl -sSLo /tmp/bank.zip "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
unzip -o /tmp/bank.zip bank.zip -d /tmp
unzip -o /tmp/bank.zip -d /tmp/bank
cp /tmp/bank/bank-full.csv data/raw/
```

Downloading is a one-time manual step; the file is gitignored. Expected:
45,211 rows × 17 columns, `;`-separated, target `y`, 11.7% `yes`.

**Facebook ads (campaign module)** — Kaggle
`loveall/clicks-conversion-tracking`, file `KAG_conversion_data.csv`
(1,143 rows × 11 columns). Place it in `data/raw/`.

If a file is missing, the loader scans `data/raw/` for a CSV with the required
columns and raises a clear error naming the expected columns.

---

## 2. Lead conversion: four algorithms, one fair comparison

### Data preparation

- Target: `y` → `yes = 1`, `no = 0`.
- **`duration` is excluded.** It is the call length, known only after the call,
  and it separates the classes almost perfectly (mean 537 s for subscribers vs
  221 s for non-subscribers). Including it would be target leakage for a
  prediction made *before* contacting the customer. A hard guard
  (`src/data/schema.py`) fails training if `duration`, `y` or any other
  post-outcome column reaches the feature matrix.
- 16 features: age, job, marital, education, default, balance, housing, loan,
  contact, day, month, campaign, pdays, previous, poutcome, plus
  `previously_contacted`.
- `pdays = -1` means "never contacted". It is turned into a flag plus a nullable
  recency value; the median imputer runs **inside** the pipeline.
- Categorical `unknown` is kept as a real category (job 288, education 1,857,
  contact 13,020, poutcome 36,959).
- Missing values: none in the file besides the `pdays` sentinel.
- Duplicates: 0.
- Class imbalance: 11.7% positive, documented and handled with
  `class_weight="balanced"` in the SVM grids rather than silently resampled.

Preprocessing is a single `Pipeline`/`ColumnTransformer`: median imputation +
`StandardScaler` for numeric columns, most-frequent imputation + one-hot
encoding for categorical columns. It is fitted on the training split only.
The same stratified split (seed 42, 20% test) is shared by all four models.

### Sample size

RBF SVC is roughly O(n²), so a **stratified 12,000-row sample** of the 45,211
rows is used by training and both experiments. It is drawn once with seed 42,
so every model sees exactly the same rows. Full data:

```bash
python3 -m src.train_conversion --max-rows 0   # slow
```

### Hyperparameter tuning

`GridSearchCV`, 5-fold stratified CV, scoring = **positive-class F1**:

| Model | Grid |
|---|---|
| KNN | `n_neighbors` 3–15, `weights` uniform/distance, `metric` euclidean/manhattan |
| GaussianNB | `var_smoothing` 1e-11 … 1e-5 |
| Linear SVM | `C` 0.01–10, `class_weight` None/balanced |
| RBF SVM | `C` 0.1–10, `gamma` scale/0.01, `class_weight` None/balanced |

Tuning is done without `probability=True` (fast). The chosen hyperparameters are
then refitted once with Platt calibration so the shipped artifact, the reported
metrics and the app agree.

### Results (held-out test split, seed 42)

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | CV F1 |
|---|---|---|---|---|---|---|---|
| KNN | 0.8692 | 0.3830 | 0.1922 | 0.2559 | 0.5921 | 0.6564 | 0.2837 |
| GaussianNB | 0.8237 | 0.3301 | 0.4911 | 0.3948 | 0.6459 | 0.7250 | 0.3823 |
| Linear SVM | 0.7796 | 0.2825 | 0.5730 | 0.3784 | 0.6222 | 0.7363 | 0.3799 |
| **RBF SVM** | **0.8021** | **0.3210** | **0.6192** | **0.4228** | **0.6517** | **0.7672** | **0.4236** |

Majority-class baseline accuracy: **0.8829**. Best model: **RBF SVM**, selected
on test positive-class F1 (recall breaks ties).

**Why accuracy is not the selection metric.** Predicting "no" for everyone
scores 88.3% accuracy and finds none of the 11.7% who subscribe. A model at
80.2% accuracy can therefore be more useful than one at 86.9%: RBF SVM catches
61.9% of subscribers, KNN only 19.2%. F1 balances precision against recall and
is 0 for the all-negative classifier, so it cannot be gamed by the majority
class. Confusion matrices are in `reports/figures/conversion_*.png`.

### K sweep and test sizes (`src/experiments.py`)

- KNN `k = 1…15`, uniform weights, euclidean distance, 5-fold CV. The curve is
  flat; the highest mean F1 is k = 1 (0.297), only ~0.014 above the chosen k = 3.
  That gap is well inside fold-to-fold noise (±0.03), and 1-NN memorises the
  training set, so the grid searched 3–15 and `GridSearchCV` chose k = 3
  (euclidean, uniform) on the 12k sample. The sweep figure is the evidence, not
  a fixed k.
- Test sizes 0.3 / 0.2 / 0.1 with **fixed** hyperparameters, so the split is the
  only variable. RBFSVM F1: 0.408 / 0.423 / 0.425; KNN: 0.276 / 0.256 / 0.234.
  A 0.1 split leaves 1,200 test rows, so differences of a few thousandths are
  noise. 0.2 is kept as the balanced default, not because it "won".

Results are stored in `models/experiments.json`; the app reads that file and
never retrains.

---

## 3. Campaign intelligence (honest by design)

Descriptive analysis of 1,143 real ads (915 after a $1 minimum spend):

- spend, impressions, clicks, CTR, CPC, CPM, approved conversions, CPA,
  conversion rate, campaign/ad/age/gender/interest comparisons;
- 44.7% of ads never convert; median CPA of converting ads is **$30.59**;
- one row is **one ad**, not a campaign; there is no budget, duration, location,
  creative, CTA or revenue column.

Measured with 5-fold CV on the four planning fields, conversions per dollar are
**not learnable** (R² −0.03 KNN, −0.14 SVR). CPM (R² 0.43–0.49) and CTR (0.48)
are. A regressor that printed "predicted CPA = $0.238" would therefore be
decoration. Instead the planner:

1. finds the 25 most similar historical ads (one-hot on age, gender, interest,
   campaign);
2. reports the observed P10–P90 CPA / conversion efficiency, how many of those
   ads met the user's target, and the sample size;
3. scales budget linearly, stating that diminishing returns cannot be modelled;
4. switches the output panel by objective — Awareness → CPM/impressions,
   Traffic → CTR/CPC/clicks, Lead Generation/Sales → conversions/CPA;
5. converts `Target CPA = 1 / Target Leads per Dollar` and never claims a target
   is achievable.

The UI tags every field: **supported** (trained feature), **calculated**
(budget, objective, target), **context only** (brand, creative, CTA, offer,
location, duration, previous metrics — saved but never seen by a model) or
**not available** (reach, frequency, ROAS). Full audit: `docs/data-audit.md`.

---

## 4. Customer segmentation

K-Means and Ward hierarchical clustering on customer profile and relationship
history (age, job, marital, education, default, balance, housing, loan, pdays,
previous, poutcome, previously contacted). Dropped: `duration` and `y`
(post-outcome), and `day`, `month`, `contact`, `campaign` (campaign-operation
fields, not a customer portrait).

- Scaling, imputation and one-hot encoding inside the preprocessor;
- elbow + silhouette for k = 2…10, highest silhouette chosen: **k = 3**
  (silhouette 0.378, ARI vs Ward 0.965);
- profiles, conversion rate per segment and business tiers in
  `models/clusters.json`; PCA views and a truncated dendrogram in
  `reports/figures/`.

The three segments: never contacted (81.7% of customers, 9.2% subscribe),
previously contacted with prior failure (8.0%, 13.3%), previously contacted with
higher balance and recent contact (10.2%, **30.7%**). The silhouette is low and
the curve flat, so these are coarse portraits, not well-separated groups. The
target is used only to describe clusters, never to form them.

---

## 5. MarketMind AI UI (9 pages)

A persistent sidebar groups the pages:

- **Overview** — Dashboard, Campaign Intelligence, Lead Conversion Prediction,
  Customer Segmentation
- **Machine Learning** — Model Comparison, Dataset & Training History,
  Prediction History
- **Workspace** — Settings, About

1. **Dashboard** — dataset records, best-model F1, saved predictions and the
   observed conversion rate, followed by model performance, class distribution,
   campaign summary, recent predictions, training status and quick actions.
2. **Campaign Intelligence** — dataset overview and charts, planning form with
   coverage tags, objective-specific benchmark, save to history.
3. **Lead Conversion Prediction** — one of the four models or compare all four
   on the same customer; probability, class, priority band and decision score,
   with a per-model explanation of where the probability comes from.
4. **Customer Segmentation** — evidence, profiles, PCA views, dendrogram.
5. **Model Comparison** — metric table and chart, confusion matrices,
   GridSearchCV results, K sweep, test-size comparison.
6. **Dataset & Training History** — sources, rows/columns, excluded features,
   preprocessing, split, hyperparameters, metrics, experiment tables, artifact
   paths.
7. **Prediction History** — saved conversion predictions with editable actual
   outcomes, plus saved campaign analyses.
8. **Settings** — artifact status, storage paths, history download, palette.
9. **About** — modules, datasets, method notes and limitations.

The palette lives once in `src/config.py:PALETTE` and is imported by
`app/theme.py`, which owns the CSS, reusable KPI/badge/stat helpers and the
matplotlib chart builders. Saved figures in `reports/figures/` use the same
colours. Plotly is not required; every chart is matplotlib.

Probabilities are shown for all four because all four are fitted with
`predict_proba`: SVCs use Platt scaling, GaussianNB uses its independence
posterior (often overconfident), KNN uses neighbour frequency (coarse). The
disclaimer and the source note appear next to every score.

**Priority bands** are relative, not guarantees: High ≥ 30% and Medium ≥ 10%,
against an 11.7% cohort subscription rate.

### SQLite history

`data/history.db` (gitignored), created automatically with `sqlite3`:

- `predictions`: timestamp, dataset, model, input JSON, predicted class,
  probability, decision score, priority, actual outcome (editable later);
- `campaign_analyses`: timestamp, objective, full context JSON, result JSON,
  methodology.

Delete the file to reset history.

---

## 6. Project layout

```
app/streamlit_app.py      9-page sidebar UI (reads artifacts, never trains)
app/theme.py              palette, CSS, cards, matplotlib chart builders
.streamlit/config.toml    light Streamlit theme
src/config.py             paths, seed, sample size, selection metric, bands, palette
src/data/loaders.py       bank + campaign adapters, derived metrics
src/data/preprocess.py    split + sklearn preprocessing pipeline
src/data/schema.py        feature contract + leakage guard
src/train_common.py       the four models, grids, selection
src/train_conversion.py   CLI entry point for training
src/evaluate.py           GridSearchCV, metrics, confusion matrices
src/experiments.py        K sweep + test-size comparison
src/campaign.py           descriptive analytics + NN benchmark
src/clustering.py         K-Means, Ward, profiles, figures
src/predict.py            one-model and all-model scoring
src/history.py            SQLite prediction/campaign history
tests/smoke.py            fixtures, leakage guard, pipeline, SQLite
docs/data-audit.md        measured support/limitations of the ad data
models/                   trained pipelines + JSON results (committed)
reports/figures/          PNG figures shown in the app (committed)
```

Artifacts are committed so the demo works after cloning; the raw datasets are
not (licences and file size).

---

## 7. Limitations

- The bank sample is 12k of 45,211 rows because RBF SVC scales poorly; numbers
  will move slightly on the full set.
- The task is imbalanced and the best F1 is 0.42: this is a screening aid, not
  an automated decision system.
- `duration` is deliberately absent, which costs accuracy but keeps the
  prediction usable before the call.
- The ad dataset is one advertiser, one platform, no dates, no revenue: no
  reliable CPA prediction, no ROAS, no reach/frequency.
- Clusters are weakly separated; segment names are interpretations.
- Actual outcomes are only stored once a user enters them; no live CRM feed.
