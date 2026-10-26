# Data audit — what the campaign data can and cannot support

Written before building the richer campaign-prediction UI, so the design is
driven by measurements instead of assumptions. Every number below was produced
from `data/raw/KAG_conversion_data.csv` with the project's own preprocessing.

## 1. Shape of the training data

`KAG_conversion_data.csv` — 1,143 rows × 11 columns:
`ad_id, xyz_campaign_id, fb_campaign_id, age, gender, interest, Impressions,
Clicks, Spent, Total_Conversion, Approved_Conversion`.

**One row is one ad, not one campaign.** No date, objective, budget, duration,
creative, location or revenue column exists.

After the `MIN_SPEND = 1` floor: **915 rows**, 3 campaigns, 40 interest codes,
400 distinct audience/campaign combinations. 186 ads have a unique combination;
577 sit in combinations of 3 or more.

## 2. CPA distribution (the business scale)

| Statistic | Value |
|---|---|
| Ads with zero approved conversions | 44.7% (CPA = infinite → label 1) |
| Cheapest CPA observed | $0.80 |
| P10 / P25 / median / P75 / P90 | $2.49 / $8.31 / **$30.59** / $60.86 / $123.94 |
| Median conversion efficiency | 0.008 leads per $1 (≈ $125 CPA) |
| Best campaign (916) efficiency | 0.107 leads per $1 (≈ $9.3 CPA) |
| Ads with CPA under $1 | 0.1% (1 of 915) |

Consequence: a "5 leads per $1" target (CPA $0.20) is about 150× cheaper than
the cheapest ad in the dataset and ~40× better than the best campaign. The UI
must accept such a target but warn that nothing historical supports it; the
sensible default is the cohort benchmark, $30.59.

## 3. Predictability of each candidate target

5-fold CV, one-hot age/gender/interest/campaign, R² on held-out folds:

| Target | KNN(5) | RBF SVR | Verdict |
|---|---|---|---|
| log(conversions per $1) | −0.03 | −0.14 | **unlearnable** — worse than the mean |
| log(CPA), converting ads only | +0.38 | +0.53 | looks fine but drops the 45% that never converted |
| log(CPM) | +0.43 | +0.49 | learnable |
| log(CPC) | +0.22 | +0.26 | weak |
| CTR | +0.48 | −14.0 | learnable by KNN, SVR diverges |

So: **no point-estimate regression for CPA or leads**. CPM and CTR are the only
quantities these features genuinely explain, which is why Awareness (CPM →
impressions) and Traffic (CTR/CPC → clicks) outputs are defensible while
"predicted CPA = $0.238" is not.

## 4. Field support matrix

| Input group | Kaggle data | Meta sync (if an account exists) |
|---|---|---|
| Age, Gender, Interest | supported (4 / 2 / 40 values) | supported |
| Campaign | supported (3 IDs) | supported |
| Objective | absent — cannot train objective-specific models | supported |
| Platform, placement | absent (all Facebook) | supported |
| Location | absent | supported |
| Budget, duration, schedule | absent | supported (budget is current value, may have been edited) |
| Creative format, CTA, offer, creatives | absent | partial — needs an extra `/ads?fields=creative{...}` call |
| Bid strategy | absent | supported |
| Reach, frequency, audience size | absent | absent (approximate, 13-month cap) |
| Previous CTR/CPC/CPM/CVR/CPA | absent (one advertiser, no dates) | only from prior windows, no leakage guarantee |
| Brand, domain, industry, campaign type, funnel | constant / absent | constant / absent |
| Lead quality | absent (CRM-side) | absent |
| ROAS | **impossible** — no revenue column anywhere | impossible |

The UI therefore exposes only age, gender, interest and campaign as model
inputs. Budget multiplies a historical efficiency; objective selects which
metrics are displayed. Everything else is saved as context for the history page
and labelled "not used by the model".

## 5. Corrections to earlier assumptions

- **K = 3 is not assumed.** The K sweep (k = 1…15, 5-fold CV) is flat; its
  highest F1 is k = 1, only ~0.014 above k = 3, which is inside fold noise. The
  main grid searched 3–15 (1-NN memorises the training set) and chose k = 3.
- **Test-size comparison** shows test F1 *and* the note that a 0.1 split leaves
  only 1,200 test rows. The conclusion is "0.2 kept for a balanced split", not
  "0.2 scored highest".
- **The classifier cannot judge a user's own target.** It predicts subscription
  probability, not CPA. Arbitrary CPA targets are answered by the empirical CPA
  distribution of the nearest historical ads, with the matching sample size
  stated.

## 6. Proposed prediction logic

1. User enters audience + campaign, plus either a target CPA or a target
   leads-per-$ (converted both ways, `CPA = 1 / (leads per $)`).
2. Find the 25 most similar historical ads in the existing one-hot space.
3. Report: share meeting the target, P10–P90 CPA of those ads, and
   `leads ≈ budget × median efficiency` with the linearity assumption stated.
4. Objective switches the panel: CPA/leads (Lead Gen, Sales), CTR/CPC/clicks
   (Traffic), CPM/impressions (Awareness). Reach, frequency and ROAS render an
   explicit "not supported by the training data" panel.
5. Every prediction is written to SQLite (`data/history.db`, stdlib) with the
   full input snapshot, result and model name.
