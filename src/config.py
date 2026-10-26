"""Project-wide settings. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
HISTORY_DB = ROOT / "data" / "history.db"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TEST_SIZES = (0.3, 0.2, 0.1)

# RBF SVC is roughly O(n^2) and the bank data has 45,211 rows. One stratified
# sample is drawn once, so all four algorithms (and every experiment) see
# exactly the same rows. Raise it if you have an hour to spare.
MAX_BANK_ROWS = 12_000

# 11.7% positives. Selection uses positive-class F1: "always predict no" scores
# F1 = 0, so it cannot game the metric (macro-F1 would reward the majority
# class). Macro-F1, precision, recall and the confusion matrix are still shown.
SELECTION_METRIC = "f1"

# Predicted probability -> priority band (lead-conversion page). High is about
# 2.5x the 11.7% cohort subscription rate, so the bands describe relative risk,
# not a claim that the customer will subscribe.
BANDS = {"High": 0.30, "Medium": 0.10}

# Campaign module: rows below this spend are placeholder ads with no delivery.
MIN_SPEND = 1.0
# Nearest-neighbour benchmark size for campaign planning.
NEIGHBOURS = 25

BANK_CSV = DATA_RAW / "bank-full.csv"
CAMPAIGN_CSV = DATA_RAW / "KAG_conversion_data.csv"

# One palette for the app theme, saved figures and report charts.
PALETTE = {
    "workspace": "#FFFFFF",
    "sidebar": "#F7F7FA",
    "page": "#F9FAFC",
    "purple": "#8275E8",
    "purple_deep": "#6659D9",
    "lavender": "#EAE6FF",
    "gradient_from": "#7969E8",
    "gradient_to": "#E4A5F5",
    "text": "#171717",
    "muted": "#737780",
    "border": "#E8E9EE",
    "card": "#FFFFFF",
    "orange": "#F28C45",
    "green": "#24B47E",
    "blue": "#657FE8",
    "pink": "#E66C9B",
    "hover": "#F0EEFA",
}

