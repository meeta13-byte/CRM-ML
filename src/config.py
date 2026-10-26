"""Project-wide settings. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# SVC is O(n^2) and probability=True adds an internal CV, so Stage 1 is capped.
MAX_STAGE1_ROWS = 12_000

# Stage 1 label: "did this campaign beat its target cost per acquisition?"
# The threshold is the median CPA of the same cohort (business KPI in a real
# account). Every run reports it, and the report includes a sensitivity band.
# Spend values in the public export are mostly small; $1 keeps 80% of the ads
# while still dropping placeholder rows with zero delivery.
MIN_SPEND = 1.0
TARGET_CPA_QUANTILE = 0.50
TARGET_CPA_OVERRIDE = None  # e.g. 250.0 once you have a real account KPI

# Imbalanced classes: select on macro-F1 so "predict the majority class"
# cannot win. Per-class F1/precision/recall are still reported.
SELECTION_METRIC = "f1_macro"

# Predicted probability -> priority band (both stages).
BANDS = {"High": 0.65, "Medium": 0.40}

# Default locations; the loaders also search data/raw for a matching CSV.
STAGE1_KAGGLE = DATA_RAW / "KAG_conversion_data.csv"
STAGE2_XEDUCATION = DATA_RAW / "Lead Scoring.csv"
STAGE1_META = DATA_RAW / "stage1_meta.csv"

META_API_VERSION = "v26.0"
