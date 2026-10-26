"""Lead-conversion model: will this bank customer subscribe to a term deposit?

    python3 -m src.train_conversion                 # stratified 12k sample
    python3 -m src.train_conversion --max-rows 0    # full 45,211 rows (slow)
"""
import argparse

from .config import MAX_BANK_ROWS
from .data.loaders import load_bank
from .data.schema import BANK_FORBIDDEN
from .train_common import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows", type=int, default=MAX_BANK_ROWS,
        help="stratified sample size; 0 uses every row (RBF SVC becomes very slow)",
    )
    args = parser.parse_args()

    X, y, info = load_bank()
    print(f"Lead conversion | {info['name']} | rows: {len(X)} | "
          f"features: {len(X.columns)} | subscribed: {y.mean():.1%} | "
          f"excluded: {info['excluded']}")
    run("Lead conversion", X, y, info, BANK_FORBIDDEN, "conversion",
        max_rows=args.max_rows or None)


if __name__ == "__main__":
    main()
