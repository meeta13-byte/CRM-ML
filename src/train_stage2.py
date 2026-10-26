"""Stage 2: predict lead conversion from information known at capture time.

Usage:
    python -m src.train_stage2
    python -m src.train_stage2 --include-snapshot-fields   # leakage demo
"""
import argparse

from .data.loaders import load_stage2
from .data.schema import STAGE2_FORBIDDEN
from .train_common import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-snapshot-fields",
        action="store_true",
        help="keep Last Activity / Asymmetrique scores (comparison run only)",
    )
    args = parser.parse_args()

    X, y, info = load_stage2(include_snapshot_fields=args.include_snapshot_fields)
    print(f"Stage 2 rows: {len(X)} | converted: {y.mean():.1%} | "
          f"columns: {len(X.columns)}")
    run("Stage 2", X, y, info, STAGE2_FORBIDDEN, "stage2")


if __name__ == "__main__":
    main()
