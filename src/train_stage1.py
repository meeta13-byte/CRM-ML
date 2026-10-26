"""Stage 1: predict whether a planned campaign beats its target CPA.

Usage:
    python -m src.train_stage1                 # Kaggle Facebook-ads export
    python -m src.train_stage1 --source meta   # your own synced ad account
"""
import argparse

from .config import MAX_STAGE1_ROWS
from .data.loaders import load_stage1_kaggle, load_stage1_meta
from .data.schema import META_FORBIDDEN, STAGE1_FORBIDDEN
from .train_common import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["kaggle", "meta"], default="kaggle")
    args = parser.parse_args()

    if args.source == "kaggle":
        X, y, info = load_stage1_kaggle()
        forbidden = STAGE1_FORBIDDEN
    else:
        X, y, info = load_stage1_meta()
        forbidden = META_FORBIDDEN

    print(f"Stage 1 source: {info['source']} | rows: {len(X)} | "
          f"target CPA: {info.get('target_cpa')}")
    run("Stage 1", X, y, info, forbidden, "stage1", MAX_STAGE1_ROWS)


if __name__ == "__main__":
    main()
