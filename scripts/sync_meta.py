"""Pull your own Meta ad account into data/raw/stage1_meta.csv.

    python scripts/sync_meta.py --list-accounts
    python scripts/sync_meta.py --since 2024-01-01 --until 2026-09-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_RAW, STAGE1_META
from src.connectors.meta import MetaAdsClient, build_stage1_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="YYYY-MM-DD, at most 37 months back")
    parser.add_argument("--until", help="YYYY-MM-DD")
    parser.add_argument("--out", default=str(STAGE1_META))
    parser.add_argument("--list-accounts", action="store_true")
    args = parser.parse_args()

    client = MetaAdsClient()

    if args.list_accounts:
        for account in client.ad_accounts():
            print(account)
        return

    if not args.since or not args.until:
        parser.error("--since and --until are required unless --list-accounts")

    frame = build_stage1_frame(client, args.since, args.until)
    if frame.empty:
        raise SystemExit(
            "No ad sets with spend in that window. Check the account id, the date "
            "range (max 37 months), and that the token has ads_read."
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"{len(frame)} ad sets -> {args.out}")
    print(frame[["objective", "spend", "conversions"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
