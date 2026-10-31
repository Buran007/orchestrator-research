"""Write a fixed random subset of the Search-R1 test split for frequent validation.

Usage: python research/search_r1/make_val_subset.py DATA_DIR [--size 512] [--seed 42]
Reads DATA_DIR/test.parquet, writes DATA_DIR/val_<size>.parquet.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = pq.read_table(args.data_dir / "test.parquet")
    rows = sorted(random.Random(args.seed).sample(range(table.num_rows), args.size))
    out = args.data_dir / f"val_{args.size}.parquet"
    pq.write_table(table.take(rows), out)
    print(f"{out}: {len(rows)} of {table.num_rows} rows, seed {args.seed}")


if __name__ == "__main__":
    main()
