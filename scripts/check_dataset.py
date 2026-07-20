from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_lmdb import load_expected_sources, validate_lmdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully validate a FiSeR LMDB")
    parser.add_argument("lmdb_path", type=Path)
    parser.add_argument("--mode", choices=["full", "keys"], default="full")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--expected-sources", type=Path)
    args = parser.parse_args()
    result = validate_lmdb(
        args.lmdb_path,
        mode=args.mode,
        expected_count=args.expected_count,
        expected_source_count=args.expected_source_count,
        expected_sources=load_expected_sources(args.expected_sources),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
