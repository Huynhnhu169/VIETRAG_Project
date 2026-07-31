"""Generate query noise only from a frozen split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vietrag.data.robustness import create_robustness_variants
from vietrag.schemas import QueryRecord
from vietrag.utils.io import read_json, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/processed/queries.with_splits.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/context_disjoint.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/robustness_queries.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/interim/robustness_audit.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    queries = [QueryRecord.from_dict(row) for row in read_jsonl(args.queries)]
    variants, audit = create_robustness_variants(
        queries, read_json(args.manifest), seed=args.seed
    )
    write_jsonl(args.output, (variant.to_dict() for variant in variants))
    write_json(args.audit_output, audit)
    print(
        f"Created {len(variants)} deterministic variants after split freeze; "
        f"{audit['flagged_count']} require manual review."
    )


if __name__ == "__main__":
    main()
