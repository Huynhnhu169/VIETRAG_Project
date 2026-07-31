"""Audit an already prepared corpus and query file."""

from __future__ import annotations

import argparse
from pathlib import Path

from vietrag.data.preparation import build_audit_report
from vietrag.schemas import DocumentRecord, QueryRecord
from vietrag.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/processed/queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/audit.json"))
    args = parser.parse_args()
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(args.corpus)]
    queries = [QueryRecord.from_dict(row) for row in read_jsonl(args.queries)]
    report = build_audit_report(documents, queries)
    write_json(args.output, report)
    print(f"Wrote audit report to {args.output}")


if __name__ == "__main__":
    main()
