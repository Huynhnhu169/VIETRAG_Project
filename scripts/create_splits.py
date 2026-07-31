"""Create frozen context-disjoint and document-disjoint manifests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vietrag.data.splits import (
    assert_manifest_integrity,
    create_context_disjoint_manifest,
    create_document_fold_manifest,
)
from vietrag.schemas import DocumentRecord, QueryRecord
from vietrag.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/processed/queries.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--document-folds", type=int, default=5)
    parser.add_argument(
        "--context-output",
        type=Path,
        default=Path("data/manifests/context_disjoint.json"),
    )
    parser.add_argument(
        "--document-output",
        type=Path,
        default=Path("data/manifests/document_disjoint_folds.json"),
    )
    parser.add_argument(
        "--queries-output",
        type=Path,
        default=Path("data/processed/queries.with_splits.jsonl"),
    )
    args = parser.parse_args()
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(args.corpus)]
    queries = [QueryRecord.from_dict(row) for row in read_jsonl(args.queries)]
    context_manifest = create_context_disjoint_manifest(
        documents, queries, seed=args.seed
    )
    document_manifest = create_document_fold_manifest(
        documents, queries, seed=args.seed, n_folds=args.document_folds
    )
    assert_manifest_integrity(context_manifest, documents, queries)
    assert_manifest_integrity(document_manifest, documents, queries)
    split_queries = [
        replace(query, split=context_manifest["query_assignments"][query.query_id])
        for query in queries
    ]
    write_json(args.context_output, context_manifest)
    write_json(args.document_output, document_manifest)
    write_jsonl(args.queries_output, (query.to_dict() for query in split_queries))
    print("Frozen leakage-aware manifests after successful integrity checks.")


if __name__ == "__main__":
    main()
