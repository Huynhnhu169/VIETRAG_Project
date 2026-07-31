"""Parse, canonicalize, hash, and deduplicate ViRHE4QA-style input."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vietrag.data.preparation import (
    build_audit_report,
    deduplicate_documents,
    remap_query_contexts,
)
from vietrag.data.virhe4qa import parse_virhe4qa
from vietrag.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-name",
        default=None,
        help="Defaults to synthetic for data/samples inputs and ViRHE4QA otherwise.",
    )
    parser.add_argument("--corpus-output", type=Path, default=Path("data/processed/corpus.jsonl"))
    parser.add_argument("--queries-output", type=Path, default=Path("data/processed/queries.jsonl"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/interim/audit.json"))
    args = parser.parse_args()

    source_name = args.source_name or (
        "synthetic"
        if "samples" in {part.lower() for part in args.input.parts}
        else "ViRHE4QA"
    )
    raw_documents, queries, row_count = parse_virhe4qa(
        args.input, source_name=source_name
    )
    documents, aliases = deduplicate_documents(raw_documents)
    queries = remap_query_contexts(queries, aliases)
    audit = build_audit_report(
        documents, queries, input_document_count=len(raw_documents)
    )
    audit["source_rows"] = row_count
    audit["repeated_context_rows"] = max(0, row_count - len(raw_documents))
    before_counts = Counter(document.context_hash for document in raw_documents)
    audit["exact_duplicate_hashes_before_merge"] = {
        digest: count
        for digest, count in sorted(before_counts.items())
        if count > 1
    }
    audit["deduplicated_context_records"] = len(raw_documents) - len(documents)
    write_jsonl(args.corpus_output, (item.to_dict() for item in documents))
    write_jsonl(args.queries_output, (item.to_dict() for item in queries))
    write_json(args.audit_output, audit)
    print(
        f"Prepared {len(documents)} unique contexts and {len(queries)} queries; "
        f"audit: {args.audit_output}"
    )


if __name__ == "__main__":
    main()
