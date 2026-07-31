"""Canonicalization, exact deduplication, hashing, and dataset auditing."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any, Iterable

from vietrag.preprocessing import canonicalize_context, lexical_normalize
from vietrag.schemas import DocumentRecord, QueryRecord


def compute_context_hash(text: str) -> str:
    canonical = canonicalize_context(text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def prepare_document(
    *,
    doc_id: str,
    article_id: str,
    parent_id: str,
    title: str,
    raw_text: str,
    source: str,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    canonical = canonicalize_context(raw_text)
    return DocumentRecord(
        doc_id=doc_id,
        article_id=article_id,
        parent_id=parent_id,
        title=title,
        raw_text=raw_text,
        normalized_text=lexical_normalize(raw_text),
        source=source,
        version=version,
        context_hash=compute_context_hash(canonical),
        metadata=dict(metadata or {}),
    )


def deduplicate_documents(
    documents: Iterable[DocumentRecord],
) -> tuple[list[DocumentRecord], dict[str, str]]:
    """Merge exact canonical-context duplicates and return old-to-new ID aliases."""
    by_hash: dict[str, DocumentRecord] = {}
    aliases: dict[str, str] = {}
    provenance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for document in documents:
        expected_hash = compute_context_hash(document.raw_text)
        if document.context_hash != expected_hash:
            raise ValueError(
                f"context hash mismatch for {document.article_id}: "
                f"{document.context_hash} != {expected_hash}"
            )
        provenance[expected_hash].append(
            {
                "article_id": document.article_id,
                "doc_id": document.doc_id,
                "source": document.source,
            }
        )
        if expected_hash not in by_hash:
            by_hash[expected_hash] = document
        aliases[document.article_id] = by_hash[expected_hash].article_id

    deduplicated: list[DocumentRecord] = []
    for digest, document in by_hash.items():
        metadata = dict(document.metadata)
        metadata["provenance"] = provenance[digest]
        deduplicated.append(replace(document, metadata=metadata))
    deduplicated.sort(key=lambda item: item.article_id)
    return deduplicated, aliases


def remap_query_contexts(
    queries: Iterable[QueryRecord], aliases: dict[str, str]
) -> list[QueryRecord]:
    remapped: list[QueryRecord] = []
    for query in queries:
        gold_ids = sorted({aliases.get(item, item) for item in query.gold_context_ids})
        remapped.append(replace(query, gold_context_ids=gold_ids))
    return remapped


def _near_duplicate_pairs(
    documents: list[DocumentRecord],
    *,
    threshold: float = 0.92,
    max_pairs: int = 100,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1 :]:
            if left.context_hash == right.context_hash:
                continue
            left_text = canonicalize_context(left.raw_text)
            right_text = canonicalize_context(right.raw_text)
            length_ratio = min(len(left_text), len(right_text)) / max(
                len(left_text), len(right_text), 1
            )
            if length_ratio < 0.7:
                continue
            similarity = SequenceMatcher(None, left_text, right_text).ratio()
            if similarity >= threshold:
                warnings.append(
                    {
                        "left_article_id": left.article_id,
                        "right_article_id": right.article_id,
                        "similarity": round(similarity, 4),
                    }
                )
                if len(warnings) >= max_pairs:
                    return warnings
    return warnings


def build_audit_report(
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    *,
    input_document_count: int | None = None,
    near_duplicate_threshold: float = 0.92,
) -> dict[str, Any]:
    missing = Counter()
    for document in documents:
        for field_name in (
            "doc_id",
            "article_id",
            "title",
            "raw_text",
            "normalized_text",
            "context_hash",
        ):
            if not getattr(document, field_name):
                missing[f"document.{field_name}"] += 1
    for query in queries:
        for field_name in ("query_id", "base_query_id", "query", "clean_query"):
            if not getattr(query, field_name):
                missing[f"query.{field_name}"] += 1
        if query.answerable and not query.gold_context_ids:
            missing["query.gold_context_ids"] += 1

    hash_counts = Counter(document.context_hash for document in documents)
    duplicate_hashes = {
        digest: count for digest, count in sorted(hash_counts.items()) if count > 1
    }
    return {
        "record_counts": {
            "input_documents": input_document_count
            if input_document_count is not None
            else len(documents),
            "output_documents": len(documents),
            "queries": len(queries),
        },
        "unique_contexts": len(hash_counts),
        "unique_documents": len({document.doc_id for document in documents}),
        "missing_values": dict(sorted(missing.items())),
        "exact_duplicate_hashes": duplicate_hashes,
        "near_duplicate_warnings": _near_duplicate_pairs(
            documents, threshold=near_duplicate_threshold
        ),
    }
