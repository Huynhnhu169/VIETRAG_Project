from dataclasses import replace

import pytest

from vietrag.data.robustness import (
    create_robustness_variants,
    protected_expressions,
)
from vietrag.data.splits import create_context_disjoint_manifest


def test_noise_requires_frozen_manifest(make_documents_and_queries) -> None:
    _, queries = make_documents_and_queries
    with pytest.raises(ValueError):
        create_robustness_variants(queries, {"frozen": False}, seed=42)


def test_noise_rejects_mutated_frozen_manifest(
    make_documents_and_queries,
) -> None:
    documents, queries = make_documents_and_queries
    manifest = create_context_disjoint_manifest(documents, queries, seed=42)
    manifest["query_assignments"]["Q1"] = "test"
    with pytest.raises(ValueError):
        create_robustness_variants(queries, manifest, seed=42)


def test_variants_preserve_split_gold_and_protected_values(
    make_documents_and_queries,
) -> None:
    documents, queries = make_documents_and_queries
    manifest = create_context_disjoint_manifest(documents, queries, seed=42)
    queries = [
        replace(query, split=manifest["query_assignments"][query.query_id])
        for query in queries
    ]
    variants, audit = create_robustness_variants(queries, manifest, seed=42)
    assert len({variant.query_id for variant in variants}) == len(variants)
    base_by_id = {query.base_query_id: query for query in queries}
    for variant in variants:
        base = base_by_id[variant.base_query_id]
        assert variant.split == base.split
        assert variant.gold_context_ids == base.gold_context_ids
        assert protected_expressions(variant.query) == protected_expressions(
            base.clean_query
        )
    assert all(
        item["reason"] == "no reliable paraphraser configured"
        for item in audit["skipped"]
    )
