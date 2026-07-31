"""Leakage-aware split and grouped-fold manifests."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Hashable, Iterable

from vietrag.schemas import DocumentRecord, QueryRecord


class _UnionFind:
    def __init__(self, values: Iterable[Hashable]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Hashable) -> Hashable:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            previous = self.parent[value]
            self.parent[value] = root
            value = previous
        return root

    def union(self, left: Hashable, right: Hashable) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _manifest_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _finalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["frozen"] = True
    payload["manifest_checksum"] = _manifest_checksum(payload)
    return payload


def verify_manifest_checksum(manifest: dict[str, Any]) -> bool:
    expected = manifest.get("manifest_checksum")
    if not isinstance(expected, str):
        return False
    payload = {key: value for key, value in manifest.items() if key != "manifest_checksum"}
    return expected == _manifest_checksum(payload)


def _partition(
    groups: dict[str, list[str]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    required = ("train", "validation", "test")
    if set(ratios) != set(required):
        raise ValueError(f"ratios must contain exactly {required}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("all split ratios must be positive")
    if len(groups) < 3:
        raise ValueError(
            "context-disjoint train/validation/test requires at least three "
            "independent connected context groups"
        )

    rng = random.Random(seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda item: len(groups[item]), reverse=True)
    totals = {split: 0 for split in required}
    target = {
        split: ratios[split] * sum(len(values) for values in groups.values())
        for split in required
    }
    assignments: dict[str, str] = {}
    for index, group_id in enumerate(group_ids):
        empty_splits = [
            split
            for split in required
            if totals[split] == 0 and len(group_ids) - index <= sum(v == 0 for v in totals.values())
        ]
        if empty_splits:
            selected = empty_splits[0]
        else:
            selected = min(
                required,
                key=lambda split: (
                    totals[split] / max(target[split], 1e-12),
                    totals[split],
                    split,
                ),
            )
        assignments[group_id] = selected
        totals[selected] += len(groups[group_id])
    return assignments


def _connect_query_groups(
    values: list[str],
    queries: list[QueryRecord],
    article_to_value: dict[str, str],
) -> dict[str, list[str]]:
    union_find = _UnionFind(values)
    by_family: dict[str, set[str]] = defaultdict(set)
    for query in queries:
        for article_id in query.gold_context_ids:
            if article_id not in article_to_value:
                raise ValueError(
                    f"query {query.query_id} refers to unknown context {article_id}"
                )
            by_family[query.base_query_id].add(article_to_value[article_id])
    for connected in by_family.values():
        ordered = sorted(connected)
        for value in ordered[1:]:
            union_find.union(ordered[0], value)
    components: dict[str, list[str]] = defaultdict(list)
    for value in values:
        components[str(union_find.find(value))].append(value)
    return dict(components)


def create_context_disjoint_manifest(
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    ratios = ratios or {"train": 0.7, "validation": 0.15, "test": 0.15}
    article_to_hash = {
        document.article_id: document.context_hash for document in documents
    }
    hashes = sorted(set(article_to_hash.values()))
    components = _connect_query_groups(hashes, queries, article_to_hash)
    component_splits = _partition(components, ratios, seed)
    hash_to_component = {
        context_hash: component
        for component, members in components.items()
        for context_hash in members
    }
    article_assignments = {
        article_id: component_splits[hash_to_component[context_hash]]
        for article_id, context_hash in article_to_hash.items()
    }
    query_assignments: dict[str, str] = {}
    for query in queries:
        gold_splits = {
            article_assignments[article_id] for article_id in query.gold_context_ids
        }
        if len(gold_splits) != 1:
            raise AssertionError("connected contexts were assigned to multiple splits")
        query_assignments[query.query_id] = next(iter(gold_splits))
    return _finalize_manifest(
        {
            "protocol": "context-disjoint",
            "seed": seed,
            "ratios": ratios,
            "article_assignments": article_assignments,
            "query_assignments": query_assignments,
            "context_hash_assignments": {
                context_hash: component_splits[hash_to_component[context_hash]]
                for context_hash in hashes
            },
        }
    )


def create_document_fold_manifest(
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    *,
    seed: int,
    n_folds: int = 5,
) -> dict[str, Any]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    article_to_doc = {document.article_id: document.doc_id for document in documents}
    doc_ids = sorted(set(article_to_doc.values()))
    components = _connect_query_groups(doc_ids, queries, article_to_doc)
    rng = random.Random(seed)
    component_ids = sorted(components)
    rng.shuffle(component_ids)
    component_ids.sort(key=lambda item: len(components[item]), reverse=True)
    fold_sizes = [0] * n_folds
    component_fold: dict[str, int] = {}
    for component in component_ids:
        fold = min(range(n_folds), key=lambda item: (fold_sizes[item], item))
        component_fold[component] = fold
        fold_sizes[fold] += len(components[component])
    doc_to_component = {
        doc_id: component
        for component, members in components.items()
        for doc_id in members
    }
    document_assignments = {
        doc_id: component_fold[doc_to_component[doc_id]] for doc_id in doc_ids
    }
    article_assignments = {
        article_id: document_assignments[doc_id]
        for article_id, doc_id in article_to_doc.items()
    }
    query_assignments: dict[str, int] = {}
    for query in queries:
        folds = {
            article_assignments[article_id] for article_id in query.gold_context_ids
        }
        if len(folds) != 1:
            raise AssertionError("connected documents were assigned to multiple folds")
        query_assignments[query.query_id] = next(iter(folds))
    return _finalize_manifest(
        {
            "protocol": "document-disjoint-grouped-folds",
            "seed": seed,
            "n_folds": n_folds,
            "document_assignments": document_assignments,
            "article_assignments": article_assignments,
            "query_assignments": query_assignments,
            "fold_sizes": fold_sizes,
        }
    )


def assert_manifest_integrity(
    manifest: dict[str, Any],
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
) -> None:
    if not manifest.get("frozen"):
        raise AssertionError("split manifest is not frozen")
    if not verify_manifest_checksum(manifest):
        raise AssertionError("split manifest checksum mismatch")
    article_assignments = manifest["article_assignments"]
    if set(article_assignments) != {document.article_id for document in documents}:
        raise AssertionError("manifest does not cover the corpus exactly")
    query_assignments = manifest["query_assignments"]
    if set(query_assignments) != {query.query_id for query in queries}:
        raise AssertionError("manifest does not cover all queries")

    if manifest["protocol"] == "context-disjoint":
        context_splits: dict[str, set[str]] = defaultdict(set)
        for document in documents:
            context_splits[document.context_hash].add(
                article_assignments[document.article_id]
            )
        if any(len(splits) != 1 for splits in context_splits.values()):
            raise AssertionError("a context hash crosses split boundaries")
    elif manifest["protocol"] == "document-disjoint-grouped-folds":
        document_folds: dict[str, set[int]] = defaultdict(set)
        for document in documents:
            document_folds[document.doc_id].add(
                article_assignments[document.article_id]
            )
        if any(len(folds) != 1 for folds in document_folds.values()):
            raise AssertionError("a document crosses fold boundaries")
    else:
        raise AssertionError(f"unknown protocol: {manifest['protocol']}")

    family_assignments: dict[str, set[Any]] = defaultdict(set)
    for query in queries:
        assignment = query_assignments[query.query_id]
        family_assignments[query.base_query_id].add(assignment)
        gold_assignments = {
            article_assignments[article_id] for article_id in query.gold_context_ids
        }
        if gold_assignments != {assignment}:
            raise AssertionError(
                f"query {query.query_id} is separated from its gold evidence"
            )
    if any(len(assignments) != 1 for assignments in family_assignments.values()):
        raise AssertionError("a base_query_id family crosses split boundaries")
