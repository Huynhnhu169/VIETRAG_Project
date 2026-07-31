"""Dependency-free retrieval metrics."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(
    ranked_context_ids: list[str], gold_context_ids: Iterable[str], k: int
) -> float:
    gold = set(gold_context_ids)
    if not gold:
        return 0.0
    return len(gold & set(ranked_context_ids[:k])) / len(gold)


def reciprocal_rank(
    ranked_context_ids: list[str], gold_context_ids: Iterable[str], k: int
) -> float:
    gold = set(gold_context_ids)
    for rank, context_id in enumerate(ranked_context_ids[:k], start=1):
        if context_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_context_ids: list[str], gold_context_ids: Iterable[str], k: int
) -> float:
    gold = set(gold_context_ids)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, context_id in enumerate(ranked_context_ids[:k], start=1)
        if context_id in gold
    )
    ideal_hits = min(len(gold), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_ranking(
    ranked_context_ids: list[str],
    gold_context_ids: Iterable[str],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics = {
        f"recall@{cutoff}": recall_at_k(
            ranked_context_ids, gold_context_ids, cutoff
        )
        for cutoff in cutoffs
    }
    metrics["mrr@10"] = reciprocal_rank(ranked_context_ids, gold_context_ids, 10)
    metrics["ndcg@10"] = ndcg_at_k(ranked_context_ids, gold_context_ids, 10)
    return metrics
