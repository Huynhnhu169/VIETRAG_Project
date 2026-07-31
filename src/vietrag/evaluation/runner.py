"""Query-level evaluation and aggregate robustness summaries."""

from __future__ import annotations

import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from vietrag.evaluation.metrics import evaluate_ranking
from vietrag.reranking import BaseReranker
from vietrag.retrieval import BaseRetriever
from vietrag.schemas import QueryRecord
from vietrag.utils.io import write_json, write_jsonl


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_names = sorted(rows[0]["metrics"])
    result = {
        metric: statistics.fmean(row["metrics"][metric] for row in rows)
        for metric in metric_names
    }
    latencies = [row["latency_ms"]["total"] for row in rows]
    result.update(
        {
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "query_count": len(rows),
        }
    )
    return result


def _robustness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["noise_type"]].append(row)
    by_noise = {name: _aggregate(values) for name, values in sorted(grouped.items())}
    clean = by_noise.get("clean", {})
    drops: dict[str, dict[str, float | None]] = {}
    for name, metrics in by_noise.items():
        if name == "clean":
            continue
        noise_drops: dict[str, float | None] = {}
        for metric_name in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
            if metric_name not in clean or metric_name not in metrics:
                continue
            absolute = clean[metric_name] - metrics[metric_name]
            relative = absolute / clean[metric_name] if clean[metric_name] else None
            noise_drops[f"{metric_name}_absolute_drop"] = absolute
            noise_drops[f"{metric_name}_relative_drop"] = relative
        drops[name] = noise_drops
    candidates = {
        name: metrics["mrr@10"]
        for name, metrics in by_noise.items()
        if name != "clean" and "mrr@10" in metrics
    }
    worst = min(candidates, key=lambda item: (candidates[item], item)) if candidates else None
    return {
        "by_noise_type": by_noise,
        "drops_from_clean": drops,
        "worst_group_by_mrr@10": worst,
    }


def evaluate_retriever(
    retriever: BaseRetriever,
    queries: list[QueryRecord],
    *,
    top_k: int = 10,
    cutoffs: tuple[int, ...] = (1, 5, 10),
    reranker: BaseReranker | None = None,
    reranker_input_k: int = 20,
    reranker_output_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        candidates = retriever.search(
            query.query, max(top_k, reranker_input_k if reranker else top_k)
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        reranking_ms = 0.0
        if reranker:
            reranking_started = time.perf_counter()
            candidates = reranker.rerank(
                query.query,
                candidates[:reranker_input_k],
                top_k=reranker_output_k,
            )
            reranking_ms = (time.perf_counter() - reranking_started) * 1000
        ranked_ids = [candidate.context_id for candidate in candidates]
        total_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "query_id": query.query_id,
                "base_query_id": query.base_query_id,
                "query": query.query,
                "noise_type": query.noise_type,
                "split": query.split,
                "gold_context_ids": query.gold_context_ids,
                "retrieved_context_ids": ranked_ids,
                "retrieval_scores": [candidate.score for candidate in candidates],
                "component_scores": [
                    candidate.component_scores for candidate in candidates
                ],
                "metrics": evaluate_ranking(
                    ranked_ids, query.gold_context_ids, cutoffs=cutoffs
                ),
                "latency_ms": {
                    "retrieval": retrieval_ms,
                    "reranking": reranking_ms,
                    "total": total_ms,
                },
            }
        )
    summary = {
        "system_id": (
            f"{retriever.system_id}+{reranker.reranker_id}"
            if reranker
            else retriever.system_id
        ),
        "aggregate": _aggregate(rows),
        "robustness": _robustness_summary(rows),
    }
    return rows, summary


def save_evaluation_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", rows)
    write_json(output / "metrics.json", summary)
    write_json(output / "run_metadata.json", run_metadata)
    aggregate = summary.get("aggregate", {})
    with (output / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
