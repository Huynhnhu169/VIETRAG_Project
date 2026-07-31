"""Shared, UI-independent retrieval and grounded-answer pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vietrag.generation import ExtractiveProvider, GroundedAnswerer, GroundedResult
from vietrag.generation.providers import OpenAICompatibleProvider
from vietrag.preprocessing import safe_normalize
from vietrag.retrieval.factory import build_reranker, build_retriever
from vietrag.schemas import DocumentRecord, Prediction, RetrievedPassage


@dataclass(slots=True)
class PipelineResult:
    prediction: Prediction
    passages: list[RetrievedPassage]
    grounded: GroundedResult


class RAGPipeline:
    def __init__(
        self,
        documents: list[DocumentRecord],
        config: dict[str, Any],
        *,
        retrieval_mode: str | None = None,
    ) -> None:
        self.config = config
        self.retriever = build_retriever(
            documents, config, mode=retrieval_mode
        )
        self.reranker = build_reranker(config)
        generation = config["generation"]
        provider_name = generation.get("provider", "extractive")
        if provider_name == "extractive":
            provider = ExtractiveProvider()
        elif provider_name == "openai_compatible":
            provider = OpenAICompatibleProvider()
        else:
            raise ValueError(f"unsupported generation provider: {provider_name}")
        self.answerer = GroundedAnswerer(
            provider,
            abstention_threshold=float(generation["abstention_threshold"]),
            max_evidence=int(generation.get("max_evidence", 3)),
            abstention_message=str(generation["abstention_message"]),
            clarification_message=str(generation["clarification_message"]),
        )

    def ask(
        self,
        query: str,
        *,
        query_id: str = "interactive",
        ambiguous: bool = False,
    ) -> PipelineResult:
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        normalized_query = safe_normalize(query)
        normalization_ms = (time.perf_counter() - stage_started) * 1000
        if not normalized_query:
            raise ValueError("query cannot be empty")

        retrieval = self.config["retrieval"]
        reranking = self.config.get("reranking", {})
        stage_started = time.perf_counter()
        candidates = self.retriever.search(
            normalized_query,
            int(
                reranking.get("input_k", retrieval.get("candidate_k", 20))
                if self.reranker
                else retrieval.get("top_k", 10)
            ),
        )
        retrieval_ms = (time.perf_counter() - stage_started) * 1000

        reranking_ms = 0.0
        if self.reranker:
            stage_started = time.perf_counter()
            candidates = self.reranker.rerank(
                normalized_query,
                candidates,
                top_k=int(reranking.get("output_k", 5)),
            )
            reranking_ms = (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        grounded = self.answerer.answer(
            normalized_query, candidates, ambiguous=ambiguous
        )
        generation_ms = (time.perf_counter() - stage_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        prediction = Prediction(
            query_id=query_id,
            system_id=(
                f"{self.retriever.system_id}+{self.reranker.reranker_id}"
                if self.reranker
                else self.retriever.system_id
            ),
            normalized_query=normalized_query,
            retrieved_context_ids=[
                passage.context_id for passage in candidates
            ],
            retrieval_scores=[passage.score for passage in candidates],
            selected_evidence=[
                passage.document.raw_text
                for passage in grounded.selected_evidence
            ],
            answer=grounded.answer,
            citations=grounded.citations,
            abstained=grounded.abstained,
            latency_ms={
                "normalization": normalization_ms,
                "retrieval": retrieval_ms,
                "reranking": reranking_ms,
                "generation": generation_ms,
                "total": total_ms,
            },
        )
        return PipelineResult(
            prediction=prediction,
            passages=candidates,
            grounded=grounded,
        )
