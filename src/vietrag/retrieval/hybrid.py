"""Reciprocal-rank and validation-tuned weighted fusion."""

from __future__ import annotations

from typing import Iterable

from vietrag.retrieval.base import BaseRetriever
from vietrag.schemas import DocumentRecord, RetrievedPassage


def reciprocal_rank_fusion(
    result_sets: dict[str, list[RetrievedPassage]],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[RetrievedPassage]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    documents: dict[str, DocumentRecord] = {}
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for system_name, passages in sorted(result_sets.items()):
        for passage in passages:
            context_id = passage.context_id
            documents[context_id] = passage.document
            contribution = 1.0 / (rrf_k + passage.rank)
            scores[context_id] = scores.get(context_id, 0.0) + contribution
            components.setdefault(context_id, {})[f"{system_name}_rrf"] = contribution
            components[context_id][system_name] = passage.score
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [
        RetrievedPassage(
            context_id=context_id,
            document=documents[context_id],
            rank=rank,
            score=scores[context_id],
            component_scores=components[context_id],
        )
        for rank, context_id in enumerate(ordered, start=1)
    ]


def _min_max_scores(passages: Iterable[RetrievedPassage]) -> dict[str, float]:
    passages = list(passages)
    if not passages:
        return {}
    values = [passage.score for passage in passages]
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return {passage.context_id: 1.0 for passage in passages}
    return {
        passage.context_id: (passage.score - minimum) / (maximum - minimum)
        for passage in passages
    }


def weighted_score_fusion(
    lexical: list[RetrievedPassage],
    dense: list[RetrievedPassage],
    *,
    alpha: float,
    top_k: int = 10,
) -> list[RetrievedPassage]:
    """Fuse normalized scores; alpha must be selected on validation only."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    lexical_scores = _min_max_scores(lexical)
    dense_scores = _min_max_scores(dense)
    passage_by_id = {
        passage.context_id: passage for passage in [*lexical, *dense]
    }
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for context_id in passage_by_id:
        lexical_score = lexical_scores.get(context_id, 0.0)
        dense_score = dense_scores.get(context_id, 0.0)
        scores[context_id] = alpha * lexical_score + (1 - alpha) * dense_score
        components[context_id] = {
            "bm25_normalized": lexical_score,
            "dense_normalized": dense_score,
        }
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [
        RetrievedPassage(
            context_id=context_id,
            document=passage_by_id[context_id].document,
            rank=rank,
            score=scores[context_id],
            component_scores=components[context_id],
        )
        for rank, context_id in enumerate(ordered, start=1)
    ]


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        lexical: BaseRetriever,
        dense: BaseRetriever,
        *,
        mode: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_k: int = 20,
    ) -> None:
        if mode not in {"rrf", "alpha"}:
            raise ValueError("hybrid mode must be 'rrf' or 'alpha'")
        self.lexical = lexical
        self.dense = dense
        self.mode = mode
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.candidate_k = candidate_k
        self.system_id = f"hybrid_{mode}"

    @property
    def documents(self) -> list[DocumentRecord]:
        return self.lexical.documents

    def index(self, documents: Iterable[DocumentRecord]) -> None:
        documents = list(documents)
        self.lexical.index(documents)
        self.dense.index(documents)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedPassage]:
        lexical = self.lexical.search(query, self.candidate_k)
        dense = self.dense.search(query, self.candidate_k)
        if self.mode == "rrf":
            return reciprocal_rank_fusion(
                {"bm25": lexical, "dense": dense},
                rrf_k=self.rrf_k,
                top_k=top_k,
            )
        return weighted_score_fusion(
            lexical, dense, alpha=self.alpha, top_k=top_k
        )
