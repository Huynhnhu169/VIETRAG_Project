"""Common reranker contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vietrag.schemas import RetrievedPassage


class BaseReranker(ABC):
    reranker_id: str

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedPassage],
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        raise NotImplementedError
