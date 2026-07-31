"""Common retriever contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from vietrag.schemas import DocumentRecord, RetrievedPassage


class BaseRetriever(ABC):
    """All retrieval systems index documents and return the same passage schema."""

    system_id: str

    @abstractmethod
    def index(self, documents: Iterable[DocumentRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievedPassage]:
        raise NotImplementedError

    @property
    @abstractmethod
    def documents(self) -> list[DocumentRecord]:
        raise NotImplementedError


def stable_rank(
    scored: Iterable[tuple[DocumentRecord, float, dict[str, float]]],
    top_k: int,
) -> list[RetrievedPassage]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ordered = sorted(
        scored,
        key=lambda item: (-float(item[1]), item[0].article_id),
    )[:top_k]
    return [
        RetrievedPassage(
            context_id=document.article_id,
            document=document,
            rank=rank,
            score=float(score),
            component_scores=components,
        )
        for rank, (document, score, components) in enumerate(ordered, start=1)
    ]
