"""Typed, dependency-free data contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _require(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    article_id: str
    parent_id: str
    title: str
    raw_text: str
    normalized_text: str
    source: str
    context_hash: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "doc_id",
            "article_id",
            "parent_id",
            "title",
            "raw_text",
            "normalized_text",
            "source",
            "context_hash",
        ):
            _require(getattr(self, name), name)
        if not self.context_hash.startswith("sha256:"):
            raise ValueError("context_hash must use the sha256:<hex> format")
        if len(self.context_hash.removeprefix("sha256:")) != 64:
            raise ValueError("context_hash must contain a 64-character digest")
        self.metadata.setdefault("document_name", self.doc_id)
        self.metadata.setdefault("article_number", "")
        self.metadata.setdefault("source_url", None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentRecord":
        return cls(**value)


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    base_query_id: str
    query: str
    clean_query: str
    noise_type: str
    answerable: bool
    ambiguous: bool
    gold_context_ids: list[str]
    reference_answer: str
    source_dataset: str
    split: str

    def __post_init__(self) -> None:
        for name in (
            "query_id",
            "base_query_id",
            "query",
            "clean_query",
            "noise_type",
            "source_dataset",
            "split",
        ):
            _require(getattr(self, name), name)
        if self.split not in {"train", "validation", "test", "unassigned"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.answerable and not self.gold_context_ids:
            raise ValueError("answerable queries require at least one gold context")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryRecord":
        return cls(**value)


@dataclass(slots=True)
class RetrievedPassage:
    context_id: str
    document: DocumentRecord
    rank: int
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.context_id, "context_id")
        if self.context_id != self.document.article_id:
            raise ValueError("context_id must equal document.article_id")
        if self.rank < 1:
            raise ValueError("rank must be one-based")

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "rank": self.rank,
            "score": self.score,
            "component_scores": dict(self.component_scores),
            "document": self.document.to_dict(),
        }


@dataclass(slots=True)
class Citation:
    doc_id: str
    article_id: str
    title: str
    document_name: str
    article_number: str
    evidence_snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Prediction:
    query_id: str
    system_id: str
    normalized_query: str
    retrieved_context_ids: list[str]
    retrieval_scores: list[float]
    selected_evidence: list[str]
    answer: str
    citations: list[Citation]
    abstained: bool
    latency_ms: dict[str, float]
    noise_type: str = "clean"
    split: str = "unassigned"

    def __post_init__(self) -> None:
        if len(self.retrieved_context_ids) != len(self.retrieval_scores):
            raise ValueError("retrieved context IDs and scores must have equal length")
        if self.abstained and self.citations:
            raise ValueError("abstained predictions cannot include citations")
        if any(value < 0 for value in self.latency_ms.values()):
            raise ValueError("latency values cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["citations"] = [citation.to_dict() for citation in self.citations]
        return value
