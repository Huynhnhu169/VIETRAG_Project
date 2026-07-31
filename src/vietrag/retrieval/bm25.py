"""Dependency-free Okapi BM25 implementation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from vietrag.retrieval.base import BaseRetriever, stable_rank
from vietrag.retrieval.tokenization import tokenize
from vietrag.schemas import DocumentRecord, RetrievedPassage


class BM25Retriever(BaseRetriever):
    system_id = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._documents: list[DocumentRecord] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: dict[str, int] = {}
        self._lengths: list[int] = []
        self._average_length = 0.0

    @property
    def documents(self) -> list[DocumentRecord]:
        return list(self._documents)

    def index(self, documents: Iterable[DocumentRecord]) -> None:
        self._documents = sorted(documents, key=lambda item: item.article_id)
        if not self._documents:
            raise ValueError("cannot build a BM25 index with no documents")
        self._term_frequencies = []
        document_frequency: dict[str, int] = defaultdict(int)
        self._lengths = []
        for document in self._documents:
            frequencies = Counter(tokenize(document.normalized_text))
            self._term_frequencies.append(frequencies)
            self._lengths.append(sum(frequencies.values()))
            for term in frequencies:
                document_frequency[term] += 1
        self._document_frequency = dict(document_frequency)
        self._average_length = sum(self._lengths) / len(self._lengths)

    def _idf(self, term: str) -> float:
        count = self._document_frequency.get(term, 0)
        total = len(self._documents)
        return math.log(1.0 + (total - count + 0.5) / (count + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[RetrievedPassage]:
        if not self._documents:
            raise RuntimeError("index() must be called before search()")
        query_terms = tokenize(query)
        scored = []
        for document, frequencies, length in zip(
            self._documents, self._term_frequencies, self._lengths
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * length / max(self._average_length, 1e-12)
                )
                score += self._idf(term) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scored.append((document, score, {"bm25": score}))
        return stable_rank(scored, min(top_k, len(self._documents)))

    def state(self) -> dict:
        return {
            "system_id": self.system_id,
            "k1": self.k1,
            "b": self.b,
            "average_length": self._average_length,
            "document_frequency": self._document_frequency,
            "document_lengths": self._lengths,
        }
