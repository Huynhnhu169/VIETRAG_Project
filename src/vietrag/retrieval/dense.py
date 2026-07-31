"""Configurable dense retrieval with an offline deterministic fallback."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from vietrag.retrieval.base import BaseRetriever, stable_rank
from vietrag.retrieval.tokenization import tokenize
from vietrag.schemas import DocumentRecord, RetrievedPassage


class TextEncoder(ABC):
    encoder_id: str
    revision: str | None = None

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class HashingEncoder(TextEncoder):
    """Feature-hashing encoder for offline smoke tests, not a trained model."""

    encoder_id = "hashing"

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 16:
            raise ValueError("hashing dimensions must be at least 16")
        self.dimensions = dimensions

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        features = [f"w:{word}" for word in words]
        for word in words:
            padded = f"<{word}>"
            features.extend(
                f"c:{padded[index:index + 3]}"
                for index in range(max(0, len(padded) - 2))
            )
        return features

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for feature in self._features(text):
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vector[index] += sign
            output.append(_normalize(vector))
        return output


class SentenceTransformerEncoder(TextEncoder):
    """Lazy optional Sentence Transformers backend."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        self.encoder_id = model_id
        self.revision = revision
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is optional. Install requirements-ml.txt "
                    "or set retrieval.dense.backend=hashing for offline execution."
                ) from exc
            device = None if self.device == "auto" else self.device
            self._model = SentenceTransformer(
                self.encoder_id, revision=self.revision, device=device
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._load().encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()


class DenseRetriever(BaseRetriever):
    system_id = "dense"

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self._documents: list[DocumentRecord] = []
        self._vectors: list[list[float]] = []

    @property
    def documents(self) -> list[DocumentRecord]:
        return list(self._documents)

    @property
    def vectors(self) -> list[list[float]]:
        return [list(vector) for vector in self._vectors]

    def index(self, documents: Iterable[DocumentRecord]) -> None:
        self._documents = sorted(documents, key=lambda item: item.article_id)
        if not self._documents:
            raise ValueError("cannot build a dense index with no documents")
        texts = [
            f"{document.title}\n{document.raw_text}" for document in self._documents
        ]
        self._vectors = self.encoder.encode(texts)
        if len(self._vectors) != len(self._documents):
            raise ValueError("encoder returned the wrong number of vectors")

    def search(self, query: str, top_k: int = 10) -> list[RetrievedPassage]:
        if not self._documents:
            raise RuntimeError("index() must be called before search()")
        query_vector = self.encoder.encode([query])[0]
        scored = []
        for document, vector in zip(self._documents, self._vectors):
            score = sum(left * right for left, right in zip(query_vector, vector))
            scored.append((document, score, {"dense": score}))
        return stable_rank(scored, min(top_k, len(self._documents)))

    def state(self) -> dict:
        return {
            "system_id": self.system_id,
            "encoder_id": self.encoder.encoder_id,
            "model_revision": self.encoder.revision,
            "dimensions": len(self._vectors[0]) if self._vectors else 0,
            "vectors": self._vectors,
        }
