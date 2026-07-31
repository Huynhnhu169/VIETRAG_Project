"""Configuration-driven retrieval and reranking factories."""

from __future__ import annotations

from typing import Any

from vietrag.reranking import CrossEncoderReranker, LexicalReranker
from vietrag.retrieval.bm25 import BM25Retriever
from vietrag.retrieval.dense import (
    DenseRetriever,
    HashingEncoder,
    SentenceTransformerEncoder,
)
from vietrag.retrieval.hybrid import HybridRetriever
from vietrag.schemas import DocumentRecord


def build_retriever(
    documents: list[DocumentRecord],
    config: dict[str, Any],
    *,
    mode: str | None = None,
):
    retrieval = config["retrieval"]
    mode = mode or retrieval["mode"]
    bm25_config = retrieval.get("bm25", {})
    bm25 = BM25Retriever(
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    dense_config = retrieval.get("dense", {})
    backend = dense_config.get("backend", "hashing")
    if backend == "hashing":
        encoder = HashingEncoder(int(dense_config.get("dimensions", 384)))
    elif backend == "sentence_transformers":
        encoder = SentenceTransformerEncoder(
            str(dense_config["model_id"]),
            revision=dense_config.get("model_revision"),
            normalize_embeddings=bool(
                dense_config.get("normalize_embeddings", True)
            ),
            device=str(dense_config.get("device", "auto")),
            batch_size=int(dense_config.get("batch_size", 8)),
        )
    else:
        raise ValueError(f"unsupported dense backend: {backend}")
    dense = DenseRetriever(encoder)
    fusion = retrieval.get("fusion", {})
    candidate_k = int(retrieval.get("candidate_k", 20))
    if mode == "bm25":
        retriever = bm25
    elif mode == "dense":
        retriever = dense
    elif mode in {"hybrid_rrf", "rrf"}:
        retriever = HybridRetriever(
            bm25,
            dense,
            mode="rrf",
            rrf_k=int(fusion.get("rrf_k", 60)),
            candidate_k=candidate_k,
        )
    elif mode in {"hybrid_alpha", "alpha"}:
        retriever = HybridRetriever(
            bm25,
            dense,
            mode="alpha",
            alpha=float(fusion.get("alpha", 0.5)),
            candidate_k=candidate_k,
        )
    else:
        raise ValueError(f"unsupported retrieval mode: {mode}")
    retriever.index(documents)
    return retriever


def build_reranker(config: dict[str, Any]):
    reranking = config.get("reranking", {})
    if not reranking.get("enabled", False):
        return None
    backend = reranking.get("backend", "lexical")
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross_encoder":
        return CrossEncoderReranker(
            str(reranking["model_id"]),
            revision=reranking.get("model_revision"),
            device=str(reranking.get("device", "auto")),
            batch_size=int(reranking.get("batch_size", 4)),
        )
    raise ValueError(f"unsupported reranking backend: {backend}")
