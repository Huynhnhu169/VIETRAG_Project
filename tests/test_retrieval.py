import pytest

from vietrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HashingEncoder,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from vietrag.schemas import RetrievedPassage


def test_retrieval_output_schema_and_deterministic_ranking(
    make_documents_and_queries,
) -> None:
    documents, _ = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(reversed(documents))
    first = retriever.search("100 tín chỉ", top_k=3)
    second = retriever.search("100 tín chỉ", top_k=3)
    assert [item.context_id for item in first] == [
        item.context_id for item in second
    ]
    assert first[0].context_id == "DOC_A_ART_1"
    assert all(isinstance(item, RetrievedPassage) for item in first)
    assert [item.rank for item in first] == [1, 2, 3]


def test_hashing_dense_and_hybrid_run_offline(make_documents_and_queries) -> None:
    documents, _ = make_documents_and_queries
    bm25 = BM25Retriever()
    dense = DenseRetriever(HashingEncoder(dimensions=64))
    hybrid = HybridRetriever(bm25, dense, mode="rrf", candidate_k=4)
    hybrid.index(documents)
    results = hybrid.search("hạn học phí 15/09/2026", top_k=2)
    assert len(results) == 2
    assert results[0].context_id == "DOC_B_ART_1"
    assert {"bm25", "dense"}.issubset(results[0].component_scores)


def test_rrf_uses_rank_formula(make_documents_and_queries) -> None:
    documents, _ = make_documents_and_queries
    bm25 = BM25Retriever()
    bm25.index(documents)
    first = bm25.search("tín chỉ", top_k=4)
    fused = reciprocal_rank_fusion({"a": first, "b": first}, rrf_k=60, top_k=1)
    assert fused[0].score == pytest.approx(2 / 61)


def test_weighted_alpha_is_validated(make_documents_and_queries) -> None:
    documents, _ = make_documents_and_queries
    hybrid = HybridRetriever(
        BM25Retriever(),
        DenseRetriever(HashingEncoder(64)),
        mode="alpha",
        alpha=1.2,
    )
    hybrid.index(documents)
    with pytest.raises(ValueError):
        hybrid.search("test")
