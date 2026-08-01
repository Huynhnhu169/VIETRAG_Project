from vietrag.pipeline import RAGPipeline


def _config():
    return {
        "retrieval": {
            "mode": "hybrid_rrf",
            "top_k": 4,
            "candidate_k": 4,
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense": {"backend": "hashing", "dimensions": 64},
            "fusion": {"rrf_k": 60, "alpha": 0.5},
        },
        "reranking": {
            "enabled": True,
            "backend": "lexical",
            "input_k": 4,
            "output_k": 3,
        },
        "generation": {
            "provider": "extractive",
            "max_evidence": 2,
            "abstention_threshold": 0.1,
            "abstention_message": "Không đủ bằng chứng.",
            "clarification_message": "Vui lòng làm rõ.",
        },
    }


def test_end_to_end_offline_smoke_pipeline(make_documents_and_queries) -> None:
    documents, _ = make_documents_and_queries
    pipeline = RAGPipeline(documents, _config())
    result = pipeline.ask("Cần bao nhiêu tín chỉ?", query_id="SMOKE")
    assert result.prediction.query_id == "SMOKE"
    assert result.prediction.answer
    assert not result.prediction.abstained
    assert result.prediction.citations
    retrieved = set(result.prediction.retrieved_context_ids)
    assert all(
        citation.article_id in retrieved
        for citation in result.prediction.citations
    )
    assert result.prediction.latency_ms["total"] >= 0


def test_end_to_end_pipeline_abstains_for_unrelated_query(
    make_documents_and_queries,
) -> None:
    documents, _ = make_documents_and_queries
    pipeline = RAGPipeline(documents, _config())
    result = pipeline.ask("Thời tiết ngày mai thế nào?")
    assert result.prediction.abstained
    assert result.prediction.citations == []
