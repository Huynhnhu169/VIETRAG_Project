import pytest

from vietrag.evaluation.metrics import evaluate_ranking
from vietrag.evaluation.runner import evaluate_retriever, save_evaluation_artifacts
from vietrag.retrieval import BM25Retriever
from vietrag.reranking import LexicalReranker


def test_ranking_metrics() -> None:
    metrics = evaluate_ranking(["D2", "D1", "D3"], ["D1"])
    assert metrics["recall@1"] == 0
    assert metrics["recall@5"] == 1
    assert metrics["mrr@10"] == pytest.approx(0.5)
    assert 0 < metrics["ndcg@10"] < 1


def test_evaluation_writes_query_and_aggregate_artifacts(
    make_documents_and_queries, tmp_path
) -> None:
    documents, queries = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    rows, summary = evaluate_retriever(retriever, queries, top_k=4)
    save_evaluation_artifacts(tmp_path, rows, summary, {"seed": 42})
    assert (tmp_path / "predictions.jsonl").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "run_metadata.json").is_file()
    assert summary["aggregate"]["query_count"] == len(queries)


def test_reranked_evaluation_retains_largest_metric_cutoff(
    make_documents_and_queries,
) -> None:
    documents, queries = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    rows, _ = evaluate_retriever(
        retriever,
        queries,
        cutoffs=(1, 2),
        reranker=LexicalReranker(),
        reranker_input_k=2,
        reranker_output_k=1,
    )
    assert all(len(row["retrieved_context_ids"]) == 2 for row in rows)
    assert all("recall@2" in row["metrics"] for row in rows)


def test_evaluation_rejects_empty_cutoffs(make_documents_and_queries) -> None:
    documents, queries = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    with pytest.raises(ValueError, match="cutoffs"):
        evaluate_retriever(retriever, queries, cutoffs=())
