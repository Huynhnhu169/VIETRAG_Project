from vietrag.generation import ExtractiveProvider, GroundedAnswerer
from vietrag.retrieval import BM25Retriever


def test_citations_always_refer_to_selected_evidence(
    make_documents_and_queries,
) -> None:
    documents, _ = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    passages = retriever.search("Cần bao nhiêu tín chỉ?", top_k=3)
    answerer = GroundedAnswerer(
        ExtractiveProvider(), abstention_threshold=0.1, max_evidence=2
    )
    result = answerer.answer("Cần bao nhiêu tín chỉ?", passages)
    selected_ids = {
        passage.context_id for passage in result.selected_evidence
    }
    assert not result.abstained
    assert result.citations
    assert {citation.article_id for citation in result.citations} <= selected_ids
    assert all(
        citation.evidence_snippet in passage.document.raw_text
        for citation, passage in zip(result.citations, result.selected_evidence)
    )


def test_abstention_when_evidence_is_insufficient(
    make_documents_and_queries,
) -> None:
    documents, _ = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    passages = retriever.search("Thời tiết ngày mai thế nào?", top_k=3)
    result = GroundedAnswerer(
        ExtractiveProvider(), abstention_threshold=0.5
    ).answer("Thời tiết ngày mai thế nào?", passages)
    assert result.abstained
    assert result.citations == []
    assert result.selected_evidence == []


def test_ambiguous_question_requests_clarification(
    make_documents_and_queries,
) -> None:
    documents, _ = make_documents_and_queries
    retriever = BM25Retriever()
    retriever.index(documents)
    result = GroundedAnswerer(
        ExtractiveProvider(), abstention_threshold=0.1
    ).answer("Điều kiện này?", retriever.search("Điều kiện này?", top_k=2))
    assert result.abstained
    assert result.clarification_requested
