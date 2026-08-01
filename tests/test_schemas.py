import pytest

from vietrag.schemas import DocumentRecord, QueryRecord


def test_document_requires_stable_sha256_hash() -> None:
    with pytest.raises(ValueError):
        DocumentRecord(
            doc_id="D1",
            article_id="D1_A1",
            parent_id="D1",
            title="Điều 1",
            raw_text="Nội dung",
            normalized_text="nội dung",
            source="synthetic",
            context_hash="bad",
        )


def test_answerable_query_requires_gold_evidence() -> None:
    with pytest.raises(ValueError):
        QueryRecord(
            query_id="Q1",
            base_query_id="Q1",
            query="Câu hỏi?",
            clean_query="Câu hỏi?",
            noise_type="clean",
            answerable=True,
            ambiguous=False,
            gold_context_ids=[],
            reference_answer="",
            source_dataset="synthetic",
            split="train",
        )
