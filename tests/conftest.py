from __future__ import annotations

import pytest

from vietrag.data.preparation import prepare_document
from vietrag.schemas import QueryRecord


@pytest.fixture
def make_documents_and_queries():
    documents = [
        prepare_document(
            doc_id="DOC_A",
            article_id="DOC_A_ART_1",
            parent_id="DOC_A",
            title="Điều 1",
            raw_text="Sinh viên cần tích lũy 100 tín chỉ.",
            source="synthetic",
            metadata={"document_name": "A", "article_number": "1"},
        ),
        prepare_document(
            doc_id="DOC_A",
            article_id="DOC_A_ART_2",
            parent_id="DOC_A",
            title="Điều 2",
            raw_text="Điểm trung bình phải từ 2,50 trở lên.",
            source="synthetic",
            metadata={"document_name": "A", "article_number": "2"},
        ),
        prepare_document(
            doc_id="DOC_B",
            article_id="DOC_B_ART_1",
            parent_id="DOC_B",
            title="Điều 1",
            raw_text="Hạn đóng học phí là ngày 15/09/2026.",
            source="synthetic",
            metadata={"document_name": "B", "article_number": "1"},
        ),
        prepare_document(
            doc_id="DOC_C",
            article_id="DOC_C_ART_1",
            parent_id="DOC_C",
            title="Điều 1",
            raw_text="Học phần SE1234 yêu cầu học phần SE1001.",
            source="synthetic",
            metadata={"document_name": "C", "article_number": "1"},
        ),
    ]
    queries = [
        QueryRecord(
            query_id="Q1",
            base_query_id="Q1",
            query="Cần bao nhiêu tín chỉ?",
            clean_query="Cần bao nhiêu tín chỉ?",
            noise_type="clean",
            answerable=True,
            ambiguous=False,
            gold_context_ids=["DOC_A_ART_1"],
            reference_answer="100 tín chỉ",
            source_dataset="synthetic",
            split="unassigned",
        ),
        QueryRecord(
            query_id="Q1_typo",
            base_query_id="Q1",
            query="Can bao nhieu tin chi?",
            clean_query="Cần bao nhiêu tín chỉ?",
            noise_type="typo",
            answerable=True,
            ambiguous=False,
            gold_context_ids=["DOC_A_ART_1"],
            reference_answer="100 tín chỉ",
            source_dataset="synthetic",
            split="unassigned",
        ),
        QueryRecord(
            query_id="Q2",
            base_query_id="Q2",
            query="Hạn đóng học phí?",
            clean_query="Hạn đóng học phí?",
            noise_type="clean",
            answerable=True,
            ambiguous=False,
            gold_context_ids=["DOC_B_ART_1"],
            reference_answer="15/09/2026",
            source_dataset="synthetic",
            split="unassigned",
        ),
        QueryRecord(
            query_id="Q3",
            base_query_id="Q3",
            query="Điều kiện học SE1234?",
            clean_query="Điều kiện học SE1234?",
            noise_type="clean",
            answerable=True,
            ambiguous=False,
            gold_context_ids=["DOC_C_ART_1"],
            reference_answer="SE1001",
            source_dataset="synthetic",
            split="unassigned",
        ),
    ]
    return documents, queries
