"""Evidence sufficiency, citation construction, clarification, and abstention."""

from __future__ import annotations

from dataclasses import dataclass

from vietrag.chunking import split_sentences
from vietrag.generation.providers import AnswerProvider
from vietrag.retrieval.tokenization import tokenize
from vietrag.schemas import Citation, RetrievedPassage

_AMBIGUOUS_MARKERS = {
    "điều kiện này",
    "quy định này",
    "trường hợp đó",
    "cái này",
}
_CONTENT_STOPWORDS = {
    "ạ",
    "ai",
    "bao",
    "các",
    "cái",
    "cho",
    "có",
    "của",
    "da",
    "dang",
    "dau",
    "de",
    "dieu",
    "do",
    "đã",
    "đang",
    "đâu",
    "để",
    "điều",
    "đó",
    "được",
    "duoc",
    "gì",
    "gi",
    "khi",
    "không",
    "khong",
    "là",
    "la",
    "mai",
    "một",
    "mot",
    "nào",
    "nao",
    "này",
    "nay",
    "ngày",
    "ngay",
    "những",
    "nhung",
    "ở",
    "o",
    "phải",
    "phai",
    "thế",
    "the",
    "thì",
    "thi",
    "thời",
    "thoi",
    "trong",
    "từ",
    "tu",
    "và",
    "va",
    "về",
    "ve",
    "với",
    "voi",
}


@dataclass(slots=True)
class GroundedResult:
    answer: str
    citations: list[Citation]
    selected_evidence: list[RetrievedPassage]
    evidence_sufficiency: float
    abstained: bool
    clarification_requested: bool = False


def evidence_sufficiency(query: str, evidence: list[RetrievedPassage]) -> float:
    query_tokens = {
        token for token in tokenize(query) if token not in _CONTENT_STOPWORDS
    }
    if not query_tokens or not evidence:
        return 0.0
    best = 0.0
    for passage in evidence:
        evidence_tokens = set(
            tokenize(f"{passage.document.title} {passage.document.raw_text}")
        )
        overlap = len(query_tokens & evidence_tokens) / len(query_tokens)
        best = max(best, overlap)
    return best


def is_obviously_ambiguous(query: str) -> bool:
    normalized = " ".join(tokenize(query))
    return len(tokenize(query)) <= 4 and any(
        marker in normalized for marker in _AMBIGUOUS_MARKERS
    )


def _best_snippet(query: str, passage: RetrievedPassage) -> str:
    query_tokens = set(tokenize(query))
    sentences = split_sentences(passage.document.raw_text)
    if not sentences:
        return passage.document.raw_text
    return max(
        sentences,
        key=lambda sentence: (
            len(query_tokens & set(tokenize(sentence))),
            -sentences.index(sentence),
        ),
    )


class GroundedAnswerer:
    def __init__(
        self,
        provider: AnswerProvider,
        *,
        abstention_threshold: float,
        max_evidence: int = 3,
        abstention_message: str = (
            "Tôi chưa tìm thấy đủ bằng chứng trong các văn bản hiện có "
            "để trả lời câu hỏi này."
        ),
        clarification_message: str = (
            "Câu hỏi chưa đủ thông tin. Bạn vui lòng nêu rõ đối tượng, "
            "chương trình hoặc thời điểm áp dụng."
        ),
    ) -> None:
        if not 0 <= abstention_threshold <= 1:
            raise ValueError("abstention_threshold must be between 0 and 1")
        self.provider = provider
        self.abstention_threshold = abstention_threshold
        self.max_evidence = max_evidence
        self.abstention_message = abstention_message
        self.clarification_message = clarification_message

    def answer(
        self,
        query: str,
        passages: list[RetrievedPassage],
        *,
        ambiguous: bool = False,
    ) -> GroundedResult:
        if ambiguous or is_obviously_ambiguous(query):
            return GroundedResult(
                answer=self.clarification_message,
                citations=[],
                selected_evidence=[],
                evidence_sufficiency=0.0,
                abstained=True,
                clarification_requested=True,
            )
        selected = passages[: self.max_evidence]
        sufficiency = evidence_sufficiency(query, selected)
        if sufficiency < self.abstention_threshold:
            return GroundedResult(
                answer=self.abstention_message,
                citations=[],
                selected_evidence=[],
                evidence_sufficiency=sufficiency,
                abstained=True,
            )
        answer = self.provider.generate(query, selected)
        if not answer.strip():
            return GroundedResult(
                answer=self.abstention_message,
                citations=[],
                selected_evidence=[],
                evidence_sufficiency=sufficiency,
                abstained=True,
            )
        # Extractive providers return a verbatim sentence. In that case, cite
        # only passages that contain the answer; generative providers retain
        # the full selected evidence set because claim-level attribution is
        # unavailable without a separate verifier.
        directly_supporting = [
            passage for passage in selected if answer in passage.document.raw_text
        ]
        cited_evidence = directly_supporting or selected
        citations = [
            Citation(
                doc_id=passage.document.doc_id,
                article_id=passage.document.article_id,
                title=passage.document.title,
                document_name=str(
                    passage.document.metadata.get(
                        "document_name", passage.document.doc_id
                    )
                ),
                article_number=str(
                    passage.document.metadata.get("article_number", "")
                ),
                evidence_snippet=_best_snippet(query, passage),
            )
            for passage in cited_evidence
        ]
        return GroundedResult(
            answer=answer,
            citations=citations,
            selected_evidence=cited_evidence,
            evidence_sufficiency=sufficiency,
            abstained=False,
        )
