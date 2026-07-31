"""Streamlit demo backed by the same RAGPipeline used by the CLI and tests."""

from __future__ import annotations

from pathlib import Path

from vietrag.config import load_config
from vietrag.data.preparation import deduplicate_documents, remap_query_contexts
from vietrag.data.virhe4qa import parse_virhe4qa
from vietrag.pipeline import RAGPipeline
from vietrag.schemas import DocumentRecord
from vietrag.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parent


def load_demo_documents() -> tuple[list[DocumentRecord], str]:
    processed = ROOT / "data" / "processed" / "corpus.jsonl"
    if processed.is_file():
        return (
            [DocumentRecord.from_dict(row) for row in read_jsonl(processed)],
            "local processed corpus",
        )
    synthetic = ROOT / "data" / "samples" / "raw_synthetic.jsonl"
    documents, queries, _ = parse_virhe4qa(
        synthetic, source_name="synthetic"
    )
    documents, aliases = deduplicate_documents(documents)
    remap_query_contexts(queries, aliases)
    return documents, "tracked synthetic smoke fixture"


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    st.set_page_config(
        page_title="VietRAG Policy Assistant",
        page_icon="📚",
        layout="wide",
    )
    st.title("VietRAG Policy Assistant")
    st.caption(
        "Vietnamese university-regulation retrieval with grounded citations "
        "and evidence-based abstention."
    )
    documents, corpus_label = load_demo_documents()
    st.info(f"Corpus: {corpus_label} · {len(documents)} unique contexts")

    mode_label = st.selectbox(
        "Retrieval mode",
        ("Hybrid RRF", "BM25", "Dense (offline hashing)", "Hybrid alpha"),
    )
    mode = {
        "Hybrid RRF": "hybrid_rrf",
        "BM25": "bm25",
        "Dense (offline hashing)": "dense",
        "Hybrid alpha": "hybrid_alpha",
    }[mode_label]
    query = st.text_input(
        "Câu hỏi",
        value="Điều kiện để đăng ký khóa luận tốt nghiệp là gì?",
    )
    compare = st.checkbox("Compare BM25, dense, and hybrid rankings")
    if not st.button("Ask", type="primary") or not query.strip():
        return

    config = load_config(ROOT / "configs" / "default.yaml")
    pipeline = RAGPipeline(documents, config, retrieval_mode=mode)
    result = pipeline.ask(query)
    st.markdown("#### Safely normalized query")
    st.code(result.prediction.normalized_query, language=None)
    if result.prediction.abstained:
        st.warning(result.prediction.answer)
    else:
        st.success(result.prediction.answer)
        st.markdown("#### Verifiable citations")
        for citation in result.prediction.citations:
            label = (
                f"{citation.document_name} · {citation.title} "
                f"(article {citation.article_number or 'n/a'})"
            )
            with st.expander(label):
                st.write(citation.evidence_snippet)
                st.caption(f"Stable evidence ID: {citation.article_id}")

    st.markdown("#### Retrieved evidence")
    for passage in result.passages:
        with st.expander(
            f"#{passage.rank} · {passage.document.title} · score {passage.score:.4f}"
        ):
            st.write(passage.document.raw_text)
            st.json(passage.component_scores)
    st.caption(
        f"Total latency: {result.prediction.latency_ms['total']:.2f} ms · "
        f"evidence sufficiency: {result.grounded.evidence_sufficiency:.3f}"
    )

    if compare:
        comparison = {}
        for comparison_mode in ("bm25", "dense", "hybrid_rrf"):
            candidate_pipeline = RAGPipeline(
                documents, config, retrieval_mode=comparison_mode
            )
            candidate_result = candidate_pipeline.ask(query)
            comparison[comparison_mode] = [
                {
                    "rank": item.rank,
                    "context_id": item.context_id,
                    "title": item.document.title,
                    "score": round(item.score, 6),
                }
                for item in candidate_result.passages
            ]
        st.markdown("#### Retriever comparison")
        st.json(comparison)


if __name__ == "__main__":
    main()
