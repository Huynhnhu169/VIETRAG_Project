from vietrag.data.preparation import (
    compute_context_hash,
    deduplicate_documents,
    prepare_document,
)


def test_context_hash_is_stable_after_safe_canonicalization() -> None:
    assert compute_context_hash("Điều 1.\n Nội dung") == compute_context_hash(
        "Điều 1.  Nội dung"
    )


def test_exact_deduplication_returns_alias_mapping() -> None:
    left = prepare_document(
        doc_id="D1",
        article_id="D1_A1",
        parent_id="D1",
        title="Điều 1",
        raw_text="Nội dung giống nhau.",
        source="synthetic",
    )
    right = prepare_document(
        doc_id="D2",
        article_id="D2_A9",
        parent_id="D2",
        title="Điều 9",
        raw_text="Nội dung  giống nhau.",
        source="synthetic",
    )
    documents, aliases = deduplicate_documents([left, right])
    assert len(documents) == 1
    assert aliases == {"D1_A1": "D1_A1", "D2_A9": "D1_A1"}
    assert len(documents[0].metadata["provenance"]) == 2
