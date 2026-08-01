from vietrag.data.splits import (
    assert_manifest_integrity,
    create_context_disjoint_manifest,
    create_document_fold_manifest,
)


def test_context_disjoint_and_query_family_integrity(
    make_documents_and_queries,
) -> None:
    documents, queries = make_documents_and_queries
    manifest = create_context_disjoint_manifest(
        documents,
        queries,
        seed=42,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
    )
    assert_manifest_integrity(manifest, documents, queries)
    assert manifest["query_assignments"]["Q1"] == manifest["query_assignments"]["Q1_typo"]
    assert manifest["frozen"] is True


def test_document_disjoint_grouped_folds(make_documents_and_queries) -> None:
    documents, queries = make_documents_and_queries
    manifest = create_document_fold_manifest(
        documents, queries, seed=42, n_folds=3
    )
    assert_manifest_integrity(manifest, documents, queries)
    doc_a_folds = {
        manifest["article_assignments"][document.article_id]
        for document in documents
        if document.doc_id == "DOC_A"
    }
    assert len(doc_a_folds) == 1


def test_manifest_checksum_detects_mutation(make_documents_and_queries) -> None:
    documents, queries = make_documents_and_queries
    manifest = create_context_disjoint_manifest(documents, queries, seed=42)
    manifest["article_assignments"]["DOC_A_ART_1"] = "test"
    try:
        assert_manifest_integrity(manifest, documents, queries)
    except AssertionError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("mutated manifest was accepted")
