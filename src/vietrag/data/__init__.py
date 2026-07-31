"""Dataset parsing, preparation, auditing, and split generation."""

from .preparation import (
    build_audit_report,
    compute_context_hash,
    deduplicate_documents,
)
from .splits import (
    assert_manifest_integrity,
    create_context_disjoint_manifest,
    create_document_fold_manifest,
    verify_manifest_checksum,
)

__all__ = [
    "assert_manifest_integrity",
    "build_audit_report",
    "compute_context_hash",
    "create_context_disjoint_manifest",
    "create_document_fold_manifest",
    "deduplicate_documents",
    "verify_manifest_checksum",
]
