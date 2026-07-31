"""Persist corpus metadata separately from backend-specific index state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vietrag.retrieval.base import BaseRetriever
from vietrag.schemas import DocumentRecord
from vietrag.utils.io import read_json, read_jsonl, write_json, write_jsonl


def save_index(
    directory: str | Path,
    retriever: BaseRetriever,
    *,
    configuration: dict[str, Any],
) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "documents.jsonl",
        (document.to_dict() for document in retriever.documents),
    )
    state = retriever.state() if hasattr(retriever, "state") else {}
    write_json(
        output / "index.json",
        {
            "system_id": retriever.system_id,
            "configuration": configuration,
            "state": state,
        },
    )


def load_index_documents(directory: str | Path) -> list[DocumentRecord]:
    return [
        DocumentRecord.from_dict(row)
        for row in read_jsonl(Path(directory) / "documents.jsonl")
    ]


def load_index_metadata(directory: str | Path) -> dict[str, Any]:
    return read_json(Path(directory) / "index.json")
