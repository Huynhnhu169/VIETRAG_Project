"""VietRAG policy assistant package."""

from .schemas import (
    Citation,
    DocumentRecord,
    Prediction,
    QueryRecord,
    RetrievedPassage,
)

__all__ = [
    "Citation",
    "DocumentRecord",
    "Prediction",
    "QueryRecord",
    "RetrievedPassage",
]

__version__ = "0.1.0"
