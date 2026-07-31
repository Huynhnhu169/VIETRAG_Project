"""Text preprocessing utilities."""

from .normalization import (
    canonicalize_context,
    lexical_normalize,
    remove_diacritics,
    safe_normalize,
)

__all__ = [
    "canonicalize_context",
    "lexical_normalize",
    "remove_diacritics",
    "safe_normalize",
]
