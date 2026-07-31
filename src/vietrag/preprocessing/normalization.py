"""Conservative Vietnamese text normalization.

The canonical text remains Vietnamese and keeps all visible legal content.
Only compatibility normalization, invisible formatting characters, and
whitespace are changed.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_INVISIBLE_CODEPOINTS = {
    "\u00ad",  # soft hyphen
    "\u034f",  # combining grapheme joiner
    "\u061c",  # Arabic letter mark
    "\u180e",  # Mongolian vowel separator
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte order mark
}


def _remove_invisible_characters(text: str) -> str:
    return "".join(
        char
        for char in text
        if char not in _INVISIBLE_CODEPOINTS
        and unicodedata.category(char) not in {"Cf", "Cc"}
        or char in {"\n", "\r", "\t"}
    )


def safe_normalize(text: str) -> str:
    """Normalize format without correcting or paraphrasing content."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _remove_invisible_characters(normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def canonicalize_context(text: str) -> str:
    """Return the content used for deterministic deduplication and hashing."""
    return safe_normalize(text)


def lexical_normalize(text: str) -> str:
    """Return a conservative lowercase copy for lexical indexing only."""
    return safe_normalize(text).casefold()


def remove_diacritics(text: str) -> str:
    """Create an explicitly separate no-diacritic query variant."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")
