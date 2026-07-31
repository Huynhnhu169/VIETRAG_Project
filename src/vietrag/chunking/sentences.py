"""Conservative sentence splitting for evidence selection."""

from __future__ import annotations

import re

_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-ỸĐ0-9])")


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _BOUNDARY_RE.split(text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])
