"""Small deterministic tokenizer for lexical baselines."""

from __future__ import annotations

import re

from vietrag.preprocessing import lexical_normalize

_TOKEN_RE = re.compile(r"[^\W_]+(?:[./,-][^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(lexical_normalize(text))
