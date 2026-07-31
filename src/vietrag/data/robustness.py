"""Deterministic, auditable Vietnamese query robustness variants."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from vietrag.data.splits import verify_manifest_checksum
from vietrag.preprocessing import remove_diacritics, safe_normalize
from vietrag.schemas import QueryRecord

_PROTECTED_RE = re.compile(
    r"(?<!\w)[A-ZĐ]{2,}\d+[A-Z0-9-]*(?!\w)"
    r"|(?<!\w)\d+(?:[.,]\d+)*(?:%|/[0-9./-]+)?(?!\w)",
    flags=re.UNICODE,
)
_TOKEN_RE = re.compile(r"\w+|\W+", flags=re.UNICODE)

_ABBREVIATIONS = (
    ("khóa luận tốt nghiệp", "KLTN"),
    ("điểm trung bình", "ĐTB"),
    ("sinh viên", "SV"),
    ("học phí", "HP"),
    ("điều kiện", "ĐK"),
    ("không", "ko"),
)


def protected_expressions(text: str) -> list[str]:
    return _PROTECTED_RE.findall(text)


def _typo(text: str, rng: random.Random) -> str:
    tokens = _TOKEN_RE.findall(text)
    candidates = [
        index
        for index, token in enumerate(tokens)
        if token.isalpha() and len(token) >= 5 and token.upper() != token
    ]
    if not candidates:
        return text
    index = rng.choice(candidates)
    token = tokens[index]
    position = rng.randrange(1, len(token) - 1)
    tokens[index] = (
        token[:position]
        + token[position + 1]
        + token[position]
        + token[position + 2 :]
    )
    return "".join(tokens)


def _abbreviate(text: str) -> str:
    output = text
    for phrase, abbreviation in _ABBREVIATIONS:
        output = re.sub(
            rf"\b{re.escape(phrase)}\b",
            abbreviation,
            output,
            flags=re.IGNORECASE,
        )
    return output


def _informal(text: str) -> str:
    core = text.rstrip(" ?")
    return f"cho mình hỏi {core} là sao ạ?"


def _audit_variant(base: QueryRecord, variant: QueryRecord) -> list[str]:
    flags: list[str] = []
    if not variant.query.strip():
        flags.append("empty_variant")
    if protected_expressions(base.clean_query) != protected_expressions(variant.query):
        flags.append("protected_expression_changed")
    if variant.query == base.clean_query and variant.noise_type != "clean":
        flags.append("transformation_had_no_effect")
    if variant.gold_context_ids != base.gold_context_ids:
        flags.append("gold_evidence_changed")
    if variant.split != base.split:
        flags.append("split_changed")
    return flags


def create_robustness_variants(
    queries: list[QueryRecord],
    manifest: dict[str, Any],
    *,
    seed: int,
    paraphraser: Callable[[str], str] | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    if not manifest.get("frozen") or not verify_manifest_checksum(manifest):
        raise ValueError("noise generation requires a frozen split manifest")
    manifest_assignments = manifest.get("query_assignments", {})
    base_queries: dict[str, QueryRecord] = {}
    for query in queries:
        existing = base_queries.get(query.base_query_id)
        if existing is None or query.noise_type == "clean":
            if (
                existing is not None
                and existing.noise_type == "clean"
                and existing.clean_query != query.clean_query
            ):
                raise ValueError(
                    f"conflicting clean queries for {query.base_query_id}"
                )
            base_queries[query.base_query_id] = query
    variants: list[QueryRecord] = []
    audit_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for base in sorted(base_queries.values(), key=lambda item: item.base_query_id):
        expected_split = manifest_assignments.get(base.query_id)
        if expected_split is None:
            raise ValueError(f"query {base.query_id} is absent from the split manifest")
        clean = replace(base, split=str(expected_split), noise_type="clean")
        transforms: list[tuple[str, str]] = [
            ("clean", clean.clean_query),
            ("no_diacritic", remove_diacritics(clean.clean_query)),
            (
                "typo",
                _typo(
                    clean.clean_query,
                    random.Random(f"{seed}:{clean.base_query_id}:typo"),
                ),
            ),
            ("abbreviation", _abbreviate(clean.clean_query)),
            ("informal", _informal(clean.clean_query)),
        ]
        if paraphraser is not None:
            transforms.append(("paraphrase", safe_normalize(paraphraser(clean.clean_query))))
        else:
            skipped.append(
                {
                    "base_query_id": clean.base_query_id,
                    "noise_type": "paraphrase",
                    "reason": "no reliable paraphraser configured",
                }
            )
        for noise_type, query_text in transforms:
            query_id = (
                clean.query_id
                if noise_type == "clean"
                else f"{clean.base_query_id}_{noise_type}"
            )
            variant = replace(
                clean,
                query_id=query_id,
                base_query_id=clean.base_query_id,
                query=safe_normalize(query_text),
                noise_type=noise_type,
            )
            flags = _audit_variant(clean, variant)
            variants.append(variant)
            audit_rows.append(
                {
                    "query_id": query_id,
                    "base_query_id": clean.base_query_id,
                    "noise_type": noise_type,
                    "flags": flags,
                    "requires_manual_review": bool(flags),
                }
            )
    return variants, {
        "seed": seed,
        "variant_count": len(variants),
        "flagged_count": sum(row["requires_manual_review"] for row in audit_rows),
        "variants": audit_rows,
        "skipped": skipped,
    }
