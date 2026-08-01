from vietrag.preprocessing.normalization import (
    canonicalize_context,
    lexical_normalize,
    safe_normalize,
)


def test_nfkc_and_whitespace_normalization() -> None:
    assert safe_normalize("  Điều   １２  ") == "Điều 12"


def test_invisible_characters_are_removed() -> None:
    assert safe_normalize("học\u200b phí\ufeff") == "học phí"


def test_numbers_dates_percentages_and_identifiers_are_preserved() -> None:
    source = "Học phần SE1234: 50%, 12.500.000 đồng, ngày 27/07/2026."
    assert safe_normalize(source) == source
    assert "se1234" in lexical_normalize(source)


def test_canonical_normalization_is_idempotent() -> None:
    source = "Điều 12.\n  Điều kiện làm khóa luận"
    once = canonicalize_context(source)
    assert canonicalize_context(once) == once
