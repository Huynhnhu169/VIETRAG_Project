import io
import zipfile

from openpyxl import Workbook

from vietrag.data.virhe4qa import load_source_rows, parse_virhe4qa


def _xlsx_payload() -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(
        ["question", "context", "document", "article", "abstractive answer"]
    )
    worksheet.append(["Điều kiện là gì?", "Nội dung điều kiện.", "Quy chế", "Điều 1", "Đáp án"])
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def test_load_source_rows_reads_xlsx(tmp_path) -> None:
    source = tmp_path / "sample.xlsx"
    source.write_bytes(_xlsx_payload())
    rows = load_source_rows(source)
    assert rows[0]["question"] == "Điều kiện là gì?"


def test_parse_virhe4qa_reads_xlsx_inside_zip(tmp_path) -> None:
    source = tmp_path / "sample.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("raw_excel/train.xlsx", _xlsx_payload())
    documents, queries, row_count = parse_virhe4qa(source)
    assert row_count == 1
    assert len(documents) == 1
    assert queries[0].gold_context_ids == [documents[0].article_id]
    assert queries[0].reference_answer == "Đáp án"
    assert documents[0].metadata["upstream_split_ignored"] == "train"
    assert documents[0].metadata["upstream_splits_ignored"] == ["train"]
