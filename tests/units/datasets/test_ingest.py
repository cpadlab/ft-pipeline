
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from ft.datasets.ingest import PDFIngestor, _ExtractedLine


def _create_sample_pdf(path: Path, *, include_empty_page: bool = False) -> Path:
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "System Architecture", fontsize=18)
    page1.insert_text((72, 92), "Overview", fontsize=18)
    page1.insert_text((72, 130), "Node1", fontsize=11)
    page1.insert_text((72, 150), "- Bullet item with enough detail.", fontsize=11)
    page1.insert_text((72, 170), "This is a long hyphen-", fontsize=11)
    page1.insert_text((72, 190), "ated word and another", fontsize=11)
    page1.insert_text((72, 210), "continuation of the sentence that should be kept in the chunk.", fontsize=11)
    page1.insert_text((72, 240), "42", fontsize=11)

    if include_empty_page:
        doc.new_page()

    doc.save(path)
    doc.close()
    return path


def test_pdf_ingestor_init_validation() -> None:
    with pytest.raises(ValueError, match="heading_size_ratio"):
        PDFIngestor(heading_size_ratio=1.0)

    with pytest.raises(ValueError, match="min_content_chars"):
        PDFIngestor(min_content_chars=-1)


def test_ingest_path_file_and_directory(tmp_path: Path) -> None:
    pdf1 = _create_sample_pdf(tmp_path / "a.pdf", include_empty_page=True)
    _create_sample_pdf(tmp_path / "b.pdf")

    ingestor = PDFIngestor(min_content_chars=40)
    single_chunks = ingestor.ingest_path(pdf1)
    assert len(single_chunks) == 1
    assert single_chunks[0].chunk_id == "a_p001"
    assert single_chunks[0].section_title == "System Architecture Overview"
    assert "Node1" not in single_chunks[0].content
    assert "hyphenated word" in single_chunks[0].content
    assert "another continuation" in single_chunks[0].content

    dir_chunks = ingestor.ingest_path(tmp_path)
    assert len(dir_chunks) == 2


def test_ingest_path_error_cases(tmp_path: Path) -> None:
    ingestor = PDFIngestor()

    with pytest.raises(FileNotFoundError):
        ingestor.ingest_path(tmp_path / "missing.pdf")

    txt_file = tmp_path / "note.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid PDF"):
        ingestor.ingest_path(txt_file)

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No PDF files"):
        ingestor.ingest_path(empty_dir)


def test_process_page_skips_short_content(tmp_path: Path) -> None:
    pdf = _create_sample_pdf(tmp_path / "short.pdf")
    ingestor = PDFIngestor(min_content_chars=5000)
    assert ingestor.ingest_path(pdf) == []


def test_split_title_and_body_edge_cases() -> None:
    ingestor = PDFIngestor()
    lines = [
        _ExtractedLine(text="First line", max_font_size=11.0, y_top=10.0),
        _ExtractedLine(text="Large later", max_font_size=18.0, y_top=30.0),
    ]
    title, body = ingestor._split_title_and_body(lines, heading_threshold=15.0)
    assert title == "Sin título"
    assert body == ["First line", "Large later"]


def test_format_body_lines_keep_diagram_labels_and_join_paragraphs() -> None:
    ingestor = PDFIngestor(filter_diagram_labels=False)
    assert ingestor._format_body_lines([]) == ""
    assert ingestor._format_body_lines(["Label", "Next."]) == "Label\nNext."


def test_estimate_body_font_size_empty_doc() -> None:
    ingestor = PDFIngestor()
    empty_doc = pymupdf.open()
    empty_doc.new_page()
    assert ingestor._estimate_body_font_size(empty_doc) == 10.0
    empty_doc.close()

    fake_page = SimpleNamespace(
        get_text=lambda _mode: {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "   ", "size": 24.0},
                                {"text": "Normal body text.", "size": 12.0},
                            ],
                            "bbox": (0, 0, 10, 10),
                        }
                    ],
                }
            ]
        }
    )
    assert ingestor._estimate_body_font_size([fake_page]) == 12.0


def test_parse_line_dict_and_iter_text_blocks_edge_cases() -> None:
    ingestor = PDFIngestor()
    assert ingestor._parse_line_dict({"spans": []}) is None
    assert ingestor._parse_line_dict({"spans": [{"text": "  12  ", "size": 10.0}]}) is None

    fake_page = SimpleNamespace(
        get_text=lambda mode: {
            "blocks": [
                {"type": 1, "lines": []},
                {"type": 0, "lines": [{"spans": [{"text": "Hello", "size": 11.0}], "bbox": (0, 5, 10, 15)}]},
            ]
        }
    )
    blocks = list(ingestor._iter_text_blocks(fake_page))
    assert len(blocks) == 1
    lines = list(ingestor._iter_page_lines(fake_page))
    assert len(lines) == 1
    assert lines[0].text == "Hello"
