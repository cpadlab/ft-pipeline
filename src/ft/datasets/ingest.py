
from __future__ import annotations
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Final, TypedDict
import pymupdf


TEXT_BLOCK_TYPE: Final[int] = 0
DEFAULT_FONT_SIZE: Final[float] = 10.0
MIN_CONTENT_LENGTH: Final[int] = 120
MIN_PROSE_LINE_WORDS: Final[int] = 4
HEADING_VERTICAL_GAP: Final[float] = 35.0
PAGE_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+$")
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
BULLET_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[•\-*]\s+")


class _SpanDict(TypedDict, total=False):

    text: str
    size: float
    bbox: tuple[float, float, float, float]


class _LineDict(TypedDict, total=False):

    spans: list[_SpanDict]
    bbox: tuple[float, float, float, float]


class _BlockDict(TypedDict, total=False):

    type: int
    lines: list[_LineDict]


@dataclass(frozen=True, slots=True)
class _ExtractedLine:

    text: str
    max_font_size: float
    y_top: float


class DocumentChunk(BaseModel):

    chunk_id: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    section_title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)

    model_config = ConfigDict(
        frozen=True
    )


class PDFIngestor:
    """
    Extracts, cleans, and structures text chunks from PDF documents.

    Parses individual PDF files or entire directories using PyMuPDF, inferring
    section headings via font size heuristics, filtering isolated diagram labels,
    reconstructing line-wrapped paragraphs, and yielding structured `DocumentChunk`
    instances per page.

    Attributes:
        heading_size_ratio (Final[float]): Minimum ratio relative to estimated body
            font size required to classify a line as a section heading.
        min_content_chars (Final[int]): Minimum character count threshold required
            for a processed page to be emitted as a valid chunk.
        filter_diagram_labels (Final[bool]): Whether short non-prose labels
            typically found in embedded diagrams should be discarded.
    """


    def __init__(
        self,
        heading_size_ratio: float = 1.12,
        min_content_chars: int = MIN_CONTENT_LENGTH,
        filter_diagram_labels: bool = True,
    ) -> None:
        """
        Initializes the PDFIngestor with layout parsing and filtering thresholds.

        Args:
            heading_size_ratio (float, optional): Multiplier applied to the
                dominant body font size to detect headings. Must be greater than
                1.0. Defaults to 1.12.
            min_content_chars (int, optional): Minimum cleaned body character
                length required to keep a page chunk. Must be non-negative.
                Defaults to `MIN_CONTENT_LENGTH`.
            filter_diagram_labels (bool, optional): If True, removes short
                isolated lines lacking terminal punctuation that resemble diagram
                annotations. Defaults to True.

        Raises:
            ValueError: If `heading_size_ratio <= 1.0` or `min_content_chars < 0`.
        """
        if heading_size_ratio <= 1.0:
            raise ValueError("heading_size_ratio must be greater than 1.0")
        
        if min_content_chars < 0:
            raise ValueError("min_content_chars cannot be negative")

        self.heading_size_ratio: Final[float] = heading_size_ratio
        self.min_content_chars: Final[int] = min_content_chars
        self.filter_diagram_labels: Final[bool] = filter_diagram_labels


    def ingest_path(self, target_path: str | Path) -> list[DocumentChunk]:
        """
        Processes a single PDF file or all PDFs within a target directory.

        Args:
            target_path (str | Path): File path to a `.pdf` document or a
                directory containing PDF files.

        Returns:
            list[DocumentChunk]: Ordered list of extracted and cleaned page chunks
                across all resolved PDF files.

        Raises:
            FileNotFoundError: If `target_path` does not exist.
            ValueError: If `target_path` is not a `.pdf` file or is an empty
                directory without PDFs.
        """
        path = Path(target_path)
        pdf_files = self._resolve_pdf_files(path)

        chunks: list[DocumentChunk] = []
        for pdf_file in pdf_files:
            chunks.extend(self._process_single_pdf(pdf_file))

        return chunks


    @staticmethod
    def _resolve_pdf_files(path: Path) -> list[Path]:
        """
        Validates the target path and resolves a sorted list of PDF file paths.

        Args:
            path (Path): Resolved filesystem path pointing to a file or directory.

        Returns:
            list[Path]: List containing either the single validated PDF path or all
                `.pdf` files in the directory sorted alphabetically.

        Raises:
            FileNotFoundError: If `path` does not exist on disk.
            ValueError: If `path` is a file without a `.pdf` suffix or a directory
                containing zero `.pdf` files.
        """
        if not path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"The file is not a valid PDF: {path}")
            return [path]

        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files were found in: {path}")
            
        return pdf_files

    def _process_single_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        """
        Opens a PDF document, estimates typography thresholds, and extracts chunks.

        Args:
            pdf_path (Path): Path to the PDF file to process.

        Returns:
            list[DocumentChunk]: Validated `DocumentChunk` objects extracted from
                the document pages.
        """
        chunks: list[DocumentChunk] = []

        with closing(pymupdf.open(pdf_path)) as doc:
            body_font_size = self._estimate_body_font_size(doc)
            heading_threshold = body_font_size * self.heading_size_ratio

            for page_number, page in enumerate(doc, start=1):
                chunk = self._process_page(pdf_path, page, page_number, heading_threshold)
                
                if chunk is not None:
                    chunks.append(chunk)

        return chunks


    def _process_page(
        self, pdf_path: Path, page: pymupdf.Page, 
        page_number: int, heading_threshold: float,
    ) -> DocumentChunk | None:
        """
        Extracts, cleans, and structures a single PDF page into a `DocumentChunk`.

        Args:
            pdf_path (Path): Path to the source PDF file.
            page (pymupdf.Page): PyMuPDF page object to parse.
            page_number (int): 1-based page index within the document.
            heading_threshold (float): Font size cutoff used to distinguish
                section titles from body prose.

        Returns:
            DocumentChunk | None: Populated `DocumentChunk` if the page contains
                at least `min_content_chars` of cleaned text; otherwise None.
        """
        lines = list(self._iter_page_lines(page))
        if not lines:
            return None

        title, body_lines = self._split_title_and_body(lines, heading_threshold)
        cleaned_body = self._format_body_lines(body_lines)

        if len(cleaned_body) < self.min_content_chars:
            return None

        return DocumentChunk(
            chunk_id=f"{pdf_path.stem}_p{page_number:03d}",
            source_file=pdf_path.name,
            section_title=title,
            content=cleaned_body,
            page_start=page_number,
            page_end=page_number,
        )


    def _split_title_and_body(
        self, lines: list[_ExtractedLine], heading_threshold: float
    ) -> tuple[str, list[str]]:
        """
        Merges contiguous large-font lines into a title and separates body text.

        Args:
            lines (list[_ExtractedLine]): Vertically ordered text lines extracted
                from a single page.
            heading_threshold (float): Minimum font size required for a leading
                line to be treated as part of the section heading.

        Returns:
            tuple[str, list[str]]: A tuple containing:
                - **title** (`str`): Concatenated heading text, defaulting to
                  `"Sin título"` if no heading lines precede the body.
                - **body_lines** (`list[str]`): Remaining text lines belonging to
                  the page body.
        """
        title_parts: list[str] = []
        body_lines: list[str] = []
        last_title_y: float | None = None

        for line in lines:
            is_large = line.max_font_size >= heading_threshold
            is_contiguous = (
                last_title_y is None
                or (line.y_top - last_title_y) <= HEADING_VERTICAL_GAP
            )

            if is_large and is_contiguous and not body_lines:
                title_parts.append(line.text)
                last_title_y = line.y_top
            else:
                body_lines.append(line.text)

        title = " ".join(title_parts).strip() or "Sin título"
        
        return title, body_lines


    def _format_body_lines(self, body_lines: list[str]) -> str:
        """
        Filters diagram noise labels and rejoins margin-broken paragraphs.

        Args:
            body_lines (list[str]): Raw body text lines extracted from a page.

        Returns:
            str: Cleaned, paragraph-reconstructed prose string.
        """
        filtered: list[str] = []

        for line in body_lines:
            if self.filter_diagram_labels and self._is_diagram_noise(line):
                continue
            filtered.append(line)

        return self._join_broken_paragraphs(filtered)


    @staticmethod
    def _is_diagram_noise(line: str) -> bool:
        """
        Identifies short isolated labels typical of embedded diagrams or figures.

        Preserves bulleted items and lines that meet minimum prose word counts or
        end with terminal punctuation.

        Args:
            line (str): Single cleaned line of body text.

        Returns:
            bool: True if the line is classified as diagram annotation noise;
                False otherwise.
        """
        if BULLET_PATTERN.match(line):
            return False

        word_count = len(line.split())
        ends_with_punctuation = line[-1] in {".", ":", ";", "?", "!"}

        return word_count < MIN_PROSE_LINE_WORDS and not ends_with_punctuation


    @staticmethod
    def _join_broken_paragraphs(lines: list[str]) -> str:
        """
        Reconstructs paragraphs artificially split by PDF line breaks or hyphenation.

        Joins hyphenated word wraps seamlessly and concatenates consecutive lines
        when the preceding line lacks terminal punctuation and the following line
        starts with a lowercase character.

        Args:
            lines (list[str]): Filtered sequence of body text lines.

        Returns:
            str: Reconstructed text with coherent paragraphs separated by newlines.
        """
        if not lines:
            return ""

        paragraphs: list[str] = [lines[0]]
        for current in lines[1:]:
            previous = paragraphs[-1]
            
            if previous.endswith("-"):
                paragraphs[-1] = f"{previous[:-1]}{current}"
            elif previous[-1] not in {".", ":", "?", "!"} and current[0].islower():
                paragraphs[-1] = f"{previous} {current}"
            else:
                paragraphs.append(current)

        return "\n".join(paragraphs)


    def _estimate_body_font_size(self, doc: pymupdf.Document) -> float:
        """Estimates the dominant body font size weighted by character length.

        Args:
            doc (pymupdf.Document): Open PyMuPDF document instance.

        Returns:
            float: Most frequent rounded font size across all non-empty text
                spans, or `DEFAULT_FONT_SIZE` if no text spans are present.
        """
        size_counts: Counter[float] = Counter()

        for span in self._iter_text_spans(doc):
            cleaned_text = span.get("text", "").strip()
            if cleaned_text:
                font_size = round(span.get("size", DEFAULT_FONT_SIZE), 1)
                size_counts[font_size] += len(cleaned_text)

        return size_counts.most_common(1)[0][0] if size_counts else DEFAULT_FONT_SIZE


    def _iter_page_lines(self, page: pymupdf.Page) -> Iterator[_ExtractedLine]:
        """
        Yields cleaned page lines sorted top-to-bottom by vertical coordinate.

        Args:
            page (pymupdf.Page): PyMuPDF page object to inspect.

        Yields:
            _ExtractedLine: Parsed line records ordered ascending by `y_top`.
        """
        raw_lines: list[_ExtractedLine] = []

        for block in self._iter_text_blocks(page):
            for raw_line in block.get("lines", []):
                parsed = self._parse_line_dict(raw_line)
                
                if parsed is not None:
                    raw_lines.append(parsed)

        raw_lines.sort(key=lambda item: item.y_top)
        
        yield from raw_lines


    def _parse_line_dict(self, raw_line: _LineDict) -> _ExtractedLine | None:
        """
        Converts a raw PyMuPDF line dictionary into a structured `_ExtractedLine`.

        Concatenates span strings, normalizes whitespace, filters standalone page
        numbers, and records maximum span font size and vertical position.

        Args:
            raw_line (_LineDict): Dictionary representing a line within a PyMuPDF
                text block.

        Returns:
            _ExtractedLine | None: Structured line metadata, or None if the line
                contains no valid text after cleaning.
        """
        spans = raw_line.get("spans", [])
        if not spans:
            return None

        raw_text = "".join(span.get("text", "") for span in spans)
        clean_text = self._clean_line(raw_text)
        
        if not clean_text:
            return None

        max_size = max((span.get("size", 0.0) for span in spans), default=0.0)
        bbox = raw_line.get("bbox", (0.0, 0.0, 0.0, 0.0))
        return _ExtractedLine(
            text=clean_text,
            max_font_size=max_size,
            y_top=bbox[1],
        )


    def _iter_text_spans(self, doc: pymupdf.Document) -> Iterator[_SpanDict]:
        """
        Iterates lazily over all text span dictionaries across every page in a PDF.

        Args:
            doc (pymupdf.Document): Open PyMuPDF document instance.

        Yields:
            _SpanDict: Raw PyMuPDF span dictionary containing text and font metadata.
        """
        for page in doc:
            for block in self._iter_text_blocks(page):
                for line in block.get("lines", []):
                    yield from line.get("spans", [])


    @staticmethod
    def _iter_text_blocks(page: pymupdf.Page) -> Iterator[_BlockDict]:
        """
        Yields text-type layout blocks from a PyMuPDF page dictionary.

        Args:
            page (pymupdf.Page): PyMuPDF page object to extract blocks from.

        Yields:
            _BlockDict: Block dictionaries matching `TEXT_BLOCK_TYPE`.
        """
        page_dict: dict[str, list[_BlockDict]] = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == TEXT_BLOCK_TYPE:
                yield block


    @staticmethod
    def _clean_line(text: str) -> str:
        """
        Normalizes internal whitespace and discards standalone page number strings.

        Args:
            text (str): Raw concatenated line string.

        Returns:
            str: Whitespace-normalized string, or an empty string if blank or
                matched as a page number.
        """
        cleaned = WHITESPACE_PATTERN.sub(" ", text).strip()
        
        if not cleaned or PAGE_NUMBER_PATTERN.fullmatch(cleaned):
            return ""
        
        return cleaned