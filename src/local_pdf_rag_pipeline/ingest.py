"""Extract page text without logging or publishing document content."""

import re
from pathlib import Path

import pymupdf


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph and row boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[\t \u00a0]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_pages(
    path: Path,
    max_pdf_bytes: int = 100_000_000,
    max_pages: int = 1_000,
    max_text_chars: int = 10_000_000,
) -> list[dict]:
    """Return nonempty pages with one-based page numbers and a local source name."""
    path = Path(path)
    if min(max_pdf_bytes, max_pages, max_text_chars) < 1:
        raise ValueError("Resource limits must be positive.")
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Input must be an existing PDF file.")
    if path.is_symlink() or path.stat().st_size > max_pdf_bytes:
        raise ValueError("PDF input is a symlink or exceeds the configured size limit.")
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError) as exc:
        raise ValueError("Unable to extract text from this PDF.") from exc
    with document:
        if document.page_count > max_pages:
            raise ValueError("PDF exceeds the configured page limit.")
        pages = []
        total_chars = 0
        for number, page in enumerate(document, start=1):
            cleaned = clean_text(page.get_text())
            total_chars += len(cleaned)
            if total_chars > max_text_chars:
                raise ValueError("PDF exceeds the configured decoded-text limit.")
            if cleaned:
                pages.append({"source": path.name, "page": number, "text": cleaned})
    if not pages:
        raise ValueError("PDF contains no extractable text; scanned images require OCR.")
    return pages
