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


def extract_pages(path: Path) -> list[dict]:
    """Return nonempty pages with one-based page numbers and a local source name."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Input must be an existing PDF file.")
    try:
        with pymupdf.open(path) as document:
            pages = [
                {"source": path.name, "page": number, "text": cleaned}
                for number, page in enumerate(document, start=1)
                if (cleaned := clean_text(page.get_text()))
            ]
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError) as exc:
        raise ValueError("Unable to extract text from this PDF.") from exc
    if not pages:
        raise ValueError("PDF contains no extractable text; scanned images require OCR.")
    return pages
