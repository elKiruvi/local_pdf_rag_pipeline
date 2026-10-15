"""Conservative section- and table-aware character chunking."""

import re


def line_kind(line: str) -> str:
    """Classify obvious headings and delimited rows; ambiguous text stays prose."""
    line = line.strip()
    if len(line) <= 80 and re.match(r"^#{1,6} |^\d+(?:\.\d+)*[.)]?\s+[A-Z]", line):
        return "heading"
    if len(line) <= 72 and line.isupper() and re.search(r"[A-Z]", line):
        return "heading"
    if line.count("|") >= 2 or line.count("\t") >= 2:
        return "table"
    return "text"


def chunk_pages(pages: list[dict], size: int = 800, overlap: int = 100) -> list[dict]:
    """Split each page independently, carrying headings into following chunks."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    chunks = []
    for page in pages:
        if not isinstance(page.get("page"), int) or page["page"] < 1 or not page.get("source"):
            raise ValueError("Pages require a source and a positive page number.")
        heading = ""
        units = []
        for line in page.get("text", "").splitlines():
            if not line.strip():
                continue
            kind = line_kind(line)
            if kind == "heading":
                heading = line.strip()
            else:
                units.append((line.strip(), kind, heading))
        # Keep content type and heading boundaries distinct. Long units use overlap.
        groups = []
        for text, kind, section in units:
            if groups and groups[-1][1:] == (kind, section):
                groups[-1] = (groups[-1][0] + "\n" + text, kind, section)
            else:
                groups.append((text, kind, section))
        for text, kind, section in groups:
            step = size - overlap
            for start in range(0, len(text), step):
                piece = text[start : start + size]
                if not piece:
                    break
                chunks.append(
                    {
                        "id": f"chunk-{len(chunks):04d}",
                        "text": piece,
                        "source": page["source"],
                        "page": page["page"],
                        "section": section,
                        "content_type": kind,
                    }
                )
                if start + size >= len(text):
                    break
    if not chunks:
        raise ValueError("No text remains after chunking.")
    return chunks
