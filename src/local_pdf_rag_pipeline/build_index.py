"""Persist a local FAISS inner-product index and aligned, private metadata."""

import json
from pathlib import Path

import faiss
import numpy as np


def build_index(vectors: np.ndarray, chunks: list[dict], directory: Path) -> None:
    if (
        vectors.ndim != 2
        or vectors.shape[0] == 0
        or vectors.shape[1] == 0
        or vectors.shape[0] != len(chunks)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Vector dimensions/count must match nonempty chunk metadata.")
    if any(not all(k in chunk for k in ("id", "text", "source", "page")) for chunk in chunks):
        raise ValueError("Chunk metadata is incomplete.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(directory / "vectors.index"))
    (directory / "metadata.json").write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")


def load_index(directory: Path):
    directory = Path(directory)
    if not (directory / "vectors.index").is_file() or not (directory / "metadata.json").is_file():
        raise ValueError("No index found. Run the index command first.")
    try:
        chunks = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / "vectors.index"))
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise ValueError("Index files are damaged or unreadable.") from exc
    if (
        not isinstance(chunks, list)
        or not chunks
        or index.ntotal != len(chunks)
        or index.d < 1
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Index and metadata disagree or have invalid dimensions.")
    return index, chunks
