"""Persist a local FAISS inner-product index and aligned, private metadata."""

import json
from pathlib import Path

import faiss
import numpy as np

from .embed import EMBEDDING_MODEL


def build_index(
    vectors: np.ndarray, chunks: list[dict], directory: Path, model: str = EMBEDDING_MODEL
) -> None:
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
    if not model or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("Vectors must be L2-normalized and have an embedding model.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(directory / "vectors.index"))
    (directory / "metadata.json").write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"embedding_model": model, "dimension": vectors.shape[1]}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_index(directory: Path, expected_model: str | None = None):
    directory = Path(directory)
    if not all(
        (directory / name).is_file() for name in ("vectors.index", "metadata.json", "manifest.json")
    ):
        raise ValueError("No index found. Run the index command first.")
    try:
        chunks = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
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
    if (
        not isinstance(manifest, dict)
        or not manifest.get("embedding_model")
        or manifest.get("dimension") != index.d
    ):
        raise ValueError("Index manifest disagrees with stored vectors.")
    if expected_model is not None and expected_model != manifest["embedding_model"]:
        raise ValueError("Query embedding model differs from indexed model.")
    return index, chunks
