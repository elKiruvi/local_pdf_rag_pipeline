"""Persist only trusted, locally generated FAISS indexes and private metadata."""

import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import faiss
import numpy as np

from .embed import EMBEDDING_MODEL

FILES = {"vectors.index", "metadata.json", "manifest.json"}
SCHEMA_VERSION = 1
MAX_VECTORS = 100_000
MAX_DIMENSION = 16_384
MAX_FILE_BYTES = {"vectors.index": 512_000_000, "metadata.json": 50_000_000, "manifest.json": 4_096}


def _safe_directory(directory: Path) -> None:
    if ".." in directory.parts or any(
        path.is_symlink() for path in (directory.absolute(), *directory.absolute().parents)
    ):
        raise ValueError("Index directory must not traverse symlinks.")


def _metadata(chunks: object) -> bool:
    return (
        isinstance(chunks, list)
        and 0 < len(chunks) <= MAX_VECTORS
        and all(
            isinstance(chunk, dict)
            and isinstance(chunk.get("id"), str)
            and bool(chunk["id"])
            and isinstance(chunk.get("text"), str)
            and bool(chunk["text"].strip())
            and isinstance(chunk.get("source"), str)
            and bool(chunk["source"])
            and Path(chunk["source"]).name == chunk["source"]
            and "\\" not in chunk["source"]
            and type(chunk.get("page")) is int
            and chunk["page"] > 0
            and isinstance(chunk.get("section", ""), str)
            and isinstance(chunk.get("content_type", "text"), str)
            for chunk in chunks
        )
        and len({chunk["id"] for chunk in chunks}) == len(chunks)
    )


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_index(directory: Path, expected_model: str | None = None):
    """Check format and corruption before FAISS parsing; never authenticate hostile files."""
    directory = Path(directory)
    _safe_directory(directory)
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != FILES:
        raise ValueError("No complete index found. Rebuild the local index.")
    for name in FILES:
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= MAX_FILE_BYTES[name]
        ):
            raise ValueError("Index file type or size is invalid; rebuild the local index.")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION
            or not isinstance(manifest.get("embedding_model"), str)
            or not manifest["embedding_model"].strip()
            or type(manifest.get("dimension")) is not int
            or not 0 < manifest["dimension"] <= MAX_DIMENSION
            or type(manifest.get("vector_count")) is not int
            or not 0 < manifest["vector_count"] <= MAX_VECTORS
            or not isinstance(manifest.get("sha256"), dict)
            or set(manifest["sha256"]) != {"vectors.index", "metadata.json"}
        ):
            raise ValueError("Index schema is unsupported; rebuild the local index.")
        if expected_model is not None and expected_model != manifest["embedding_model"]:
            raise ValueError("Query embedding model differs from indexed model.")
        for name, digest in manifest["sha256"].items():
            if not isinstance(digest, str) or _digest(directory / name) != digest:
                raise ValueError("Index checksum mismatch; rebuild the local index.")
        chunks = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not _metadata(chunks) or len(chunks) != manifest["vector_count"]:
            raise ValueError("Index metadata is invalid; rebuild the local index.")
        # Checksums detect accidents only. Native parsing still requires a trusted index.
        index = faiss.read_index(str(directory / "vectors.index"))
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise ValueError("Index files are damaged or unreadable; rebuild the local index.") from exc
    if (
        not isinstance(faiss.downcast_index(index), faiss.IndexFlatIP)
        or index.ntotal != len(chunks)
        or index.d != manifest["dimension"]
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Index and manifest disagree; rebuild the local index.")
    return index, chunks


def build_index(
    vectors: np.ndarray, chunks: list[dict], directory: Path, model: str = EMBEDDING_MODEL
) -> None:
    if (
        vectors.ndim != 2
        or not 0 < vectors.shape[0] <= MAX_VECTORS
        or not 0 < vectors.shape[1] <= MAX_DIMENSION
        or vectors.shape[0] != len(chunks)
        or vectors.size * 4 > 256_000_000
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Vector dimensions/count must match nonempty chunk metadata.")
    if not _metadata(chunks):
        raise ValueError("Chunk metadata is incomplete or invalid.")
    if not model or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError("Vectors must be L2-normalized and have an embedding model.")
    directory = Path(directory)
    _safe_directory(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        load_index(directory)  # Do not replace unknown files or an unsupported old format.
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent))
    backup = directory.parent / f".{directory.name}.backup-{uuid.uuid4().hex}"
    try:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(staging / "vectors.index"))
        (staging / "metadata.json").write_text(
            json.dumps(chunks, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": model,
            "dimension": vectors.shape[1],
            "vector_count": len(chunks),
            "sha256": {
                name: _digest(staging / name) for name in ("vectors.index", "metadata.json")
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        load_index(staging, model)
        had_previous = directory.exists()
        if had_previous:
            directory.rename(backup)
        try:
            staging.rename(directory)
        except OSError:
            if had_previous:
                backup.rename(directory)  # If rollback fails, leave backup for manual recovery.
            raise
        if had_previous:
            shutil.rmtree(backup)  # The former index was validated before replacement.
    finally:
        if staging.exists():
            shutil.rmtree(staging)
