"""Validate local Ollama embedding responses and normalize for cosine search."""

import numpy as np
from ollama import Client, ResponseError

EMBEDDING_MODEL = "embeddinggemma"
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"


def local_client() -> Client:
    """Avoid ambient host, proxy, and API-key settings for document requests."""
    return Client(host=LOCAL_OLLAMA_HOST, headers={"Authorization": ""}, trust_env=False)


def chunk_embedding_text(chunk: dict) -> str:
    """Add section context to retrieval input without changing quoted evidence."""
    text = chunk["text"]
    section = chunk.get("section", "")
    if not isinstance(text, str) or not isinstance(section, str):
        raise ValueError("Chunk embedding input is invalid.")
    return f"{section}\n{text}" if section else text


def embed_texts(
    texts: list[str], model: str = EMBEDDING_MODEL, client=None, batch_size: int = 32
) -> np.ndarray:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input must contain nonempty strings.")
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError("Embedding batch size must be between 1 and 1024.")
    client = client or local_client()
    batches = []
    dimension = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = client.embed(model=model, input=batch)
        except (ConnectionError, ResponseError) as exc:
            raise RuntimeError("Local Ollama unavailable or embedding model missing.") from exc
        try:
            embeddings = (
                response.embeddings if hasattr(response, "embeddings") else response["embeddings"]
            )
            vectors = np.asarray(embeddings, dtype=np.float32)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Embedding response contains invalid numeric values.") from exc
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(batch)
            or vectors.shape[1] < 1
            or not np.isfinite(vectors).all()
            or (dimension is not None and vectors.shape[1] != dimension)
        ):
            raise ValueError("Embedding response shape or values are invalid.")
        dimension = vectors.shape[1]
        batches.append(vectors)
    vectors = np.concatenate(batches)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embedding response contains a zero or invalid vector.")
    return np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
