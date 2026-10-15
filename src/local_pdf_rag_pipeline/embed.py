"""Validate local Ollama embedding responses and normalize for cosine search."""

import numpy as np
from ollama import Client, ResponseError

EMBEDDING_MODEL = "embeddinggemma"
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"


def local_client() -> Client:
    """Avoid ambient host, proxy, and API-key settings for document requests."""
    return Client(host=LOCAL_OLLAMA_HOST, headers={"Authorization": ""}, trust_env=False)


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL, client=None) -> np.ndarray:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input must contain nonempty strings.")
    try:
        response = (client or local_client()).embed(model=model, input=texts)
    except (ConnectionError, ResponseError) as exc:
        raise RuntimeError(
            f"Local Ollama unavailable or embedding model '{model}' missing."
        ) from exc
    embeddings = response.embeddings if hasattr(response, "embeddings") else response["embeddings"]
    try:
        vectors = np.asarray(embeddings, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("Embedding response contains invalid numeric values.") from exc
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(texts)
        or vectors.shape[1] < 1
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Embedding response shape or values are invalid.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embedding response contains a zero or invalid vector.")
    return np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
