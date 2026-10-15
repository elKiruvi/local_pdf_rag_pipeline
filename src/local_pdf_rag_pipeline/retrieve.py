"""Rank normalized query vectors against normalized document vectors."""

import numpy as np

from .embed import EMBEDDING_MODEL, embed_texts


def retrieve(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 3,
    model: str = EMBEDDING_MODEL,
    client=None,
) -> list[dict]:
    if not question.strip() or top_k < 1:
        raise ValueError("Question must be nonempty and top_k must be positive.")
    if index.ntotal != len(chunks) or not chunks:
        raise ValueError("Index and metadata counts must agree.")
    vector = embed_texts([question], model=model, client=client)
    if vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension differs from index.")
    scores, positions = index.search(np.asarray(vector, dtype=np.float32), min(top_k, len(chunks)))
    return [
        dict(chunks[int(pos)], score=float(score))
        for score, pos in zip(scores[0], positions[0], strict=True)
        if pos >= 0
    ]
