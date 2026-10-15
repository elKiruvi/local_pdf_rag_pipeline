"""Conventional retrieval metrics over independently labeled chunk identifiers."""


def precision_at_k(found: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    return len(set(found[:k]) & relevant) / k


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    return len(set(found[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(found: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    return next((1 / rank for rank, item in enumerate(found[:k], 1) if item in relevant), 0.0)


def evaluate(cases: list[dict], search, k: int = 3) -> dict:
    if not cases or k < 1:
        raise ValueError("At least one case and positive k are required.")
    totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}
    for case in cases:
        relevant = set(case["relevant_ids"])
        found = [item["id"] for item in search(case["question"], k)]
        totals["precision_at_k"] += precision_at_k(found, relevant, k)
        totals["recall_at_k"] += recall_at_k(found, relevant, k)
        totals["mrr_at_k"] += reciprocal_rank(found, relevant, k)
    return {key: value / len(cases) for key, value in totals.items()} | {
        "cases": len(cases),
        "k": k,
    }
