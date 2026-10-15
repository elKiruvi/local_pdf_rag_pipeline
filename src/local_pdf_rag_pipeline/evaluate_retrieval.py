"""Conventional retrieval metrics over independently labeled chunk identifiers."""


def precision_at_k(found: list[str], relevant: set[str], k: int) -> float:
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive.")
    return len(set(found[:k]) & relevant) / k


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive.")
    return len(set(found[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(found: list[str], relevant: set[str], k: int) -> float:
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive.")
    return next((1 / rank for rank, item in enumerate(found[:k], 1) if item in relevant), 0.0)


def evaluate(cases: list[dict], search, k: int = 3, valid_ids: set[str] | None = None) -> dict:
    if not isinstance(cases, list) or not cases or type(k) is not int or k < 1:
        raise ValueError("At least one case and positive k are required.")
    totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("question"), str)
            or not case["question"].strip()
            or not isinstance(case.get("relevant_ids"), list)
            or any(not isinstance(id_, str) or not id_ for id_ in case["relevant_ids"])
            or len(set(case["relevant_ids"])) != len(case["relevant_ids"])
        ):
            raise ValueError("Evaluation case or labels are invalid.")
        relevant = set(case["relevant_ids"])
        if valid_ids is not None and not relevant <= valid_ids:
            raise ValueError("Evaluation labels do not match the indexed corpus.")
        found = [item["id"] for item in search(case["question"], k)]
        if len(found) != len(set(found)) or (valid_ids is not None and not set(found) <= valid_ids):
            raise ValueError("Search results contain duplicate or unknown chunk IDs.")
        totals["precision_at_k"] += precision_at_k(found, relevant, k)
        totals["recall_at_k"] += recall_at_k(found, relevant, k)
        totals["mrr_at_k"] += reciprocal_rank(found, relevant, k)
    return {key: value / len(cases) for key, value in totals.items()} | {
        "cases": len(cases),
        "k": k,
    }
