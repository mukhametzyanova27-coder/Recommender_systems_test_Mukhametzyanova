"""MAP@K as in Kaggle / ml_metrics: AP@K = sum_k P@k * rel_k / min(|truth|, K)."""
import numpy as np


def apk(actual: set, predicted, k: int = 5) -> float:
    if not actual:
        return 0.0
    hits, score = 0, 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


def mapk(truth: dict, recs: dict, users, k: int = 5) -> float:
    """Mean AP@K over `users` (users must have non-empty truth)."""
    return float(np.mean([apk(truth[u], recs.get(u, []), k) for u in users]))
