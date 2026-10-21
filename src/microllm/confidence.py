"""Official TypeSafe System-One confidence formulas, expectation math, and StartLux tournament routing.

Implements:
- jevcompat SPEC.md §4.3, §4.4, §4.5:
  - choice_confidence: (n * m - 1) / (n - 1) where m = max_i p_i
  - score_confidence: max(0, 1 - sum_i p_i |i - k| / D) where k = argmax_i p_i and D = (1/n) sum_i |i - (n-1)/2|
  - score_expectation: sum_{i=0}^{n-1} i * p_i
- StartLux-Decision (arXiv:2610.17309) hierarchical tournament for >25 options (up to 255 options).
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence


def sanitize_and_normalize_probabilities(
    raw_scores: Mapping[str, float],
) -> dict[str, float]:
    """Ensure finite non-negative probabilities summing strictly to 1.0 over identical keys."""
    keys = list(raw_scores.keys())
    if not keys:
        raise ValueError("Cannot normalize an empty option mapping.")

    cleaned: dict[str, float] = {}
    for k in keys:
        val = float(raw_scores[k])
        if not math.isfinite(val) or val < 0.0:
            val = 0.0
        cleaned[k] = val

    total = sum(cleaned.values())
    if total <= 1e-12 or not math.isfinite(total):
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}

    normalized = {k: cleaned[k] / total for k in keys}
    # Correct floating-point drift on the argmax key so sum is 1.0 within machine precision
    best_key = max(normalized, key=lambda k: normalized[k])
    other_sum = sum(v for k, v in normalized.items() if k != best_key)
    normalized[best_key] = max(0.0, min(1.0, 1.0 - other_sum))
    return normalized


def choice_confidence(probabilities: Mapping[str, float]) -> float:
    """Compute official TypeSafe choice confidence: (n * max_p - 1) / (n - 1).

    Matches `system-one-adapter-python` (`confidence_metrics.py`) and `jevcompat` §4.5.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    max_p = max(float(v) for v in probabilities.values())
    raw = (n * max_p - 1.0) / (n - 1.0)
    return float(max(0.0, min(1.0, raw)))


def score_expectation(probabilities: Mapping[str, float]) -> float:
    """Compute probability-weighted expected level: sum_{i=0}^{n-1} i * p_i."""
    total = 0.0
    for idx_str, prob in probabilities.items():
        idx = int(idx_str)
        total += idx * float(prob)
    return float(total)


def score_confidence(probabilities: Mapping[str, float]) -> float:
    """Compute official TypeSafe score confidence (`jevcompat` §4.5).

    confidence = max(0, 1 - sum_i p_i * |i - k| / D)
    where k is the mode (most likely level) and D = (1/n) * sum_i |i - (n - 1)/2|
    is the spread of a uniform distribution around the midpoint.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0

    ordered_probs = [float(probabilities[str(i)]) for i in range(n)]
    mode_k = max(range(n), key=lambda i: ordered_probs[i])
    midpoint = (n - 1.0) / 2.0
    denom_d = sum(abs(i - midpoint) for i in range(n)) / float(n)
    if denom_d <= 1e-12:
        return 1.0

    expected_dist = sum(ordered_probs[i] * abs(i - mode_k) for i in range(n))
    conf = max(0.0, 1.0 - (expected_dist / denom_d))
    return float(min(1.0, conf))


def multi_confidence(probabilities: Mapping[str, float]) -> float:
    """Compute average binary certainty across independent multi-label probabilities: (1/n) sum_k |2*p_k - 1|."""
    if not probabilities:
        return 1.0
    vals = [abs(2.0 * float(p) - 1.0) for p in probabilities.values()]
    return float(max(0.0, min(1.0, sum(vals) / len(vals))))


def earth_movers_distance_1d(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute normalized 1D Earth Mover's (Wasserstein-1) distance between ordinal distributions (IDEA-008)."""
    n = len(p)
    if n <= 1 or len(q) != n:
        return 0.0
    cdf_diff_sum = 0.0
    cum_p = 0.0
    cum_q = 0.0
    for i in range(n - 1):
        cum_p += float(p[i])
        cum_q += float(q[i])
        cdf_diff_sum += abs(cum_p - cum_q)
    return round(float(cdf_diff_sum / float(n - 1)), 6)


def unimodal_ordinal_smooth(
    probabilities: Mapping[str, float], strength: float = 0.15
) -> dict[str, float]:
    """Project an ordinal `score` distribution toward unimodal neighbor consistency (IDEA-008)."""
    n = len(probabilities)
    if n <= 2 or strength <= 0.0:
        return sanitize_and_normalize_probabilities(probabilities)
    lam = float(max(0.0, min(0.5, strength)))
    ordered = [float(probabilities[str(i)]) for i in range(n)]
    smoothed: dict[str, float] = {}
    for i in range(n):
        left = ordered[i - 1] if i > 0 else ordered[i]
        right = ordered[i + 1] if i < n - 1 else ordered[i]
        smoothed[str(i)] = (1.0 - lam) * ordered[i] + 0.5 * lam * (left + right)
    return sanitize_and_normalize_probabilities(smoothed)


def run_hierarchical_tournament(
    options: Mapping[str, str | None],
    chunk_scorer: Callable[[dict[str, str | None]], dict[str, float]],
    chunk_size: int = 25,
    top_m: int = 3,
    seed_diversity: bool = False,
) -> dict[str, float]:
    """StartLux-Decision (§3.2) hierarchical tournament for high-cardinality choices (up to 255 options).

    Splits >25 options into chunks of <=25, advances the top-M candidates from each chunk to a final
    playoff round, and rescales chunk probabilities consistently across all original options so all 255
    options receive valid, positive, normalized probabilities summing to 1.0.
    """
    orig_items = list(options.items())
    n = len(orig_items)
    if n <= chunk_size:
        return sanitize_and_normalize_probabilities(chunk_scorer(dict(orig_items)))

    if seed_diversity:
        # Snake / stride interleaving across chunks to prevent semantic group-of-death (IDEA-007)
        sorted_items = sorted(orig_items, key=lambda kv: (kv[0], str(kv[1] or "")))
        n_chunks = math.ceil(n / float(chunk_size))
        items: list[tuple[str, str | None]] = []
        for c_idx in range(n_chunks):
            items.extend(sorted_items[c_idx::n_chunks])
    else:
        items = orig_items

    chunks: list[dict[str, str | None]] = [
        dict(items[i : i + chunk_size]) for i in range(0, n, chunk_size)
    ]
    chunk_distributions: list[dict[str, float]] = []
    finalists: dict[str, str | None] = {}

    for chunk in chunks:
        c_probs = sanitize_and_normalize_probabilities(chunk_scorer(chunk))
        chunk_distributions.append(c_probs)
        ranked = sorted(c_probs.keys(), key=lambda k: c_probs[k], reverse=True)
        for k in ranked[:top_m]:
            finalists[k] = options[k]

    # If finalists still exceed chunk_size, recurse once
    if len(finalists) > chunk_size:
        final_probs = run_hierarchical_tournament(finalists, chunk_scorer, chunk_size, top_m)
    else:
        final_probs = sanitize_and_normalize_probabilities(chunk_scorer(finalists))

    # Reconstruct full distribution over all N options preserving finalist relative weights
    # and guaranteeing no eliminated non-finalist can exceed any finalist from its chunk.
    combined: dict[str, float] = {}
    for chunk, c_probs in zip(chunks, chunk_distributions):
        chunk_finalists = [k for k in chunk if k in final_probs]
        chunk_non_finalists = [k for k in chunk if k not in final_probs]
        min_finalist = min((final_probs[k] for k in chunk_finalists), default=0.0)
        max_non_finalist_local = max((c_probs[k] for k in chunk_non_finalists), default=0.0)
        denom = float(max(1, len(chunk_non_finalists)))

        for k in chunk_finalists:
            combined[k] = final_probs[k]
        for k in chunk_non_finalists:
            rel_local = (
                (c_probs[k] / max_non_finalist_local) if max_non_finalist_local > 1e-12 else 1.0
            )
            combined[k] = min_finalist * 0.25 * rel_local / denom

    return sanitize_and_normalize_probabilities(combined)

