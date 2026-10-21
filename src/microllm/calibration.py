"""Calibration, Per-K Temperature Scaling, Split Conformal Prediction, and Clopper-Pearson bounds.

Synthesizes techniques from:
- openJev-verdict-2.0: Per-cardinality (per-K) temperature scaling
- VerdictML & poorjev: Split Conformal Prediction (`conformal_set` and `abstain` with 1 - alpha coverage)
- AnyJev & OpenJev-RLCD: Expected Calibration Error (ECE), Brier Score, and Clopper-Pearson exact risk certification.
"""

from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist


def _softmax_2d(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    scaled = np.asarray(logits, dtype=np.float64) / t
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass mean squared probability error (Brier score): (1/N) sum_i ||p_i - e_{y_i}||^2."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n, k = p.shape
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def reliability_diagram_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Compute reliability diagram bin statistics (confidence vs accuracy per bin)."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(np.float64)

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    for b_idx in range(n_bins):
        low, high = float(boundaries[b_idx]), float(boundaries[b_idx + 1])
        if b_idx == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            gap = abs(avg_acc - avg_conf)
        else:
            avg_conf = 0.5 * (low + high)
            avg_acc = 0.0
            gap = 0.0
        bins.append(
            {
                "bin_low": round(low, 4),
                "bin_high": round(high, 4),
                "count": count,
                "mean_confidence": round(avg_conf, 4),
                "empirical_accuracy": round(avg_acc, 4),
                "calibration_gap": round(gap, 4),
            }
        )
    return bins


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` equal-width confidence bins."""
    n = len(labels)
    if n == 0:
        return 0.0
    bins = reliability_diagram_bins(probs, labels, n_bins=n_bins)
    ece = sum((int(b["count"]) / float(n)) * float(b["calibration_gap"]) for b in bins)
    return float(ece)


class PerKTemperatureScaler:
    """Cardinality-aware temperature scaling (openJev-verdict-2.0 & AnyJev L1 calibration).

    Stores a learned temperature T_K for each option count K, falling back to log-interpolated
    or default temperature when an unseen cardinality is queried.
    """

    def __init__(
        self,
        default_temperature: float = 1.0,
        temperatures: dict[int, float] | None = None,
    ) -> None:
        self.default_temperature = float(default_temperature)
        self.temperatures_by_k: dict[int, float] = dict(temperatures) if temperatures else {}

    def fit_k(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit optimal temperature T_K minimizing negative log-likelihood / Brier on held-out set."""
        arr = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        _, k = arr.shape

        def objective(temp: float) -> float:
            probs = _softmax_2d(arr, temperature=temp)
            eps = 1e-12
            nll = -float(np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))
            return nll + 0.5 * brier_score(probs, y)

        res = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        best_t = float(res.x) if res.success else 1.0
        self.temperatures_by_k = {**self.temperatures_by_k, k: best_t}
        return best_t

    def get_temperature(self, k: int) -> float:
        """Return calibrated temperature for cardinality `k` with log-K interpolation fallback."""
        if k in self.temperatures_by_k:
            return float(self.temperatures_by_k[k])
        if not self.temperatures_by_k:
            return self.default_temperature

        sorted_ks = sorted(self.temperatures_by_k.keys())
        if k <= sorted_ks[0]:
            return float(self.temperatures_by_k[sorted_ks[0]])
        if k >= sorted_ks[-1]:
            return float(self.temperatures_by_k[sorted_ks[-1]])

        log_k = math.log(max(1, k))
        for idx in range(len(sorted_ks) - 1):
            k_low, k_high = sorted_ks[idx], sorted_ks[idx + 1]
            if k_low <= k <= k_high:
                log_low = math.log(max(1, k_low))
                log_high = math.log(max(1, k_high))
                if abs(log_high - log_low) <= 1e-12:
                    return float(self.temperatures_by_k[k_low])
                ratio = (log_k - log_low) / (log_high - log_low)
                t_low = float(self.temperatures_by_k[k_low])
                t_high = float(self.temperatures_by_k[k_high])
                return float((1.0 - ratio) * t_low + ratio * t_high)
        return self.default_temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling for the column cardinality K of `logits`."""
        arr = np.asarray(logits, dtype=np.float64)
        k = arr.shape[-1]
        temp = self.get_temperature(int(k))
        return _softmax_2d(arr, temperature=temp)


class SplitConformalCalibrator:
    """Split Conformal Prediction for finite-sample coverage guarantees (`VerdictML` / `poorjev`).

    Given miscoverage rate `alpha` (e.g. 0.10 for 90% coverage), computes the quantile `q_hat`
    of non-conformity scores `s_i = 1 - p(y_i | x_i)` on calibration data, and returns
    `conformal_set = {y : 1 - p(y|x) <= q_hat}` plus `abstain = (len(conformal_set) != 1)`.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)
        self.q_hat: float = 1.0 - alpha

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> float:
        """Compute finite-sample conformal quantile ceiling((n+1)*(1-alpha))/n."""
        p = np.asarray(cal_probs, dtype=np.float64)
        y = np.asarray(cal_labels, dtype=np.int64)
        n = len(y)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
        non_conformity = 1.0 - p[np.arange(n), y]
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / float(n))
        self.q_hat = float(np.quantile(non_conformity, level, method="higher"))
        return self.q_hat

    def predict_set(self, probabilities: Mapping[str, float]) -> dict[str, Any]:
        """Return conformal prediction set and abstention decision."""
        min_prob = 1.0 - self.q_hat
        included = [
            opt for opt, prob in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
            if float(prob) >= min_prob
        ]
        if not included:
            best_opt = max(probabilities, key=lambda k: probabilities[k])
            included = [best_opt]
        return {
            "conformal_set": included,
            "abstain": len(included) != 1,
            "coverage_target": round(1.0 - self.alpha, 4),
            "threshold_prob": round(max(0.0, min_prob), 4),
        }


def clopper_pearson_upper_bound(num_errors: int, num_accepted: int, delta: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound on binomial error rate."""
    if num_accepted <= 0:
        return 1.0
    if num_errors >= num_accepted:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, num_errors + 1, num_accepted - num_errors))


def clopper_pearson_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_risk: float = 0.05,
    delta: float = 0.05,
) -> dict[str, float]:
    """Select the lowest confidence threshold `tau` whose Clopper-Pearson upper risk bound <= `target_risk`."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.int64)
    n = len(conf)
    if n == 0:
        return {"threshold": 1.0, "coverage": 0.0, "empirical_risk": 0.0, "upper_risk_bound": 1.0}

    candidates = np.unique(np.quantile(conf, np.linspace(0.0, 0.98, 100)))
    best = {"threshold": 1.0, "coverage": 0.0, "empirical_risk": 0.0, "upper_risk_bound": 1.0}

    for tau in candidates:
        mask = conf >= tau
        n_acc = int(np.sum(mask))
        if n_acc < 5:
            continue
        n_err = int(np.sum(1 - corr[mask]))
        emp_risk = n_err / float(n_acc)
        ucb = clopper_pearson_upper_bound(n_err, n_acc, delta=delta)
        if ucb <= target_risk:
            return {
                "threshold": float(tau),
                "coverage": float(n_acc / float(n)),
                "empirical_risk": float(emp_risk),
                "upper_risk_bound": float(ucb),
            }
    return best

