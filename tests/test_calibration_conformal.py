"""Tests for Per-K Temperature Scaling, Brier Score, ECE, Split Conformal Prediction, and Clopper-Pearson certification."""

from __future__ import annotations

import numpy as np
import pytest

from microllm.calibration import (
    PerKTemperatureScaler,
     SplitConformalCalibrator,
    brier_score,
    clopper_pearson_threshold,
    expected_calibration_error,
    reliability_diagram_bins,
)


def test_brier_and_ece_metrics_reduce_after_temperature_scaling() -> None:
    """Overconfident logits have high ECE and Brier; fitting PerKTemperatureScaler strictly reduces both."""
    rng = np.random.default_rng(42)
    n_samples, k = 300, 4
    # Generate ground-truth underlying logits with true temperature 1.0, then multiply by 4.5 (severe overconfidence)
    base_logits = rng.normal(0.0, 1.0, size=(n_samples, k))
    exp_base = np.exp(base_logits - np.max(base_logits, axis=1, keepdims=True))
    true_probs = exp_base / np.sum(exp_base, axis=1, keepdims=True)
    labels = np.array([rng.choice(k, p=true_probs[i]) for i in range(n_samples)], dtype=np.int64)

    overconfident_logits = base_logits * 4.5
    exp_over = np.exp(overconfident_logits - np.max(overconfident_logits, axis=1, keepdims=True))
    raw_probs = exp_over / np.sum(exp_over, axis=1, keepdims=True)

    raw_ece = expected_calibration_error(raw_probs, labels, n_bins=10)
    raw_brier = brier_score(raw_probs, labels)

    scaler = PerKTemperatureScaler()
    fitted_t = scaler.fit_k(overconfident_logits, labels)
    assert fitted_t > 2.0  # Must discover that logits were inflated

    cal_probs = scaler.transform(overconfident_logits)
    cal_ece = expected_calibration_error(cal_probs, labels, n_bins=10)
    cal_brier = brier_score(cal_probs, labels)

    assert cal_ece < raw_ece * 0.5
    assert cal_brier < raw_brier

    # Reliability diagram bins check
    bins = reliability_diagram_bins(cal_probs, labels, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == n_samples


def test_split_conformal_prediction_coverage_guarantee() -> None:
    """SplitConformalCalibrator guarantees empirical coverage >= 1 - alpha on exchangeable test data."""
    rng = np.random.default_rng(123)
    n_cal, n_test, k = 400, 400, 5
    logits = rng.normal(0.0, 1.4, size=(n_cal + n_test, k))
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    labels = np.array([rng.choice(k, p=probs[i]) for i in range(n_cal + n_test)], dtype=np.int64)

    cal_probs, test_probs = probs[:n_cal], probs[n_cal:]
    cal_labels, test_labels = labels[:n_cal], labels[n_cal:]

    option_names = [f"class_{i}" for i in range(k)]
    conformal = SplitConformalCalibrator(alpha=0.10)  # Target 90% coverage
    q_hat = conformal.fit(cal_probs, cal_labels)
    assert 0.0 <= q_hat <= 1.0

    covered = 0
    for i in range(n_test):
        p_dict = {option_names[j]: float(test_probs[i, j]) for j in range(k)}
        res = conformal.predict_set(p_dict)
        assert len(res["conformal_set"]) >= 1
        assert isinstance(res["abstain"], bool)
        # Abstain is True iff conformal_set does not uniquely identify 1 option
        assert res["abstain"] == (len(res["conformal_set"]) != 1)
        if option_names[int(test_labels[i])] in res["conformal_set"]:
            covered += 1

    empirical_coverage = covered / n_test
    assert empirical_coverage >= 0.86  # Finite-sample concentration around 0.90


def test_clopper_pearson_exact_risk_threshold() -> None:
    """Clopper-Pearson upper confidence bound finds a threshold automating high-confidence items at risk <= target."""
    rng = np.random.default_rng(7)
    n = 500
    confidences = rng.beta(5.0, 1.5, size=n)
    # Correct with probability equal to confidence (well-calibrated)
    correct = (rng.uniform(0.0, 1.0, size=n) <= confidences).astype(np.int64)

    cert = clopper_pearson_threshold(confidences, correct, target_risk=0.15, delta=0.05)
    assert 0.0 <= cert["threshold"] <= 1.0
    assert cert["coverage"] > 0.20
    assert cert["empirical_risk"] <= 0.15
    assert cert["upper_risk_bound"] <= 0.15


def test_edge_cases_and_boundary_values() -> None:
    """Explicitly test empty/degenerate inputs, NaN/Inf sanitization, and boundary cardinalities."""
    from microllm.backends.prefill_llm import van_der_corput_order
    from microllm.confidence import (
        choice_confidence,
        sanitize_and_normalize_probabilities,
        score_confidence,
    )

    assert van_der_corput_order(0) == []
    assert van_der_corput_order(1) == [0]
    assert choice_confidence({"only_one": 1.0}) == 1.0
    assert score_confidence({"0": 1.0}) == 1.0

    # Sanitize NaN / Inf / negative probabilities -> falls back cleanly to finite normalized distribution
    cleaned = sanitize_and_normalize_probabilities({"a": float("nan"), "b": -2.0, "c": float("inf")})
    assert abs(sum(cleaned.values()) - 1.0) < 1e-9
    assert all(np.isfinite(v) and v >= 0.0 for v in cleaned.values())

    with pytest.raises(ValueError, match="empty"):
        sanitize_and_normalize_probabilities({})

    with pytest.raises(ValueError, match="alpha"):
        SplitConformalCalibrator(alpha=1.5)

    with pytest.raises(ValueError, match="empty"):
        SplitConformalCalibrator(alpha=0.1).fit(np.zeros((0, 2)), np.zeros(0, dtype=int))

    assert expected_calibration_error(np.zeros((0, 2)), np.zeros(0, dtype=int)) == 0.0
    empty_cp = clopper_pearson_threshold(np.array([]), np.array([]))
    assert empty_cp["threshold"] == 1.0
    assert empty_cp["coverage"] == 0.0

    # PerKTemperatureScaler log-interpolation for unseen K
    scaler = PerKTemperatureScaler(default_temperature=0.85, temperatures={2: 0.70, 10: 1.00})
    t_interp = scaler.get_temperature(5)
    assert 0.70 < t_interp < 1.00
    assert scaler.get_temperature(1) == 0.70
    assert scaler.get_temperature(25) == 1.00

    # Hierarchical tournament monotonicity: finalists in a chunk never rank below eliminated non-finalists
    from microllm.confidence import run_hierarchical_tournament

    options = {f"opt_{i:03d}": f"Desc {i}" for i in range(30)}

    def mock_chunk_scorer(sub_opts: dict[str, str | None]) -> dict[str, float]:
        keys = list(sub_opts.keys())
        if len(keys) == 20:
            probs = {k: 0.46 / 16 for k in keys}
            probs["opt_000"] = 0.15
            probs["opt_001"] = 0.14
            probs["opt_002"] = 0.13
            probs["opt_003"] = 0.12
            return probs
        if len(keys) == 6:
            return {
                "opt_000": 0.55,
                "opt_001": 0.05,
                "opt_002": 0.01,
                keys[3]: 0.13,
                keys[4]: 0.13,
                keys[5]: 0.13,
            }
        return {k: 1.0 / len(keys) for k in keys}

    merged = run_hierarchical_tournament(options, mock_chunk_scorer, chunk_size=20, top_m=3)
    assert abs(sum(merged.values()) - 1.0) < 1e-6
    assert merged["opt_000"] > merged["opt_001"] > merged["opt_002"] > merged["opt_003"]


