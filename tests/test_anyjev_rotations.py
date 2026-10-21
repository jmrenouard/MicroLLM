"""Tests for AnyJev L0 Prefill Readout: log-space circular rotation, van der Corput ordering, early stopping, answer mass gate, and batch prior debiasing."""

from __future__ import annotations

import numpy as np
import pytest

from microllm.backends.prefill_llm import (
    AnyJevPrefillBackend,
    apply_prior_debiasing,
    compute_answer_mass,
    l0_circular_rotation_readout,
    van_der_corput_order,
)


def test_van_der_corput_order_properties() -> None:
    """van_der_corput_order produces a valid permutation of 0..K-1 starting with 0, K/2, K/4, 3K/4."""
    for k in (2, 3, 4, 5, 8, 10, 20):
        order = van_der_corput_order(k)
        assert len(order) == k
        assert sorted(order) == list(range(k))
        assert order[0] == 0
    # For K=4, binary-reflected order visits 0, 2, 1, 3
    assert van_der_corput_order(4) == [0, 2, 1, 3]
    # For K=20 (non-power of 2, AnyJev Section 5), first 4 shifts must bisect 0, K/2=10, K/4=5, 3K/4=15
    assert van_der_corput_order(20)[:4] == [0, 10, 5, 15]


def test_l0_log_space_mean_cancels_additive_position_bias_exactly() -> None:
    """Theorem 1 (AnyJev): If logit(option i at slot j) = true_score[i] + slot_bias[j],
    log-space mean across all K circular rotations recovers softmax(true_score) exactly,
    bringing order-flip rate to 0%.
    """
    true_scores = np.array([1.2, 1.45, 0.3, -0.5], dtype=np.float64)  # Option 1 is genuinely best
    # Severe position bias favoring slot 0 ("A") so strongly that raw single-pass flips to slot 0
    slot_bias = np.array([2.5, -0.4, -0.8, -1.3], dtype=np.float64)
    k = len(true_scores)

    def biased_slot_probs(perm: list[int]) -> tuple[np.ndarray, float]:
        # perm[j] is the original option index placed at slot j
        logits = np.array([true_scores[perm[j]] + slot_bias[j] for j in range(k)], dtype=np.float64)
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        return probs, 0.995

    # Single pass (no rotation): option 0 wins falsely because of slot_bias[0] = 2.5
    raw_probs, _ = biased_slot_probs([0, 1, 2, 3])
    assert int(np.argmax(raw_probs)) == 0  # Wrong due to position bias!

    # Full L0 rotation readout (tau=None -> all K passes)
    result = l0_circular_rotation_readout(k, biased_slot_probs, tau=None)
    expected_logits = true_scores - np.max(true_scores)
    expected_probs = np.exp(expected_logits) / np.sum(np.exp(expected_logits))

    assert result["passes_used"] == k
    assert int(np.argmax(result["probabilities"])) == 1
    np.testing.assert_allclose(result["probabilities"], expected_probs, atol=1e-7)

    # Now reverse the initial option order and verify Order-Flip Rate is 0
    rev_scores = true_scores[::-1]

    def rev_biased_slot_probs(perm: list[int]) -> tuple[np.ndarray, float]:
        logits = np.array([rev_scores[perm[j]] + slot_bias[j] for j in range(k)], dtype=np.float64)
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        return probs, 0.995

    rev_result = l0_circular_rotation_readout(k, rev_biased_slot_probs, tau=None)
    np.testing.assert_allclose(rev_result["probabilities"][::-1], result["probabilities"], atol=1e-7)


def test_l0_early_stopping_with_margin_tau() -> None:
    """When top-2 log-odds margin mu(z_bar) >= tau after >= 2 passes, readout stops early."""
    true_scores = np.array([4.0, 0.1, -1.0, -2.0, -3.0, -3.5], dtype=np.float64)
    k = len(true_scores)

    def clear_winner_probs(perm: list[int]) -> tuple[np.ndarray, float]:
        logits = np.array([true_scores[perm[j]] for j in range(k)], dtype=np.float64)
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs, 0.998

    res = l0_circular_rotation_readout(k, clear_winner_probs, tau=1.5, min_passes=2)
    assert res["passes_used"] == 2
    assert res["early_stopped"] is True
    assert res["margin"] >= 1.5
    assert int(np.argmax(res["probabilities"])) == 0


def test_answer_mass_gate_and_prior_debiasing() -> None:
    """Answer mass gate checks sum of option token probabilities in full vocab; prior debiasing corrects label prior."""
    full_vocab_logits = np.array([-10.0, 3.0, 4.0, -12.0, -11.0], dtype=np.float64)
    mass = compute_answer_mass(full_vocab_logits, option_token_ids=[1, 2])
    assert mass > 0.99

    low_mass = compute_answer_mass(np.zeros(100), option_token_ids=[1, 2])
    assert low_mass < 0.05

    # Prior debiasing: p_dbs \propto p / q_hat^alpha
    skewed_probs = np.array([0.60, 0.40], dtype=np.float64)
    empirical_prior = np.array([0.80, 0.20], dtype=np.float64)  # Model strongly prefers label 0
    debiased = apply_prior_debiasing(skewed_probs, empirical_prior, alpha=0.75)
    assert np.isclose(np.sum(debiased), 1.0)
    # After dividing out the heavy prior on index 0, index 1 becomes the winner
    assert debiased[1] > debiased[0]


def test_anyjev_prefill_backend_end_to_end() -> None:
    """AnyJevPrefillBackend evaluates noul, choice, and score with L0 rotation and exact 0.0 order-flip drift."""
    backend = AnyJevPrefillBackend(tau=1.2, empirical_prior={"billing": 0.35, "technical": 0.35, "shipping": 0.30})
    state = "Customer says: I was billed twice on my credit card, please refund my invoice."
    options = {
        "billing": "Invoices, double charges, refunds, payment issues",
        "technical": "App crashes, login bugs, server outage",
        "shipping": "Delayed package, lost parcel, courier delivery",
        "legal": "Subpoena, GDPR compliance",
        "security": "Phishing, compromised credentials",
    }
    res_forward = backend.evaluate_choice(state, "Route the support ticket", options)
    res_reverse = backend.evaluate_choice(
        state, "Route the support ticket", dict(reversed(list(options.items())))
    )
    assert res_forward["choice"] == "billing"
    assert res_reverse["choice"] == "billing"
    assert res_forward["x_l0_meta"]["answer_mass"] > 0.90
    for k in options:
        assert abs(res_forward["probabilities"][k] - res_reverse["probabilities"][k]) < 1e-12
