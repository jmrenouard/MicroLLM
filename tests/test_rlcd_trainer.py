"""Tests for OpenJev-RLCD (arXiv:2609.38850) & StartLux RLCD² (arXiv:2610.17309):
- Shifted Brier score strictly proper maximum at u = q
- Proposition 1 variance identity for group-mean vs per-rollout scoring
- Analytical Stage 1 pathwise readout gradient vs numerical finite-difference
- Stage 2 leave-one-out REINFORCE advantage with KL anchor beta
- Two-stage 'Calibrate, then Reinforce' trainer improving calibration & Brier score
"""

from __future__ import annotations

import numpy as np
import pytest

from microllm.rlcd import (
    RLCDConfig,
    RLCDDecisionHeadTrainer,
    beta_decision_weight,
    compute_leave_one_out_kl_advantages,
    group_mean_variance_identity,
    rlcd2_weighted_brier_reward,
    shifted_brier_reward,
    stage1_pathwise_loss_and_grad,
    verify_tabular_rlcd_unbiased_gradient,
)


def test_shifted_brier_strictly_proper_unique_optimum() -> None:
    """Expected shifted Brier score E_{y~q}[2 u_y - ||u||^2] is strictly maximized at u = q."""
    q = np.array([0.15, 0.65, 0.20], dtype=np.float64)
    # Expected reward under target distribution q is 2 u^T q - ||u||^2
    opt_val = float(2.0 * np.dot(q, q) - np.sum(q**2))

    rng = np.random.default_rng(11)
    for _ in range(50):
        raw = rng.dirichlet(np.ones(3))
        if not np.allclose(raw, q, atol=1e-4):
            val = float(2.0 * np.dot(raw, q) - np.sum(raw**2))
            assert val < opt_val

    # Single label y=1 gives J(u, 1) = 1 - ||u - e_1||^2
    u = np.array([0.1, 0.7, 0.2], dtype=np.float64)
    e_1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert np.isclose(shifted_brier_reward(u, 1), 1.0 - np.sum((u - e_1) ** 2))


def test_proposition_1_group_mean_variance_identity() -> None:
    """OpenJev-RLCD Proposition 1: J(E_r[u(r)], q) == E_r[J(u(r), q)] + E_r[||u(r) - p_bar||^2]."""
    rollout_probs = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.70, 0.10],
            [0.50, 0.30, 0.20],
            [0.60, 0.25, 0.15],
        ],
        dtype=np.float64,
    )
    target_q = np.array([0.60, 0.30, 0.10], dtype=np.float64)
    ident = group_mean_variance_identity(rollout_probs, target_q)
    assert ident["diversity_bonus"] > 0.0
    assert np.isclose(
        ident["score_of_mean"],
        ident["mean_of_scores"] + ident["diversity_bonus"],
        atol=1e-10,
    )


@pytest.mark.parametrize("use_weights", [False, True])
def test_stage1_pathwise_gradient_matches_finite_differences(use_weights: bool) -> None:
    """Exact analytical gradient of Stage 1 pathwise Brier loss wrt logits matches finite differences."""
    logits = np.array([0.4, -0.7, 1.8, 0.2], dtype=np.float64)
    target = 2
    loss, grad = stage1_pathwise_loss_and_grad(logits, target, use_rlcd2_weights=use_weights)
    assert loss >= 0.0
    assert np.isfinite(loss)

    eps = 1e-6
    num_grad = np.zeros_like(logits)
    for i in range(len(logits)):
        l_plus = logits.copy()
        l_minus = logits.copy()
        l_plus[i] += eps
        l_minus[i] -= eps
        loss_p, _ = stage1_pathwise_loss_and_grad(l_plus, target, use_rlcd2_weights=use_weights)
        loss_m, _ = stage1_pathwise_loss_and_grad(l_minus, target, use_rlcd2_weights=use_weights)
        num_grad[i] = (loss_p - loss_m) / (2.0 * eps)
    np.testing.assert_allclose(grad, num_grad, atol=1e-6)


def test_openjev_rlcd_appendix_a_tabular_unbiased_gradient() -> None:
    """OpenJev-RLCD Appendix A (test_rlcd_rb.py): exact enumeration verifies that the Rao-Blackwellized
    hybrid gradient (pathwise readout + REINFORCE trajectory) equals the true gradient of J(p_theta, q)
    to machine precision (< 1e-14).
    """
    res = verify_tabular_rlcd_unbiased_gradient(seed=7)
    assert res["max_policy_grad_error"] < 1e-14
    assert res["max_readout_grad_error"] < 1e-14


def test_stage2_leave_one_out_kl_anchored_advantages() -> None:
    """OpenJev-RLCD Algorithm 1 Stage 2: LOO baseline with KL anchor beta penalizes divergence from Stage 1."""
    rewards = np.array([0.85, 0.40, 0.75, 0.60], dtype=np.float64)
    log_pi_theta = np.array([-1.2, -0.5, -1.1, -0.9], dtype=np.float64)
    log_pi_ref = np.array([-1.2, -2.5, -1.1, -0.9], dtype=np.float64)  # Rollout 1 drifted far from ref!

    adv = compute_leave_one_out_kl_advantages(rewards, log_pi_theta, log_pi_ref, beta=0.04)
    assert len(adv) == 4
    # Sum of leave-one-out centered advantages is 0
    assert np.isclose(np.sum(adv), 0.0, atol=1e-10)
    # Rollout 0 has highest reward and zero KL drift -> positive advantage
    assert adv[0] > 0.0
    # Rollout 1 has lowest reward and positive KL penalty -> negative advantage
    assert adv[1] < 0.0


def test_rlcd2_beta_decision_weighting_and_two_stage_trainer() -> None:
    """StartLux RLCD² weights high-confidence regions more heavily; 2-stage trainer reduces Brier loss."""
    # Beta(8, 2) mixture weight is higher near confidence 0.9 than near 0.2
    w_high = beta_decision_weight(0.92)
    w_low = beta_decision_weight(0.25)
    assert w_high > w_low > 0.0

    u = np.array([0.85, 0.15], dtype=np.float64)
    r_weighted = rlcd2_weighted_brier_reward(u, 0)
    assert np.isfinite(r_weighted)

    # Train a local decision readout head with Algorithm 1 (Stage 1 Calibrate + Stage 2 Reinforce)
    rng = np.random.default_rng(99)
    n_samples, d_feat, k_classes = 80, 12, 3
    features = rng.normal(0.0, 1.0, size=(n_samples, d_feat))
    true_w = rng.normal(0.0, 1.2, size=(d_feat, k_classes))
    logits = features @ true_w
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    labels = np.array([rng.choice(k_classes, p=probs[i]) for i in range(n_samples)], dtype=np.int64)

    cfg = RLCDConfig(
        stage1_steps=40,
        stage2_steps=30,
        lr_stage1=0.65,
        lr_stage2=0.15,
        num_rollouts=4,
        kl_beta=0.04,
        reinforce_weight_c=0.3,
        use_rlcd2_weights=True,
    )
    trainer = RLCDDecisionHeadTrainer(d_feat=d_feat, k_classes=k_classes, config=cfg, seed=42)
    metrics = trainer.fit(features, labels)

    assert metrics["final_brier"] < metrics["initial_brier"]
    assert metrics["stage1_brier"] < metrics["initial_brier"]
    assert metrics["final_accuracy"] >= metrics["initial_accuracy"]
    assert metrics["final_accuracy"] > 0.70
    assert len(metrics["history"]) == cfg.stage1_steps + cfg.stage2_steps
