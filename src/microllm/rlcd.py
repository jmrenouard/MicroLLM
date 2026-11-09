"""OpenJev-RLCD (arXiv:2609.38850) & StartLux-Decision RLCD² (arXiv:2610.17309) Local Training Engine.

Implements:
1. Strictly proper shifted Brier scoring rule: J(u, y) = 2 u_y - ||u||^2 = 1 - ||u - e_y||^2
2. Proposition 1 (Variance identity for group-mean vs per-rollout scoring):
   J(E_r[u(r)], q) = E_r[J(u(r), q)] + E_r[||u(r) - p_bar||^2]
3. StartLux RLCD² decision-weighted Brier reward using w(c) = 0.5 * Uniform(0,1) + 0.5 * Beta(8,2)
4. Algorithm 1 ("Calibrate, then Reinforce"):
   - Stage 1: Deterministic pathwise readout calibration minimizing Brier loss
   - Stage 2: Leave-One-Out (LOO) REINFORCE with KL-divergence anchor beta to prevent System-One Collapse
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import numpy as np
from scipy.stats import beta as beta_dist

from microllm.calibration import brier_score, expected_calibration_error


def shifted_brier_reward(u: np.ndarray, target: int | np.ndarray) -> float:
    """Compute strictly proper shifted Brier score J(u, y) = 2 u^T q - ||u||^2.

    When `target` is an integer class index y, equals 2 u_y - ||u||^2 = 1 - ||u - e_y||^2.
    """
    probs = np.asarray(u, dtype=np.float64)
    if isinstance(target, (int, np.integer)):
        q = np.zeros_like(probs)
        q[int(target)] = 1.0
    else:
        q = np.asarray(target, dtype=np.float64)
    return float(2.0 * np.dot(probs, q) - np.sum(probs**2))


def group_mean_variance_identity(
    rollout_probs: np.ndarray, target_q: np.ndarray
) -> dict[str, float]:
    """Verify Proposition 1 from OpenJev-RLCD (arXiv:2609.38850):

    J(E_r[u(r)], q) = E_r[J(u(r), q)] + E_r[||u(r) - p_bar||^2]
    where the second term is the artificial diversity bonus paid by group-mean scoring.
    """
    U = np.asarray(rollout_probs, dtype=np.float64)
    q = np.asarray(target_q, dtype=np.float64)
    p_bar = np.mean(U, axis=0)

    score_of_mean = shifted_brier_reward(p_bar, q)
    per_rollout_scores = [shifted_brier_reward(U[i], q) for i in range(len(U))]
    mean_of_scores = float(np.mean(per_rollout_scores))
    diversity_bonus = float(np.mean(np.sum((U - p_bar) ** 2, axis=1)))
    return {
        "score_of_mean": score_of_mean,
        "mean_of_scores": mean_of_scores,
        "diversity_bonus": diversity_bonus,
    }


def beta_decision_weight(confidence: float, alpha: float = 8.0, beta: float = 2.0) -> float:
    """StartLux-Decision RLCD² density weight: w(c) = 0.5 * 1.0 + 0.5 * BetaPDF(c; 8, 2).

    Emphasizes calibration accuracy in the high-confidence automation zone [0.8, 1.0].
    """
    c = float(np.clip(confidence, 1e-4, 1.0 - 1e-4))
    if alpha == 8.0 and beta == 2.0:
        pdf_val = 72.0 * (c**7) * (1.0 - c)
    else:
        pdf_val = float(beta_dist.pdf(c, alpha, beta))
    return float(0.5 + 0.5 * pdf_val)


def rlcd2_weighted_brier_reward(u: np.ndarray, target: int) -> float:
    """Decision-weighted Brier reward from StartLux-Decision (arXiv:2610.17309)."""
    probs = np.asarray(u, dtype=np.float64)
    conf = float(np.max(probs))
    weight = beta_decision_weight(conf)
    base_brier_loss = 1.0 - shifted_brier_reward(probs, target)
    return float(1.0 - weight * base_brier_loss)


def stage1_pathwise_loss_and_grad(
    logits: np.ndarray,
    target: int,
    use_rlcd2_weights: bool = False,
    decision_weight: float | None = None,
) -> tuple[float, np.ndarray]:
    """Compute exact Stage 1 pathwise Brier loss L_1 and analytical gradient dL_1/dz.

    For base Brier loss B(u, y) = ||u - e_y||^2 = 1 - 2 u_y + ||u||^2 >= 0 where u = softmax(z):
    - When `decision_weight` is provided (stop-gradient sg(w(c))) or `use_rlcd2_weights=False`,
       L_1 = w * B(u, y) and dL_1/du_k = 2 * w * (u_k - 1_{k == y}).
    - When `use_rlcd2_weights=True` and `decision_weight is None`, L_1 = w(max(u)) * B(u, y)
      and dL_1/du_k includes the chain-rule derivative dw(u_{k*})/du_{k*} * B(u, y).
    """
    z = np.asarray(logits, dtype=np.float64)
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    u = exp_z / np.sum(exp_z)

    e_y = np.zeros_like(u)
    e_y[int(target)] = 1.0

    base_brier = float(1.0 - 2.0 * u[int(target)] + np.sum(u**2))
    k_star = int(np.argmax(u))
    c_raw = float(u[k_star])

    if decision_weight is not None:
        weight = float(decision_weight)
        dw_dc = 0.0
    elif use_rlcd2_weights:
        weight = beta_decision_weight(c_raw)
        if 1e-4 < c_raw < 1.0 - 1e-4:
            alpha_b, beta_b = 8.0, 2.0
            if alpha_b == 8.0 and beta_b == 2.0:
                pdf_val = 72.0 * (c_raw**7) * (1.0 - c_raw)
            else:
                pdf_val = float(beta_dist.pdf(c_raw, alpha_b, beta_b))
            dw_dc = 0.5 * pdf_val * ((alpha_b - 1.0) / c_raw - (beta_b - 1.0) / (1.0 - c_raw))
        else:
            dw_dc = 0.0
    else:
        weight = 1.0
        dw_dc = 0.0

    loss = float(weight * base_brier)
    g_u = 2.0 * weight * (u - e_y)
    if dw_dc != 0.0:
        g_u[k_star] += dw_dc * base_brier

    dot_gu = float(np.dot(u, g_u))
    grad_z = u * (g_u - dot_gu)
    return loss, grad_z


def compute_leave_one_out_kl_advantages(
    rewards: np.ndarray,
    log_pi_theta: np.ndarray,
    log_pi_ref: np.ndarray,
    beta: float = 0.04,
) -> np.ndarray:
    """OpenJev-RLCD Algorithm 1 Stage 2 Leave-One-Out (LOO) KL-anchored advantage:

    tilde_R_i = R_i - beta * (log pi_theta(r_i|x) - log pi_{theta_1}(r_i|x))
    A_i = (1/M) * (tilde_R_i - (1/(M-1)) * sum_{j != i} tilde_R_j)
    """
    r = np.asarray(rewards, dtype=np.float64)
    lp_cur = np.asarray(log_pi_theta, dtype=np.float64)
    lp_ref = np.asarray(log_pi_ref, dtype=np.float64)
    m = r.shape[-1]
    if m <= 1:
        return np.zeros_like(r)

    kl_penalized = r - float(beta) * (lp_cur - lp_ref)
    total = np.sum(kl_penalized, axis=-1, keepdims=True)
    loo_baseline = (total - kl_penalized) / float(m - 1)
    advantages = (kl_penalized - loo_baseline) / float(m)
    return advantages


def verify_tabular_rlcd_unbiased_gradient(
    r_types: int = 3,
    k_classes: int = 3,
    m_rollouts: int = 3,
    beta: float = 0.04,
    seed: int = 7,
) -> dict[str, float]:
    """Exact enumeration check (`test_rlcd_rb.py`) from OpenJev-RLCD Appendix A (arXiv:2609.38850, p. 11).

    Enumerates all R^M rationale tuples and all K outcomes for a tabular policy pi_theta(r) and
    readout u_theta(r), comparing the expected Stage 1 pathwise + Stage 2 LOO KL-shaped REINFORCE
    surrogate gradient against the exact analytical gradient of:
        L(theta) = -E_{Y~q} E_{r~pi_theta} [ J(u_theta(r), Y) - beta * log(pi_theta(r) / pi_ref(r)) ].
    """
    import itertools

    rng = np.random.default_rng(abs(int(seed)))
    pol_logits = rng.normal(0.0, 0.7, size=r_types)
    ref_logits = rng.normal(0.0, 0.7, size=r_types)
    readout_logits = rng.normal(0.0, 0.8, size=(r_types, k_classes))
    q_raw = rng.uniform(0.1, 1.0, size=k_classes)
    target_q = q_raw / np.sum(q_raw)

    pi_theta = np.exp(pol_logits - np.max(pol_logits))
    pi_theta /= np.sum(pi_theta)
    pi_ref = np.exp(ref_logits - np.max(ref_logits))
    pi_ref /= np.sum(pi_ref)

    log_pi_theta = np.log(pi_theta)
    log_pi_ref = np.log(pi_ref)

    u_table = np.zeros((r_types, k_classes), dtype=np.float64)
    for r in range(r_types):
        z = readout_logits[r]
        ez = np.exp(z - np.max(z))
        u_table[r] = ez / np.sum(ez)

    # 1. Exact analytical gradient of L(theta) wrt pol_logits and readout_logits
    per_r_expected_reward = np.array(
        [shifted_brier_reward(u_table[r], target_q) for r in range(r_types)], dtype=np.float64
    )
    shaped_r = per_r_expected_reward - beta * (log_pi_theta - log_pi_ref)
    # Note: d/d(pol_logits) E_{r~pi}[R(r) - beta * log(pi(r)/pi_ref(r))]
    # = sum_r pi_r * (shaped_r - beta) * (e_r - pi)
    effective_pol_val = shaped_r - beta
    mean_pol_val = float(np.dot(pi_theta, effective_pol_val))
    exact_grad_pol = -pi_theta * (effective_pol_val - mean_pol_val)

    exact_grad_readout = np.zeros_like(readout_logits)
    for r in range(r_types):
        for y in range(k_classes):
            _, gz = stage1_pathwise_loss_and_grad(readout_logits[r], y, use_rlcd2_weights=False)
            exact_grad_readout[r] += pi_theta[r] * target_q[y] * gz

    # 2. Expected surrogate gradient by enumerating all R^M tuples and K outcomes
    enum_grad_pol = np.zeros_like(pol_logits)
    enum_grad_readout = np.zeros_like(readout_logits)

    for tuple_r in itertools.product(range(r_types), repeat=m_rollouts):
        p_tuple = 1.0
        for r_idx in tuple_r:
            p_tuple *= float(pi_theta[r_idx])
        lp_cur = np.array([log_pi_theta[r_idx] for r_idx in tuple_r], dtype=np.float64)
        lp_ref = np.array([log_pi_ref[r_idx] for r_idx in tuple_r], dtype=np.float64)

        for y in range(k_classes):
            weight_ty = p_tuple * float(target_q[y])
            rewards = np.array(
                [shifted_brier_reward(u_table[r_idx], y) for r_idx in tuple_r], dtype=np.float64
            )
            adv = compute_leave_one_out_kl_advantages(rewards, lp_cur, lp_ref, beta=beta)

            for m_idx, r_idx in enumerate(tuple_r):
                # d log pi_theta(r_idx) / d pol_logits = e_{r_idx} - pi_theta
                d_log_pi = -pi_theta.copy()
                d_log_pi[r_idx] += 1.0
                # Surrogate loss has - sum_m sg(adv_m) * log pi_theta(r_m) +KL term inside adv_m
                # Plus the direct d/d(theta) of -beta * log(pi_theta/pi_ref) is handled when beta shaping is in R_i
                enum_grad_pol += weight_ty * (-(adv[m_idx] - beta / float(m_rollouts)) * d_log_pi)

                _, gz = stage1_pathwise_loss_and_grad(
                    readout_logits[r_idx], y, use_rlcd2_weights=False
                )
                enum_grad_readout[r_idx] += weight_ty * (gz / float(m_rollouts))

    pol_diff = float(np.max(np.abs(exact_grad_pol - enum_grad_pol)))
    readout_diff = float(np.max(np.abs(exact_grad_readout - enum_grad_readout)))
    return {
        "max_policy_grad_error": pol_diff,
        "max_readout_grad_error": readout_diff,
    }


@dataclass(frozen=True)
class RLCDConfig:
    """Hyperparameters for OpenJev-RLCD 2-stage 'Calibrate, then Reinforce' training."""

    stage1_steps: int = 40
    stage2_steps: int = 30
    lr_stage1: float = 0.85
    lr_stage2: float = 0.25
    num_rollouts: int = 4
    kl_beta: float = 0.04
    reinforce_weight_c: float = 0.30
    use_rlcd2_weights: bool = True


class RLCDDecisionHeadTrainer:
    """Local CPU trainer implementing Algorithm 1 ('Calibrate, then Reinforce') from OpenJev-RLCD.

    Models a stochastic latent rationale modulator `r_i ~ N(mu_theta(x), sigma^2)` followed by
    a calibrated readout projection `u_theta(x, r_i)`.
    - Stage 1 trains the readout pathwise on the strictly proper Brier loss.
    - Stage 2 freezes the Stage-1 anchor `theta_1` for KL regularization (`beta=0.04`) and
      jointly updates readout + rationale policy using Leave-One-Out REINFORCE (`c=0.3`).
    """

    def __init__(
        self,
        d_feat: int,
        k_classes: int,
        config: RLCDConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.d_feat = int(d_feat)
        self.k_classes = int(k_classes)
        self.config = config or RLCDConfig()
        self.rng = np.random.default_rng(abs(int(seed)))

        # Initialize uncalibrated weights
        self.w_readout = self.rng.normal(0.0, 0.25, size=(self.d_feat, self.k_classes))
        self.b_readout = np.zeros(self.k_classes, dtype=np.float64)
        self.w_policy = np.zeros((self.d_feat, self.d_feat), dtype=np.float64)
        self.w_policy_anchor = self.w_policy.copy()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Compute calibrated readout probabilities for input feature matrix."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(
                f"features must be a 2D array of shape (n_samples, d_feat), got ndim={x.ndim} shape={x.shape}"
            )
        if x.shape[1] != self.d_feat:
            raise ValueError(
                f"Feature dimension mismatch: expected d_feat={self.d_feat}, got {x.shape[1]}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("features contains NaN or Infinite values.")
        modulated = x + 0.25 * np.tanh(x @ self.w_policy)
        logits = modulated @ self.w_readout + self.b_readout
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Execute Stage 1 (Calibrate readout) followed by Stage 2 (KL-anchored LOO Reinforce)."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if x.ndim != 2:
            raise ValueError(
                f"features must be a 2D array of shape (n_samples, d_feat), got ndim={x.ndim} shape={x.shape}"
            )
        if x.shape[0] == 0:
            raise ValueError("features and labels must be non-empty (0 samples provided).")
        if x.shape[1] != self.d_feat:
            raise ValueError(
                f"Feature dimension mismatch: expected d_feat={self.d_feat}, got {x.shape[1]}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("features contains NaN or Infinite values.")

        if y.ndim != 1:
            raise ValueError(
                f"labels must be a 1D array of shape (n_samples,), got ndim={y.ndim} shape={y.shape}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: features has {x.shape[0]} samples, labels has {y.shape[0]} items."
            )
        if np.any(y < 0):
            raise ValueError(
                f"labels must be non-negative integers, got min={int(np.min(y))}."
            )
        if np.any(y >= self.k_classes):
            raise ValueError(
                f"labels must be in range [0, {self.k_classes - 1}], got max={int(np.max(y))}."
            )
        n = len(y)

        init_probs = self.predict_proba(x)
        init_brier = brier_score(init_probs, y)
        init_ece = expected_calibration_error(init_probs, y)
        init_acc = float(np.mean(np.argmax(init_probs, axis=1) == y))

        history: list[dict[str, float | int | str]] = []

        # Stage 1: Pathwise Readout Calibration (Algorithm 1 lines 2-7)
        e_y = np.zeros((n, self.k_classes), dtype=np.float64)
        e_y[np.arange(n), y] = 1.0

        for step in range(self.config.stage1_steps):
            time.sleep(0.0045)
            logits = x @ self.w_readout + self.b_readout
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_z = np.exp(shifted)
            u = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            if self.config.use_rlcd2_weights:
                c_raw = np.max(u, axis=1)
                c_clip = np.clip(c_raw, 1e-4, 1.0 - 1e-4)
                weights = 0.5 + 0.5 * 72.0 * (c_clip**7) * (1.0 - c_clip)
            else:
                weights = np.ones(n, dtype=np.float64)

            g_u = 2.0 * weights[:, None] * (u - e_y)
            dot_gu = np.sum(u * g_u, axis=1, keepdims=True)
            grad_z = u * (g_u - dot_gu)

            grad_w = x.T @ grad_z
            grad_b = np.sum(grad_z, axis=0)

            self.w_readout -= self.config.lr_stage1 * (grad_w / float(n))
            self.b_readout -= self.config.lr_stage1 * (grad_b / float(n))

            cur_probs = self.predict_proba(x)
            history.append(
                {
                    "step": step + 1,
                    "stage": "stage1_calibrate",
                    "brier": round(brier_score(cur_probs, y), 4),
                    "ece": round(expected_calibration_error(cur_probs, y), 4),
                }
            )

        stage1_probs = self.predict_proba(x)
        stage1_brier = brier_score(stage1_probs, y)
        self.w_policy_anchor = self.w_policy.copy()

        # Stage 2: KL-Anchored Leave-One-Out REINFORCE + Pathwise Readout (Algorithm 1 lines 8-19)
        m_roll = self.config.num_rollouts
        sigma = 0.25
        beta = self.config.kl_beta
        c_reinforce = self.config.reinforce_weight_c

        e_y_stage2 = np.zeros((n, m_roll, self.k_classes), dtype=np.float64)
        e_y_stage2[np.arange(n)[:, None], np.arange(m_roll)[None, :], y[:, None]] = 1.0

        for step in range(self.config.stage2_steps):
            time.sleep(0.0045)
            mu_cur_all = np.tanh(x @ self.w_policy)
            mu_ref_all = np.tanh(x @ self.w_policy_anchor)
            noise_all = self.rng.normal(0.0, sigma, size=(n, m_roll, self.d_feat))
            r_samples_all = mu_cur_all[:, None, :] + noise_all

            log_pi_cur_all = -0.5 * np.sum((noise_all / sigma) ** 2, axis=2)
            log_pi_ref_all = -0.5 * np.sum(((r_samples_all - mu_ref_all[:, None, :]) / sigma) ** 2, axis=2)

            mod_x_vec = x[:, None, :] + 0.25 * r_samples_all
            mod_x_flat = mod_x_vec.reshape(n * m_roll, self.d_feat)
            logits_flat = mod_x_flat @ self.w_readout + self.b_readout
            logits_vec = logits_flat.reshape(n, m_roll, self.k_classes)
            shifted_vec = logits_vec - np.max(logits_vec, axis=2, keepdims=True)
            exp_z_vec = np.exp(shifted_vec)
            probs_vec = exp_z_vec / np.sum(exp_z_vec, axis=2, keepdims=True)

            if self.config.use_rlcd2_weights:
                c_max_vec = np.max(probs_vec, axis=2)
                c_clip_vec = np.clip(c_max_vec, 1e-4, 1.0 - 1e-4)
                w_sg_vec = 0.5 + 0.5 * 72.0 * (c_clip_vec**7) * (1.0 - c_clip_vec)
            else:
                w_sg_vec = np.ones((n, m_roll), dtype=np.float64)

            g_u_vec = 2.0 * w_sg_vec[:, :, None] * (probs_vec - e_y_stage2)
            dot_gu_vec = np.sum(probs_vec * g_u_vec, axis=2, keepdims=True)
            gz_vec = probs_vec * (g_u_vec - dot_gu_vec)
            gz_flat = gz_vec.reshape(n * m_roll, self.k_classes)

            grad_w = (mod_x_flat.T @ gz_flat) / float(m_roll)
            grad_b = np.sum(gz_vec, axis=(0, 1)) / float(m_roll)

            probs_y_vec = probs_vec[np.arange(n)[:, None], np.arange(m_roll)[None, :], y[:, None]]
            base_brier_vec = 1.0 - 2.0 * probs_y_vec + np.sum(probs_vec**2, axis=2)
            rewards_vec = (
                (1.0 - w_sg_vec * base_brier_vec)
                if self.config.use_rlcd2_weights
                else (2.0 * probs_y_vec - np.sum(probs_vec**2, axis=2))
            )

            kl_pen_vec = rewards_vec - beta * (log_pi_cur_all - log_pi_ref_all)
            tot_vec = np.sum(kl_pen_vec, axis=1, keepdims=True)
            loo_vec = (tot_vec - kl_pen_vec) / float(m_roll - 1)
            adv_vec = (kl_pen_vec - loo_vec) / float(m_roll)

            dtanh_vec = 1.0 - mu_cur_all**2
            sum_adv_noise_vec = np.sum(adv_vec[:, :, None] * noise_all, axis=1)
            score_grad_sum_vec = (sum_adv_noise_vec / (sigma**2)) * dtanh_vec
            grad_pol = - c_reinforce * (x.T @ score_grad_sum_vec)

            self.w_readout -= self.config.lr_stage2 * (grad_w / float(n))
            self.b_readout -= self.config.lr_stage2 * (grad_b / float(n))
            self.w_policy -= self.config.lr_stage2 * (grad_pol / float(n))

            cur_probs = self.predict_proba(x)
            history.append(
                {
                    "step": self.config.stage1_steps + step + 1,
                    "stage": "stage2_reinforce",
                    "brier": round(brier_score(cur_probs, y), 4),
                    "ece": round(expected_calibration_error(cur_probs, y), 4),
                }
            )

        final_probs = self.predict_proba(x)
        return {
            "initial_brier": round(init_brier, 4),
            "stage1_brier": round(stage1_brier, 4),
            "final_brier": round(brier_score(final_probs, y), 4),
            "initial_ece": round(init_ece, 4),
            "final_ece": round(expected_calibration_error(final_probs, y), 4),
            "initial_accuracy": round(init_acc, 4),
            "final_accuracy": round(float(np.mean(np.argmax(final_probs, axis=1) == y)), 4),
            "history": history,
        }


class RLCDReplayBuffer:
    """Continual calibration & active learning ring buffer prioritizing uncertain examples (IDEA-009)."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = max(8, int(capacity))
        self._items: list[tuple[np.ndarray, int, float]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, features: np.ndarray, label: int, confidence: float = 0.5) -> None:
        """Record a feature vector, gold label, and prediction confidence."""
        feat = np.asarray(features, dtype=np.float64).reshape(-1)
        entry = (feat, int(label), float(confidence))
        if len(self._items) < self.capacity:
            self._items.append(entry)
        else:
            # Replace highest-confidence (least informative) item
            max_idx = max(range(len(self._items)), key=lambda idx: self._items[idx][2])
            if float(confidence) <= self._items[max_idx][2]:
                self._items[max_idx] = entry

    def sample_uncertain(self, max_samples: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Return up to `max_samples` lowest-confidence examples for RLCD fine-tuning."""
        if not self._items:
            raise ValueError("RLCDReplayBuffer is empty.")
        ordered = sorted(self._items, key=lambda item: item[2])[: max(1, int(max_samples))]
        x = np.stack([item[0] for item in ordered], axis=0)
        y = np.array([item[1] for item in ordered], dtype=np.int64)
        return x, y

    def fit_trainer(
        self, trainer: RLCDDecisionHeadTrainer, max_samples: int = 128
    ) -> dict[str, Any]:
        """Fine-tune `trainer` on the lowest-confidence examples in the buffer."""
        x, y = self.sample_uncertain(max_samples=max_samples)
        return trainer.fit(x, y)
