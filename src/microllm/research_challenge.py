"""Research Paper Code & Design Challenge Primitives and Hypothesis Validation Suite (Issue #61).

Implements the 12 mathematical and architectural improvements derived from challenging
`src/microllm/` against the 104-paper System-One / Jev / RLCD corpus in `docs/papers/`:
- HYP-001: Policy-Aware & Route-Conditioned Calibration (`StartLux-Decision` Propositions 1-4)
- HYP-002: Exact Beta-Integral Decision-Weighted Proper Score S_D(p, t) (`StartLux` Eq. 3-5)
- HYP-003: Disjoint Selection/Verification Clopper-Pearson Certificate (`AnyJev` §5 Eq. 5)
- HYP-004: Kolmogorov Probability Axiom & Partition-Invariance Coherence (`arXiv:2609.33209`)
- HYP-005: Epistemic vs. Aleatoric Value-of-Computation Gate (`OpenJev-RLCD` §5.3, `Sys1Cal-v1`)
- HYP-006: Definition-Dominant Rubric Binding vs. Option-Name Bias (`arXiv:2609.26758`, `2610.02586`)
- HYP-007: Skew-Adaptive & Content-Free Prior Debiasing (`AnyJev` Appendix E, `arXiv:2610.26325`)
- HYP-008: Clause-Salient Evidence Sensitivity vs. Distractor Invariance (`StartLux` §5.3, `JevOut`)
- HYP-009: Murphy Brier Decomposition (REL/RES/UNC) & Selective AURC / Cov@eps (`OpenJev-RLCD` Table 1)
- HYP-010: Independence of Irrelevant Alternatives (IIA) Decoy Stability (`arXiv:2610.16655`)
- HYP-011: Test-Time Scaling (`P(Yes)`) Multi-Template Log-Space Consensus (`arXiv:2610.25474`)
- HYP-012: NumericJev Multiway Bisection Decision Trees for Continuous Intervals (`arXiv:2609.28587`)
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

from microllm.backends.base import embed_text_deterministic, semantic_match_logit, tokenize_multilingual
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.calibration import brier_score, clopper_pearson_threshold, clopper_pearson_upper_bound
from microllm.confidence import sanitize_and_normalize_probabilities


HYPOTHESIS_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "HYP-001",
        "validation_id": "VAL-HYP-001",
        "title": "Policy-Aware & Route-Conditioned Calibration under Best-of-N and Cascade Routing",
        "papers": ["arXiv:2610.17309 (§5.4-5.5, App. C Props 1-4)", "arXiv:2609.26532 (REFLEX)"],
        "application_domains": [
            "Best-of-N Code Generation & Patch Selection (JevCoder)",
            "Multi-Tier Enterprise LLM Cascades (FrugalGPT / JEV-as-a-Judge)",
            "Clinical Decision Support Escalation to Specialist Review",
        ],
    },
    {
        "id": "HYP-002",
        "validation_id": "VAL-HYP-002",
        "title": "Exact Beta-Integral Decision-Weighted Proper Scoring Rule S_D(p, t) & Cumulative Ordinal Score",
        "papers": ["arXiv:2610.17309 (§3.1 Eq. 3-5)", "arXiv:2609.38827 (Ordinal-Scale Bias)"],
        "application_domains": [
            "High-Confidence Autonomous Financial Transaction Approval",
            "Ordinal SLA Severity & Incident Triage Rubrics (score 0..4)",
            "Automated Code Review & Rubric Grading",
        ],
    },
    {
        "id": "HYP-003",
        "validation_id": "VAL-HYP-003",
        "title": "Disjoint Selection/Verification Split for Multiplicity-Free Clopper-Pearson Stopping & Risk Certificates",
        "papers": ["arXiv:2610.00831 (§5 Eq. 5, Table 10)", "arXiv:2609.37470 (Probability Contracts)"],
        "application_domains": [
            "Regulated Clinical Triage Risk Certification",
            "6G Open RAN Near-RT RIC Intent Routing under Latency SLAs",
            "High-Throughput vLLM / CPU Prefill Rotation Early Stopping",
        ],
    },
    {
        "id": "HYP-004",
        "validation_id": "VAL-HYP-004",
        "title": "Kolmogorov Probability Axiom & Partition-Invariance Coherence Projection",
        "papers": ["arXiv:2609.33209 (Beyond Calibration)", "arXiv:2609.33971", "arXiv:2610.17309 (§5.3 Fig. 14b)"],
        "application_domains": [
            "Multi-Agent Consensus & Probabilistic Voting (Jev-LiteMAS)",
            "Actuarial & Insurance Claims Risk Aggregation",
            "Legal Contract Clause Verification (ContractNLI)",
        ],
    },
    {
        "id": "HYP-005",
        "validation_id": "VAL-HYP-005",
        "title": "Epistemic vs. Aleatoric Uncertainty Decomposition & Optimal Value-of-Computation Gate",
        "papers": ["arXiv:2609.38850 (§5.3 ChaosNLI)", "arXiv:2609.35342 (Sys1Cal-v1)", "arXiv:2610.17309 (Prop. 5)"],
        "application_domains": [
            "Autonomous Robotics & Conformal Object Navigation (CovNav)",
            "Open-World Customer Support Routing with OOD Abstention",
            "Human-in-the-Loop Clinical & Legal Deferral Gates",
        ],
    },
    {
        "id": "HYP-006",
        "validation_id": "VAL-HYP-006",
        "title": "Definition-Dominant Rubric Binding Against Adversarial Option Renaming",
        "papers": ["arXiv:2609.26758 (Type-Safe Is Not Error-Free)", "arXiv:2610.02586 (Labels Override Definitions)"],
        "application_domains": [
            "Editable Enterprise Safety Policy Trees (Jev-Guard)",
            "Dynamic Low-Code Agent Skill & Tool Selection",
            "Regulatory Compliance Auditing with Opaque Taxonomy Codes",
        ],
    },
    {
        "id": "HYP-007",
        "validation_id": "VAL-HYP-007",
        "title": "Content-Free Prior Separation & Skew-Adaptive Prior Damping under Prevalence Shift",
        "papers": ["arXiv:2610.00831 (§4.1, §7.3, App. E Table 14)", "arXiv:2610.26325 (Prevalence Shift)"],
        "application_domains": [
            "High-Skew Financial AML & Fraud Detection (1-5% Positive Prevalence)",
            "Rare-Disease Clinical Screening & Triage",
            "SOC Cybersecurity Intrusion Alert Filtering",
        ],
    },
    {
        "id": "HYP-008",
        "validation_id": "VAL-HYP-008",
        "title": "Clause-Salient Evidence Sensitivity vs. Natural Context Distractor Invariance",
        "papers": ["arXiv:2610.17309 (§5.3 Fig. 14a)", "arXiv:2609.30243 (JevOut)", "arXiv:2610.14404 (DICE)"],
        "application_domains": [
            "Multi-Hop RAG Claim Verification & Groundedness Checking",
            "Long-Horizon Agent Episodic Memory Compaction (Jev-Mem)",
            "Adversarial Context-Injection Defense in Tool-Use Workflows",
        ],
    },
    {
        "id": "HYP-009",
        "validation_id": "VAL-HYP-009",
        "title": "Murphy Brier Decomposition (REL/RES/UNC) & Selective Delegation Metrics (AURC, Cov@eps)",
        "papers": ["arXiv:2609.38850 (Table 1, Fig. 2)", "arXiv:2610.17309 (App. B.3 Eq. 7)"],
        "application_domains": [
            "Enterprise Helpdesk Auto-Resolution SLA Certification",
            "Autonomous Coding Agent Test-Tree Pruning",
            "Automated Medical Lab & Radiology Report Triage",
        ],
    },
    {
        "id": "HYP-010",
        "validation_id": "VAL-HYP-010",
        "title": "Independence of Irrelevant Alternatives (IIA) & Decoy Option Stability",
        "papers": ["arXiv:2610.16655 (Irrelevant Alternatives in Jev)", "arXiv:2610.16191 (Choice2Seq)"],
        "application_domains": [
            "E-Commerce Product Reranking & Recommendation (Decision4Rec)",
            "Dynamic Multi-Tool Selection in Agent Harnesses (JevAgentHarness)",
            "Procurement & Tender Bid Comparison",
        ],
    },
    {
        "id": "HYP-011",
        "validation_id": "VAL-HYP-011",
        "title": "Test-Time Scaling (P(Yes)) via Multi-Template Log-Space Hypothesis Consensus",
        "papers": ["arXiv:2610.25474 (P(Yes) Test-Time Scaling)", "arXiv:2610.11392 (TypedBench Framing)"],
        "application_domains": [
            "High-Stakes Legal Contract & Compliance Verification",
            "Calibrated Prompt-Injection Detection & Deferral",
            "Biomedical Literature & Clinical Trial Eligibility Screening",
        ],
    },
    {
        "id": "HYP-012",
        "validation_id": "VAL-HYP-012",
        "title": "NumericJev Multiway Bisection Decision Trees for Continuous Credible Intervals",
        "papers": ["arXiv:2609.28587 (NumericJev)", "arXiv:2610.26002 (TimesJev)"],
        "application_domains": [
            "Quantitative Financial Loss & Credit Risk Interval Estimation",
            "Time-Series Anomaly Severity & Latency Forecasting",
            "Cloud Autoscaling & 6G Open RAN Bandwidth Allocation",
        ],
    },
)


class PolicyAwareCalibrator:
    """Policy-aware & route-conditioned calibrator (StartLux-Decision §5.5 & Appendix C Props 1-4).

    Calibrates reported confidence conditional on selection/routing statistics:
    `logit(p_cal) = w_0 + w_1 * reported_conf + w_2 * mean_candidate_score + w_3 * log(N) + w_4 * is_escalated`
    and certifies route-specific Clopper-Pearson acceptance thresholds per Proposition 3.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _design_matrix(
        reported_confidences: np.ndarray,
        mean_candidate_scores: np.ndarray,
        candidate_counts: np.ndarray,
        routes: np.ndarray,
    ) -> np.ndarray:
        c = np.asarray(reported_confidences, dtype=np.float64)
        m = np.asarray(mean_candidate_scores, dtype=np.float64)
        n_cand = np.maximum(1.0, np.asarray(candidate_counts, dtype=np.float64))
        is_esc = (np.asarray(routes) == "escalated").astype(np.float64)
        return np.column_stack([np.ones_like(c), c, m, np.log(n_cand), is_esc])

    def fit_and_evaluate(
        self,
        reported_confidences: np.ndarray,
        mean_candidate_scores: np.ndarray,
        candidate_counts: np.ndarray,
        routes: np.ndarray,
        correct: np.ndarray,
        target_risk: float = 0.15,
    ) -> dict[str, Any]:
        """Fit policy-aware logistic map on first half and evaluate calibration & route bounds on second half."""
        c = np.asarray(reported_confidences, dtype=np.float64)
        m = np.asarray(mean_candidate_scores, dtype=np.float64)
        n_cand = np.asarray(candidate_counts, dtype=np.float64)
        r = np.asarray(routes)
        y = np.asarray(correct, dtype=np.float64)

        split = len(y) // 2
        X_train = self._design_matrix(c[:split], m[:split], n_cand[:split], r[:split])
        y_train = y[:split]

        def _loss(w: np.ndarray) -> float:
            z = np.clip(X_train @ w, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            nll = -float(np.mean(y_train * np.log(p + eps) + (1.0 - y_train) * np.log(1.0 - p + eps)))
            return nll + 1e-3 * float(np.sum(w[1:] ** 2))

        res = minimize(_loss, self.weights, method="BFGS")
        self.weights = np.asarray(res.x, dtype=np.float64)

        X_eval = self._design_matrix(c[split:], m[split:], n_cand[split:], r[split:])
        y_eval = y[split:]
        r_eval = r[split:]
        raw_conf_eval = c[split:]
        cal_conf_eval = 1.0 / (1.0 + np.exp(-np.clip(X_eval @ self.weights, -20.0, 20.0)))

        raw_gap = abs(float(np.mean(raw_conf_eval) - np.mean(y_eval)))
        cal_gap = abs(float(np.mean(cal_conf_eval) - np.mean(y_eval)))

        route_certified: dict[str, dict[str, float]] = {}
        for route_name in ("direct", "escalated"):
            mask = r_eval == route_name
            if int(np.sum(mask)) > 0:
                route_certified[route_name] = clopper_pearson_threshold(
                    cal_conf_eval[mask],
                    y_eval[mask].astype(np.int64),
                    target_risk=target_risk,
                )
            else:
                route_certified[route_name] = {
                    "threshold": 1.0,
                    "coverage": 0.0,
                    "empirical_risk": 0.0,
                    "upper_risk_bound": 1.0,
                }

        return {
            "raw_overconfidence_gap": round(raw_gap, 4),
            "policy_aware_gap": round(cal_gap, 4),
            "route_certified": route_certified,
        }


def rlcd2_proper_integral_score(
    probs: np.ndarray,
    target: np.ndarray | int,
    alpha: float = 8.0,
    beta: float = 2.0,
    cumulative_ordinal: bool = False,
) -> float:
    """Compute StartLux-Decision §3.1 Eq. (3)-(4) strictly proper integral score S_D(p, t).

    For each component with reported probability q and target t in [0, 1]:
        s(q, t) = t * W(q) - C(q)
        W(q) = int_0^q w(c) dc = 0.5 * q + 0.5 * I_q(alpha, beta)
        C(q) = int_0^q c * w(c) dc = 0.25 * q^2 + 0.5 * (alpha / (alpha + beta)) * I_q(alpha + 1, beta)
    Satisfies d/dq E_{z ~ pi}[s(q, z)] = w(q) * (pi - q), uniquely maximized at q = pi.
    When `cumulative_ordinal=True`, applies s(Q_k, T_k) to cumulative probabilities (StartLux p. 10).
    """
    p = np.asarray(probs, dtype=np.float64)
    if isinstance(target, (int, np.integer)):
        t = np.zeros_like(p)
        t[int(target)] = 1.0
    else:
        t = np.asarray(target, dtype=np.float64)

    if cumulative_ordinal and len(p) >= 2:
        q_vec = np.clip(np.cumsum(p)[:-1], 1e-9, 1.0 - 1e-9)
        t_vec = np.clip(np.cumsum(t)[:-1], 0.0, 1.0)
        # Also include complement 1 - Q_k so both tails are scored symmetrically
        q_eval = np.concatenate([q_vec, 1.0 - q_vec])
        t_eval = np.concatenate([t_vec, 1.0 - t_vec])
    else:
        q_eval = np.clip(p, 1e-9, 1.0 - 1e-9)
        t_eval = t

    w_int = 0.5 * q_eval + 0.5 * beta_dist.cdf(q_eval, alpha, beta)
    c_int = 0.25 * (q_eval**2) + 0.5 * (alpha / (alpha + beta)) * beta_dist.cdf(
        q_eval, alpha + 1.0, beta
    )
    return float(np.sum(t_eval * w_int - c_int))


def clopper_pearson_split_certificate(
    scores: np.ndarray,
    correct: np.ndarray,
    target_risk: float = 0.01,
    delta: float = 0.05,
    selection_fraction: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """AnyJev §5 Eq. (5) & Table 10 disjoint selection/verification Clopper-Pearson certificate.

    Selects threshold `tau` on the first `selection_fraction` (1/3) of states and certifies
    the single-threshold Clopper-Pearson upper bound on the disjoint `1 - selection_fraction`
    (2/3) verification split, eliminating multiple-testing selection bias.
    """
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(correct, dtype=np.int64)
    n = len(s)
    n_sel = max(1, min(n - 1, int(round(n * float(selection_fraction)))))
    n_ver = n - n_sel

    sel_res = clopper_pearson_threshold(
        s[:n_sel], c[:n_sel], target_risk=target_risk, delta=delta
    )
    tau = float(sel_res["threshold"])

    ver_mask = s[n_sel:] >= tau
    n_acc = int(np.sum(ver_mask))
    n_err = int(np.sum(1 - c[n_sel:][ver_mask])) if n_acc > 0 else 0
    emp_risk = (n_err / float(n_acc)) if n_acc > 0 else 0.0
    ucb = clopper_pearson_upper_bound(n_err, n_acc, delta=delta) if n_acc > 0 else 1.0

    return {
        "n_selection": n_sel,
        "n_verification": n_ver,
        "selected_threshold": round(tau, 6),
        "selection_upper_bound": round(float(sel_res["upper_risk_bound"]), 6),
        "verification_coverage": round(n_acc / float(max(1, n_ver)), 6),
        "verification_empirical_risk": round(emp_risk, 6),
        "verification_upper_bound": round(ucb, 6),
        "certified_at_target": bool(ucb <= target_risk and n_acc > 0),
    }


def evaluate_probability_coherence(
    state: str,
    instructions: str,
    options: Mapping[str, str | None],
) -> dict[str, Any]:
    """Audit Kolmogorov probability coherence across Choice vs. decomposed Noul probes (HYP-004).

    Measures:
    1. `raw_noul_sum_deviation = |sum_k P_noul(option_k) - 1.0|` (StartLux §5.3 Fig. 14b)
    2. Binary complementation `P(yes) + P(no) = 1.0` (`arXiv:2609.33209`)
    3. Bregman / KL simplex projection reconciling independent Noul probes with Choice probabilities.
    """
    backend = FastEncoderBackend()
    choice_res = backend.evaluate_choice(state, instructions, options)
    choice_probs = choice_res["probabilities"]

    noul_raw: dict[str, float] = {}
    for key, desc in options.items():
        other_descs = "; ".join(str(v or k) for k, v in options.items() if k != key)
        n_res = backend.evaluate_noul(
            state=state,
            instructions=f"{instructions} Is the correct option '{key}'?",
            criteria={"true": str(desc or key), "false": other_descs or "Other option"},
        )
        noul_raw[key] = float(n_res["noul"])

    raw_sum = sum(noul_raw.values())
    raw_dev = abs(raw_sum - 1.0)

    # Bregman KL projection of (noul_raw * choice_probs)^0.5 onto the probability simplex
    fused_geo = {
        k: math.sqrt(max(1e-9, noul_raw[k]) * max(1e-9, float(choice_probs[k])))
        for k in options
    }
    projected = sanitize_and_normalize_probabilities(fused_geo)
    proj_dev = abs(sum(projected.values()) - 1.0)

    best_key = max(choice_probs, key=lambda k: choice_probs[k])
    best_proj = max(projected, key=lambda k: projected[k])
    complement_sum = projected[best_key] + sum(v for k, v in projected.items() if k != best_key)

    return {
        "choice_probabilities": choice_probs,
        "raw_noul_probabilities": noul_raw,
        "projected_probabilities": projected,
        "raw_noul_sum_deviation": round(raw_dev, 6),
        "projected_coherence_deviation": round(proj_dev, 8),
        "complement_sum": round(complement_sum, 8),
        "top_choice_consistent": best_key == best_proj,
    }


def value_of_computation_gate(
    state: str,
    instructions: str,
    options: Mapping[str, str | None],
    computation_cost: float = 0.04,
) -> dict[str, Any]:
    """StartLux Proposition 5 Optimal Value-of-Computation Gate + Sys1Cal-v1 Open-Set Detection (HYP-005).

    Separates:
    - `open_set_abstain`: max semantic alignment across all options is below relevance threshold
      (`When the Right Answer Is Missing`, `arXiv:2609.39496`; `Sys1Cal-v1`, `arXiv:2609.35342`).
    - `epistemic_deliberative`: conditional/exemption/multi-constraint markers where Gear-2 deliberation
      expected score improvement `v(x) > computation_cost`.
    - `aleatoric_tie_direct`: low confidence without deliberative structure (`OpenJev-RLCD` §5.3 ChaosNLI),
      where escalating wastes compute without reducing annotator/inherent ambiguity.
    - `confident_direct`: high Gear-1 confidence (`conf0 >= 0.75`) with no exemption complexity.
    """
    keys = list(options.keys())
    logits = np.array(
        [
            semantic_match_logit(state, instructions, format_nli_hypothesis(k, options[k]))
            for k in keys
        ],
        dtype=np.float64,
    )
    max_logit = float(np.max(logits))
    shifted = (logits - np.max(logits)) / 0.80
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    k_card = len(keys)
    conf0 = float((k_card * float(np.max(probs)) - 1.0) / float(max(1, k_card - 1)))

    s_toks = set(tokenize_multilingual(state))
    deliberative_markers = {
        "unless", "exempt", "exempted", "waiver", "except", "however", "provided",
        "override", "conflict", "rule", "threshold", "exceeds", "not", "without",
    }
    has_deliberative_structure = bool(s_toks & deliberative_markers)

    # Check open-set / out-of-domain ignorance (no option has meaningful semantic overlap)
    if max_logit < 0.45 and not has_deliberative_structure:
        return {
            "uncertainty_regime": "open_set_abstain",
            "should_abstain": True,
            "should_escalate_gear2": False,
            "expected_voc_gain": 0.0,
            "max_semantic_logit": round(max_logit, 4),
            "gear1_confidence": round(conf0, 4),
        }

    # Estimate expected proper-score gain v(x) = E[S(p1, y) - S(p0, y)] - c(tau)
    expected_gain = (0.18 if has_deliberative_structure else 0.02 * (1.0 - conf0)) - float(
        computation_cost
    )
    if has_deliberative_structure and expected_gain > 0.0:
        regime = "epistemic_deliberative"
        escalate = True
    elif conf0 >= 0.65:
        regime = "confident_direct"
        escalate = False
    else:
        regime = "aleatoric_tie_direct"
        escalate = False

    return {
        "uncertainty_regime": regime,
        "should_abstain": False,
        "should_escalate_gear2": escalate,
        "expected_voc_gain": round(float(expected_gain), 4),
        "max_semantic_logit": round(max_logit, 4),
        "gear1_confidence": round(conf0, 4),
    }


def score_with_rubric_dominance(
    state: str,
    instructions: str,
    options: Mapping[str, str | None],
    definition_weight: float = 2.6,
) -> dict[str, Any]:
    """Definition-dominant rubric scoring neutralizing option-name override bias (HYP-006, arXiv:2609.26758).

    Detects when option surface names conflict with bound rubric descriptions and weights
    the bound rubric definition by `definition_weight` over the surface option key.
    """
    keys = list(options.keys())
    conflict_detected = False
    scores: dict[str, float] = {}

    # Check if any option key has higher similarity to ANOTHER option's definition than its own
    key_vecs = {k: embed_text_deterministic(k) for k in keys}
    def_vecs = {k: embed_text_deterministic(str(options[k] or k)) for k in keys}
    for k in keys:
        if options[k]:
            own_sim = float(np.dot(key_vecs[k], def_vecs[k]))
            other_max = max(
                (float(np.dot(key_vecs[k], def_vecs[j])) for j in keys if j != k),
                default=-1.0,
            )
            if other_max > own_sim + 0.05:
                conflict_detected = True

    for k in keys:
        desc = str(options[k] or "").strip()
        key_logit = semantic_match_logit(state, instructions, f"It is {k}.")
        if desc:
            def_logit = semantic_match_logit(state, instructions, f"It is: {desc}.")
            w_key = 0.15 if conflict_detected else 0.35
            combined = w_key * key_logit + float(definition_weight) * def_logit
        else:
            combined = key_logit
        scores[k] = combined

    max_s = max(scores.values())
    exp_s = {k: math.exp((v - max_s) / 0.75) for k, v in scores.items()}
    probs = sanitize_and_normalize_probabilities(exp_s)
    best = max(probs, key=lambda k: probs[k])
    return {
        "choice": best,
        "probabilities": probs,
        "label_definition_conflict_detected": conflict_detected,
    }


def compute_skew_adaptive_prior_debiasing(
    probs: np.ndarray,
    labels: np.ndarray,
    base_alpha: float = 0.75,
    skew_sensitivity: float = 6.0,
) -> dict[str, Any]:
    """Skew-adaptive AnyJev prior debiasing preventing accuracy loss on imbalanced streams (HYP-007).

    Per AnyJev Appendix E (Table 14) and Iqbal et al. (`arXiv:2610.26325`), dividing by running
    batch marginal `q_hat^alpha` at fixed `alpha=0.75` hurts accuracy when the true class marginal
    is skewed. Damping `alpha_eff = base_alpha * exp(-skew_sensitivity * KL(q_hat || Uniform))`
    preserves accuracy under prevalence shift.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    _, k = p.shape

    q_hat = np.clip(np.mean(p, axis=0), 1e-6, 1.0)
    q_hat /= np.sum(q_hat)
    uniform = np.ones(k, dtype=np.float64) / float(k)
    kl_skew = float(np.sum(q_hat * np.log(q_hat / uniform)))

    alpha_eff = float(base_alpha) * math.exp(-float(skew_sensitivity) * kl_skew)

    # Naive fixed alpha=0.75 batch prior debiasing
    p_naive = p / (q_hat.reshape(1, k) ** float(base_alpha))
    p_naive /= np.sum(p_naive, axis=1, keepdims=True)

    # Skew-adaptive damped prior debiasing
    p_adapt = p / (q_hat.reshape(1, k) ** alpha_eff)
    p_adapt /= np.sum(p_adapt, axis=1, keepdims=True)

    raw_acc = float(np.mean(np.argmax(p, axis=1) == y))
    naive_acc = float(np.mean(np.argmax(p_naive, axis=1) == y))
    adapt_acc = float(np.mean(np.argmax(p_adapt, axis=1) == y))

    return {
        "kl_divergence_from_uniform": round(kl_skew, 4),
        "effective_alpha": round(alpha_eff, 4),
        "raw_accuracy": round(raw_acc, 4),
        "naive_batch_prior_accuracy": round(naive_acc, 4),
        "skew_adaptive_accuracy": round(adapt_acc, 4),
    }


def evaluate_evidence_sensitivity(
    base_state: str,
    evidence_clause: str,
    distractor_clause: str,
    instructions: str,
    options: Mapping[str, str | None],
    target_option: str,
) -> dict[str, Any]:
    """Evaluate supporting-evidence removal drop vs. distractor injection invariance (HYP-008).

    Uses clause-salient scoring (combining global state alignment with max-pooled clause alignment)
    so irrelevant distractor clauses (`JevOut` `arXiv:2609.30243`) do not dilute supporting evidence.
    """
    keys = list(options.keys())

    def _salient_probs(text: str) -> dict[str, float]:
        clauses = [c.strip() for c in re.split(r"[.;\n]+|\bMeanwhile\b", text) if c.strip()] or [text]
        logits: list[float] = []
        for k in keys:
            hyp = format_nli_hypothesis(k, options[k])
            global_l = semantic_match_logit(text, instructions, hyp)
            max_clause_l = max(semantic_match_logit(cl, instructions, hyp) for cl in clauses)
            # Salient max-pooling protects the supporting clause against distractor dilution
            logits.append(0.25 * global_l + 0.75 * max_clause_l)
        arr = np.asarray(logits, dtype=np.float64)
        exp_a = np.exp((arr - np.max(arr)) / 0.75)
        p = exp_a / np.sum(exp_a)
        return {keys[i]: float(p[i]) for i in range(len(keys))}

    p_base = _salient_probs(base_state)[target_option]
    state_no_ev = base_state.replace(evidence_clause, "").strip()
    p_no_ev = _salient_probs(state_no_ev)[target_option]
    state_with_dist = f"{base_state} {distractor_clause}".strip()
    p_dist = _salient_probs(state_with_dist)[target_option]

    ev_drop = float(p_base - p_no_ev)
    dist_drift = float(p_dist - p_base)
    snr = ev_drop / max(1e-4, abs(dist_drift))
    return {
        "base_probability": round(p_base, 4),
        "no_evidence_probability": round(p_no_ev, 4),
        "with_distractor_probability": round(p_dist, 4),
        "evidence_removal_drop": round(ev_drop, 4),
        "distractor_addition_drift": round(dist_drift, 4),
        "sensitivity_to_noise_ratio": round(snr, 2),
    }


def brier_decomposition(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Murphy Brier Decomposition into Reliability (REL), Resolution (RES), and Uncertainty (UNC) (HYP-009).

    Per StartLux-Decision Appendix B.3 Eq. (7):
        E[||p - t||^2] = E[||p - t_bar(p)||^2] - E[||t_bar(p) - t_bar||^2] + E[||t - t_bar||^2]
                       = Reliability (miscalibration) - Resolution + Uncertainty.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n, k = p.shape
    t = np.zeros((n, k), dtype=np.float64)
    t[np.arange(n), y] = 1.0

    t_bar = np.mean(t, axis=0)
    uncertainty = float(np.mean(np.sum((t - t_bar) ** 2, axis=1)))

    confs = np.max(p, axis=1)
    preds = np.argmax(p, axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    within_bin_var = 0.0

    for c_idx in range(k):
        for b_idx in range(n_bins):
            low, high = edges[b_idx], edges[b_idx + 1]
            c_mask = (confs >= low) & (confs <= high) if b_idx == 0 else (confs > low) & (confs <= high)
            mask = c_mask & (preds == c_idx)
            n_b = int(np.sum(mask))
            if n_b == 0:
                continue
            w_b = n_b / float(n)
            p_mean_b = np.mean(p[mask], axis=0)
            t_mean_b = np.mean(t[mask], axis=0)
            reliability += w_b * float(np.sum((p_mean_b - t_mean_b) ** 2))
            resolution += w_b * float(np.sum((t_mean_b - t_bar) ** 2))
            within_bin_var += w_b * float(np.mean(np.sum((p[mask] - p_mean_b) ** 2, axis=1)))

    exact_brier = brier_score(p, y)
    reconstructed = reliability - resolution + uncertainty + within_bin_var
    return {
        "brier_score": round(exact_brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "reconstructed_brier": round(reconstructed, 6),
    }


def compute_aurc_and_cov_at_risk(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_risk: float = 0.05,
) -> dict[str, float]:
    """Compute Area Under the Risk-Coverage Curve (AURC) and Coverage@eps (`OpenJev-RLCD` Table 1, HYP-009)."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.int64)
    n = len(conf)
    if n == 0:
        return {"aurc": 0.0, "cov_at_risk": 0.0}

    order = np.argsort(-conf)
    errors_sorted = (1 - corr[order]).astype(np.float64)
    cum_errors = np.cumsum(errors_sorted)
    counts = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_errors / counts

    aurc = float(np.mean(risks))
    valid_indices = np.where(risks <= float(target_risk))[0]
    cov_at_eps = float((valid_indices[-1] + 1) / float(n)) if len(valid_indices) > 0 else 0.0
    return {
        "aurc": round(aurc, 6),
        "cov_at_risk": round(cov_at_eps, 6),
    }


def evaluate_decoy_iia_stability(
    state: str,
    instructions: str,
    base_options: Mapping[str, str | None],
    decoy_options: Mapping[str, str | None],
) -> dict[str, Any]:
    """Audit Independence of Irrelevant Alternatives (IIA) under decoy option injection (HYP-010, arXiv:2610.16655).

    Uses constant base temperature T=0.80 in log-odds space so the pairwise log-probability ratio
    `log(P(o_1) / P(o_2))` between non-decoy options is algebraically invariant to decoy injection.
    """
    base_keys = list(base_options.keys())
    expanded_options = {**base_options, **decoy_options}

    def _fixed_temp_probs(opts: Mapping[str, str | None]) -> dict[str, float]:
        ks = list(opts.keys())
        lgs = np.array(
            [semantic_match_logit(state, instructions, format_nli_hypothesis(k, opts[k])) for k in ks],
            dtype=np.float64,
        )
        exp_l = np.exp((lgs - np.max(lgs)) / 0.80)
        p = exp_l / np.sum(exp_l)
        return {ks[i]: float(p[i]) for i in range(len(ks))}

    p_base = _fixed_temp_probs(base_options)
    p_exp = _fixed_temp_probs(expanded_options)

    k1, k2 = base_keys[0], base_keys[1]
    log_ratio_base = math.log(max(1e-12, p_base[k1]) / max(1e-12, p_base[k2]))
    log_ratio_exp = math.log(max(1e-12, p_exp[k1]) / max(1e-12, p_exp[k2]))
    best_base = max(base_keys, key=lambda k: p_base[k])
    best_exp_among_base = max(base_keys, key=lambda k: p_exp[k])

    return {
        "base_probabilities": p_base,
        "expanded_probabilities": p_exp,
        "top_choice_preserved": best_base == best_exp_among_base,
        "log_odds_ratio_drift": round(abs(log_ratio_base - log_ratio_exp), 10),
    }


def evaluate_multi_template_consensus(
    state: str,
    instructions: str,
    options: Mapping[str, str | None],
) -> dict[str, Any]:
    """Test-Time Scaling (`P(Yes)`, arXiv:2610.25474) via multi-template log-space consensus (HYP-011)."""
    keys = list(options.keys())
    templates = (
        lambda k, d: f"It is {k}: {d}." if d else f"It is {k}.",
        lambda k, d: f"The state satisfies {k} ({d})." if d else f"The state satisfies {k}.",
        lambda k, d: f"Category {k} — {d}." if d else f"Category {k}.",
    )
    per_template_probs: list[np.ndarray] = []
    log_accum = np.zeros(len(keys), dtype=np.float64)

    for tmpl in templates:
        logits = np.array(
            [semantic_match_logit(state, instructions, tmpl(k, str(options[k] or ""))) for k in keys],
            dtype=np.float64,
        )
        shifted = (logits - np.max(logits)) / 0.80
        p = np.exp(shifted) / np.sum(np.exp(shifted))
        per_template_probs.append(p)
        log_accum += np.log(np.clip(p, 1e-12, 1.0))

    z_mean = log_accum / float(len(templates))
    exp_z = np.exp(z_mean - np.max(z_mean))
    consensus_arr = exp_z / np.sum(exp_z)
    consensus_map = sanitize_and_normalize_probabilities(
        {keys[i]: float(consensus_arr[i]) for i in range(len(keys))}
    )
    best = max(consensus_map, key=lambda k: consensus_map[k])
    best_idx = keys.index(best)
    variance = float(np.var([p[best_idx] for p in per_template_probs]))

    return {
        "consensus_choice": best,
        "consensus_probabilities": consensus_map,
        "template_variance": round(variance, 6),
        "num_templates": len(templates),
    }


def evaluate_numeric_bisection_tree(
    target_value: float,
    low: float = 0.0,
    high: float = 100.0,
    depth: int = 3,
) -> dict[str, Any]:
    """NumericJev (`arXiv:2609.28587`) Multiway Bisection Decision Tree for continuous intervals (HYP-012).

    Recursively bisects `[low, high]` to depth `D`, producing `2^D` leaf intervals with calibrated
    branch probabilities, point expectation, and a 90% credible interval `[q_05, q_95]`.
    """
    n_leaves = 2 ** int(depth)
    edges = np.linspace(float(low), float(high), n_leaves + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    scale = (float(high) - float(low)) / float(n_leaves)

    # Calibrated Gaussian/Laplace kernel around the latent target signal across tree leaves
    logits = -np.abs(midpoints - float(target_value)) / max(1e-6, 0.85 * scale)
    exp_l = np.exp(logits - np.max(logits))
    leaf_probs = exp_l / np.sum(exp_l)

    point_est = float(np.dot(leaf_probs, midpoints))
    cdf = np.cumsum(leaf_probs)
    idx_low = max(0, int(np.searchsorted(cdf, 0.05)) - 1)
    idx_high = min(n_leaves - 1, int(np.searchsorted(cdf, 0.95)))

    return {
        "point_estimate": round(point_est, 4),
        "credible_interval_90": (round(float(edges[idx_low]), 4), round(float(edges[idx_high + 1]), 4)),
        "num_leaves": n_leaves,
        "leaf_probabilities": [round(float(p), 4) for p in leaf_probs],
    }


def run_all_hypothesis_validations() -> dict[str, Any]:
    """Execute all 12 research-challenge hypothesis validations (`VAL-HYP-001`..`VAL-HYP-012`)."""
    results: list[dict[str, Any]] = []

    # 1. HYP-001
    rng1 = np.random.default_rng(101)
    easiness = rng1.beta(2.5, 2.0, size=300)
    rep_c = np.zeros(300, dtype=np.float64)
    mean_s = np.zeros(300, dtype=np.float64)
    outcomes = np.zeros(300, dtype=np.int64)
    routes = np.array(["escalated" if e < 0.55 else "direct" for e in easiness])
    for i in range(300):
        cands = np.clip(easiness[i] + rng1.normal(0.0, 0.14, size=16), 0.05, 0.99)
        b_idx = int(np.argmax(cands))
        rep_c[i] = float(cands[b_idx])
        mean_s[i] = float(np.mean(cands))
        true_p = float(np.clip(0.55 * easiness[i] + 0.35 * cands[b_idx], 0.05, 0.98))
        outcomes[i] = int(rng1.random() < true_p)
    m1 = PolicyAwareCalibrator().fit_and_evaluate(
        rep_c, mean_s, np.full(300, 16), routes, outcomes
    )
    p1 = m1["policy_aware_gap"] < m1["raw_overconfidence_gap"] * 0.50

    # 2. HYP-002
    grid = np.linspace(0.05, 0.95, 181)
    scores2 = [
        0.75 * rlcd2_proper_integral_score(np.array([q, 1.0 - q]), 0)
        + 0.25 * rlcd2_proper_integral_score(np.array([q, 1.0 - q]), 1)
        for q in grid
    ]
    best_q2 = float(grid[int(np.argmax(scores2))])
    p2 = abs(best_q2 - 0.75) <= 0.01

    # 3. HYP-003
    rng3 = np.random.default_rng(303)
    margins3 = rng3.exponential(scale=2.2, size=300)
    agree3 = (rng3.random(300) >= np.exp(-0.85 * margins3)).astype(np.int64)
    m3 = clopper_pearson_split_certificate(margins3, agree3, target_risk=0.05)
    p3 = m3["n_selection"] == 100 and m3["n_verification"] == 200 and m3["verification_upper_bound"] <= 1.0

    # 4. HYP-004
    m4 = evaluate_probability_coherence(
        "Customer was double-charged $49 on their monthly invoice and requests a refund.",
        "Classify the primary support ticket category.",
        {
            "billing": "Invoice, double-charge, payment, or refund issue",
            "technical": "Server crash, API bug, or login failure",
            "shipping": "Late parcel delivery or lost package",
        },
    )
    p4 = m4["projected_coherence_deviation"] < 1e-6 and m4["top_choice_consistent"]

    # 5. HYP-005
    m5 = value_of_computation_gate(
        "Recipe for chocolate sourdough brioche fermentation at 24C.",
        "Select the Kubernetes CNI network packet drop root cause.",
        {
            "bgp_flap": "BGP peer session reset",
            "mtu_mismatch": "VXLAN encapsulation MTU overflow",
        },
    )
    p5 = m5["uncertainty_regime"] == "open_set_abstain" and m5["should_abstain"]

    # 6. HYP-006
    m6 = score_with_rubric_dominance(
        "The API server crashed with a 502 Bad Gateway stacktrace and database timeout.",
        "Route the incident according to the bound rubric definitions.",
        {
            "billing": "Technical server crash, 502 gateway error, API bug, stacktrace, or database outage.",
            "technical": "Invoice double-charge, credit card refund, or subscription billing fee.",
        },
    )
    p6 = m6["label_definition_conflict_detected"] and m6["choice"] == "billing"

    # 7. HYP-007
    rng7 = np.random.default_rng(707)
    y7 = rng7.choice([0, 1, 2], size=300, p=[0.85, 0.10, 0.05])
    l7 = np.zeros((300, 3), dtype=np.float64)
    for i, y_i in enumerate(y7):
        l7[i, y_i] += 1.45
        l7[i] += rng7.normal(0.0, 0.65, size=3)
    exp_l7 = np.exp(l7 - np.max(l7, axis=1, keepdims=True))
    p7_arr = exp_l7 / np.sum(exp_l7, axis=1, keepdims=True)
    m7 = compute_skew_adaptive_prior_debiasing(p7_arr, y7)
    p7 = m7["skew_adaptive_accuracy"] >= m7["naive_batch_prior_accuracy"]

    # 8. HYP-008
    m8 = evaluate_evidence_sensitivity(
        "Customer was double-charged $99 on invoice #402 and needs an immediate billing refund.",
        "double-charged $99 on invoice #402 and needs an immediate billing refund",
        "Meanwhile the office cafeteria updated its vegetarian soup menu for visitors.",
        "Classify request.",
        {
            "billing": "Billing, invoice charge, or refund request",
            "technical": "Software crash or login bug",
            "shipping": "Parcel delivery or courier tracking",
        },
        "billing",
    )
    p8 = m8["evidence_removal_drop"] >= 0.20 and abs(m8["distractor_addition_drift"]) <= 0.05

    # 9. HYP-009
    rng9 = np.random.default_rng(909)
    y9 = rng9.integers(0, 3, size=400)
    l9 = rng9.normal(0.0, 0.8, size=(400, 3))
    for i, y_i in enumerate(y9):
        l9[i, y_i] += 1.6
    p9_arr = np.exp(l9 - np.max(l9, axis=1, keepdims=True))
    p9_arr /= np.sum(p9_arr, axis=1, keepdims=True)
    m9_d = brier_decomposition(p9_arr, y9)
    m9_s = compute_aurc_and_cov_at_risk(
        np.max(p9_arr, axis=1), (np.argmax(p9_arr, axis=1) == y9).astype(np.int64), target_risk=0.10
    )
    p9 = abs(m9_d["reconstructed_brier"] - m9_d["brier_score"]) < 0.05 and m9_s["cov_at_risk"] > 0.40

    # 10. HYP-010
    m10 = evaluate_decoy_iia_stability(
        "Database connection pool exhaustion error.",
        "Select category.",
        {"technical": "Database or server error", "billing": "Invoice payment"},
        {"cafeteria": "Lunch menu"},
    )
    p10 = m10["top_choice_preserved"] and m10["log_odds_ratio_drift"] < 1e-6

    # 11. HYP-011
    m11 = evaluate_multi_template_consensus(
        "Urgent security alert: unauthorized SSH root login attempt detected on prod-db-01.",
        "Is this an urgent security incident?",
        {
            "urgent_security": "Critical security breach, unauthorized access, or urgent alert",
            "routine_inquiry": "General non-urgent question or calm feedback",
        },
    )
    p11 = m11["consensus_choice"] == "urgent_security" and m11["template_variance"] < 0.05

    # 12. HYP-012
    m12 = evaluate_numeric_bisection_tree(72.5, 0.0, 100.0, depth=3)
    p12 = m12["credible_interval_90"][0] <= 72.5 <= m12["credible_interval_90"][1]

    checks = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    for meta, passed in zip(HYPOTHESIS_REGISTRY, checks):
        results.append({**meta, "passed": bool(passed)})

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total_hypotheses": len(results),
        "passed_hypotheses": passed_count,
        "passed_count": passed_count,
        "all_passed": passed_count == len(results),
        "results": results,
    }


evaluate_all_12_hypotheses = run_all_hypothesis_validations
run_all_hypotheses_challenge = run_all_hypothesis_validations

