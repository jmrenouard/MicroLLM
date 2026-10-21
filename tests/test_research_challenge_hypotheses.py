"""TDD validation suite for the Research Paper Code & Design Challenge and RCA Hypotheses (Issue #61).

Verifies:
1. `docs/RCA_REGISTER.md` contains the complete Research Paper Design & Code Challenge (`RCA-016`,
   `IMP-006`..`IMP-008`, `APP-008`..`APP-010`) and Section 6 documenting all 12 Improvement Hypotheses
   (`HYP-001` through `HYP-012`), their Validation Test Protocols (`VAL-HYP-001`..`VAL-HYP-012`), and
   their Target Application Domains.
2. Empirical and mathematical validation of all 12 Improvement Hypotheses (`HYP-001`..`HYP-012`)
   against `src/microllm/research_challenge.py`:
   - `HYP-001`: Policy-Aware & Route-Conditioned Calibration (`StartLux-Decision` Propositions 1–4)
   - `HYP-002`: Exact Integral Decision-Weighted Proper Scoring Rule S_D(p, t) (`StartLux-Decision` Eq. 3–5)
   - `HYP-003`: Disjoint Selection/Verification Split Clopper-Pearson Certificate (`AnyJev` §5 Eq. 5)
   - `HYP-004`: Kolmogorov Probability Axiom & Partition-Invariance Coherence (`arXiv:2609.33209`, `2609.33971`)
   - `HYP-005`: Epistemic vs. Aleatoric Uncertainty & Value-of-Computation Gate (`OpenJev-RLCD` §5.3, `Sys1Cal-v1`)
   - `HYP-006`: Definition-Dominant Rubric Binding vs. Adversarial Option Renaming (`arXiv:2609.26758`, `2610.02586`)
   - `HYP-007`: Content-Free Prior Separation & Skew-Adaptive Prior Damping (`AnyJev` Appendix E, `2610.26325`)
   - `HYP-008`: Supporting-Evidence Sensitivity vs. Distractor Invariance (`StartLux` §5.3, `JevOut` `2609.30243`)
   - `HYP-009`: Murphy Brier Decomposition (REL/RES/UNC) & Selective Delegation (`AURC`, `Cov@5%`)
   - `HYP-010`: Independence of Irrelevant Alternatives (IIA) & Decoy Stability (`arXiv:2610.16655`)
   - `HYP-011`: Test-Time Scaling (`P(Yes)`) Multi-Template Log-Space Consensus (`arXiv:2610.25474`)
   - `HYP-012`: NumericJev Multiway Bisection Decision Trees for Continuous Intervals (`arXiv:2609.28587`)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from microllm.research_challenge import (
    HYPOTHESIS_REGISTRY,
    PolicyAwareCalibrator,
    brier_decomposition,
    clopper_pearson_split_certificate,
    compute_aurc_and_cov_at_risk,
    compute_skew_adaptive_prior_debiasing,
    evaluate_decoy_iia_stability,
    evaluate_evidence_sensitivity,
    evaluate_multi_template_consensus,
    evaluate_numeric_bisection_tree,
    evaluate_probability_coherence,
    rlcd2_proper_integral_score,
    run_all_hypothesis_validations,
    score_with_rubric_dominance,
    value_of_computation_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rca_register_enriched_with_research_challenge_hypotheses_and_domains() -> None:
    """Verify `docs/RCA_REGISTER.md` contains all 12 hypotheses, test protocols, and application domains."""
    rca_path = REPO_ROOT / "docs" / "RCA_REGISTER.md"
    assert rca_path.is_file(), "Missing docs/RCA_REGISTER.md"
    content = rca_path.read_text(encoding="utf-8")

    # New RCA, IMP, and APP entries from challenging code against recent papers
    for entry_id in (
        "RCA-016",
        "IMP-006",
        "IMP-007",
        "IMP-008",
        "APP-008",
        "APP-009",
        "APP-010",
    ):
        assert entry_id in content, f"Missing {entry_id} in docs/RCA_REGISTER.md"

    # All 12 Improvement Hypotheses and their Validation Test Protocols
    for idx in range(1, 13):
        hyp_id = f"HYP-{idx:03d}"
        val_id = f"VAL-HYP-{idx:03d}"
        assert hyp_id in content, f"Missing hypothesis {hyp_id} in docs/RCA_REGISTER.md"
        assert val_id in content, f"Missing validation test protocol {val_id} in docs/RCA_REGISTER.md"

    # Required structural sections for the research challenge
    for phrase in (
        "Research Paper Code & Design Challenge",
        "Improvement Hypotheses",
        "Validation Test Protocol",
        "Application Domains",
        "Policy-Aware Calibration",
        "Kolmogorov Probability Axiom",
        "Clopper-Pearson",
        "Value-of-Computation Gate",
        "NumericJev",
    ):
        assert phrase in content, f"Missing '{phrase}' in docs/RCA_REGISTER.md"


def test_hyp001_policy_aware_and_route_conditioned_calibration() -> None:
    """Validate HYP-001: Policy-aware calibration mitigates Best-of-N & route-conditional miscalibration."""
    rng = np.random.default_rng(101)
    # Simulate 300 Best-of-N candidate selection problems (StartLux Proposition 1)
    n_problems = 300
    n_candidates = 16
    easiness = rng.beta(2.5, 2.0, size=n_problems)

    reported_confs = np.zeros(n_problems, dtype=np.float64)
    mean_scores = np.zeros(n_problems, dtype=np.float64)
    outcomes = np.zeros(n_problems, dtype=np.int64)
    routes = np.array(["escalated" if e < 0.55 else "direct" for e in easiness])

    for i in range(n_problems):
        cand_scores = np.clip(easiness[i] + rng.normal(0.0, 0.14, size=n_candidates), 0.05, 0.99)
        best_idx = int(np.argmax(cand_scores))
        reported_confs[i] = float(cand_scores[best_idx])
        mean_scores[i] = float(np.mean(cand_scores))
        # True correctness depends on latent easiness, not upward noise spike (Optimizer's Curse)
        true_p = float(np.clip(0.55 * easiness[i] + 0.35 * cand_scores[best_idx], 0.05, 0.98))
        outcomes[i] = int(rng.random() < true_p)

    cal = PolicyAwareCalibrator()
    metrics = cal.fit_and_evaluate(
        reported_confidences=reported_confs,
        mean_candidate_scores=mean_scores,
        candidate_counts=np.full(n_problems, n_candidates, dtype=np.int64),
        routes=routes,
        correct=outcomes,
        target_risk=0.15,
    )
    assert metrics["raw_overconfidence_gap"] > 0.08
    assert metrics["policy_aware_gap"] < metrics["raw_overconfidence_gap"] * 0.50
    assert metrics["route_certified"]["direct"]["threshold"] >= 0.50
    assert metrics["route_certified"]["escalated"]["threshold"] >= 0.50


def test_hyp002_exact_integral_decision_weighted_proper_scoring_rule() -> None:
    """Validate HYP-002: StartLux Eq. (3)-(5) integral score S_D(p, t) is strictly proper at p = pi."""
    for true_pi in (0.15, 0.35, 0.50, 0.75, 0.90):
        grid = np.linspace(0.01, 0.99, 991)
        expected_scores = [
            true_pi * rlcd2_proper_integral_score(np.array([q, 1.0 - q]), np.array([1.0, 0.0]))
            + (1.0 - true_pi)
            * rlcd2_proper_integral_score(np.array([q, 1.0 - q]), np.array([0.0, 1.0]))
            for q in grid
        ]
        best_q = float(grid[int(np.argmax(expected_scores))])
        assert abs(best_q - true_pi) <= 0.005, f"Improper maximizer {best_q} != {true_pi}"

    # Ordinal cumulative scoring check: closer ordinal prediction must score strictly higher
    target_level = 3
    one_hot = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    p_near = np.array([0.02, 0.08, 0.30, 0.50, 0.10])
    p_far = np.array([0.30, 0.08, 0.02, 0.50, 0.10])
    s_near = rlcd2_proper_integral_score(p_near, one_hot, cumulative_ordinal=True)
    s_far = rlcd2_proper_integral_score(p_far, one_hot, cumulative_ordinal=True)
    assert s_near > s_far


def test_hyp003_disjoint_clopper_pearson_split_certificate() -> None:
    """Validate HYP-003: Disjoint (1/3 select, 2/3 verify) certificate eliminates multiplicity bias."""
    rng = np.random.default_rng(303)
    n = 600
    margins = rng.exponential(scale=2.2, size=n)
    # Disagreement occurs with higher probability when margin is small
    disagreements = (rng.random(n) < np.exp(-0.85 * margins)).astype(np.int64)
    agreements = 1 - disagreements

    cert = clopper_pearson_split_certificate(
        scores=margins,
        correct=agreements,
        target_risk=0.05,
        delta=0.05,
        selection_fraction=1.0 / 3.0,
    )
    assert cert["n_selection"] == 200
    assert cert["n_verification"] == 400
    assert cert["selected_threshold"] > 0.0
    assert 0.0 <= cert["verification_empirical_risk"] <= cert["verification_upper_bound"] <= 1.0


def test_hyp004_kolmogorov_probability_coherence_and_projection() -> None:
    """Validate HYP-004: Cross-formulation coherence audit and Bregman simplex projection."""
    res = evaluate_probability_coherence(
        state="Customer was double-charged $49 on their monthly invoice and requests a refund.",
        instructions="Classify the primary support ticket category.",
        options={
            "billing": "Invoice, double-charge, payment, or refund issue",
            "technical": "Server crash, API bug, or login failure",
            "shipping": "Late parcel delivery or lost package",
        },
    )
    assert res["raw_noul_sum_deviation"] >= 0.0
    assert res["projected_coherence_deviation"] < 1e-6
    assert abs(res["complement_sum"] - 1.0) < 1e-6
    assert res["top_choice_consistent"] is True


def test_hyp005_epistemic_vs_aleatoric_value_of_computation_gate() -> None:
    """Validate HYP-005: Value-of-Computation gate separates epistemic, aleatoric, and open-set states."""
    # 1. Epistemic policy complexity (exemption clause requiring deliberation)
    epistemic = value_of_computation_gate(
        state="Deployment lacks secondary sign-off unless exempted by a signed emergency waiver.",
        instructions="Is this deployment policy-compliant?",
        options={"compliant": "Meets policy or valid waiver", "violation": "Prohibited or missing approval"},
    )
    # 2. Open-set / out-of-domain uninformative state (Sys1Cal-v1 missing right answer)
    open_set = value_of_computation_gate(
        state=" Recipe for chocolate sourdough brioche fermentation at 24C.",
        instructions="Select the Kubernetes CNI network packet drop root cause.",
        options={"bgp_flap": "BGP peer session reset", "mtu_mismatch": "VXLAN encapsulation MTU overflow"},
    )
    assert epistemic["uncertainty_regime"] in ("epistemic_deliberative", "confident_direct")
    assert open_set["uncertainty_regime"] == "open_set_abstain"
    assert open_set["should_abstain"] is True


def test_hyp006_definition_dominant_rubric_binding_under_swapped_labels() -> None:
    """Validate HYP-006: Definition-dominant scoring follows bound definitions when option keys conflict."""
    state = "The API server crashed with a 502 Bad Gateway stacktrace and database timeout."
    instructions = "Route the incident according to the bound rubric definitions."
    # Adversarial swapped keys: key 'billing' is bound to the technical crash definition!
    swapped_options = {
        "billing": "Technical server crash, 502 gateway error, API bug, stacktrace, or database outage.",
        "technical": "Invoice double-charge, credit card refund, or subscription billing fee.",
    }
    res = score_with_rubric_dominance(state, instructions, swapped_options)
    assert res["label_definition_conflict_detected"] is True
    assert res["choice"] == "billing"
    assert res["probabilities"]["billing"] > 0.60


def test_hyp007_skew_adaptive_and_content_free_prior_debiasing() -> None:
    """Validate HYP-007: Skew-adaptive prior damping avoids accuracy loss on imbalanced streams."""
    rng = np.random.default_rng(707)
    # 85% class 0, 10% class 1, 5% class 2 prevalence
    n = 300
    labels = rng.choice([0, 1, 2], size=n, p=[0.85, 0.10, 0.05])
    logits = np.zeros((n, 3), dtype=np.float64)
    for i, y in enumerate(labels):
        logits[i, y] += 1.45
        logits[i] += rng.normal(0.0, 0.65, size=3)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    res = compute_skew_adaptive_prior_debiasing(probs, labels, base_alpha=0.75)
    # Naive batch prior at alpha=0.75 hurts accuracy under 85/10/5 skew; skew-adaptive preserves it
    assert res["skew_adaptive_accuracy"] >= res["naive_batch_prior_accuracy"]
    assert res["effective_alpha"] < 0.40


def test_hyp008_evidence_sensitivity_vs_distractor_invariance() -> None:
    """Validate HYP-008: Salient scoring responds strongly to evidence removal and ignores distractors."""
    res = evaluate_evidence_sensitivity(
        base_state="Customer was double-charged $99 on invoice #402 and needs an immediate billing refund.",
        evidence_clause="double-charged $99 on invoice #402 and needs an immediate billing refund",
        distractor_clause=(
            "Meanwhile the office cafeteria updated its Tuesday vegetarian menu with roasted pumpkin soup "
            "and artisanal sourdough bread for all visitors in building B."
        ),
        instructions="Classify the customer request.",
        options={
            "billing": "Billing, invoice charge, or refund request",
            "technical": "Software crash or login bug",
            "shipping": "Parcel delivery or courier tracking",
        },
        target_option="billing",
    )
    assert res["evidence_removal_drop"] >= 0.20
    assert abs(res["distractor_addition_drift"]) <= 0.05
    assert res["sensitivity_to_noise_ratio"] >= 4.0


def test_hyp009_murphy_brier_decomposition_and_selective_aurc() -> None:
    """Validate HYP-009: Exact Murphy Brier decomposition (REL - RES + UNC) and AURC / Cov@5%."""
    rng = np.random.default_rng(909)
    n = 400
    labels = rng.integers(0, 3, size=n)
    logits = rng.normal(0.0, 0.8, size=(n, 3))
    for i, y in enumerate(labels):
        logits[i, y] += 1.6
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    decomp = brier_decomposition(probs, labels, n_bins=10)
    # Binned Murphy identity: reliability - resolution + uncertainty ~= brier_score
    assert abs(decomp["reconstructed_brier"] - decomp["brier_score"]) < 0.05
    assert decomp["resolution"] > decomp["reliability"]

    confs = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(np.int64)
    sel = compute_aurc_and_cov_at_risk(confs, correct, target_risk=0.10)
    assert 0.0 <= sel["aurc"] <= 0.50
    assert sel["cov_at_risk"] > 0.40


def test_hyp010_iia_decoy_stability_and_hyp011_multi_template_and_hyp012_numeric_tree() -> None:
    """Validate HYP-010 (IIA decoy stability), HYP-011 (P(Yes) scaling), and HYP-012 (NumericJev tree)."""
    # HYP-010: IIA decoy stability
    iia = evaluate_decoy_iia_stability(
        state="Production PostgreSQL database is throwing connection pool exhaustion errors.",
        instructions="Select the incident category.",
        base_options={
            "technical": "Database, server, API, or software infrastructure error",
            "billing": "Invoice, charge, or subscription payment issue",
        },
        decoy_options={
            "cafeteria": "Lunch menu and coffee machine maintenance",
            "parking": "Visitor parking badge validation",
        },
    )
    assert iia["top_choice_preserved"] is True
    assert iia["log_odds_ratio_drift"] < 1e-6

    # HYP-011: Multi-template P(Yes) test-time scaling
    tts = evaluate_multi_template_consensus(
        state="Urgent security alert: unauthorized SSH root login attempt detected on prod-db-01.",
        instructions="Is this an urgent security incident?",
        options={
            "urgent_security": "Critical security breach, unauthorized access, or urgent alert",
            "routine_inquiry": "General non-urgent question or calm feedback",
        },
    )
    assert tts["consensus_choice"] == "urgent_security"
    assert tts["template_variance"] < 0.05

    # HYP-012: NumericJev multiway bisection tree
    num_res = evaluate_numeric_bisection_tree(
        target_value=72.5,
        low=0.0,
        high=100.0,
        depth=3,
    )
    assert abs(num_res["point_estimate"] - 72.5) <= 12.5
    assert num_res["credible_interval_90"][0] <= 72.5 <= num_res["credible_interval_90"][1]


def test_run_all_12_hypothesis_validations_master_suite() -> None:
    """Execute the full 12-hypothesis validation suite and assert 100% pass rate."""
    assert len(HYPOTHESIS_REGISTRY) == 12
    summary = run_all_hypothesis_validations()
    assert summary["total_hypotheses"] == 12
    assert summary["passed_hypotheses"] == 12
    assert summary["all_passed"] is True
    for item in summary["results"]:
        assert item["passed"] is True, f"Hypothesis {item['id']} failed: {item}"
        assert len(item["application_domains"]) >= 2
