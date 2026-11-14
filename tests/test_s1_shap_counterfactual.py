"""TDD tests for logit-space S1-SHAP counterfactual attribution (arXiv:2611.12521, Issue #249)."""

from __future__ import annotations

import pytest

from microllm.backends.base import explain_counterfactual


def test_explain_counterfactual_exposes_log_odds_and_logit_drop() -> None:
    """Verify legacy keys plus pre-softmax original_log_odds and per-token logit_drop."""
    cf = explain_counterfactual(
        state_text="Customer was double charged twice on credit card invoice payment.",
        instructions_text="Route support ticket to the appropriate department.",
        options={
            "billing": "Invoices, double charges, credit card billing payments",
            "technical": "API crashes, software bugs, server outages",
            "shipping": "Package delivery tracking and courier delays",
        },
        top_k=5,
    )
    assert cf["original_choice"] == "billing"
    assert isinstance(cf["original_log_odds"], float)
    assert cf["original_log_odds"] > 0.0
    assert len(cf["token_impacts"]) > 0
    first = cf["token_impacts"][0]
    assert "probability_drop" in first
    assert "logit_drop" in first
    assert isinstance(first["logit_drop"], float)
    assert any(item["logit_drop"] > 0.0 for item in cf["token_impacts"])


def test_s1_shap_recovers_saturated_disjunctive_or_gate_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify logit_drop ranks true causal triggers even when softmax probability saturates at 1.0."""

    def _saturated_additive_scorer(state: str, instructions: str, hypothesis: str) -> float:
        toks = set(state.lower().split())
        if "billing" in hypothesis.lower():
            # Each trigger contributes +20.0 logit so presence of both hits +40.0 (softmax=1.0),
            # and ablating one leaves +20.0 (softmax still rounds to 1.0000 in prob space).
            return 20.0 * float("chargeback" in toks) + 20.0 * float("overcharged" in toks)
        return 0.0

    monkeypatch.setattr("microllm.backends.base.semantic_match_logit", _saturated_additive_scorer)

    cf = explain_counterfactual(
        state_text="hello please review chargeback overcharged urgent today",
        instructions_text="Route ticket",
        options={
            "billing": "Chargebacks, double charges, billing disputes",
            "shipping": "Delivery status and parcel tracking",
        },
        top_k=4,
    )
    assert cf["original_choice"] == "billing"
    assert cf["original_probability"] == 1.0
    assert cf["original_log_odds"] == 40.0

    # In probability space every token has ~0.0 probability_drop due to saturation,
    # whereas logit-link S1-SHAP assigns logit_drop=20.0 strictly to the two OR-triggers.
    top_two_tokens = {item["token"] for item in cf["token_impacts"][:2]}
    assert top_two_tokens == {"chargeback", "overcharged"}
    for item in cf["token_impacts"][:2]:
        assert item["probability_drop"] == 0.0
        assert item["logit_drop"] == 20.0

    filler_impacts = [item for item in cf["token_impacts"][2:]]
    for item in filler_impacts:
        assert item["logit_drop"] == 0.0


def test_explain_counterfactual_rejects_fewer_than_two_options() -> None:
    """Verify ValueError is raised when fewer than two options are passed."""
    with pytest.raises(ValueError, match="at least 2 options"):
        explain_counterfactual("billing issue", "Route ticket", {"billing": "Billing"})
