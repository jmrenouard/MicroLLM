"""TDD tests for logit-space S1-SHAP counterfactual attribution (arXiv:2611.12521, Issue #249)."""

from __future__ import annotations

import numpy as np
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


def test_explain_counterfactual_empty_and_subminimal_inputs() -> None:
    """Verify empty, whitespace-only, punctuation-only, and single-char inputs handle gracefully."""
    options = {
        "billing": "Invoice disputes, double charges, refunds, credit card billing",
        "shipping": "Courier delivery, parcel tracking, logistics, customs",
    }
    for subminimal in ["", "   \n\t  ", ". ! ? , -", "a b c d"]:
        cf = explain_counterfactual(
            state_text=subminimal,
            instructions_text="Route ticket",
            options=options,
        )
        assert cf["original_choice"] in options
        assert isinstance(cf["original_log_odds"], float)
        assert np.isfinite(cf["original_log_odds"])
        assert cf["token_impacts"] == []
        assert cf["decision_flip_tokens"] == []


def test_explain_counterfactual_token_deduplication_and_punctuation() -> None:
    """Verify duplicate words are deduplicated and punctuation is stripped while preserving hyphens."""
    options = {
        "billing": "Billing invoices, charges, and payment dispute handling",
        "shipping": "Courier delivery and logistics tracking",
    }
    cf = explain_counterfactual(
        state_text="Urgent billing billing! Double billing issue on billing-invoice.",
        instructions_text="Route ticket",
        options=options,
        top_k=10,
    )
    tokens = [item["token"] for item in cf["token_impacts"]]
    assert len(tokens) == len(set(tokens))
    assert tokens.count("billing") == 1
    assert "billing!" not in tokens
    assert "billing-invoice" in tokens


def test_explain_counterfactual_unicode_and_multilingual() -> None:
    """Verify multi-lingual inputs in French, German, and Spanish preserve non-ASCII characters."""
    # French
    cf_fr = explain_counterfactual(
        state_text="Facturation contestée pour prélèvement indu et litige bancaire",
        instructions_text="Route support ticket",
        options={
            "billing": "Facturation prélèvement litige bancaire",
            "shipping": "Livraison colis transporteur",
        },
        top_k=8,
    )
    tokens_fr = [item["token"] for item in cf_fr["token_impacts"]]
    assert "contestée" in tokens_fr
    assert "prélèvement" in tokens_fr
    assert cf_fr["original_choice"] == "billing"
    assert any(item["logit_drop"] > 0.0 for item in cf_fr["token_impacts"])

    # German
    cf_de = explain_counterfactual(
        state_text="Rechnungsprüfung für Doppelbuchung",
        instructions_text="Route support ticket",
        options={
            "billing": "Rechnungsprüfung und Doppelbuchung",
            "shipping": "Versand und Paket",
        },
    )
    tokens_de = [item["token"] for item in cf_de["token_impacts"]]
    assert "rechnungsprüfung" in tokens_de
    assert "doppelbuchung" in tokens_de
    assert cf_de["original_choice"] == "billing"

    # Spanish
    cf_es = explain_counterfactual(
        state_text="Facturación duplicada y reclamo bancario",
        instructions_text="Route support ticket",
        options={
            "billing": "Facturación y reclamos",
            "shipping": "Envío de paquetes",
        },
    )
    tokens_es = [item["token"] for item in cf_es["token_impacts"]]
    assert "facturación" in tokens_es
    assert cf_es["original_choice"] == "billing"


def test_explain_counterfactual_options_with_none_descriptions_and_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify options with None descriptions format safely and handle exact logit ties."""
    options = {"billing": None, "shipping": None, "technical": None}
    cf = explain_counterfactual(
        state_text="Customer needs assistance with an account inquiry.",
        instructions_text="Route ticket",
        options=options,
    )
    assert cf["original_choice"] in options
    assert isinstance(cf["original_log_odds"], float)
    assert len(cf["token_impacts"]) > 0

    # Exact logit tie across options
    monkeypatch.setattr("microllm.backends.base.semantic_match_logit", lambda s, i, h: 5.0)
    cf_tie = explain_counterfactual(
        state_text="Neutral input text here.",
        instructions_text="Route ticket",
        options=options,
    )
    assert cf_tie["original_log_odds"] == 0.0
    assert round(cf_tie["original_probability"], 4) == round(1.0 / len(options), 4)


def test_explain_counterfactual_top_k_bounds_and_decision_flips() -> None:
    """Verify top_k bounds clipping and detection of decision-flipping tokens."""
    options = {
        "billing": "billing charge and invoice",
        "shipping": "courier delivery package",
    }
    # top_k = 1
    cf_1 = explain_counterfactual(
        state_text="billing courier delivery",
        instructions_text="Route ticket",
        options=options,
        top_k=1,
    )
    assert len(cf_1["token_impacts"]) <= 1

    # top_k <= 0 clipped to max(1, top_k)
    cf_zero = explain_counterfactual(
        state_text="billing courier delivery",
        instructions_text="Route ticket",
        options=options,
        top_k=0,
    )
    assert len(cf_zero["token_impacts"]) <= 1

    cf_neg = explain_counterfactual(
        state_text="billing courier delivery",
        instructions_text="Route ticket",
        options=options,
        top_k=-5,
    )
    assert len(cf_neg["token_impacts"]) <= 1

    # top_k = 100 on short sentence
    cf_large = explain_counterfactual(
        state_text="billing courier delivery",
        instructions_text="Route ticket",
        options=options,
        top_k=100,
    )
    assert len(cf_large["token_impacts"]) == 3

    # Decision flip verification
    assert cf_large["original_choice"] == "shipping"
    flips = [item for item in cf_large["token_impacts"] if item["flips_decision"]]
    assert len(flips) >= 1
    assert set(cf_large["decision_flip_tokens"]) == {item["token"] for item in flips}
    assert any(item["new_choice"] == "billing" for item in flips)


def test_explain_counterfactual_k_ary_scaling_and_ranking_consistency() -> None:
    """Verify k-ary scaling across 8 options and strict ordering of impacts."""
    options = {
        "billing": "Invoice disputes, double charges, refunds, credit card billing",
        "shipping": "Courier delivery, parcel tracking, logistics, customs",
        "technical": "Software bugs, API crashes, stack traces, compiler errors",
        "auth": "Password resets, MFA tokens, login failures, OAuth",
        "sales": "Enterprise quotes, discounts, licensing contracts, sales reps",
        "hr": "Employee benefits, payroll, onboarding, health insurance",
        "legal": "Terms of service, privacy policy, GDPR compliance, NDAs",
        "infra": "Kubernetes clusters, database replicas, AWS cloud outages",
    }
    cf = explain_counterfactual(
        state_text="Kubernetes clusters and database replicas experienced an AWS cloud outage",
        instructions_text="Route incident ticket",
        options=options,
        top_k=8,
    )
    assert cf["original_choice"] == "infra"
    assert len(cf["token_impacts"]) == 8

    # Strict ranking by (probability_drop, logit_drop) descending
    for idx in range(len(cf["token_impacts"]) - 1):
        curr = cf["token_impacts"][idx]
        nxt = cf["token_impacts"][idx + 1]
        assert (round(curr["probability_drop"], 4), curr["logit_drop"]) >= (
            round(nxt["probability_drop"], 4),
            nxt["logit_drop"],
        )
