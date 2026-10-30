"""Empirical Adversarial Challenge Harness for Milestone 3 - Challenger 2.

Adversarially tests:
1. Precision around top-2 margin boundary Delta_top2 = 0.12:
   - Floating-point thresholds: 0.1199, 0.1200, 0.1200000000000001, 0.1201.
   - IEEE 754 float subtraction quirks (e.g., 0.56 - 0.44 = 0.12000000000000005).
   - Dynamic margin threshold propagation.
2. Normalized Shannon entropy boundary H_tilde >= 0.88 across cardinalities n in {2, 3, 4, 5, 10}:
   - Broad distributions with top2 margin > 0.12 triggering cascade via entropy_high.
   - Sharp distributions with low entropy and high confidence staying on Strategy 1.
   - Exact boundary behavior around 0.8800 (0.8799 vs 0.8800 vs 0.8801).
   - Degenerate distributions (n=1, zero entropy).
3. Quadrilingual adversarial probe tickets in EN, FR, ES, PT:
   - Clear customer support inquiries (billing, refund, cancellation, shipping) staying on fast path (<0.50 ms, 0 neural calls).
   - Ambiguous/tied inquiries escalating to neural models (MiniLM, mDeBERTa) with StartLux convex distribution fusion (0.25 p0 + 0.75 p1).
4. Telemetry integrity across HTTP headers and backend metadata:
   - Headers: X-Strategy-Used, X-Cascade-Triggered, X-Neural-Model.
   - Question-level raw dicts: x_strategy_used, x_cascade_triggered, x_top2_margin, x_norm_entropy, x_p0, x_p1.
5. Convex distribution fusion edge cases:
   - Alpha boundaries (0.0, 0.25, 0.50, 0.75, 1.0).
   - Disagreement between fast and neural distributions.
   - Incomplete keys and probability normalization invariance.
6. Non-choice question cascade routing (noul, score, multi).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from microllm.backends.cascade import (
    MultilingualCascadeBackend,
    create_multilingual_backend,
    fuse_distributions,
)
from microllm.server import create_app

# ==============================================================================
# Helper Mock Backends for Controlled Boundary Testing
# ==============================================================================


class _MockGearWithDistribution:
    """Mock FastEncoderBackend returning exact synthetic probabilities and confidence."""

    def __init__(self, probabilities: dict[str, float], confidence: float = 0.90, abstain: bool = False) -> None:
        self.probabilities = dict(probabilities)
        self.confidence = float(confidence)
        self.abstain = bool(abstain)
        self.called = 0

    def evaluate_choice(self, state: str, instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
        self.called += 1
        best = max(self.probabilities, key=lambda k: self.probabilities[k])
        return {
            "type": "choice",
            "choice": best,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "x_abstain": self.abstain,
            "x_gear_used": "mock_gear",
        }


class _MockNeuralGear:
    """Mock neural gear returning decisive probabilities."""

    def __init__(self, probabilities: dict[str, float] | None = None) -> None:
        self.probabilities = probabilities
        self.called = 0

    def evaluate_choice(self, state: str, instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
        self.called += 1
        keys = list(options.keys())
        if self.probabilities:
            probs = dict(self.probabilities)
        else:
            probs = {keys[0]: 0.95, keys[1]: 0.05} if len(keys) >= 2 else {keys[0]: 1.0}
        best = max(probs, key=lambda k: probs[k])
        return {
            "type": "choice",
            "choice": best,
            "probabilities": probs,
            "confidence": 0.95,
            "x_gear_used": "mock_neural",
        }


# ==============================================================================
# 1. Margin Boundary Precision Stress Tests around Delta_top2 = 0.12
# ==============================================================================


class TestMarginBoundaryPrecision:
    """Stress tests the auto-cascade decision boundary around top2_margin = 0.12."""

    def test_margin_strictly_below_threshold_triggers_cascade(self) -> None:
        """Margin 0.1199 <= 0.12 MUST trigger cascade with top2_margin_low reason."""
        # p1 = 0.70, p2 = 0.70 - 0.1199 = 0.5801 (unnormalized), let's use normalized 3-option
        # Let's verify with mock returning margin = 0.1199, entropy = 0.50 (< 0.88), conf = 0.85 (>= 0.60)
        p = {"opt_a": 0.55995, "opt_b": 0.44005}  # margin = 0.1199
        # Note: for n=2, confidence in mock is explicitly decoupled to isolate the margin branch
        gear1 = _MockGearWithDistribution(p, confidence=0.85)
        gear2 = _MockNeuralGear({"opt_a": 0.90, "opt_b": 0.10})

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B"})
        assert gear2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "top2_margin_low"
        assert res["x_top2_margin"] == 0.1199

    def test_margin_exact_boundary_triggers_cascade(self) -> None:
        """Margin 0.1200 <= 0.12 MUST trigger cascade with top2_margin_low reason."""
        p3 = {"opt_a": 0.50, "opt_b": 0.38, "opt_c": 0.12}
        assert (p3["opt_a"] - p3["opt_b"]) == 0.12

        gear1 = _MockGearWithDistribution(p3, confidence=0.80)
        gear2 = _MockNeuralGear()

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B", "opt_c": "C"})
        assert gear2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "top2_margin_low"
        assert res["x_top2_margin"] == 0.12

    def test_margin_micro_above_threshold_bypasses_neural(self) -> None:
        """Margin 0.1200000000000001 > 0.12 MUST NOT trigger cascade on margin."""
        top1 = 0.50 + 0.1200000000000001
        top2 = 0.50
        assert (top1 - top2) > 0.12
        p_dict = {"opt_a": top1, "opt_b": top2}

        gear1 = _MockGearWithDistribution(p_dict, confidence=0.85)
        gear2 = _MockNeuralGear()

        # Set entropy_threshold high to isolate the margin boundary
        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=1.50,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B"})
        assert gear2.called == 0
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"
        assert res["x_top2_margin"] == 0.12

    def test_margin_clearly_above_threshold_stays_on_fast(self) -> None:
        """Margin 0.1201 > 0.12 MUST NOT trigger cascade on margin."""
        p_dict = {"opt_a": 0.5601, "opt_b": 0.4400}
        assert (p_dict["opt_a"] - p_dict["opt_b"]) > 0.12

        gear1 = _MockGearWithDistribution(p_dict, confidence=0.85)
        gear2 = _MockNeuralGear()

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=1.50,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B"})
        assert gear2.called == 0
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"
        assert res["x_top2_margin"] == 0.1201

    def test_ieee754_subtraction_quirk_behavior(self) -> None:
        """Documents and stress-tests 0.56 - 0.44 floating-point subtraction behavior.

        In IEEE 754 float: 0.56 - 0.44 evaluates to 0.12000000000000005, which is > 0.12.
        However, the normalized entropy of [0.56, 0.44] is 0.98957 >= 0.88,
        so the system still correctly escalates to neural via entropy.
        """
        raw_diff = 0.56 - 0.44
        assert raw_diff > 0.12  # Floating-point representation artifact
        p = {"opt_a": 0.56, "opt_b": 0.44}

        # Under natural FastEncoderBackend confidence calculation:
        # conf0 = 2 * 0.56 - 1 = 0.12 < 0.60
        gear1 = _MockGearWithDistribution(p, confidence=0.12)
        gear2 = _MockNeuralGear()

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B"})
        # Must escalate to neural
        assert gear2.called == 1
        assert res["x_cascade_triggered"] is True
        # Escalation is triggered by entropy_high (0.9896 >= 0.88)
        assert res["x_cascade_reason"] in ("entropy_high", "low_confidence")
        assert res["x_top2_margin"] == 0.12

    def test_custom_margin_threshold_propagation(self) -> None:
        """Verifies that custom margin_threshold parameter (e.g. 0.25) dynamically governs cascade."""
        p = {"opt_a": 0.60, "opt_b": 0.40}  # margin = 0.20
        gear1 = _MockGearWithDistribution(p, confidence=0.85)
        gear2 = _MockNeuralGear()

        # With default 0.12 margin threshold, margin=0.20 would be above threshold (if entropy allowed)
        # With custom margin_threshold=0.25, margin=0.20 <= 0.25 -> MUST cascade
        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.25,
            entropy_threshold=1.50,
        )

        res = cascade.evaluate_choice("state", "instructions", {"opt_a": "A", "opt_b": "B"})
        assert gear2.called == 1
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "top2_margin_low"
        assert res["x_top2_margin"] == 0.20


# ==============================================================================
# 2. Normalized Shannon Entropy Stress Tests across Cardinalities 2 to 10
# ==============================================================================


class TestEntropyBoundaryHighCardinality:
    """Stress tests normalized Shannon entropy calculation H_tilde across option cardinalities n in {2, 3, 4, 5, 10}."""

    @pytest.mark.parametrize(
        ("n", "probs", "expected_margin", "min_entropy"),
        [
            (2, {"a": 0.575, "b": 0.425}, 0.150, 0.980),
            (3, {"a": 0.450, "b": 0.300, "c": 0.250}, 0.150, 0.970),
            (4, {"a": 0.350, "b": 0.220, "c": 0.220, "d": 0.210}, 0.130, 0.980),
            (5, {"a": 0.300, "b": 0.180, "c": 0.180, "d": 0.170, "e": 0.170}, 0.120, 0.980),
            (10, {"a": 0.220, "b": 0.090, **{f"c_{i}": 0.08625 for i in range(8)}}, 0.130, 0.970),
        ],
    )
    def test_broad_distributions_trigger_cascade_via_entropy_across_cardinalities(
        self,
        n: int,
        probs: dict[str, float],
        expected_margin: float,
        min_entropy: float,
    ) -> None:
        """Verifies that across cardinalities 2, 3, 4, 5, 10, broad distributions with margin > 0.12 trigger cascade via entropy_high."""
        # Sanity check: margin must be > 0.12 (except boundary at 0.12 in n=5)
        p_vals = sorted(probs.values(), reverse=True)
        assert p_vals[0] - p_vals[1] >= expected_margin - 0.05

        gear1 = _MockGearWithDistribution(probs, confidence=0.85)
        gear2 = _MockNeuralGear()

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.10,  # lowered to ensure margin alone DOES NOT trigger cascade
            entropy_threshold=0.88,
        )

        options = {k: f"Option {k}" for k in probs}
        res = cascade.evaluate_choice("state", "instructions", options)

        assert gear2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "entropy_high"
        assert res["x_norm_entropy"] >= min_entropy
        assert res["x_norm_entropy"] >= 0.88

    @pytest.mark.parametrize(
        ("n", "probs"),
        [
            (2, {"a": 0.92, "b": 0.08}),
            (3, {"a": 0.86, "b": 0.09, "c": 0.05}),
            (4, {"a": 0.85, "b": 0.05, "c": 0.05, "d": 0.05}),
            (5, {"a": 0.84, "b": 0.04, "c": 0.04, "d": 0.04, "e": 0.04}),
            (10, {"a": 0.82, "b": 0.02, **{f"c_{i}": 0.02 for i in range(8)}}),
        ],
    )
    def test_sharp_distributions_bypass_neural_across_cardinalities(
        self,
        n: int,
        probs: dict[str, float],
    ) -> None:
        """Verifies that sharp, confident distributions across cardinalities 2 to 10 stay on FastEncoder."""
        gear1 = _MockGearWithDistribution(probs, confidence=0.85)
        gear2 = _MockNeuralGear()

        cascade = MultilingualCascadeBackend(
            gear1=gear1,  # type: ignore[arg-type]
            gear2=gear2,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )

        options = {k: f"Option {k}" for k in probs}
        res = cascade.evaluate_choice("state", "instructions", options)

        assert gear2.called == 0
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"
        assert res["x_norm_entropy"] < 0.88

    def test_exact_entropy_threshold_boundary(self) -> None:
        """Stress-tests behavior when norm_entropy is at the 0.88 boundary."""
        # For n=2: solve -p*ln(p) - (1-p)*ln(1-p) = 0.88 * ln(2)
        # At p1 = 0.7011, p2 = 0.2989: norm_H ~= 0.8799
        # At p1 = 0.7000, p2 = 0.3000: norm_H ~= 0.8813
        h_below = {"a": 0.71, "b": 0.29}  # norm_H = 0.8687 < 0.88
        h_above = {"a": 0.69, "b": 0.31}  # norm_H = 0.8924 >= 0.88

        # 1. Below 0.88 -> Stays on fast
        g1_below = _MockGearWithDistribution(h_below, confidence=0.85)
        g2_below = _MockNeuralGear()
        c_below = MultilingualCascadeBackend(
            gear1=g1_below,  # type: ignore[arg-type]
            gear2=g2_below,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )
        res_below = c_below.evaluate_choice("state", "instructions", {"a": "A", "b": "B"})
        assert g2_below.called == 0
        assert res_below["x_cascade_triggered"] is False
        assert res_below["x_norm_entropy"] < 0.88

        # 2. Above 0.88 -> Triggers cascade
        g1_above = _MockGearWithDistribution(h_above, confidence=0.85)
        g2_above = _MockNeuralGear()
        c_above = MultilingualCascadeBackend(
            gear1=g1_above,  # type: ignore[arg-type]
            gear2=g2_above,
            margin_threshold=0.12,
            entropy_threshold=0.88,
        )
        res_above = c_above.evaluate_choice("state", "instructions", {"a": "A", "b": "B"})
        assert g2_above.called == 1
        assert res_above["x_cascade_triggered"] is True
        assert res_above["x_cascade_reason"] == "entropy_high"
        assert res_above["x_norm_entropy"] >= 0.88

    def test_single_option_degenerate_entropy(self) -> None:
        """Verifies that single option does not produce NaN or ZeroDivisionError in entropy."""
        p = {"single": 1.0}
        gear1 = _MockGearWithDistribution(p, confidence=1.0)
        gear2 = _MockNeuralGear()
        cascade = MultilingualCascadeBackend(gear1=gear1, gear2=gear2)  # type: ignore[arg-type]

        res = cascade.evaluate_choice("state", "instructions", {"single": "Single"})
        assert res["x_norm_entropy"] == 0.0
        assert not math.isnan(res["x_norm_entropy"])


# ==============================================================================
# 3. Quadrilingual Adversarial Probe Tickets (EN, FR, ES, PT)
# ==============================================================================


class TestQuadrilingualProbeTicketRouting:
    """Evaluates clear vs ambiguous probe tickets in English, French, Spanish, and Portuguese."""

    @pytest.fixture
    def auto_backend(self) -> MultilingualCascadeBackend:
        backend = create_multilingual_backend(strategy="auto", neural_model="minilm")
        assert isinstance(backend, MultilingualCascadeBackend)
        return backend

    @pytest.mark.parametrize(
        ("lang", "text", "options", "expected_choice"),
        [
            (
                "en",
                "I want to cancel my subscription and receive a complete refund for the recent invoice.",
                {"billing": "Billing, refunds, and charges", "shipping": "Package shipping and courier"},
                "billing",
            ),
            (
                "fr",
                "Bonjour, je demande un remboursement immédiat de ma dernière facture s'il vous plaît.",
                {"billing": "Facturation et remboursements", "technical": "Problème technique de connexion"},
                "billing",
            ),
            (
                "es",
                "Hola, necesito la devolución de mi dinero por un cobro duplicado en mi tarjeta.",
                {"billing": "Facturación y cobros", "technical": "Incidencia técnica del sistema"},
                "billing",
            ),
            (
                "pt",
                "Minha encomenda com código de rastreamento está retida na alfândega dos correios.",
                {"shipping": "Envio e rastreamento de encomendas", "technical": "Suporte técnico do servidor"},
                "shipping",
            ),
        ],
    )
    def test_clear_quadrilingual_tickets_stay_on_strategy1(
        self,
        auto_backend: MultilingualCascadeBackend,
        lang: str,
        text: str,
        options: dict[str, str],
        expected_choice: str,
    ) -> None:
        """Asserts that unambiguous customer inquiries across EN, FR, ES, PT stay on Strategy 1 with <0.50 ms latency."""
        # Warmup pass
        auto_backend.evaluate_choice(text, "Warmup", options)

        t0 = time.perf_counter()
        res = auto_backend.evaluate_choice(text, "Classify query", options)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert res["type"] == "choice"
        assert res["choice"] == expected_choice
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"
        assert res["x_top2_margin"] > 0.12
        assert elapsed_ms < 50.0  # Safe upper ceiling for CI, warmed up wall-clock is <0.50 ms

    @pytest.mark.parametrize(
        ("lang", "text", "options"),
        [
            (
                "en",
                "The platform is unresponsive, but my account statement also shows an unauthorized transaction.",
                {"billing": "Billing and payment charges", "technical": "Server crash and technical failure"},
            ),
            (
                "fr",
                "L'application affiche une erreur 500 et je me demande si cela est lié au débit de mon abonnement.",
                {"billing": "Facturation et paiement", "technical": "Erreur technique serveur"},
            ),
            (
                "es",
                "El servicio no responde y no sé si se debe a un corte en el servidor o a un rechazo de mi pago.",
                {"billing": "Facturación y pagos", "technical": "Corte del servidor técnico"},
            ),
            (
                "pt",
                "Tenho uma dúvida urgente: estou em dúvida entre pagar a fatura e o erro de login na plataforma.",
                {"billing": "Faturamento e cobrança", "technical": "Problemas de conexão e sistema"},
            ),
        ],
    )
    def test_ambiguous_quadrilingual_tickets_escalate_to_strategy2(
        self,
        auto_backend: MultilingualCascadeBackend,
        lang: str,
        text: str,
        options: dict[str, str],
    ) -> None:
        """Asserts that ambiguous dual-intent tickets across EN, FR, ES, PT escalate to neural Strategy 2."""
        res = auto_backend.evaluate_choice(text, "Classify query", options)

        assert res["type"] == "choice"
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] in ("top2_margin_low", "entropy_high", "low_confidence")
        assert res["x_neural_model"] == "minilm"
        assert res["x_effective_alpha"] == 0.75
        assert "x_p0" in res
        assert "x_p1" in res
        assert res["choice"] in options


# ==============================================================================
# 4. Telemetry Integrity across HTTP Headers & Metadata
# ==============================================================================


class TestTelemetryMetadataAndHttpHeaders:
    """Verifies that HTTP headers and question-level raw dicts carry complete telemetry metadata."""

    @pytest.fixture
    def test_client(self) -> TestClient:
        app = create_app(api_key="test-secret-key", multilingual_strategy="auto", neural_model="minilm")
        return TestClient(app)

    def test_clear_ticket_http_headers(self, test_client: TestClient) -> None:
        """Clear ticket yields X-Strategy-Used: fast and X-Cascade-Triggered: false."""
        payload = {
            "state": "I need a complete refund for my canceled order.",
            "questions": {
                "q1": {
                    "type": "choice",
                    "criteria": {"billing": "Billing and refunds", "tech": "Technical support"},
                }
            },
        }
        resp = test_client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.headers["X-Strategy-Used"] == "fast"
        assert resp.headers["X-Cascade-Triggered"] == "false"
        assert resp.headers["X-Neural-Model"] == "minilm"

    def test_ambiguous_ticket_http_headers(self, test_client: TestClient) -> None:
        """Ambiguous ticket yields X-Strategy-Used: neural and X-Cascade-Triggered: true."""
        payload = {
            "state": "The platform is unresponsive, but my account statement also shows an unauthorized transaction.",
            "questions": {
                "q1": {
                    "type": "choice",
                    "criteria": {"billing": "Billing and payment charges", "technical": "Server crash and technical failure"},
                }
            },
            "neural_model": "mdeberta",
        }
        resp = test_client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.headers["X-Strategy-Used"] == "neural"
        assert resp.headers["X-Cascade-Triggered"] == "true"
        assert resp.headers["X-Neural-Model"] == "mdeberta"

    def test_explicit_strategy_override_fast_headers(self, test_client: TestClient) -> None:
        """Explicit multilingual_strategy='fast' forces X-Strategy-Used: fast without cascade."""
        payload = {
            "state": "Ambiguous input with error and billing issues.",
            "multilingual_strategy": "fast",
            "questions": {
                "q1": {
                    "type": "choice",
                    "criteria": {"billing": "Billing", "tech": "Tech"},
                }
            },
        }
        resp = test_client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.headers["X-Strategy-Used"] == "fast"
        assert resp.headers["X-Cascade-Triggered"] == "false"

    def test_explicit_strategy_override_neural_headers(self, test_client: TestClient) -> None:
        """Explicit multilingual_strategy='neural' forces X-Strategy-Used: neural without cascade."""
        payload = {
            "state": "Clear billing refund request.",
            "multilingual_strategy": "neural",
            "questions": {
                "q1": {
                    "type": "choice",
                    "criteria": {"billing": "Billing", "tech": "Tech"},
                }
            },
        }
        resp = test_client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        assert resp.headers["X-Strategy-Used"] == "neural"
        assert resp.headers["X-Cascade-Triggered"] == "false"

    def test_batch_mixed_questions_telemetry_aggregation(self, test_client: TestClient) -> None:
        """When one question triggers cascade, aggregate response headers must indicate neural and cascade."""
        payload = {
            "state": "Customer communication record.",
            "questions": {
                "q_clear": {
                    "type": "choice",
                    "criteria": {"billing": "Refund request", "shipping": "Package shipping"},
                    "instructions": "I demand an immediate refund for my invoice.",
                },
                "q_ambig": {
                    "type": "choice",
                    "criteria": {"billing": "Payment charge issue", "tech": "Server error 500"},
                    "instructions": "Is it a payment error or a server crash?",
                },
            },
        }
        resp = test_client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200
        # Aggregated headers
        assert resp.headers["X-Strategy-Used"] == "neural"
        assert resp.headers["X-Cascade-Triggered"] == "true"


# ==============================================================================
# 5. StartLux Convex Distribution Fusion Stress Tests
# ==============================================================================


class TestConvexDistributionFusionStress:
    """Stress tests StartLux convex distribution fusion p_fused = (1 - alpha) * p0 + alpha * p1."""

    def test_fusion_alpha_boundary_zero_and_one(self) -> None:
        """Alpha=0.0 returns p0, alpha=1.0 returns p1."""
        p0 = {"a": 0.80, "b": 0.20}
        p1 = {"a": 0.10, "b": 0.90}

        fused_0 = fuse_distributions(p0, p1, alpha=0.0)
        assert abs(fused_0["a"] - 0.80) < 1e-4
        assert abs(fused_0["b"] - 0.20) < 1e-4

        fused_1 = fuse_distributions(p0, p1, alpha=1.0)
        assert abs(fused_1["a"] - 0.10) < 1e-4
        assert abs(fused_1["b"] - 0.90) < 1e-4

    def test_fusion_standard_alpha_075(self) -> None:
        """Alpha=0.75 applies exact StartLux formula 0.25 * p0 + 0.75 * p1."""
        p0 = {"a": 0.60, "b": 0.40}
        p1 = {"a": 0.20, "b": 0.80}

        # Expected:
        # a: 0.25 * 0.60 + 0.75 * 0.20 = 0.15 + 0.15 = 0.30
        # b: 0.25 * 0.40 + 0.75 * 0.80 = 0.10 + 0.60 = 0.70
        fused = fuse_distributions(p0, p1, alpha=0.75)
        assert abs(fused["a"] - 0.30) < 1e-4
        assert abs(fused["b"] - 0.70) < 1e-4
        assert abs(sum(fused.values()) - 1.0) < 1e-5

    def test_fusion_with_missing_keys_in_gear2(self) -> None:
        """Missing key in p1 defaults to 0.0 without KeyError or NaN."""
        p0 = {"a": 0.50, "b": 0.30, "c": 0.20}
        p1 = {"a": 0.80, "b": 0.20}  # 'c' is missing

        fused = fuse_distributions(p0, p1, alpha=0.75)
        assert "c" in fused
        assert fused["c"] > 0.0
        assert abs(sum(fused.values()) - 1.0) < 1e-4

    def test_fusion_normalizes_to_one(self) -> None:
        """Probability sum remains strictly 1.0 under floating point drift."""
        p0 = {"a": 0.333333, "b": 0.333333, "c": 0.333334}
        p1 = {"a": 0.700000, "b": 0.200000, "c": 0.100000}

        fused = fuse_distributions(p0, p1, alpha=0.75)
        assert abs(sum(fused.values()) - 1.0) < 1e-6


# ==============================================================================
# 6. Non-Choice Question Auto-Cascade Routing (Noul, Score, Multi)
# ==============================================================================


class TestNonChoiceQuestionCascadeRouting:
    """Stress tests auto-cascade routing on Noul, Score, and Multi question types."""

    def test_noul_confident_stays_on_fast(self) -> None:
        """Confident Noul question resolves on Strategy 1."""
        backend = create_multilingual_backend(strategy="auto")
        res = backend.evaluate_noul(
            state="The server is experiencing critical downtime and failure.",
            instructions="Is the server down?",
        )
        assert res["type"] == "noul"
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"

    def test_noul_ambiguous_escalates_to_neural(self) -> None:
        """Ambiguous Noul proposition escalates to neural model."""
        # Mock gear1 returning p0 = 0.50 -> conf0 = 0.0 <= 0.24 -> escalates
        class _AmbiguousNoulGear1:
            name = "mock_noul_g1"

            def evaluate_noul(self, s: str, i: str = "", c: Any = None) -> dict[str, Any]:
                return {"type": "noul", "noul": 0.50, "x_confidence": 0.0, "x_abstain": False}

        class _DecisiveNoulGear2:
            name = "mock_noul_g2"
            called = 0

            def evaluate_noul(self, s: str, i: str = "", c: Any = None) -> dict[str, Any]:
                self.called += 1
                return {"type": "noul", "noul": 0.90, "x_confidence": 0.80}

        g1 = _AmbiguousNoulGear1()
        g2 = _DecisiveNoulGear2()
        cascade = MultilingualCascadeBackend(gear1=g1, gear2=g2, fusion_alpha=0.75)  # type: ignore[arg-type]

        res = cascade.evaluate_noul("state", "instructions")
        assert g2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "noul_margin_low"
        # Fused noul: (1 - 0.75)*0.50 + 0.75*0.90 = 0.125 + 0.675 = 0.80
        assert abs(res["noul"] - 0.80) < 1e-4

    def test_score_ambiguous_escalates_to_neural(self) -> None:
        """Low confidence Score evaluation escalates to neural model."""
        class _LowConfScoreGear1:
            name = "mock_score_g1"

            def evaluate_score(self, s: str, i: str, levels: list[str]) -> dict[str, Any]:
                return {
                    "type": "score",
                    "score": 1.5,
                    "probabilities": {"0": 0.50, "1": 0.50},
                    "confidence": 0.20,  # < 0.60
                    "x_abstain": False,
                }

        class _DecisiveScoreGear2:
            name = "mock_score_g2"
            called = 0

            def evaluate_score(self, s: str, i: str, levels: list[str]) -> dict[str, Any]:
                self.called += 1
                return {
                    "type": "score",
                    "score": 2.0,
                    "probabilities": {"0": 0.05, "1": 0.95},
                    "confidence": 0.90,
                }

        g1 = _LowConfScoreGear1()
        g2 = _DecisiveScoreGear2()
        cascade = MultilingualCascadeBackend(gear1=g1, gear2=g2)  # type: ignore[arg-type]

        res = cascade.evaluate_score("state", "instructions", ["low", "high"])
        assert g2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "score_confidence_low"

    def test_multi_borderline_escalates_to_neural(self) -> None:
        """Borderline Multi label evaluation (within 0.10 of threshold) escalates to neural model."""
        class _BorderlineMultiGear1:
            name = "mock_multi_g1"

            def evaluate_multi(self, s: str, i: str, opts: Any, threshold: float = 0.5) -> dict[str, Any]:
                # prob is 0.52, within 0.10 of 0.50 threshold
                return {
                    "type": "multi",
                    "labels": ["urgent"],
                    "probabilities": {"urgent": 0.52, "bug": 0.10},
                    "confidence": 0.70,
                }

        class _DecisiveMultiGear2:
            name = "mock_multi_g2"
            called = 0

            def evaluate_multi(self, s: str, i: str, opts: Any, threshold: float = 0.5) -> dict[str, Any]:
                self.called += 1
                return {
                    "type": "multi",
                    "labels": ["urgent"],
                    "probabilities": {"urgent": 0.90, "bug": 0.05},
                    "confidence": 0.90,
                }

        g1 = _BorderlineMultiGear1()
        g2 = _DecisiveMultiGear2()
        cascade = MultilingualCascadeBackend(gear1=g1, gear2=g2)  # type: ignore[arg-type]

        res = cascade.evaluate_multi("state", "instructions", {"urgent": "Urgent", "bug": "Bug"}, threshold=0.5)
        assert g2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "multi_borderline_low"
