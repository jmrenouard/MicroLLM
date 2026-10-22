"""StartLux-Decision (`arXiv:2610.17309`) 'Read-or-Think' Dual-Gear Cascade Router.

Implements:
- Fast pass `p_0` via Gear 1 (`FastEncoderBackend`).
- Confidence & Conformal Abstention Gate: if `confidence(p_0) >= gate_threshold` and not abstaining,
  returns `p_0` immediately (<15 ms).
- Otherwise escalates to Gear 2 (`AnyJevPrefillBackend`) to obtain `p_1`, and fuses distributions:
  `p_fused = (1 - alpha) * p_0 + alpha * p_1` with `alpha = 0.75` (proven by StartLux to minimize
  Brier score and curb reasoning overconfidence).
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from microllm.backends.base import semantic_match_logit, tokenize_multilingual
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.backends.prefill_llm import AnyJevPrefillBackend
from microllm.confidence import (
    choice_confidence,
    multi_confidence,
    sanitize_and_normalize_probabilities,
    score_confidence,
    score_expectation,
)


def fuse_distributions(
    p0: Mapping[str, float],
    p1: Mapping[str, float],
    alpha: float = 0.75,
) -> dict[str, float]:
    """Compute StartLux weighted probability fusion: `p = (1 - alpha) * p0 + alpha * p1`."""
    w1 = float(max(0.0, min(1.0, alpha)))
    w0 = 1.0 - w1
    raw = {k: w0 * float(p0[k]) + w1 * float(p1.get(k, 0.0)) for k in p0}
    return sanitize_and_normalize_probabilities(raw)


def compute_adaptive_alpha(
    conf0: float,
    conf1: float,
    answer_mass: float = 1.0,
    base_alpha: float = 0.75,
) -> float:
    """Compute entropy- & answer-mass-aware adaptive cascade fusion weight alpha(x) (IDEA-002)."""
    c0 = float(max(0.0, min(1.0, conf0)))
    c1 = float(max(0.0, min(1.0, conf1)))
    mass = float(max(0.05, min(1.0, answer_mass)))
    delta_conf = c1 - c0
    raw_alpha = float(base_alpha) + 0.30 * delta_conf
    mass_Penalty = 1.0 if mass >= 0.50 else (0.45 + 1.10 * mass)
    adjusted = raw_alpha * mass_Penalty
    return round(float(max(0.15, min(0.95, adjusted))), 4)


def evaluate_voc_gate_regime(
    state: str,
    instructions: str,
    options: Mapping[str, str | None],
    conf0: float,
    computation_cost: float = 0.04,
) -> dict[str, Any]:
    """Evaluate StartLux Proposition 5 Value-of-Computation Gate & Sys1Cal-v1 Open-Set regime (HYP-005)."""
    keys = list(options.keys())
    max_logit = max(
        (
            semantic_match_logit(state, instructions, format_nli_hypothesis(k, options[k]))
            for k in keys
        ),
        default=0.0,
    )
    s_toks = set(tokenize_multilingual(state))
    deliberative_markers = {
        "unless", "exempt", "exempted", "waiver", "except", "however", "provided",
        "override", "conflict", "threshold", "exceeds",
    }
    has_deliberative = bool(s_toks & deliberative_markers)

    if max_logit < 0.45 and not has_deliberative:
        return {
            "uncertainty_regime": "open_set_abstain",
            "should_abstain": True,
            "should_escalate_gear2": False,
            "expected_voc_gain": 0.0,
            "max_semantic_logit": round(float(max_logit), 4),
        }

    expected_gain = (0.18 if has_deliberative else 0.06 * (1.0 - float(conf0))) - float(
        computation_cost
    )
    if has_deliberative and expected_gain > 0.0:
        regime = "epistemic_deliberative"
        escalate = True
    elif float(conf0) >= 0.65:
        regime = "confident_direct"
        escalate = False
    else:
        regime = "aleatoric_tie_direct"
        escalate = expected_gain > 0.0

    return {
        "uncertainty_regime": regime,
        "should_abstain": False,
        "should_escalate_gear2": escalate,
        "expected_voc_gain": round(float(expected_gain), 4),
        "max_semantic_logit": round(float(max_logit), 4),
    }


class DualGearCascadeBackend:
    """Hybrid System-One Cascade combining Gear 1 Fast Encoder and Gear 2 AnyJev L0 Prefill."""

    name: str = "microllm-dual-gear-cascade"

    def __init__(
        self,
        gear1: FastEncoderBackend | None = None,
        gear2: AnyJevPrefillBackend | None = None,
        gate_threshold: float = 0.75,
        alpha: float = 0.75,
        adaptive_alpha: bool = False,
    ) -> None:
        self.gear1 = gear1 or FastEncoderBackend()
        self.gear2 = gear2 or AnyJevPrefillBackend()
        self.gate_threshold = float(gate_threshold)
        self.alpha = float(alpha)
        self.adaptive_alpha = bool(adaptive_alpha)

    def evaluate_noul(
        self,
        state: str,
        instructions: str = "",
        criteria: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Cascade evaluation for `noul` propositions with HYP-005 VoC gate telemetry."""
        t0 = time.perf_counter()
        res0 = self.gear1.evaluate_noul(state, instructions, criteria)
        p0 = float(res0["noul"])
        conf0 = abs(2.0 * p0 - 1.0)
        true_desc = (criteria or {}).get("true") or instructions or "True, affirmative, yes"
        false_desc = (criteria or {}).get("false") or "False, no, negative, none, calm, normal"
        voc = evaluate_voc_gate_regime(
            state,
            instructions,
            {"true": true_desc, "false": false_desc},
            conf0=conf0,
        )
        res0["x_voc_regime"] = voc["uncertainty_regime"]
        res0["x_voc_gain"] = voc["expected_voc_gain"]
        if conf0 >= self.gate_threshold and not (
            0.15 < self.gate_threshold < 0.99
            and voc["uncertainty_regime"] == "epistemic_deliberative"
        ):
            return res0

        res1 = self.gear2.evaluate_noul(state, instructions, criteria)
        p1 = float(res1["noul"])
        conf1 = abs(2.0 * p1 - 1.0)
        eff_alpha = (
            compute_adaptive_alpha(conf0, conf1, base_alpha=self.alpha)
            if self.adaptive_alpha
            else self.alpha
        )
        p_fused = (1.0 - eff_alpha) * p0 + eff_alpha * p1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "noul",
            "noul": float(max(0.0, min(1.0, p_fused))),
            "x_confidence": round(abs(2.0 * p_fused - 1.0), 6),
            "x_abstain": bool(abs(2.0 * p_fused - 1.0) < 0.15 or voc["should_abstain"]),
            "x_voc_regime": voc["uncertainty_regime"],
            "x_voc_gain": voc["expected_voc_gain"],
            "x_effective_alpha": round(float(eff_alpha), 4),
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "cascade_fused",
            "x_p0": round(p0, 4),
            "x_p1": round(p1, 4),
        }

    def evaluate_choice(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, str | None],
    ) -> dict[str, Any]:
        """Cascade evaluation for `choice` questions with (1 - alpha)p0 + alpha*p1 fusion and HYP-005 VoC gate."""
        t0 = time.perf_counter()
        res0 = self.gear1.evaluate_choice(state, instructions, options)
        conf0 = float(res0["confidence"])
        voc = evaluate_voc_gate_regime(state, instructions, options, conf0=conf0)
        res0["x_voc_regime"] = voc["uncertainty_regime"]
        res0["x_voc_gain"] = voc["expected_voc_gain"]

        if (
            conf0 >= self.gate_threshold
            and not bool(res0.get("x_abstain", False))
            and not (
                0.15 < self.gate_threshold < 0.99
                and voc["uncertainty_regime"] == "epistemic_deliberative"
            )
        ):
            return res0

        res1 = self.gear2.evaluate_choice(state, instructions, options)
        conf1 = float(res1.get("confidence", conf0))
        l0_meta = res1.get("x_l0_meta") or {}
        ans_mass = float(l0_meta.get("answer_mass", 1.0)) if isinstance(l0_meta, Mapping) else 1.0
        eff_alpha = (
            compute_adaptive_alpha(conf0, conf1, answer_mass=ans_mass, base_alpha=self.alpha)
            if self.adaptive_alpha
            else self.alpha
        )
        fused_probs = fuse_distributions(res0["probabilities"], res1["probabilities"], alpha=eff_alpha)
        best_choice = max(fused_probs, key=lambda k: fused_probs[k])
        conf = choice_confidence(fused_probs)
        conf_info = self.gear1.conformal.predict_set(fused_probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "choice",
            "choice": best_choice,
            "probabilities": fused_probs,
            "confidence": conf,
            "x_conformal_set": conf_info["conformal_set"],
            "x_abstain": bool(conf_info["abstain"] or voc["should_abstain"]),
            "x_voc_regime": voc["uncertainty_regime"],
            "x_voc_gain": voc["expected_voc_gain"],
            "x_effective_alpha": round(float(eff_alpha), 4),
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "cascade_fused",
            "x_l0_meta": res1.get("x_l0_meta"),
        }

    def evaluate_score(
        self,
        state: str,
        instructions: str,
        levels: list[str],
    ) -> dict[str, Any]:
        """Cascade evaluation for ordinal `score` questions."""
        t0 = time.perf_counter()
        res0 = self.gear1.evaluate_score(state, instructions, levels)
        if float(res0["confidence"]) >= self.gate_threshold and not bool(
            res0.get("x_abstain", False)
        ):
            return res0

        res1 = self.gear2.evaluate_score(state, instructions, levels)
        fused_probs = fuse_distributions(res0["probabilities"], res1["probabilities"], alpha=self.alpha)
        exp_score = score_expectation(fused_probs)
        conf = score_confidence(fused_probs)
        conf_info = self.gear1.conformal.predict_set(fused_probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "score",
            "score": exp_score,
            "probabilities": fused_probs,
            "confidence": conf,
            "x_conformal_set": conf_info["conformal_set"],
            "x_abstain": conf_info["abstain"],
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "cascade_fused",
        }

    def evaluate_multi(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, str | None],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Cascade evaluation for independent multi-label `multi` questions."""
        t0 = time.perf_counter()
        res0 = self.gear1.evaluate_multi(state, instructions, options, threshold=threshold)
        if float(res0["confidence"]) >= self.gate_threshold:
            return res0

        res1 = self.gear2.evaluate_multi(state, instructions, options, threshold=threshold)
        p0 = res0["probabilities"]
        p1 = res1["probabilities"]
        w1 = float(max(0.0, min(1.0, self.alpha)))
        w0 = 1.0 - w1
        fused_probs = {
            k: round(float(max(0.001, min(0.999, w0 * float(p0[k]) + w1 * float(p1.get(k, 0.0))))), 6)
            for k in p0
        }
        selected = [k for k, p in fused_probs.items() if p >= float(threshold)]
        conf = multi_confidence(fused_probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "multi",
            "labels": selected,
            "probabilities": fused_probs,
            "confidence": conf,
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "cascade_fused",
        }

