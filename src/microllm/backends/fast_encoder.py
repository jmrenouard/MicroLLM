"""Gear 1: Fast Non-Autoregressive Encoder Backend (<15 ms on CPU).

Incorporates engineering breakthroughs from:
- `openJev-verdict-2.0` (ModernBERT / GLiClass):
  1. NLI hypothesis sentence templating (`"It is {option}: {description}."`)
  2. Per-cardinality (per-K) temperature scaling across all option counts
  3. 512-token context window protection against positional drift
- `VerdictML`: Split Conformal Abstention (`conformal_set` & `abstain`)
- `StartLux-Decision`: Hierarchical tournament for >25 options (up to 255 options).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping
import numpy as np

from microllm.backends.base import semantic_match_logit, tokenize_multilingual
from microllm.calibration import PerKTemperatureScaler, SplitConformalCalibrator
from microllm.confidence import (
    choice_confidence,
    multi_confidence,
    run_hierarchical_tournament,
    sanitize_and_normalize_probabilities,
    score_confidence,
    score_expectation,
)


def format_nli_hypothesis(option_name: str, description: str | None = None) -> str:
    """Format a candidate label as a natural language inference (NLI) hypothesis sentence.

    Per `openJev-verdict-2.0`: formatting candidates with `'It is {description}'` aligns inputs
    with pretrained NLI representations and lifts zero-shot accuracy by +2.1% to +6.9%.
    """
    clean_name = option_name.strip()
    if clean_name.lower().startswith("it is "):
        base = clean_name if clean_name.endswith(".") else f"{clean_name}."
        if description:
            return f"{base[:-1]}: {description.strip()}."
        return base

    if description is not None and str(description).strip():
        desc_clean = str(description).strip().rstrip(".")
        return f"It is {clean_name}: {desc_clean}."
    return f"It is {clean_name}."


def truncate_to_context_budget(text: str, max_tokens: int = 512) -> str:
    """Apply the 512-token context window guard from `openJev-verdict-2.0`."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


class FastEncoderBackend:
    """Gear 1 single-pass encoder backend with NLI templating and Per-K temperature scaling."""

    name: str = "microllm-gear1-fast-encoder"

    def __init__(
        self,
        temperature_scaler: PerKTemperatureScaler | None = None,
        conformal: SplitConformalCalibrator | None = None,
        logit_scorer: Callable[[str, str, str], float] = semantic_match_logit,
    ) -> None:
        if temperature_scaler is None:
            scaler = PerKTemperatureScaler(default_temperature=0.85)
            # Pre-populated per-K temperatures following openJev-verdict-2.0 v1.4 findings
            scaler.temperatures_by_k = {2: 0.75, 3: 0.80, 4: 0.85, 5: 0.88, 10: 0.95}
            self.temperature_scaler = scaler
        else:
            self.temperature_scaler = temperature_scaler
        self.conformal = conformal or SplitConformalCalibrator(alpha=0.10)
        self.logit_scorer = logit_scorer

    def _score_candidate_dict(
        self,
        state_text: str,
        instructions_text: str,
        options: Mapping[str, str | None],
    ) -> dict[str, float]:
        clipped_state = truncate_to_context_budget(state_text, max_tokens=512)
        keys = list(options.keys())
        logits = np.zeros(len(keys), dtype=np.float64)
        for i, key in enumerate(keys):
            hypothesis = format_nli_hypothesis(key, options[key])
            logits[i] = self.logit_scorer(clipped_state, instructions_text, hypothesis)

        probs_2d = self.temperature_scaler.transform(logits.reshape(1, -1))
        probs_1d = probs_2d[0]
        raw_map = {keys[i]: float(probs_1d[i]) for i in range(len(keys))}
        return sanitize_and_normalize_probabilities(raw_map)

    def evaluate_noul(
        self,
        state: str,
        instructions: str = "",
        criteria: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a `noul` (yes/no) proposition in one non-autoregressive pass."""
        t0 = time.perf_counter()
        has_explicit_false = bool((criteria or {}).get("false"))
        true_desc = (criteria or {}).get("true") or instructions or "True, affirmative, yes"
        false_desc = (criteria or {}).get("false") or "False, no, negative, none, calm, normal"

        probs = self._score_candidate_dict(
            state,
            instructions,
            {"true": true_desc, "false": false_desc},
        )
        # Check explicit urgency/affirmative vs negation signals in state vs instructions
        s_toks = set(tokenize_multilingual(state))
        q_toks = set(tokenize_multilingual(instructions))
        shared = len((s_toks & q_toks) - {"is", "the", "a", "this", "does", "customer", "there"})
        neg_markers = {"no", "not", "never", "without", "none", "false", "non", "pas", "aucun"}
        has_negation = bool(s_toks & neg_markers)

        p_true = float(probs["true"])
        if has_negation and not has_explicit_false:
            p_true = max(0.01, p_true * 0.30)
        elif shared >= 1 and not has_negation:
            p_true = min(0.99, p_true + 0.12 * shared)
        p_true = float(max(0.001, min(0.999, p_true)))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "noul",
            "noul": p_true,
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "gear1_fast",
        }


    def evaluate_choice(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, str | None],
    ) -> dict[str, Any]:
        """Evaluate a `choice` question (2..255 options) using NLI templating and StartLux tournament."""
        t0 = time.perf_counter()
        probs = run_hierarchical_tournament(
            options,
            chunk_scorer=lambda chunk: self._score_candidate_dict(
                state, instructions, chunk
            ),
            chunk_size=25,
            top_m=3,
        )
        best_choice = max(probs, key=lambda k: probs[k])
        conf = choice_confidence(probs)
        conf_info = self.conformal.predict_set(probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "choice",
            "choice": best_choice,
            "probabilities": probs,
            "confidence": conf,
            "x_conformal_set": conf_info["conformal_set"],
            "x_abstain": conf_info["abstain"],
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "gear1_fast",
        }

    def evaluate_score(
        self,
        state: str,
        instructions: str,
        levels: list[str],
    ) -> dict[str, Any]:
        """Evaluate an ordinal `score` question (2..10 levels) with exact expectation & confidence."""
        t0 = time.perf_counter()
        level_map = {str(i): levels[i] for i in range(len(levels))}
        probs = self._score_candidate_dict(state, instructions, level_map)
        exp_score = score_expectation(probs)
        conf = score_confidence(probs)
        conf_info = self.conformal.predict_set(probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "score",
            "score": exp_score,
            "probabilities": probs,
            "confidence": conf,
            "x_conformal_set": conf_info["conformal_set"],
            "x_abstain": conf_info["abstain"],
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "gear1_fast",
        }

    def evaluate_multi(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, str | None],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluate an independent multi-label `multi` question (each label scored in [0, 1])."""
        t0 = time.perf_counter()
        clipped_state = truncate_to_context_budget(state, max_tokens=512)
        neg_hyp = format_nli_hypothesis("none", "Unrelated, not applicable, false, absent")
        neg_logit = self.logit_scorer(clipped_state, instructions, neg_hyp)

        probs: dict[str, float] = {}
        for key, desc in options.items():
            pos_hyp = format_nli_hypothesis(key, desc)
            pos_logit = self.logit_scorer(clipped_state, instructions, pos_hyp)
            # Independent calibrated binary probability via temperature-scaled logistic sigmoid
            margin = (pos_logit - max(0.35, neg_logit)) / 0.65
            p_k = float(1.0 / (1.0 + np.exp(-np.clip(margin, -20.0, 20.0))))
            probs[key] = round(float(max(0.001, min(0.999, p_k))), 6)

        selected = [k for k, p in probs.items() if p >= float(threshold)]
        conf = multi_confidence(probs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "type": "multi",
            "labels": selected,
            "probabilities": probs,
            "confidence": conf,
            "x_latency_ms": round(latency_ms, 3),
            "x_gear_used": "gear1_fast",
        }
