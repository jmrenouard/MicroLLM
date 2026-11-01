"""Comprehensive Unit Tests for Tier-3 deberta-large-int8 Model Profile (IDEA-048 / Issue #175).

Validates:
1. `MODEL_PROFILES["deberta-large-int8"]` registration and configuration.
2. `PROFILE_ALIASES` resolution for all 6 DeBERTa-v3-large aliases.
3. `resolve_nli_entailment_idx` dynamic NLI label index resolution.
4. Non-regression of `OFFICIAL_MICROLLM_HF_MODELS` (strict len == 4 requirement).
5. Offline bundle extraction via `pull_profile` with deterministic seed 505 and SHA-256 manifest.
6. Offline deterministic scoring via `load_profile_scorer`.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from microllm.registry import (
    MODEL_PROFILES,
    PROFILE_ALIASES,
    OFFICIAL_MICROLLM_HF_MODELS,
    _HF_MODEL_EXTRA_METADATA,
    resolve_nli_entailment_idx,
    pull_profile,
    load_profile_scorer,
)


def test_deberta_large_int8_profile_metadata() -> None:
    """Verify that deberta-large-int8 is registered in MODEL_PROFILES with exact metadata."""
    assert "deberta-large-int8" in MODEL_PROFILES
    spec = MODEL_PROFILES["deberta-large-int8"]

    assert spec["hf_repo"] == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    assert spec["microllm_hf_repo"] == "jmrenouard/microllm-deberta-large-int8"
    assert spec["pipeline_tag"] == "zero-shot-classification"
    assert spec["onnx_subpath"] == "onnx/model_quantized.onnx"
    assert spec["quantization"] == "INT8_DYNAMIC"
    assert spec["size_mb"] == 435.0
    assert spec["latency_target_ms"] == 32.0
    assert spec["recommended_temperature"] == 0.85
    assert spec["entailment_idx"] == 0


def test_deberta_large_int8_aliases_resolution() -> None:
    """Verify that all 6 aliases resolve cleanly to 'deberta-large-int8'."""
    expected_aliases = [
        "deberta-large",
        "deberta-large-int8",
        "deberta-v3-large",
        "deberta-v3-large-int8",
        "microllm-deberta-large-int8",
        "jmrenouard/microllm-deberta-large-int8",
    ]
    for alias in expected_aliases:
        assert alias in PROFILE_ALIASES, f"Missing alias: {alias}"
        assert PROFILE_ALIASES[alias] == "deberta-large-int8"


def test_resolve_nli_entailment_idx() -> None:
    """Verify dynamic entailment index resolution across model families."""
    # Moritz Laurer multi-NLI DeBERTa model family heuristic resolves to 0
    assert (
        resolve_nli_entailment_idx(
            tokenizer_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        )
        == 0
    )
    # Moritz Laurer mDeBERTa / MiniLM resolve to 0
    assert (
        resolve_nli_entailment_idx(tokenizer_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        == 0
    )
    # RoBERTa / BART MNLI resolve to 2
    assert resolve_nli_entailment_idx(tokenizer_name="roberta-large-mnli") == 2
    assert resolve_nli_entailment_idx(tokenizer_name="facebook/bart-large-mnli") == 2
    # Fallback default returns -1
    assert resolve_nli_entailment_idx(tokenizer_name="unknown-model/custom-nli") == -1


def test_official_hf_models_regression_guard() -> None:
    """Strictly assert that OFFICIAL_MICROLLM_HF_MODELS retains exactly 4 published models."""
    assert len(OFFICIAL_MICROLLM_HF_MODELS) == 4
    # Ensure extra metadata exists for community / Tier-3 discovery
    assert "jmrenouard/microllm-deberta-large-int8" in _HF_MODEL_EXTRA_METADATA


def test_pull_profile_offline_bundle_generation(tmp_path: Path) -> None:
    """Verify pull_profile creates a valid offline bundle without downloading remote weights."""
    manifest = pull_profile("deberta-large-int8", cache_dir=tmp_path, download_weights=False)
    assert isinstance(manifest, dict)
    assert manifest["installed"] is True
    assert manifest["profile"] == "deberta-large-int8"
    assert manifest["hf_repo"] == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    assert manifest["entailment_idx"] == 0
    assert "sha256_checksum" in manifest

    local_dir = Path(manifest["local_dir"])
    assert local_dir.exists()
    assert (local_dir / "model_quantized_int8.npz").exists()
    assert (local_dir / "profile_manifest.json").exists()


def test_load_profile_scorer_offline(tmp_path: Path) -> None:
    """Verify load_profile_scorer returns a deterministic offline scoring callable."""
    pull_profile("deberta-large-int8", cache_dir=tmp_path, download_weights=False)
    scorer = load_profile_scorer("deberta-large-int8", cache_dir=tmp_path)
    assert callable(scorer)

    premise = "The patient shows acute respiratory distress and severe fever."
    hyp_entail = "This is a medical emergency requiring critical care."
    hyp_contradict = "The customer wishes to renew their gym membership."

    logit_entail = scorer(premise, "Evaluate clinical triage.", hyp_entail)
    logit_contradict = scorer(premise, "Evaluate clinical triage.", hyp_contradict)

    assert isinstance(logit_entail, float)
    assert isinstance(logit_contradict, float)
    assert logit_entail > logit_contradict
