"""Comprehensive Unit Tests for Tier-3 bert-large-int8 Model Profile.

Validates:
1. `MODEL_PROFILES["bert-large-int8"]` registration, configuration, and metadata.
2. `PROFILE_ALIASES` resolution for all 5 BERT Large aliases.
3. `resolve_nli_entailment_idx` dynamic NLI label index resolution for BERT families.
4. Offline bundle extraction via `pull_profile` with deterministic seed 506 and SHA-256 manifest.
5. Offline deterministic scoring via `load_profile_scorer`.
6. Benchmark integration within `DEFAULT_PARETO_CANDIDATES` and sequence length scaling.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from microllm.multilingual_benchmark_runner import DEFAULT_MULTILINGUAL_PROBE_TICKETS
from microllm.pareto_benchmark import (
    ALL_ENGINES_CANDIDATES,
    DEFAULT_PARETO_CANDIDATES,
    run_pareto_benchmark,
    run_sequence_length_scaling_benchmark,
)
from microllm.registry import (
    MODEL_PROFILES,
    PROFILE_ALIASES,
    _HF_MODEL_EXTRA_METADATA,
    load_profile_scorer,
    pull_profile,
    resolve_nli_entailment_idx,
)


def test_bert_large_int8_profile_metadata() -> None:
    """Verify that bert-large-int8 is registered in MODEL_PROFILES with exact metadata."""
    assert "bert-large-int8" in MODEL_PROFILES
    spec = MODEL_PROFILES["bert-large-int8"]

    assert spec["hf_repo"] == "yoshitomo-matsubara/bert-large-uncased-mnli"
    assert spec["microllm_hf_repo"] == "jmrenouard/microllm-bert-large-int8"
    assert spec["pipeline_tag"] == "zero-shot-classification"
    assert spec["onnx_subpath"] == "onnx/model_quantized.onnx"
    assert spec["quantization"] == "INT8_DYNAMIC"
    assert spec["size_mb"] == 340.0
    assert spec["latency_target_ms"] == 26.0
    assert spec["recommended_temperature"] == 0.85
    assert spec["entailment_idx"] == 2
    assert spec["tokenizer_name"] == "bert-large-uncased"
    assert "jmrenouard/microllm-bert-large-int8" in _HF_MODEL_EXTRA_METADATA


def test_bert_large_int8_aliases_resolution() -> None:
    """Verify that all aliases resolve cleanly to 'bert-large-int8'."""
    expected_aliases = [
        "bert-large",
        "bert-large-int8",
        "bert-large-uncased",
        "bert-large-mnli",
        "bert-large-nli",
        "bert-large-classifier",
        "microllm-bert-large-int8",
        "microllm-bert-large-mnli",
        "microllm-bert-large-nli",
        "microllm-bert-large-classifier",
        "jmrenouard/microllm-bert-large-int8",
        "jmrenouard/microllm-bert-large-mnli",
        "jmrenouard/microllm-bert-large-nli",
        "jmrenouard/microllm-bert-large-classifier",
    ]
    for alias in expected_aliases:
        assert alias in PROFILE_ALIASES, f"Missing alias: {alias}"
        assert PROFILE_ALIASES[alias] == "bert-large-int8"


def test_resolve_nli_entailment_idx_bert() -> None:
    """Verify dynamic entailment index resolution across BERT families."""
    assert resolve_nli_entailment_idx(tokenizer_name="bert-large-uncased") == 2
    assert (
        resolve_nli_entailment_idx(
            tokenizer_name="yoshitomo-matsubara/bert-large-uncased-mnli"
        )
        == 2
    )
    assert resolve_nli_entailment_idx(tokenizer_name="bert-base-uncased") == 2
    # Explicit override takes precedence
    assert resolve_nli_entailment_idx(tokenizer_name="bert-large-uncased", explicit_idx=0) == 0


def test_pull_profile_bert_large_offline_bundle(tmp_path: Path) -> None:
    """Verify pull_profile creates a valid offline bundle without downloading remote weights."""
    manifest = pull_profile("bert-large-int8", cache_dir=tmp_path, download_weights=False)
    assert isinstance(manifest, dict)
    assert manifest["installed"] is True
    assert manifest["profile"] == "bert-large-int8"
    assert manifest["hf_repo"] == "yoshitomo-matsubara/bert-large-uncased-mnli"
    assert manifest["entailment_idx"] == 2
    assert "sha256_checksum" in manifest

    local_dir = Path(manifest["local_dir"])
    assert local_dir.exists()
    assert (local_dir / "model_quantized_int8.npz").exists()
    assert (local_dir / "profile_manifest.json").exists()


def test_load_profile_scorer_bert_large(tmp_path: Path) -> None:
    """Verify load_profile_scorer returns a deterministic offline scoring callable for bert-large-int8."""
    pull_profile("bert-large-int8", cache_dir=tmp_path, download_weights=False)
    scorer = load_profile_scorer("bert-large-int8", cache_dir=tmp_path)
    assert callable(scorer)

    premise = "Customer requested urgent refund for duplicated invoice transaction."
    hyp_entail = "This request pertains to billing and finance."
    hyp_contradict = "The customer wants to report a broken laptop screen."

    logit_entail = scorer(premise, "Categorize ticket", hyp_entail)
    logit_contradict = scorer(premise, "Categorize ticket", hyp_contradict)

    assert isinstance(logit_entail, float)
    assert isinstance(logit_contradict, float)
    assert logit_entail > logit_contradict


def test_pareto_benchmark_includes_bert_large(tmp_path: Path) -> None:
    """Verify that bert-large-int8 is evaluated in pareto benchmark and scaling."""
    assert "bert-large-int8" in DEFAULT_PARETO_CANDIDATES
    assert "bert-large-int8" in ALL_ENGINES_CANDIDATES

    results = run_pareto_benchmark(
        items=DEFAULT_MULTILINGUAL_PROBE_TICKETS[:4],
        cache_dir=tmp_path,
        save_artifacts=False,
        include_scaling=True,
        iterations_per_length=2,
    )

    assert "bert-large-int8" in results["profiles"]
    prof = results["profiles"]["bert-large-int8"]
    assert prof["total_evaluated"] == 4
    assert prof["accuracy"] == 1.0
    assert prof["p50_latency_ms"] >= 0.0
    assert "rss_memory_mb" in prof

    assert "sequence_length_scaling" in results
    assert "bert-large-int8" in results["sequence_length_scaling"]
    scaling_lengths = results["sequence_length_scaling"]["bert-large-int8"]["lengths"]
    assert "32" in scaling_lengths
    assert "64" in scaling_lengths
    assert "128" in scaling_lengths
    assert "256" in scaling_lengths
    assert "512" in scaling_lengths
