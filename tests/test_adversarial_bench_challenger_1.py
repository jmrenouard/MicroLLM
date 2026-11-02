"""Empirical Stress Tests and Boundary Edge-Case Harness for Pareto & Sequence Scaling Benchmarks.

Adversarial Challenger 1 (Benchmark Empirical Stress Challenger).
Tests:
1. Scaling benchmark edge cases: empty sequence (L=0), single token (L=1), exact budget boundary (L=512), over-budget (L=1024).
2. Telemetry validation: absence of NaN/Inf, positive RSS memory values, valid QPS/tok/s metrics.
3. Empty sequence_lengths list handling and default fallback behavior.
4. FastEncoder sub-linear vs attention scaling behavior under load.
5. Markdown formatting resilience when given non-standard sequence length sets.
6. Zero iteration edge-case detection (IndexError in un-guarded np.percentile).
"""

from __future__ import annotations

import math
from typing import Any
import pytest

from microllm.pareto_benchmark import (
    DEFAULT_PARETO_CANDIDATES,
    SEQUENCE_LENGTHS,
    format_sequence_scaling_markdown,
    run_pareto_benchmark,
    run_sequence_length_scaling_benchmark,
)


def test_sequence_length_scaling_edge_cases_0_1_512() -> None:
    """Empirically test run_sequence_length_scaling_benchmark with boundary sequence lengths (0, 1, 512, 1024)."""
    scaling = run_sequence_length_scaling_benchmark(
        candidates={"fast-cpu-int8": DEFAULT_PARETO_CANDIDATES["fast-cpu-int8"]},
        sequence_lengths=[0, 1, 512, 1024],
        iterations_per_length=2,
    )

    assert "fast-cpu-int8" in scaling
    lengths = scaling["fast-cpu-int8"]["lengths"]
    assert set(lengths.keys()) == {"0", "1", "512", "1024"}

    # Validate L=0 (empty sequence: seq_text is "")
    l0 = lengths["0"]
    assert l0["sequence_length"] == 0
    assert l0["p50_latency_ms"] >= 0.0
    assert not math.isnan(l0["p50_latency_ms"])
    assert l0["throughput_tokens_per_sec"] == 0.0  # 0 length * qps
    assert l0["rss_memory_mb"] > 0.0

    # Validate L=1 (single token: seq_text is "customer")
    l1 = lengths["1"]
    assert l1["sequence_length"] == 1
    assert l1["p50_latency_ms"] > 0.0
    assert not math.isnan(l1["p50_latency_ms"])
    assert l1["throughput_tokens_per_sec"] > 0.0
    assert l1["rss_memory_mb"] > 0.0

    # Validate L=512 (exact budget boundary)
    l512 = lengths["512"]
    assert l512["sequence_length"] == 512
    assert l512["p50_latency_ms"] > 0.0
    assert not math.isnan(l512["p50_latency_ms"])
    assert l512["throughput_tokens_per_sec"] > 0.0
    assert l512["rss_memory_mb"] > 0.0

    # Validate L=1024 (over-budget, truncated by 512-token context guard)
    l1024 = lengths["1024"]
    assert l1024["sequence_length"] == 1024
    assert l1024["p50_latency_ms"] > 0.0
    assert not math.isnan(l1024["p50_latency_ms"])
    assert l1024["rss_memory_mb"] > 0.0


def test_sequence_length_scaling_empty_lengths_list_fallback() -> None:
    """Document and verify that run_sequence_length_scaling_benchmark falls back to SEQUENCE_LENGTHS when given []."""
    scaling = run_sequence_length_scaling_benchmark(
        candidates={"fast-cpu-int8": DEFAULT_PARETO_CANDIDATES["fast-cpu-int8"]},
        sequence_lengths=[],
        iterations_per_length=2,
    )
    assert "fast-cpu-int8" in scaling
    # Because sequence_lengths or SEQUENCE_LENGTHS treats [] as falsy, it defaults to SEQUENCE_LENGTHS
    assert set(scaling["fast-cpu-int8"]["lengths"].keys()) == {str(l) for l in SEQUENCE_LENGTHS}


def test_sequence_length_scaling_zero_iterations_resilience() -> None:
    """Verify that iterations_per_length=0 is handled defensively without crash."""
    scaling = run_sequence_length_scaling_benchmark(
        candidates={"fast-cpu-int8": DEFAULT_PARETO_CANDIDATES["fast-cpu-int8"]},
        sequence_lengths=[32],
        iterations_per_length=0,
    )
    assert "fast-cpu-int8" in scaling
    l32 = scaling["fast-cpu-int8"]["lengths"]["32"]
    assert l32["p50_latency_ms"] == 0.0
    assert l32["mean_latency_ms"] == 0.0
    assert l32["throughput_qps"] == 0.0
    assert l32["throughput_tokens_per_sec"] == 0.0
    assert l32["rss_memory_mb"] > 0.0
    assert l32["rss_delta_mb"] >= 0.0


def test_all_pareto_profiles_scaling_no_nan_and_positive_rss() -> None:
    """Stress test all 5 Pareto profiles across sequence lengths L=[16, 64, 256]."""
    scaling = run_sequence_length_scaling_benchmark(
        sequence_lengths=[16, 64, 256],
        iterations_per_length=2,
    )

    expected_profiles = [
        "fast-cpu-int8",
        "multilingual",
        "bert-large-int8",
        "deberta-large-int8",
        "decision-gate-cascade",
    ]

    for prof in expected_profiles:
        assert prof in scaling, f"Profile {prof} missing from scaling benchmark"
        lengths = scaling[prof]["lengths"]
        for l_str in ["16", "64", "256"]:
            assert l_str in lengths
            metrics = lengths[l_str]
            p50 = metrics["p50_latency_ms"]
            mean_lat = metrics["mean_latency_ms"]
            qps = metrics["throughput_qps"]
            tok_s = metrics["throughput_tokens_per_sec"]
            rss = metrics["rss_memory_mb"]

            # Assert no NaNs or Infs
            for k, val in metrics.items():
                if isinstance(val, float):
                    assert not math.isnan(val), f"{prof} L={l_str} metric {k} is NaN"
                    assert not math.isinf(val), f"{prof} L={l_str} metric {k} is Inf"

            assert p50 > 0.0
            assert mean_lat > 0.0
            assert qps > 0.0
            assert tok_s > 0.0
            assert rss > 0.0, f"{prof} L={l_str} RSS memory must be strictly positive"


def test_format_sequence_scaling_markdown_resilience() -> None:
    """Verify that formatting markdown table handles non-standard sequence lengths without crashing."""
    custom_scaling = {
        "fast-cpu-int8": {
            "tier": "Tier-1",
            "family": "DeBERTa-v3-xsmall",
            "lengths": {
                "0": {"p50_latency_ms": 0.12, "throughput_tokens_per_sec": 0.0},
                "1": {"p50_latency_ms": 0.15, "throughput_tokens_per_sec": 6666.7},
                "512": {"p50_latency_ms": 0.35, "throughput_tokens_per_sec": 1462857.1},
            },
        }
    }
    table = format_sequence_scaling_markdown(custom_scaling)
    assert "### Sequence Length Latency & Throughput Scaling" in table
    assert "| `fast-cpu-int8` |" in table
    assert "O(L) Sub-linear" in table


def test_pareto_benchmark_quick_full_telemetry_consistency() -> None:
    """Empirically evaluate run_pareto_benchmark with quick slice and check all telemetry properties."""
    from microllm.multilingual_benchmark_runner import DEFAULT_MULTILINGUAL_PROBE_TICKETS

    res = run_pareto_benchmark(
        items=DEFAULT_MULTILINGUAL_PROBE_TICKETS[:8],
        save_artifacts=False,
        include_scaling=True,
        iterations_per_length=2,
    )

    profiles = res["profiles"]
    for p_name, p_data in profiles.items():
        assert p_data["rss_memory_mb"] > 0.0
        assert not math.isnan(p_data["p50_latency_ms"])
        assert not math.isnan(p_data["accuracy"])
        assert not math.isnan(p_data["selective_accuracy"])
        assert 0.0 <= p_data["accuracy"] <= 1.0
        assert 0.0 <= p_data["selective_accuracy"] <= 1.0

    scaling = res["sequence_length_scaling"]
    assert len(scaling) == 5
    for p_name, s_data in scaling.items():
        for l_key, l_data in s_data["lengths"].items():
            assert l_data["rss_memory_mb"] > 0.0
            assert l_data["p50_latency_ms"] > 0.0
            assert not math.isnan(l_data["p50_latency_ms"])
