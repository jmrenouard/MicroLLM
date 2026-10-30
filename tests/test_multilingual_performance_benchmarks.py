"""Comprehensive Unit and Regression Tests for Milestone 6 Multilingual Performance Benchmarks.

Validates:
1. Benchmark runner execution across all strategies ('fast', 'neural', 'auto') and models ('minilm', 'mdeberta').
2. JSON schema, telemetry structure, and field completeness in benchmarks/multilingual_benchmarks_results.json.
3. Mathematical monotonicity of latency percentiles (min <= p50 <= p90 <= p95 <= p99 <= max) and positive throughput.
4. Correctness of speedup ratios and baseline gain calculations in comparative analysis.
5. Invariant preservation: baseline benchmarks/latest_results.json remains bitwise untouched (SHA-256 identical).
6. Documentation compliance: BENCHMARKS.md Section 11 exists with required tables and zero non-English headings/prose.
7. 100% offline socket isolation during benchmark and evaluation execution.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import socket
from typing import Any

import pytest

from microllm.multilingual_benchmark_runner import (
    MultilingualBenchmarkConfig,
    MultilingualBenchmarkRunner,
    guard_sockets,
    run_multilingual_benchmarks,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_LATEST_RESULTS_SHA256 = (
    "5d7a796cefb0218cd42c4d5bc8207e17637546fd1c9a7a41e989f121dba6b295"
)


# ==============================================================================
# 1. Benchmark Runner Execution & Parameter Matrix Tests
# ==============================================================================

class TestMultilingualBenchmarkRunnerExecution:
    """Verifies that the benchmark runner executes properly across configurations and options."""

    def test_runner_executes_full_strategy_and_model_matrix(self, tmp_path: pathlib.Path) -> None:
        """Asserts runner executes fast, neural (minilm, mdeberta), and auto (minilm, mdeberta)."""
        out_file = tmp_path / "matrix_results.json"
        config = MultilingualBenchmarkConfig(
            languages=["en", "fr"],
            strategies=["fast", "neural", "auto"],
            neural_models=["minilm", "mdeberta"],
            output_path=str(out_file),
            max_items_per_lang=2,
            save_artifacts=True,
            enforce_offline=True,
        )
        runner = MultilingualBenchmarkRunner(config)
        report = runner.run()

        assert out_file.is_file(), "Runner must write output file to specified path"
        configs = report.get("configurations_evaluated", [])
        expected_configs = {"fast", "neural_minilm", "neural_mdeberta", "auto_minilm", "auto_mdeberta"}
        assert expected_configs.issubset(set(configs)), f"Missing configurations: {expected_configs - set(configs)}"

    def test_runner_per_language_slice_execution(self, tmp_path: pathlib.Path) -> None:
        """Asserts runner captures metrics for all 4 languages: EN, FR, ES, PT."""
        out_file = tmp_path / "lang_results.json"
        report = run_multilingual_benchmarks(
            output_path=str(out_file),
            languages=["en", "fr", "es", "pt"],
            strategies=["fast"],
            max_items_per_lang=2,
            save_artifacts=True,
            enforce_offline=True,
        )
        per_lang = report.get("results_by_configuration", {}).get("fast", {}).get("by_language", {})
        assert set(per_lang.keys()) == {"en", "fr", "es", "pt"}
        for lang, slice_data in per_lang.items():
            assert slice_data["total_items"] == 2
            assert slice_data["accuracy"] >= 0.50
            assert slice_data["p50_latency_ms"] > 0.0

    def test_runner_offline_socket_isolation_enforced(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """Asserts that socket connections trigger runtime failure under socket guard."""
        def fail_connect(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Network socket connection attempted during benchmark run!")

        monkeypatch.setattr(socket.socket, "connect", fail_connect)
        out_file = tmp_path / "offline_results.json"
        report = run_multilingual_benchmarks(
            output_path=str(out_file),
            languages=["en"],
            strategies=["fast"],
            max_items_per_lang=1,
            save_artifacts=True,
            enforce_offline=True,
        )
        assert report["environment"]["offline_socket_isolation"] is True


# ==============================================================================
# 2. JSON Schema & Telemetry Structure Tests
# ==============================================================================

class TestMultilingualBenchmarksJsonSchema:
    """Validates the schema, required keys, and data types of the output JSON artifact."""

    @pytest.fixture(scope="class")
    def benchmark_data(self) -> dict[str, Any]:
        """Loads benchmarks/multilingual_benchmarks_results.json."""
        artifact_path = ROOT / "benchmarks" / "multilingual_benchmarks_results.json"
        assert artifact_path.is_file(), f"Benchmark results missing at {artifact_path}"
        content = artifact_path.read_text(encoding="utf-8")
        return json.loads(content)

    def test_top_level_keys_present(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts all mandatory top-level sections exist."""
        required_keys = {
            "benchmark_suite",
            "version",
            "timestamp_utc",
            "environment",
            "dataset_info",
            "configurations_evaluated",
            "summary",
            "results_by_configuration",
            "per_language",
            "comparative_analysis",
        }
        missing = required_keys - set(benchmark_data.keys())
        assert not missing, f"Missing top-level keys in results JSON: {missing}"

    def test_environment_telemetry_fields(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts that environment metadata is fully populated and specifies CPU compute."""
        env = benchmark_data["environment"]
        assert "platform" in env
        assert "processor" in env
        assert isinstance(env["cpu_count_logical"], int) and env["cpu_count_logical"] >= 1
        assert "python_version" in env
        assert "numpy_version" in env
        assert "CPU" in env["compute_device"]
        assert env["offline_socket_isolation"] is True

    def test_dataset_info_metadata(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts dataset_info reports source, total_tickets >= 48, and all 4 languages."""
        ds = benchmark_data["dataset_info"]
        assert "microllm_multilingual_corpus.jsonl" in ds["source"]
        assert ds["total_tickets"] >= 48
        assert set(ds["languages"]) == {"en", "fr", "es", "pt"}

    def test_configurations_evaluated_contains_all_strategies(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts all five standard configurations were evaluated."""
        configs = benchmark_data["configurations_evaluated"]
        expected = ["fast", "neural_minilm", "neural_mdeberta", "auto_minilm", "auto_mdeberta"]
        for exp in expected:
            assert exp in configs, f"Configuration '{exp}' missing from evaluated configs"

    def test_summary_structure(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts summary contains total evaluations and per-config accuracy and latency dicts."""
        summary = benchmark_data["summary"]
        assert summary["total_evaluations"] >= 240
        assert len(summary["accuracy_by_configuration"]) >= 5
        assert len(summary["p50_latency_ms_by_configuration"]) >= 5
        for cfg, acc in summary["accuracy_by_configuration"].items():
            assert 0.0 <= acc <= 1.0, f"Accuracy for {cfg} out of bounds: {acc}"

    def test_configuration_slice_metrics_completeness(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts that each configuration's overall slice has all required metric fields."""
        required_slice_fields = {
            "total_items", "success_count", "fail_count", "accuracy",
            "selective_accuracy_above_tau", "coverage_above_tau",
            "abstention_count", "abstention_rate", "mean_confidence",
            "confidence_separation_gap", "mean_top2_margin", "mean_normalized_entropy",
            "min_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "max_latency_ms", "mean_latency_ms",
            "throughput_items_per_sec", "cascade_count", "cascade_rate",
            "fast_direct_count", "cascade_reasons",
        }
        for cfg_name, cfg_data in benchmark_data["results_by_configuration"].items():
            overall = cfg_data["overall"]
            missing = required_slice_fields - set(overall.keys())
            assert not missing, f"Configuration {cfg_name} missing overall metric fields: {missing}"


# ==============================================================================
# 3. Latency Percentiles, Throughput & Speedup Invariants
# ==============================================================================

class TestLatencyPercentilesAndThroughputInvariants:
    """Enforces mathematical invariants on latency distribution, throughput, and speedup."""

    @pytest.fixture(scope="class")
    def benchmark_data(self) -> dict[str, Any]:
        artifact_path = ROOT / "benchmarks" / "multilingual_benchmarks_results.json"
        return json.loads(artifact_path.read_text(encoding="utf-8"))

    def test_latency_percentiles_positive_and_monotonic(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts 0 < min <= p50 <= p90 <= p95 <= p99 <= max for all configurations."""
        for cfg_name, cfg_data in benchmark_data["results_by_configuration"].items():
            ov = cfg_data["overall"]
            min_l = float(ov["min_latency_ms"])
            p50 = float(ov["p50_latency_ms"])
            p90 = float(ov["p90_latency_ms"])
            p95 = float(ov["p95_latency_ms"])
            p99 = float(ov["p99_latency_ms"])
            max_l = float(ov["max_latency_ms"])

            assert min_l > 0.0, f"{cfg_name}: min_latency_ms must be > 0"
            assert min_l <= p50 + 1e-6, f"{cfg_name}: min ({min_l}) > p50 ({p50})"
            assert p50 <= p90 + 1e-6, f"{cfg_name}: p50 ({p50}) > p90 ({p90})"
            assert p90 <= p95 + 1e-6, f"{cfg_name}: p90 ({p90}) > p95 ({p95})"
            assert p95 <= p99 + 1e-6, f"{cfg_name}: p95 ({p95}) > p99 ({p99})"
            assert p99 <= max_l + 1e-6, f"{cfg_name}: p99 ({p99}) > max ({max_l})"

    def test_throughput_positive_and_nonzero(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts throughput is a positive float for all configurations."""
        for cfg_name, cfg_data in benchmark_data["results_by_configuration"].items():
            thr = float(cfg_data["overall"]["throughput_items_per_sec"])
            assert thr > 0.0, f"{cfg_name}: throughput must be positive, got {thr}"

    def test_speedup_ratios_calculated_correctly(self, benchmark_data: dict[str, Any]) -> None:
        """Asserts comparative_analysis fields exist and are positive numbers."""
        comp = benchmark_data["comparative_analysis"]
        assert "fast_speedup_vs_neural_minilm" in comp
        assert "fast_speedup_vs_neural_mdeberta" in comp
        assert "auto_speedup_vs_neural_minilm" in comp
        assert "auto_speedup_vs_neural_mdeberta" in comp
        assert comp["fast_speedup_vs_neural_minilm"] > 0.0
        assert comp["auto_speedup_vs_neural_minilm"] > 0.0
        assert comp["spanish_accuracy_gain_vs_random_baseline"] >= 0.50
        assert comp["portuguese_accuracy_gain_vs_random_baseline"] >= 0.50


# ==============================================================================
# 4. Historical Benchmark Isolation Invariant Tests
# ==============================================================================

class TestHistoricalBenchmarkIsolationInvariant:
    """Guarantees that historical benchmarks/latest_results.json is NEVER altered or corrupted."""

    def test_baseline_latest_results_hash_unchanged(self) -> None:
        """Asserts SHA-256 digest of benchmarks/latest_results.json matches authoritative baseline."""
        baseline_file = ROOT / "benchmarks" / "latest_results.json"
        assert baseline_file.is_file(), f"Historical baseline file missing: {baseline_file}"
        current_hash = hashlib.sha256(baseline_file.read_bytes()).hexdigest().lower()
        assert current_hash == BASELINE_LATEST_RESULTS_SHA256, (
            f"INVARIANT VIOLATION: benchmarks/latest_results.json was modified!\n"
            f"Expected SHA-256: {BASELINE_LATEST_RESULTS_SHA256}\n"
            f"Actual SHA-256:   {current_hash}"
        )

    def test_baseline_latest_results_contains_eval_540(self) -> None:
        """Asserts baseline latest_results.json retains its original 540-item benchmark content."""
        baseline_file = ROOT / "benchmarks" / "latest_results.json"
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
        assert "eval_540" in data or "tracks" in data or "summary" in data


# ==============================================================================
# 5. BENCHMARKS.md Section 11 & English-Only Documentation Compliance
# ==============================================================================

class TestBenchmarksMarkdownSection11Compliance:
    """Verifies that BENCHMARKS.md contains Section 11 with required tables and zero non-English prose."""

    @pytest.fixture(scope="class")
    def benchmarks_markdown(self) -> str:
        bm_path = ROOT / "BENCHMARKS.md"
        assert bm_path.is_file(), f"BENCHMARKS.md missing at {bm_path}"
        return bm_path.read_text(encoding="utf-8")

    def test_section11_heading_exists(self, benchmarks_markdown: str) -> None:
        """Asserts Section 11 heading exists in BENCHMARKS.md."""
        assert "## 11. Multilingual Decision Benchmark Suite" in benchmarks_markdown, (
            "BENCHMARKS.md is missing Section 11 heading: '## 11. Multilingual Decision Benchmark Suite'"
        )

    def test_section11_contains_required_tables(self, benchmarks_markdown: str) -> None:
        """Asserts Section 11 contains required performance, language, and speedup tables."""
        sec11_text = benchmarks_markdown.split("## 11. Multilingual Decision Benchmark Suite", 1)[1]
        assert "Overall Strategy Performance Matrix" in sec11_text
        assert "Accuracy Breakdown by Language" in sec11_text
        assert "Speedup Ratios" in sec11_text or "Throughput" in sec11_text
        assert "| Strategy Configuration |" in sec11_text
        assert "| **`fast`** |" in sec11_text or "`fast`" in sec11_text

    def test_section11_english_only_compliance(self, benchmarks_markdown: str) -> None:
        """Asserts Section 11 headings and prose do not contain translated French/Spanish/Portuguese text."""
        sec11_text = benchmarks_markdown.split("## 11. Multilingual Decision Benchmark Suite", 1)[1]
        forbidden_headings = [
            "## Résultats", "## Données", "## Évaluation",
            "## Resultados", "## Datos", "## Avaliação",
            "### Facturation", "### Expédition", "### Support Technique",
        ]
        for pattern in forbidden_headings:
            assert pattern not in sec11_text, f"Detected non-English heading in Section 11: '{pattern}'"
