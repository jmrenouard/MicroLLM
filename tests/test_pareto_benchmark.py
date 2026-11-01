"""Comprehensive Tests for Pareto Latency-Accuracy Benchmark Runner (IDEA-048 / Issue #175).

Validates:
1. Execution of `run_pareto_benchmark` across candidate profiles (`fast-cpu-int8`, `multilingual`,
   `deberta-large-int8`, and `decision-gate-cascade`).
2. Percentile metrics (p50, p90, p95, p99), selective accuracy under tau=0.45, escalation rate,
   and Dual-Gear speedup calculation.
3. Markdown report formatting (`format_pareto_benchmark_markdown`).
4. Artifact generation and validation.
5. CLI `--pareto` invocation handling in `microllm.server` and `microllm.pareto_benchmark`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from microllm.pareto_benchmark import (
    DEFAULT_MULTILINGUAL_PROBE_TICKETS,
    format_pareto_benchmark_markdown,
    main as pareto_cli_main,
    run_pareto_benchmark,
)
from microllm.server import main as server_cli_main


def test_pareto_benchmark_quick_run(tmp_path: Path) -> None:
    """Verify Pareto benchmark execution on a slice of probe tickets without disk artifacts."""
    probe_slice = DEFAULT_MULTILINGUAL_PROBE_TICKETS[:8]
    results = run_pareto_benchmark(items=probe_slice, save_artifacts=False)

    assert results["total_items"] == 8
    assert "profiles" in results
    assert "summary_findings" in results
    assert "markdown_table" in results

    summary = results["summary_findings"]
    assert "cascade_speedup_vs_tier3" in summary
    assert "cascade_escalation_rate" in summary
    assert "pareto_optimal_candidates" in summary

    expected_profiles = [
        "fast-cpu-int8",
        "multilingual",
        "deberta-large-int8",
        "decision-gate-cascade",
    ]
    for prof in expected_profiles:
        assert prof in results["profiles"]
        metrics = results["profiles"][prof]
        assert "mean_latency_ms" in metrics
        assert "p50_latency_ms" in metrics
        assert "p90_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics
        assert "selective_accuracy" in metrics
        assert 0.0 <= metrics["selective_accuracy"] <= 1.0
        assert metrics["mean_latency_ms"] >= 0.0
        assert metrics["p50_latency_ms"] >= 0.0

    # Ensure decision-gate-cascade is identified as Pareto optimal
    assert "decision-gate-cascade" in summary["pareto_optimal_candidates"]
    assert 0.0 <= summary["cascade_escalation_rate"] <= 1.0
    assert summary["cascade_speedup_vs_tier3"] > 0.0


def test_format_pareto_benchmark_markdown() -> None:
    """Verify that format_pareto_benchmark_markdown generates a complete, valid table."""
    probe_slice = DEFAULT_MULTILINGUAL_PROBE_TICKETS[:4]
    results = run_pareto_benchmark(items=probe_slice, save_artifacts=False)
    md = format_pareto_benchmark_markdown(results["profiles"], results["summary_findings"])

    assert "Comparative Latency & Accuracy Pareto Benchmark (IDEA-048)" in md
    assert "| Profile / Architecture | Tier / Model Family" in md
    assert "decision-gate-cascade" in md
    assert "deberta-large-int8" in md
    assert "multilingual" in md
    assert "fast-cpu-int8" in md
    assert "Pareto Analysis & Engineering Takeaways" in md


def test_pareto_benchmark_artifact_persistence(tmp_path: Path) -> None:
    """Verify saving benchmark results to JSON file."""
    output_path = tmp_path / "pareto_results.json"
    results = run_pareto_benchmark(
        items=DEFAULT_MULTILINGUAL_PROBE_TICKETS[:4],
        save_artifacts=True,
        output_path=output_path,
    )

    assert output_path.exists()
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["total_items"] == 4
    assert saved_data["profiles"] == results["profiles"]


def test_pareto_standalone_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify standalone `microllm.pareto_benchmark --quick` CLI execution."""
    exit_code = pareto_cli_main(["--quick"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Comparative Latency & Accuracy Pareto Benchmark" in captured.out
    assert "decision-gate-cascade" in captured.out


def test_server_cli_pareto_flag_invocation(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI `microllm benchmark --pareto --quick` executes via server.py."""
    code = server_cli_main(["benchmark", "--pareto", "--quick"])
    assert code == 0

    captured = capsys.readouterr()
    assert "Comparative Latency & Accuracy Pareto Benchmark" in captured.out
    assert "decision-gate-cascade" in captured.out
