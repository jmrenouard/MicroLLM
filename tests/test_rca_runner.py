"""Unit tests for the Automated Sequential RCA Verification Runner (src/microllm/rca_runner.py)."""

from __future__ import annotations

import pytest

from microllm.rca_runner import (
    RCA_CATALOG,
    RCARecord,
    RCARunner,
    RCARunnerResult,
)


def test_rca_catalog_has_all_15_registered_rcas() -> None:
    """Verify that RCA_CATALOG contains RCA-001 through RCA-015."""
    assert len(RCA_CATALOG) == 15
    for i in range(1, 16):
        rca_id = f"RCA-{i:03d}"
        assert rca_id in RCA_CATALOG, f"Missing {rca_id} in catalog"
        rec = RCA_CATALOG[rca_id]
        assert isinstance(rec, RCARecord)
        assert rec.id == rca_id
        assert len(rec.title) > 0
        assert len(rec.target_module) > 0
        assert rec.accuracy_sla >= 0.90
        assert rec.latency_p50_sla_ms > 0.0
        assert callable(rec.verifier)


def test_rca_runner_list_rcas() -> None:
    """Verify list_rcas() returns catalog summaries."""
    runner = RCARunner()
    items = runner.list_rcas()
    assert len(items) == 15
    first = items[0]
    assert first["id"] == "RCA-001"
    assert "accuracy_sla" in first
    assert "latency_p50_sla_ms" in first


def test_rca_runner_verify_single_rca_001() -> None:
    """Verify non-regression for RCA-001 (exemption clauses)."""
    runner = RCARunner()
    res = runner.verify_single("RCA-001", warmup=1, iterations=3)
    assert isinstance(res, RCARunnerResult)
    assert res.rca_id == "RCA-001"
    assert res.status == "PASS"
    assert res.non_regression_passed is True
    assert res.accuracy == 1.0
    assert res.latency_p50_ms < 5.0
    assert res.details.get("passed") is True


def test_rca_runner_verify_single_rca_002() -> None:
    """Verify non-regression for RCA-002 (order invariance & zero drift)."""
    runner = RCARunner()
    res = runner.verify_single("RCA-002", warmup=1, iterations=3)
    assert res.rca_id == "RCA-002"
    assert res.status == "PASS"
    assert res.non_regression_passed is True
    assert res.details.get("same_choice") is True
    assert res.details.get("max_drift", 1.0) < 1e-6


def test_rca_runner_verify_single_rca_004() -> None:
    """Verify non-regression for RCA-004 (NaN / Infinity rejection)."""
    runner = RCARunner()
    res = runner.verify_single("RCA-004", warmup=1, iterations=2)
    assert res.rca_id == "RCA-004"
    assert res.status == "PASS"
    assert res.details.get("nan_rejected") is True
    assert res.details.get("inf_rejected") is True


def test_rca_runner_sequential_execution_and_markdown() -> None:
    """Verify sequential execution of a subset with callback and markdown formatting."""
    runner = RCARunner()
    completed: list[str] = []

    def on_complete(r: RCARunnerResult) -> None:
        completed.append(r.rca_id)

    subset = ["RCA-001", "RCA-003", "RCA-004", "RCA-011", "RCA-013"]
    results = runner.run_sequential(
        rca_ids=subset, iterations_per_rca=2, on_rca_complete=on_complete
    )

    assert len(results) == 5
    assert completed == subset
    assert all(r.status == "PASS" for r in results)

    md = runner.format_report_markdown(results)
    assert "# MicroLLM RCA Non-Regression Verification Report" in md
    for rca_id in subset:
        assert rca_id in md
    assert "✅ PASS" in md
    assert "Overall Summary: 5/5 Passed" in md


def test_rca_runner_unknown_rca_raises_key_error() -> None:
    """Verify KeyError when requesting non-existent RCA."""
    runner = RCARunner()
    with pytest.raises(KeyError, match="Unknown RCA 'RCA-999'"):
        runner.verify_single("RCA-999")


def test_rca_cli_list_and_single_run(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify microllm rca CLI subcommand for --list and single RCA run."""
    from microllm.server import run_cli

    code_list = run_cli(["rca", "--list"])
    assert code_list == 0
    out_list = capsys.readouterr().out
    assert "RCA-001" in out_list
    assert "RCA-015" in out_list

    code_single = run_cli(["rca", "--rca", "RCA-002", "--iterations", "1"])
    assert code_single == 0
    out_single = capsys.readouterr().out
    assert "RCA-002" in out_single
    assert "GUARANTEED" in out_single


def test_rca_runner_verify_single_rca_012_uses_per_decision_latency_under_coverage() -> None:
    """Verify RCA-012 uses per-decision p50/p95 latency and does not fail when suite wall-clock exceeds 12s under coverage."""
    runner = RCARunner()
    rec_012 = RCA_CATALOG["RCA-012"]
    assert rec_012.latency_p50_sla_ms == 10.0

    res = runner.verify_single("RCA-012", warmup=0, iterations=1)
    assert res.rca_id == "RCA-012"
    assert res.status == "PASS"
    assert res.non_regression_passed is True
    assert res.latency_p50_ms <= 10.0
    assert res.latency_p95_ms <= 30.0
    assert res.details.get("passed") is True


