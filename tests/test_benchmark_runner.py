"""Unit and integration tests for the unified benchmark automation runner (`microllm.benchmark_runner`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microllm.benchmark_runner import (
    ALL_BENCHMARK_SUITE_IDS,
    FAST_BENCHMARK_SUITE_IDS,
    collect_system_environment,
    format_master_benchmark_markdown,
    run_all_benchmark_suites,
)
from microllm.server import run_cli


def test_collect_system_environment_returns_valid_telemetry() -> None:
    """Verify hardware and runtime telemetry dictionary structure."""
    env = collect_system_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "cpu_count_logical" in env
    assert int(env["cpu_count_logical"]) >= 1
    assert "numpy_version" in env
    assert "timestamp_utc" in env


def test_suite_id_constants_are_consistent() -> None:
    """Verify suite registry constants and fast-mode subset."""
    assert "eval_540" in ALL_BENCHMARK_SUITE_IDS
    assert "quality_750" in ALL_BENCHMARK_SUITE_IDS
    assert "quality_1000" in ALL_BENCHMARK_SUITE_IDS
    assert "test_cases_1000" in ALL_BENCHMARK_SUITE_IDS
    assert "test_cases_11000" in ALL_BENCHMARK_SUITE_IDS
    assert "web_external_curated" in ALL_BENCHMARK_SUITE_IDS
    assert "web_external_full_splits" in ALL_BENCHMARK_SUITE_IDS
    assert "research_hypotheses" in ALL_BENCHMARK_SUITE_IDS
    assert "rca_non_regression" in ALL_BENCHMARK_SUITE_IDS
    assert "web_external_full_splits" not in FAST_BENCHMARK_SUITE_IDS
    assert set(FAST_BENCHMARK_SUITE_IDS).issubset(set(ALL_BENCHMARK_SUITE_IDS))


def test_run_selected_benchmark_suites_and_format_markdown(tmp_path: Path) -> None:
    """Execute a targeted subset of suites into a temporary directory and verify artifacts."""
    report = run_all_benchmark_suites(
        suites=["eval_540", "research_hypotheses", "rca_non_regression"],
        include_full_splits=False,
        output_dir=tmp_path,
        save_artifacts=True,
    )
    assert report["all_suites_passed"] is True
    assert report["total_suites_executed"] == 3
    assert "eval_540" in report["suites"]
    assert "research_hypotheses" in report["suites"]
    assert "rca_non_regression" in report["suites"]

    master_file = tmp_path / "all_benchmarks_master_report.json"
    assert master_file.exists()
    loaded = json.loads(master_file.read_text(encoding="utf-8"))
    assert loaded["total_suites_executed"] == 3

    md = format_master_benchmark_markdown(report)
    assert "MicroLLM Unified Master Benchmark Report" in md
    assert "eval_540" in md
    assert "research_hypotheses" in md
    assert "rca_non_regression" in md


def test_run_all_benchmark_suites_rejects_unknown_suite(tmp_path: Path) -> None:
    """Ensure unknown suite identifiers raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown benchmark suite"):
        run_all_benchmark_suites(
            suites=["non_existent_suite"],
            output_dir=tmp_path,
            save_artifacts=False,
        )


def test_cli_benchmark_all_suites_fast_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `microllm benchmark --all-suites --suites ...` CLI integration."""
    exit_code = run_cli(
        [
            "benchmark",
            "--all-suites",
            "--suites",
            "research_hypotheses,rca_non_regression",
            "--output",
            str(tmp_path / "custom_master.json"),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "MicroLLM Unified Master Benchmark Report" in captured
    assert (tmp_path / "custom_master.json").exists()


def test_automation_scripts_exist_in_scripts_dir() -> None:
    """Ensure reusable automation scripts exist in `scripts/` so benchmarks can be launched in 1 command."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "scripts" / "run_all_benchmarks.py").exists()
    assert (repo_root / "scripts" / "run_all_benchmarks.ps1").exists()
    assert (repo_root / "scripts" / "run_all_benchmarks.sh").exists()


def test_load_full_external_dataset_split_preserves_cache_on_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RCA-029: Missing optional `pyarrow` (`ImportError`) must never unlink cached Parquet files."""
    import microllm.dataset_adapters as da

    cached_parquet = tmp_path / "google__boolq.parquet"
    cached_parquet.write_bytes(b"PAR1_FAKE_VALID_CACHE_PAYLOAD")

    def _raise_import_error(file_path: Path, fmt: str) -> list[dict[str, object]]:
        raise ModuleNotFoundError("No module named 'pyarrow'")

    monkeypatch.setattr(da, "_parse_cached_split_file", _raise_import_error)
    rows = da.load_full_external_dataset_split(
        "google/boolq",
        cache_dir=tmp_path,
        allow_network=False,
    )
    assert len(rows) == 10
    assert cached_parquet.exists(), "Cached Parquet split must not be unlinked on ImportError!"

