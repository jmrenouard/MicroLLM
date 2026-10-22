"""Unit and integration tests for the unified benchmark automation runner (`microllm.benchmark_runner`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microllm.benchmark_runner import (
    ALL_BENCHMARK_SUITE_IDS,
    ALL_BENCHMARK_SUITES_SPEC,
    FAST_BENCHMARK_SUITE_IDS,
    collect_system_environment,
    format_master_benchmark_markdown,
     main as runner_main,
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


def test_suite_id_constants_and_spec_are_consistent() -> None:
    """Verify suite registry constants, ALL_BENCHMARK_SUITES_SPEC, and fast-mode subset."""
    import microllm

    assert hasattr(microllm, "ALL_BENCHMARK_SUITES_SPEC")
    assert len(ALL_BENCHMARK_SUITES_SPEC) == 10
    assert tuple(s["suite_id"] for s in ALL_BENCHMARK_SUITES_SPEC) == ALL_BENCHMARK_SUITE_IDS
    assert "eval_540" in ALL_BENCHMARK_SUITE_IDS
    assert "quality_750" in ALL_BENCHMARK_SUITE_IDS
    assert "quality_1000" in ALL_BENCHMARK_SUITE_IDS
    assert "test_cases_1000" in ALL_BENCHMARK_SUITE_IDS
    assert "test_cases_11000" in ALL_BENCHMARK_SUITE_IDS
    assert "web_external_curated" in ALL_BENCHMARK_SUITE_IDS
    assert "hf_jev_models_ecosystem" in ALL_BENCHMARK_SUITE_IDS
    assert "web_external_full_splits" in ALL_BENCHMARK_SUITE_IDS
    assert "research_hypotheses" in ALL_BENCHMARK_SUITE_IDS
    assert "rca_non_regression" in ALL_BENCHMARK_SUITE_IDS
    assert "hf_jev_models_ecosystem" in FAST_BENCHMARK_SUITE_IDS
    assert "web_external_full_splits" not in FAST_BENCHMARK_SUITE_IDS
    assert set(FAST_BENCHMARK_SUITE_IDS).issubset(set(ALL_BENCHMARK_SUITE_IDS))


def test_run_selected_benchmark_suites_and_format_markdown(tmp_path: Path) -> None:
    """Execute a targeted subset of suites into a temporary directory and verify artifacts and callbacks."""
    visited: list[str] = []
    custom_out = tmp_path / "custom_out.json"
    report = run_all_benchmark_suites(
        suites=["eval_540", "research_hypotheses", "rca_non_regression"],
        quick=True,
        output_dir=tmp_path,
        output_path=custom_out,
        save_artifacts=True,
        progress_callback=lambda s_id, summary: visited.append(s_id),
    )
    assert report["all_suites_passed"] is True
    assert report["total_suites_executed"] == 3
    assert visited == ["eval_540", "research_hypotheses", "rca_non_regression"]
    assert "eval_540" in report["suites"]
    assert "research_hypotheses" in report["suites"]
    assert "rca_non_regression" in report["suites"]

    assert custom_out.exists()
    loaded = json.loads(custom_out.read_text(encoding="utf-8"))
    assert loaded["total_suites_executed"] == 3

    md = format_master_benchmark_markdown(report)
    assert "MicroLLM Unified Master Benchmark Report" in md
    assert "eval_540" in md
    assert "research_hypotheses" in md
    assert "rca_non_regression" in md


def test_partial_suite_run_does_not_clobber_canonical_master_report() -> None:
    """Ensure running a filtered subset without --output does not overwrite benchmarks/all_benchmarks_master_report.json."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_master = repo_root / "benchmarks" / "all_benchmarks_master_report.json"
    before_text = canonical_master.read_text(encoding="utf-8")

    report = run_all_benchmark_suites(
        suites=["research_hypotheses"],
        save_artifacts=True,
    )
    assert report["total_suites_executed"] == 1
    after_text = canonical_master.read_text(encoding="utf-8")
    assert after_text == before_text
    after_json = json.loads(after_text)
    assert after_json["total_suites_executed"] == 10
    assert "hf_jev_models_ecosystem" in after_json["suites"]


def test_hf_jev_models_ecosystem_suite_exports_corpora_and_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `hf_jev_models_ecosystem` suite and `--hf-jev-models` flag export JSONL corpora and JSON results."""
    report = run_all_benchmark_suites(
        suites=["hf_jev_models_ecosystem"],
        output_dir=tmp_path,
        save_artifacts=True,
    )
    assert report["all_suites_passed"] is True
    assert report["total_suites_executed"] == 1
    suite_info = report["suites"]["hf_jev_models_ecosystem"]
    assert suite_info["total_items"] == 120
    assert suite_info["primary_accuracy"] == 1.0
    assert suite_info["passed"] is True

    results_json = tmp_path / "hf_jev_models_benchmarks_results.json"
    jev_corpus_jsonl = tmp_path / "hf_jev_models_corpus.jsonl"
    combined_28_jsonl = tmp_path / "external_datasets_28_corpus.jsonl"
    assert results_json.is_file()
    assert jev_corpus_jsonl.is_file()
    assert combined_28_jsonl.is_file()

    jev_lines = [ln for ln in jev_corpus_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    combined_lines = [ln for ln in combined_28_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(jev_lines) == 480
    assert len(combined_lines) == 1120

    # Also verify `--hf-jev-models` CLI flag on benchmark_runner
    out_dir_2 = tmp_path / "cli_run"
    rc = runner_main(["--hf-jev-models", "--output-dir", str(out_dir_2), "--format", "json"])
    assert rc == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["total_suites_executed"] == 1
    assert "hf_jev_models_ecosystem" in cli_payload["suites"]
    assert (out_dir_2 / "hf_jev_models_corpus.jsonl").is_file()
    assert (out_dir_2 / "external_datasets_28_corpus.jsonl").is_file()


def test_run_all_benchmark_suites_rejects_unknown_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ensure unknown suite identifiers raise ValueError programmatically and return exit code 2 via CLI."""
    with pytest.raises(ValueError, match="Unknown benchmark suite"):
        run_all_benchmark_suites(
            suites=["non_existent_suite"],
            output_dir=tmp_path,
            save_artifacts=False,
        )

    cli_code = run_cli(["benchmark", "--all-suites", "--suites", "non_existent_suite"])
    assert cli_code == 2
    err_out = capsys.readouterr().err
    assert "Unknown benchmark suite" in err_out

    runner_code = runner_main(["--suites", "non_existent_suite"])
    assert runner_code == 2
    err_out2 = capsys.readouterr().err
    assert "Unknown benchmark suite" in err_out2


def test_cli_benchmark_all_suites_fast_and_quick_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify `microllm benchmark --all-suites --quick --suites ...` CLI integration."""
    exit_code = run_cli(
        [
            "benchmark",
            "--all-suites",
            "--quick",
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

    md_file = tmp_path / "report.md"
    json_code = runner_main(
        [
            "--quick",
            "--suites",
            "research_hypotheses",
            "--output-dir",
            str(tmp_path),
            "--markdown-out",
            str(md_file),
            "--format",
            "json",
        ]
    )
    assert json_code == 0
    assert md_file.exists()
    json_stdout = json.loads(capsys.readouterr().out)
    assert json_stdout["total_suites_executed"] == 1


def test_automation_scripts_and_documented_aliases_exist() -> None:
    """Ensure reusable automation scripts in `scripts/` and documented helper aliases exist."""
    from microllm.quality_suite import format_quality_suite_markdown
    from microllm.research_challenge import (
        evaluate_all_12_hypotheses,
        run_all_hypotheses_challenge,
    )

    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "scripts" / "run_all_benchmarks.py").exists()
    assert (repo_root / "scripts" / "run_all_benchmarks.ps1").exists()
    assert (repo_root / "scripts" / "run_all_benchmarks.sh").exists()
    assert callable(format_quality_suite_markdown)
    hyp_res = evaluate_all_12_hypotheses()
    assert hyp_res["passed_count"] == 12
    assert hyp_res["passed_hypotheses"] == 12
    assert callable(run_all_hypotheses_challenge)


def test_load_full_external_dataset_split_preserves_valid_cache_on_import_error_and_evicts_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RCA-037: Valid PAR1 Parquet files must be preserved on ImportError while corrupt files are evicted."""
    import builtins
    import microllm.dataset_adapters as da

    real_import = builtins.__import__

    def _mock_no_pyarrow(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _mock_no_pyarrow)

    # 1. Valid PAR1 header + footer -> preserved when pyarrow is missing
    valid_parquet = tmp_path / "google__boolq.parquet"
    valid_parquet.write_bytes(b"PAR1_FAKE_VALID_CACHE_PAYLOAD_PAR1")
    rows = da.load_full_external_dataset_split(
        "google/boolq",
        cache_dir=tmp_path,
        allow_network=False,
    )
    assert len(rows) == 10
    assert valid_parquet.exists(), "Valid PAR1 cached Parquet split must not be unlinked on ImportError!"

    # 2. Corrupted / truncated file without PAR1 magic bytes -> evicted even when pyarrow is missing
    valid_parquet.write_bytes(b"NOT_A_VALID_PARQUET_FILE_TRUNCATED")
    rows_corrupt = da.load_full_external_dataset_split(
        "google/boolq",
        cache_dir=tmp_path,
        allow_network=False,
    )
    assert len(rows_corrupt) == 10
    assert not valid_parquet.exists(), "Corrupted Parquet file must be evicted even without pyarrow!"


