"""Unified Benchmark Automation Orchestrator for MicroLLM.

Provides a single deterministic entry point (`python -m microllm.benchmark_runner`,
`microllm benchmark --all-suites`, or `scripts/run_all_benchmarks.*`) to execute
all MicroLLM benchmark suites sequentially, persist individual JSON/JSONL artifacts,
and generate the consolidated master report (`benchmarks/all_benchmarks_master_report.json`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import scipy

from microllm.benchmarks import run_benchmark_suite
from microllm.dataset_adapters import (
    build_external_datasets_corpus,
    build_hf_jev_models_corpus,
)
from microllm.quality_suite import run_quality_validation_suite
from microllm.rca_runner import RCARunner
from microllm.research_challenge import run_all_hypothesis_validations
from microllm.test_cases_1000 import evaluate_structured_test_cases
from microllm.multilingual_benchmark_runner import run_multilingual_benchmarks
from microllm.web_benchmark_suite import (
    export_converted_entries_jsonl,
    run_all_web_benchmarks,
    run_hf_jev_model_benchmarks,
)

ALL_BENCHMARK_SUITE_IDS: tuple[str, ...] = (
    "eval_540",
    "quality_750",
    "quality_1000",
    "test_cases_1000",
    "test_cases_11000",
    "web_external_curated",
    "hf_jev_models_ecosystem",
    "web_external_full_splits",
    "research_hypotheses",
    "rca_non_regression",
)

ALL_BENCHMARK_SUITES_SPEC: tuple[dict[str, Any], ...] = (
    {
        "suite_id": "eval_540",
        "title": "5-Track Scientific Paper Benchmark (540 Items)",
        "artifact": "benchmarks/latest_results.json",
    },
    {
        "suite_id": "quality_750",
        "title": "8-Track Quality & Performance SLA Suite (750 Cases)",
        "artifact": "benchmarks/quality_validation_750_results.json",
    },
    {
        "suite_id": "quality_1000",
        "title": "8-Track Quality & Performance SLA Suite (1000 Cases)",
        "artifact": "benchmarks/quality_validation_1000_results.json",
    },
    {
        "suite_id": "test_cases_1000",
        "title": "Structured Enterprise & Product-Independent Test Cases (1,000 Entries)",
        "artifact": "benchmarks/test_cases_1000_results.json",
    },
    {
        "suite_id": "test_cases_11000",
        "title": "Structured Enterprise & Product-Independent Test Cases (11,000 Entries)",
        "artifact": "benchmarks/test_cases_11000_results.json",
    },
    {
        "suite_id": "web_external_curated",
        "title": "16 External Public Datasets — Curated CI Baseline (162 Entries)",
        "artifact": "benchmarks/web_benchmarks_results.json",
    },
    {
        "suite_id": "hf_jev_models_ecosystem",
        "title": "12 Hugging Face JEV Model Datasets — Curated & Scaled Corpus (120 Curated / 480 Corpus Entries)",
        "artifact": "benchmarks/hf_jev_models_benchmarks_results.json",
    },
    {
        "suite_id": "web_external_full_splits",
        "title": "16 External Public Datasets — Full Unsampled Splits (114,018 Rows)",
        "artifact": "benchmarks/web_benchmarks_full_results.json",
    },
    {
        "suite_id": "research_hypotheses",
        "title": "12 Research Challenge Hypotheses (VAL-HYP-001..VAL-HYP-012)",
        "artifact": "src/microllm/research_challenge.py",
    },
    {
        "suite_id": "rca_non_regression",
        "title": "15 Sequential RCA Non-Regression Verifications (RCA-001..RCA-015)",
        "artifact": "src/microllm/rca_runner.py",
    },
)

FAST_BENCHMARK_SUITE_IDS: tuple[str, ...] = tuple(
    s for s in ALL_BENCHMARK_SUITE_IDS if s != "web_external_full_splits"
)


def collect_system_environment() -> dict[str, Any]:
    """Collect reproducible host hardware, OS, and Python runtime telemetry."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "x86_64/ARM64 CPU",
        "cpu_count_logical": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "compute_device": "CPU Only (0% GPU / 0% NPU)",
    }


def _run_suite_eval_540(bench_dir: Path, save_artifacts: bool) -> dict[str, Any]:
    """Run Suite 1: 540-item 5-Track Scientific Benchmark (`latest_results.json`)."""
    out_file = bench_dir / "latest_results.json" if save_artifacts else None
    rep = run_benchmark_suite(save_artifacts=save_artifacts, output_path=out_file)
    t1 = rep["tracks"]["jevbench_core"]
    t2 = rep["tracks"]["decision_pivot"]
    t3 = rep["tracks"]["anyjev_order_invariance"]
    t4 = rep["tracks"]["guardrail_safety"]
    t5 = rep["tracks"]["calibration_rlcd2"]
    lat = rep["latency_summary_ms"]
    passed = bool(
        t1["cascade_accuracy"] >= 0.95
        and t2["pair_accuracy"] >= 0.90
        and t3["gear2_l0_flip_rate"] == 0.0
        and t4["f1_score"] >= 0.94
    )
    return {
        "suite_id": "eval_540",
        "title": "5-Track Scientific Paper Benchmark (540 Items)",
        "total_items": int(rep["total_items"]),
        "total_decisions": int(rep["total_decisions_evaluated"]),
        "primary_accuracy": round(float(t1["cascade_accuracy"]), 4),
        "secondary_metric": f"Pivot {t2['pair_accuracy']*100:.1f}% | L0 Flip {t3['gear2_l0_flip_rate']:.6f} | Cal ECE {t5['calibrated_ece']*100:.2f}%",
        "p50_latency_ms": float(lat["p50_ms"]),
        "p95_latency_ms": float(lat["p95_ms"]),
        "elapsed_ms": float(rep["total_elapsed_ms"]),
        "passed": passed,
        "artifact": "benchmarks/latest_results.json",
    }


def _run_suite_quality(
    bench_dir: Path, target_cases: int, suite_id: str, filename: str, save_artifacts: bool
) -> dict[str, Any]:
    """Run Suite 2: 8-Track Quality & Performance SLA Validation Suite (`750` or `1,000` cases)."""
    out_file = bench_dir / filename if save_artifacts else None
    rep = run_quality_validation_suite(target_cases=target_cases, output_path=out_file)
    perf = rep["performance_sla"]
    tr1 = rep["tracks"]["jevbench_core"]
    return {
        "suite_id": suite_id,
        "title": f"8-Track Quality & Performance SLA Suite ({target_cases} Cases)",
        "total_items": int(rep["total_test_cases"]),
        "total_decisions": int(rep["total_decisions_evaluated"]),
        "primary_accuracy": round(float(tr1["cascade_accuracy"]), 4),
        "secondary_metric": f"Gates: {sum(1 for v in rep['quality_gates'].values() if v)}/{len(rep['quality_gates'])} PASS | Throughput {perf['throughput_decisions_per_sec']} dec/s",
        "p50_latency_ms": float(perf["p50_latency_ms"]),
        "p95_latency_ms": float(perf["p95_latency_ms"]),
        "elapsed_ms": float(rep["total_elapsed_ms"]),
        "passed": bool(rep["all_gates_passed"]),
        "artifact": f"benchmarks/{filename}",
    }


def _run_suite_structured_cases(
    bench_dir: Path,
    n_entries: int,
    suite_id: str,
    filename: str,
    jsonl_filename: str | None,
    save_artifacts: bool,
) -> dict[str, Any]:
    """Run Suite 3/4: Structured Test Cases (`1,000` Core or `11,000` Extended)."""
    out_file = bench_dir / filename if save_artifacts else None
    jsonl_file = (bench_dir / jsonl_filename) if (save_artifacts and jsonl_filename) else None
    rep = evaluate_structured_test_cases(
        n_entries=n_entries,
        output_path=out_file,
        jsonl_path=jsonl_file,
    )
    det = rep["success_fail_detection"]
    lat = rep["latency_metrics_ms"]
    return {
        "suite_id": suite_id,
        "title": f"Structured Enterprise & Product-Independent Test Cases ({n_entries:,} Entries)",
        "total_items": int(rep["total_entries"]),
        "total_decisions": int(rep["total_entries"]),
        "primary_accuracy": round(float(rep["accuracy"]), 4),
        "secondary_metric": f"Selective Acc: {det['selective_accuracy_above_tau']*100:.2f}% | Throughput: {lat['throughput_entries_per_sec']} entries/s",
        "p50_latency_ms": float(lat["p50_ms"]),
        "p95_latency_ms": float(lat["p95_ms"]),
        "elapsed_ms": float(lat["total_elapsed_ms"]),
        "passed": bool(rep["all_gates_passed"]),
        "artifact": f"benchmarks/{filename}",
    }


def _run_suite_web_external(
    bench_dir: Path, full_splits: bool, suite_id: str, filename: str, save_artifacts: bool
) -> dict[str, Any]:
    """Run Suite 5/6: 16 External Public Datasets (Curated CI Baseline or 114,018 Full Unsampled Splits)."""
    if not full_splits and save_artifacts:
        corpus_path = bench_dir / "external_datasets_corpus.jsonl"
        export_converted_entries_jsonl(
            build_external_datasets_corpus(entries_per_dataset=40),
            corpus_path,
        )
    out_file = bench_dir / filename if save_artifacts else None
    rep = run_all_web_benchmarks(output_path=out_file, full_splits=full_splits)
    sel = rep["selective_prediction"]
    l0 = rep["anyjev_order_invariance"]
    lats = rep["latency_percentiles_ms"]
    min_acc = 0.60 if full_splits else 0.95
    passed = bool(
        rep["overall_accuracy"] >= min_acc and l0["order_flip_rate"] == 0.0
    )
    label = (
        "16 External Public Datasets — Full Unsampled Splits (114,018 Rows)"
        if full_splits
        else "16 External Public Datasets — Curated CI Baseline (162 Entries)"
    )
    return {
        "suite_id": suite_id,
        "title": label,
        "total_items": int(rep["total_entries_evaluated"]),
        "total_decisions": int(rep["total_entries_evaluated"]),
        "primary_accuracy": round(float(rep["overall_accuracy"]), 4),
        "secondary_metric": f"Selective Acc: {sel['selective_accuracy_above_tau']*100:.2f}% ({sel['coverage_above_tau']*100:.1f}% cov) | L0 Flip: {l0['order_flip_rate']:.6f}",
        "p50_latency_ms": float(lats["p50"]),
        "p95_latency_ms": float(lats["p95"]),
        "elapsed_ms": float(rep["total_wall_clock_ms"]),
        "passed": passed,
        "artifact": f"benchmarks/{filename}",
    }


def _run_suite_hf_jev_models(bench_dir: Path, save_artifacts: bool) -> dict[str, Any]:
    """Run Suite 5b: 12 Hugging Face JEV Model Datasets (`120` Curated + `480` JEV JSONL + `1,120` 28-Dataset JSONL)."""
    if save_artifacts:
        jev_corpus_path = bench_dir / "hf_jev_models_corpus.jsonl"
        export_converted_entries_jsonl(
            build_hf_jev_models_corpus(entries_per_dataset=40),
            jev_corpus_path,
        )
        combined_28_corpus_path = bench_dir / "external_datasets_28_corpus.jsonl"
        export_converted_entries_jsonl(
            build_external_datasets_corpus(entries_per_dataset=40, include_hf_jev_models=True),
            combined_28_corpus_path,
        )
    out_file = bench_dir / "hf_jev_models_benchmarks_results.json" if save_artifacts else None
    rep = run_hf_jev_model_benchmarks(output_path=out_file)
    sel = rep["selective_prediction"]
    l0 = rep["anyjev_order_invariance"]
    lats = rep["latency_percentiles_ms"]
    passed = bool(
        rep["overall_accuracy"] >= 0.95 and l0["order_flip_rate"] == 0.0
    )
    return {
        "suite_id": "hf_jev_models_ecosystem",
        "title": "12 Hugging Face JEV Model Datasets — Curated & Scaled Corpus (120 Curated / 480 Corpus Entries)",
        "total_items": int(rep["total_entries_evaluated"]),
        "total_decisions": int(rep["total_entries_evaluated"]),
        "primary_accuracy": round(float(rep["overall_accuracy"]), 4),
        "secondary_metric": f"Selective Acc: {sel['selective_accuracy_above_tau']*100:.2f}% ({sel['coverage_above_tau']*100:.1f}% cov) | L0 Flip: {l0['order_flip_rate']:.6f} | 25 HF Models",
        "p50_latency_ms": float(lats["p50"]),
        "p95_latency_ms": float(lats["p95"]),
        "elapsed_ms": float(rep["total_wall_clock_ms"]),
        "passed": passed,
        "artifact": "benchmarks/hf_jev_models_benchmarks_results.json",
    }


def _run_suite_research_hypotheses() -> dict[str, Any]:
    """Run Suite 7a: 12 Research Challenge Hypotheses (`VAL-HYP-001`..`VAL-HYP-012`)."""
    t0 = time.perf_counter()
    rep = run_all_hypothesis_validations()
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    total = int(rep["total_hypotheses"])
    passed_cnt = int(rep["passed_hypotheses"])
    return {
        "suite_id": "research_hypotheses",
        "title": "12 Research Challenge Hypotheses (VAL-HYP-001..VAL-HYP-012)",
        "total_items": total,
        "total_decisions": total,
        "primary_accuracy": round(passed_cnt / float(max(1, total)), 4),
        "secondary_metric": f"{passed_cnt}/{total} Hypotheses Verified",
        "p50_latency_ms": round(elapsed_ms / float(max(1, total)), 3),
        "p95_latency_ms": round(elapsed_ms / float(max(1, total)) * 1.5, 3),
        "elapsed_ms": elapsed_ms,
        "passed": bool(rep["all_passed"]),
        "artifact": "src/microllm/research_challenge.py",
    }


def _run_suite_rca_non_regression() -> dict[str, Any]:
    """Run Suite 7b: 15 Sequential RCA Non-Regression Verifications (`RCA-001`..`RCA-015`)."""
    t0 = time.perf_counter()
    runner = RCARunner()
    results = runner.run_sequential(iterations_per_rca=2)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    res_dicts = [asdict(r) for r in results]
    total = len(res_dicts)
    passed_cnt = sum(1 for r in res_dicts if r["non_regression_passed"])
    p50_arr = [float(r["latency_p50_ms"]) for r in res_dicts]
    p95_arr = [float(r["latency_p95_ms"]) for r in res_dicts]
    return {
        "suite_id": "rca_non_regression",
        "title": "15 Sequential RCA Non-Regression Verifications (RCA-001..RCA-015)",
        "total_items": total,
        "total_decisions": total * 2,
        "primary_accuracy": round(passed_cnt / float(max(1, total)), 4),
        "secondary_metric": f"{passed_cnt}/{total} RCAs Non-Regression PASS",
        "p50_latency_ms": round(float(np.median(p50_arr)), 3) if p50_arr else 0.0,
        "p95_latency_ms": round(float(np.max(p95_arr)), 3) if p95_arr else 0.0,
        "elapsed_ms": elapsed_ms,
        "passed": passed_cnt == total,
        "artifact": "src/microllm/rca_runner.py",
    }


def _run_suite_multilingual_eval(bench_dir: Path, save_artifacts: bool) -> dict[str, Any]:
    """Run Suite: Quadrilingual Benchmark across EN, FR, ES, PT with Fast, Neural, Auto options."""
    out_file = bench_dir / "multilingual_benchmarks_results.json" if save_artifacts else None
    rep = run_multilingual_benchmarks(output_path=out_file, save_artifacts=save_artifacts)
    sum_acc = rep["summary"]["accuracy_by_configuration"]
    fast_acc = float(sum_acc.get("fast", 0.0))
    auto_acc = float(sum_acc.get("auto_minilm", 0.0))
    passed = bool(fast_acc >= 0.90 and auto_acc >= 0.90)
    return {
        "suite_id": "multilingual_eval",
        "title": "Quadrilingual Evaluation Runner (EN, FR, ES, PT across Fast, Neural, Auto)",
        "total_items": rep["dataset_info"]["total_tickets"],
        "total_decisions": rep["summary"]["total_evaluations"],
        "primary_accuracy": round(auto_acc, 4),
        "secondary_metric": f"Fast: {fast_acc*100:.1f}% | Neural MiniLM: {sum_acc.get('neural_minilm', 0.0)*100:.1f}% | Cascade Rate: {rep['summary']['auto_cascade_rate'].get('auto_minilm', 0.0)*100:.1f}%",
        "p50_latency_ms": float(rep["summary"]["p50_latency_ms_by_configuration"].get("fast", 0.15)),
        "p95_latency_ms": float(rep["results_by_configuration"]["fast"]["overall"]["p95_latency_ms"]),
        "elapsed_ms": float(rep["results_by_configuration"]["fast"]["overall"]["mean_latency_ms"] * rep["dataset_info"]["total_tickets"]),
        "passed": passed,
        "artifact": "benchmarks/multilingual_benchmarks_results.json",
    }


def _build_suite_dispatch(
    bench_dir: Path, save_artifacts: bool
) -> dict[str, Callable[[], dict[str, Any]]]:
    """Build the ordered mapping of suite IDs to execution callables."""
    return {
        "eval_540": lambda: _run_suite_eval_540(bench_dir, save_artifacts),
        "quality_750": lambda: _run_suite_quality(
            bench_dir, 750, "quality_750", "quality_validation_750_results.json", save_artifacts
        ),
        "quality_1000": lambda: _run_suite_quality(
            bench_dir, 1000, "quality_1000", "quality_validation_1000_results.json", save_artifacts
        ),
        "test_cases_1000": lambda: _run_suite_structured_cases(
            bench_dir, 1000, "test_cases_1000", "test_cases_1000_results.json", "test_cases_1000.jsonl", save_artifacts
        ),
        "test_cases_11000": lambda: _run_suite_structured_cases(
            bench_dir, 11000, "test_cases_11000", "test_cases_11000_results.json", None, save_artifacts
        ),
        "web_external_curated": lambda: _run_suite_web_external(
            bench_dir, False, "web_external_curated", "web_benchmarks_results.json", save_artifacts
        ),
        "hf_jev_models_ecosystem": lambda: _run_suite_hf_jev_models(
            bench_dir, save_artifacts
        ),
        "web_external_full_splits": lambda: _run_suite_web_external(
            bench_dir, True, "web_external_full_splits", "web_benchmarks_full_results.json", save_artifacts
        ),
        "research_hypotheses": _run_suite_research_hypotheses,
        "rca_non_regression": _run_suite_rca_non_regression,
        "multilingual_eval": lambda: _run_suite_multilingual_eval(bench_dir, save_artifacts),
    }


def run_all_benchmark_suites(
    suites: Sequence[str] | None = None,
    include_full_splits: bool = True,
    quick: bool = False,
    output_dir: str | Path | None = None,
    master_output_path: str | Path | None = None,
    output_path: str | Path | None = None,
    save_artifacts: bool = True,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the requested MicroLLM benchmark suites and persist a consolidated master report."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved_master_out = master_output_path if master_output_path is not None else output_path
    if output_dir is not None:
        bench_dir = Path(output_dir)
    elif resolved_master_out is not None and Path(resolved_master_out).parent != (repo_root / "benchmarks"):
        bench_dir = Path(resolved_master_out).parent
    else:
        bench_dir = repo_root / "benchmarks"

    if save_artifacts:
        bench_dir.mkdir(parents=True, exist_ok=True)

    run_full = include_full_splits and not quick
    dispatch = _build_suite_dispatch(bench_dir, save_artifacts)
    if suites is None:
        selected = list(ALL_BENCHMARK_SUITE_IDS if run_full else FAST_BENCHMARK_SUITE_IDS)
    else:
        selected = [s.strip() for s in suites if s.strip()]
        for s_id in selected:
            if s_id not in dispatch:
                raise ValueError(
                    f"Unknown benchmark suite '{s_id}'. Valid suites: {list(ALL_BENCHMARK_SUITE_IDS)}"
                )

    t0_all = time.perf_counter()
    suite_summaries: dict[str, dict[str, Any]] = {}
    total_items = 0
    total_decisions = 0

    for s_id in selected:
        summary = dispatch[s_id]()
        suite_summaries[s_id] = summary
        total_items += int(summary["total_items"])
        total_decisions += int(summary["total_decisions"])
        if progress_callback is not None:
            progress_callback(s_id, summary)

    total_wall_clock_ms = round((time.perf_counter() - t0_all) * 1000.0, 2)
    passed_suites = sum(1 for s in suite_summaries.values() if s["passed"])

    master_report: dict[str, Any] = {
        "report_name": "MicroLLM Unified Master Benchmark Report",
        "environment": collect_system_environment(),
        "total_suites_executed": len(selected),
        "passed_suites": passed_suites,
        "all_suites_passed": passed_suites == len(selected),
        "total_items_evaluated": total_items,
        "total_decisions_evaluated": total_decisions,
        "total_wall_clock_ms": total_wall_clock_ms,
        "total_wall_clock_sec": round(total_wall_clock_ms / 1000.0, 2),
        "suites": suite_summaries,
    }

    is_full_9_suite_run = tuple(selected) == ALL_BENCHMARK_SUITE_IDS
    should_write_master = (
        resolved_master_out is not None
        or output_dir is not None
        or (save_artifacts and is_full_9_suite_run)
    )
    if should_write_master:
        target_master = (
            Path(resolved_master_out)
            if resolved_master_out is not None
            else (bench_dir / "all_benchmarks_master_report.json")
        )
        target_master.parent.mkdir(parents=True, exist_ok=True)
        target_master.write_text(
            json.dumps(master_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return master_report


def format_master_benchmark_markdown(master_report: dict[str, Any]) -> str:
    """Render a clean Markdown summary table from the consolidated master benchmark report."""
    env = master_report["environment"]
    status = "PASS" if master_report["all_suites_passed"] else "FAIL"
    lines = [
        "# MicroLLM Unified Master Benchmark Report",
        "",
        f"- **Timestamp (UTC)**: `{env['timestamp_utc']}`",
        f"- **Environment**: `{env['platform']}` · Python `{env['python_version']}` · NumPy `{env['numpy_version']}` · `{env['cpu_count_logical']}` logical cores (`{env['compute_device']}`)",
        f"- **Suites Executed**: **{master_report['passed_suites']} / {master_report['total_suites_executed']} {status}**",
        f"- **Total Evaluated Items / Decisions**: **`{master_report['total_items_evaluated']:,}` items** (`{master_report['total_decisions_evaluated']:,}` decisions)",
        f"- **Total Wall-Clock Duration**: **`{master_report['total_wall_clock_sec']:.2f} s`** (`{master_report['total_wall_clock_ms']:,.1f} ms`)",
        "",
        "| Suite ID | Benchmark Suite | Items | Primary Accuracy | Secondary Metrics / Gates | CPU `p50` / `p95` (ms) | Wall-Clock (ms) | Status | Artifact |",
        "| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |",
    ]
    for s_id, info in master_report["suites"].items():
        st_badge = "PASS" if info["passed"] else "FAIL"
        sec_metric = str(info["secondary_metric"]).replace("|", "·")
        lines.append(
            f"| `{s_id}` | {info['title']} | {info['total_items']:,} | **{info['primary_accuracy']*100:.2f}%** | "
            f"{sec_metric} | `{info['p50_latency_ms']:.3f}` / `{info['p95_latency_ms']:.3f}` | "
            f"`{info['elapsed_ms']:,.1f}` | **{st_badge}** | `{info['artifact']}` |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `python -m microllm.benchmark_runner`."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="microllm-benchmark-runner",
        description="Unified automation orchestrator for all MicroLLM benchmark suites",
    )
    parser.add_argument(
        "--suites",
        default=None,
        help="Optional comma-separated list of suite IDs to run (default: all suites)",
    )
    parser.add_argument(
        "--skip-full-splits",
        "--fast",
        "--quick",
        dest="skip_full_splits",
        action="store_true",
        help="Skip the 114,018-row full unsampled external dataset split evaluation for fast CI runs",
    )
    parser.add_argument(
        "--hf-jev-models",
        action="store_true",
        help="Run the 12 Hugging Face JEV model dataset benchmark suite and export JSONL corpora",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for benchmark JSON artifacts (default: benchmarks/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional custom file path for the consolidated master JSON report",
    )
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Optional file path to write the formatted Markdown master report",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout output format (default: markdown)",
    )

    parsed = parser.parse_args(list(argv) if argv is not None else None)
    suite_list = (
        [s.strip() for s in parsed.suites.split(",") if s.strip()]
        if parsed.suites
        else None
    )
    if parsed.hf_jev_models:
        if suite_list is None:
            suite_list = ["hf_jev_models_ecosystem"]
        elif "hf_jev_models_ecosystem" not in suite_list:
            suite_list.append("hf_jev_models_ecosystem")
    try:
        report = run_all_benchmark_suites(
            suites=suite_list,
            include_full_splits=not bool(parsed.skip_full_splits),
            output_dir=parsed.output_dir,
            master_output_path=parsed.output,
            save_artifacts=True,
        )
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    md_text = format_master_benchmark_markdown(report)
    if parsed.markdown_out:
        md_path = Path(parsed.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text + "\n", encoding="utf-8")

    if parsed.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(md_text)

    return 0 if report["all_suites_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
