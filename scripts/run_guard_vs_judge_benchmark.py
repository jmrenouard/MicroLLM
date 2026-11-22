"""Standalone reproducible runner for the `microllm-guard` vs System-2 LLM-Judge benchmark (Issue #287)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from microllm.guard_benchmark import run_guard_vs_judge_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the `microllm-guard` vs System-2 LLM-Judge benchmark and persist JSON/Markdown reports."""
    parser = argparse.ArgumentParser(
        prog="run_guard_vs_judge_benchmark",
        description="Run the reproducible MicroLLM Guard vs System-2 LLM-Judge benchmark suite.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path (default: benchmarks/guard_vs_llm_judge_results.json)",
    )
    parser.add_argument(
        "--markdown-output",
        dest="markdown_output",
        default=None,
        help="Optional Markdown report output path (e.g. docs/GUARD_VS_LLM_JUDGE_BENCHMARK.md)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print JSON report to stdout instead of Markdown summary",
    )
    parser.add_argument(
        "--live-judge-url",
        dest="live_judge_url",
        default=None,
        help="Optional opt-in OpenAI-compatible HTTP endpoint URL for live System-2 LLM-Judge comparison",
    )
    parser.add_argument(
        "--live-judge-model",
        dest="live_judge_model",
        default=None,
        help="Model identifier when --live-judge-url is enabled",
    )
    parser.add_argument(
        "--live-judge-api-key",
        dest="live_judge_api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional Bearer token for --live-judge-url",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[1]
    md_out = (
        Path(parsed.markdown_output)
        if parsed.markdown_output is not None
        else (repo_root / "docs" / "GUARD_VS_LLM_JUDGE_BENCHMARK.md")
    )
    try:
        report = run_guard_vs_judge_benchmark(
            output_path=parsed.output,
            markdown_output_path=md_out,
            save_artifacts=True,
            live_judge_url=parsed.live_judge_url,
            live_judge_model=parsed.live_judge_model,
            live_judge_api_key=parsed.live_judge_api_key,
        )
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if parsed.as_json:
        printable = {k: v for k, v in report.items() if k != "markdown_report"}
        sys.stdout.write(json.dumps(printable, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(report["markdown_report"] + "\n")
    sys.stdout.flush()
    return 0 if report["gates"]["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
