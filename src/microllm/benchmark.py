"""CLI alias module allowing `python -m microllm.benchmark` in 1 command."""

from __future__ import annotations

from microllm.benchmarks import (
    build_benchmark_corpus,
    format_markdown_benchmark_table,
    main,
    run_benchmark_suite,
    save_corpus_jsonl,
)

__all__ = [
    "build_benchmark_corpus",
    "format_markdown_benchmark_table",
    "main",
    "run_benchmark_suite",
    "save_corpus_jsonl",
]

if __name__ == "__main__":
    main()
