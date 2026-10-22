#!/usr/bin/env python3
"""One-command cross-platform automation script to run all MicroLLM benchmark suites.

Usage:
    python scripts/run_all_benchmarks.py                 # Run all 9 benchmark suites (including 114,018 full splits)
    python scripts/run_all_benchmarks.py --fast          # Skip the 114,018-row full splits for ~4s quick verification
    python scripts/run_all_benchmarks.py --suites eval_540,quality_750,test_cases_1000
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*httpx.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from microllm.benchmark_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
