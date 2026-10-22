#!/usr/bin/env sh
# POSIX Shell Automation Wrapper for MicroLLM Benchmark Suites
# Usage:
#   ./scripts/run_all_benchmarks.sh                # Run all 9 benchmark suites (including 114,018 full splits)
#   ./scripts/run_all_benchmarks.sh --fast         # Fast mode (skips 114,018-row full splits)

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_EXE="python3"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    if "$REPO_ROOT/.venv/bin/python" -c "import numpy, scipy, pyarrow" >/dev/null 2>&1; then
        PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
    fi
fi

export PYTHONPATH="$REPO_ROOT/src"
exec "$PYTHON_EXE" "$REPO_ROOT/scripts/run_all_benchmarks.py" "$@"
