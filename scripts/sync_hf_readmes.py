#!/usr/bin/env python3
"""Synchronize root README.md to all 6 official Hugging Face repositories (IDEA-038).

Usage:
    python scripts/sync_hf_readmes.py
    python scripts/sync_hf_readmes.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from microllm.registry import sync_readme_to_huggingface_hub  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run Hugging Face README synchronization."""
    parser = argparse.ArgumentParser(
        description="Synchronize root README.md to all 6 official Hugging Face repositories (IDEA-038)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build all 6 Hugging Face cards without uploading to the Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face API token (defaults to HF_TOKEN or HUGGINGFACE_TOKEN env var)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = sync_readme_to_huggingface_hub(
        readme_path=ROOT / "README.md",
        token=args.token,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
