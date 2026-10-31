"""Hugging Face Space Synchronization Engine for MicroLLM Decision Studio.

Provides bundle staging, dependency requirements generation, English Space Card
synthesis with YAML frontmatter, and Hugging Face Hub Space synchronization via HfApi.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from microllm.hf_publishing import (
    AuthenticationError,
    PublishingError,
    calculate_sha256,
    get_cache_dir,
    resolve_hf_token,
)

SPACE_REPO_ID: str = "jmrenouard/microllm-decision-studio"
DEFAULT_GRADIO_VERSION: str = "5.12.0"
DEFAULT_PYTHON_VERSION: str = "3.11"


def detect_gradio_version() -> str:
    """Detect installed Gradio version, defaulting to '5.12.0'."""
    try:
        import gradio as gr

        return str(getattr(gr, "__version__", DEFAULT_GRADIO_VERSION))
    except Exception:
        return DEFAULT_GRADIO_VERSION


def build_space_readme(
    title: str = "MicroLLM Decision Studio",
    emoji: str = "⚡",
    color_from: str = "blue",
    color_to: str = "indigo",
    sdk_version: str | None = None,
    python_version: str = DEFAULT_PYTHON_VERSION,
    license_type: str = "mit",
) -> str:
    """Generate 100% English Hugging Face Space Card README.md with YAML metadata."""
    gradio_ver = sdk_version or detect_gradio_version()
    return f"""---
title: {title}
emoji: {emoji}
colorFrom: {color_from}
colorTo: {color_to}
sdk: gradio
sdk_version: {gradio_ver}
python_version: '{python_version}'
app_file: app.py
pinned: false
license: {license_type}
short_description: Sub-millisecond calibrated System-1 decision engine on CPU.
---

# ⚡ MicroLLM — Decision Studio (Interactive System-1 Engine)

> **Interactive demonstration of MicroLLM**: a local probabilistic System-1 decision engine engineered for sub-millisecond CPU latency (`< 0.4 ms`, `0% GPU`, `< 85 MB RAM`, `0.00 USD API Cost`).

---

## 🧭 Ecosystem Navigation & Quick Links

| Resource | Purpose | Direct Link |
| :--- | :--- | :--- |
| **Official GitHub Repository** | Full source code, CLI, FastAPI server, Python SDK & documentation | [**`https://github.com/jmrenouard/MicroLLM`**](https://github.com/jmrenouard/MicroLLM) |
| **Hugging Face Space** | Live interactive browser demo (Decision Studio) | [**`jmrenouard/microllm-decision-studio`**](https://huggingface.co/spaces/jmrenouard/microllm-decision-studio) |
| **Multilingual Benchmarks** | Quadrilingual evaluation corpus (EN, FR, ES, PT) | [**`jmrenouard/microllm-multilingual-benchmarks`**](https://huggingface.co/datasets/jmrenouard/microllm-multilingual-benchmarks) |
| **Official Neural Models** | ONNX INT8 zero-shot decision models | [`minilm-int8`](https://huggingface.co/jmrenouard/microllm-multilingual-minilm-int8) · [`mdeberta-int8`](https://huggingface.co/jmrenouard/microllm-multilingual-mdeberta-int8) |

---

## 🎯 Key Features & Capabilities

- **Multilingual Support**: Select between Strategy 1 (Fast Lexicon, < 1 ms CPU) and Strategy 2 (Neural ONNX MiniLMv2 & mDeBERTa-v3) across English, French, Spanish, and Portuguese.
- **3 Workspace Modes**:
  - `✨ Simple Per-Type Tabs`: Dedicated consumer tabs for `noul` (yes/no), `choice` (routing), `multi` (tagging), and `score` (rating).
  - `⚙️ Advanced Studio & Custom JSON Mode`: Interactive visual question builder with live bidirectional JSON sync.
  - `🖥️ All-in-One Cockpit Mode`: Comprehensive side-by-side Gear 1 / Gear 2 / Cascade comparison.
- **Structured Enterprise Domain Presets**: Instant evaluation across Customer Support (Billing, Shipping, Technical in EN, FR, ES, PT), AI Safety Guardrails, Clinical Emergency, Financial Dispute, and IT Operations.
- **Calibrated Decision Heads**: Two-stage OpenJev-RLCD² calibration with conformal prediction set estimation.

---

## ⚡ Quickstart

### Launching Decision Studio Locally

```bash
# Install MicroLLM with Gradio support
pip install "microllm[gradio]"

# Launch Decision Studio locally on port 7860
microllm studio --port 7860
```
"""


def generate_space_requirements() -> str:
    """Generate minimal requirements.txt dependencies for Hugging Face Space."""
    return "\n".join([
        "gradio>=4.0.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "onnxruntime>=1.16.0",
        "",
    ])


def generate_space_app_py() -> str:
    """Generate standalone app.py entrypoint for Hugging Face Space."""
    return '''"""Root Hugging Face Spaces Entrypoint for MicroLLM Decision Studio."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure `src/` is on sys.path when launched directly on Hugging Face Spaces without `pip install -e .`
_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"

if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
elif str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from microllm.gradio_app import create_gradio_demo, zero_gpu_hook  # noqa: E402

demo = create_gradio_demo()

if __name__ == "__main__":
    _ = zero_gpu_hook()
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
'''


def stage_hf_space(
    staging_dir: Path | str | None = None,
    repo_id: str = SPACE_REPO_ID,
    sdk_version: str | None = None,
    include_src: bool = True,
    repo_root: Path | str | None = None,
    license_type: str = "mit",
) -> dict[str, Any]:
    """Stage all required files for Hugging Face Space deployment with SHA-256 manifest."""
    if staging_dir is None:
        target_dir = get_cache_dir() / "staging" / "spaces" / repo_id.replace("/", "_")
    else:
        target_dir = Path(staging_dir).resolve()

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write README.md with YAML frontmatter
    readme_content = build_space_readme(sdk_version=sdk_version, license_type=license_type)
    (target_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # 2. Write requirements.txt
    reqs_content = generate_space_requirements()
    (target_dir / "requirements.txt").write_text(reqs_content, encoding="utf-8")

    # 3. Write app.py
    app_content = generate_space_app_py()
    (target_dir / "app.py").write_text(app_content, encoding="utf-8")

    # 4. Write .gitignore
    gitignore_content = "__pycache__/\n*.pyc\n*.pyo\n.pytest_cache/\n*.egg-info/\n.DS_Store\n"
    (target_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # 5. Bundle `src/microllm`
    if include_src:
        if repo_root:
            root_path = Path(repo_root).resolve()
        else:
            root_path = Path(__file__).resolve().parent.parent.parent

        src_microllm = root_path / "src" / "microllm"
        dest_src = target_dir / "src" / "microllm"

        if src_microllm.is_dir():
            if dest_src.exists():
                shutil.rmtree(dest_src)
            shutil.copytree(
                src_microllm,
                dest_src,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
            )

    # 6. Generate SHA-256 manifest
    files_manifest: list[dict[str, Any]] = []
    for item in sorted(target_dir.rglob("*")):
        if item.is_file() and item.name not in ("staging_manifest.json", "manifest.json"):
            rel_path = item.relative_to(target_dir).as_posix()
            files_manifest.append({
                "path": rel_path,
                "bytes": item.stat().st_size,
                "sha256": calculate_sha256(item),
            })

    manifest_data = {
        "repo_id": repo_id,
        "repo_type": "space",
        "space_sdk": "gradio",
        "sdk_version": sdk_version or detect_gradio_version(),
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files_manifest),
        "total_bytes": sum(f["bytes"] for f in files_manifest),
        "files": files_manifest,
    }

    manifest_json = json.dumps(manifest_data, indent=2)
    (target_dir / "staging_manifest.json").write_text(manifest_json, encoding="utf-8")
    (target_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")

    return {
        "repo_id": repo_id,
        "repo_type": "space",
        "staging_dir": str(target_dir),
        "total_files": len(files_manifest),
        "total_bytes": sum(f["bytes"] for f in files_manifest),
        "manifest": manifest_data,
    }


# Backwards compatibility and test convenience alias
stage_space_bundle = stage_hf_space


def sync_hf_space(
    space_repo_id: str = SPACE_REPO_ID,
    staging_dir: Path | str | None = None,
    dry_run: bool = False,
    token: str | None = None,
    api: Any | None = None,
    commit_message: str | None = None,
    sdk_version: str | None = None,
    include_src: bool = True,
    repo_root: Path | str | None = None,
    license_type: str = "mit",
) -> dict[str, Any]:
    """Synchronize the MicroLLM Decision Studio bundle to Hugging Face Space."""
    stage_res = stage_hf_space(
        staging_dir=staging_dir,
        repo_id=space_repo_id,
        sdk_version=sdk_version,
        include_src=include_src,
        repo_root=repo_root,
        license_type=license_type,
    )
    staged_folder = Path(stage_res["staging_dir"])

    if bool(dry_run):
        return {
            "status": "dry_run",
            "repo_id": space_repo_id,
            "repo_type": "space",
            "staging_dir": str(staged_folder),
            "upload_attempted": False,
            "network_calls_made": 0,
            "total_files": stage_res["total_files"],
            "total_bytes": stage_res["total_bytes"],
            "manifest": stage_res["manifest"],
        }

    from microllm.hf_publishing import _upload_staged_folder_to_hf

    commit_msg = commit_message or "feat(space): synchronize Gradio Decision Studio and multilingual engine"
    source = _upload_staged_folder_to_hf(
        staged_folder=staged_folder,
        repo_id=space_repo_id,
        repo_type="space",
        token=token,
        api=api,
        commit_msg=commit_msg,
        space_sdk="gradio",
    )

    return {
        "status": "synced",
        "repo_id": space_repo_id,
        "repo_type": "space",
        "token_source": source,
        "staging_dir": str(staged_folder),
        "upload_attempted": True,
        "commit_message": commit_msg,
        "total_files": stage_res["total_files"],
        "total_bytes": stage_res["total_bytes"],
    }
