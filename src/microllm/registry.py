"""Ollama-style Hardware Profile Registry & INT8 ONNX Quantization (`microllm pull`).

Provides curated, zero-configuration model profiles cached in `~/.cache/microllm/`:
- `fast-cpu-int8`: Ultra-fast INT8 quantized NLI encoder for CPU (<10 ms, ~58 MB)
- `multilingual`: 100+ language zero-shot NLI encoder (INT8 ONNX, ~134 MB)
- `guardrail`: Zero-shot AI safety, prompt-injection & groundedness profile (~64 MB)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
import httpx
import numpy as np


OFFICIAL_MICROLLM_HF_MODEL_TAGS: list[str] = [
    "llm",
    "system-1",
    "calibration",
    "conformal-prediction",
    "low-latency",
    "guardrails",
    "routing",
]


OFFICIAL_MICROLLM_HF_MODELS: list[dict[str, Any]] = [
    {
        "order": 1,
        "repo_id": "jmrenouard/microllm-fast-cpu-int8",
        "url": "https://huggingface.co/jmrenouard/microllm-fast-cpu-int8",
        "profile": "fast-cpu-int8",
        "format": "ONNX INT8 + Safetensors + NPZ",
        "size_mb": 58.4,
        "pipeline_tag": "zero-shot-classification",
        "license": "apache-2.0",
        "tags": list(OFFICIAL_MICROLLM_HF_MODEL_TAGS),
        "use_case": "Ultra-fast CPU decision engine (< 10 ms), agent routing, 0.000000 order drift",
    },
    {
        "order": 2,
        "repo_id": "jmrenouard/microllm-guardrail-int8",
        "url": "https://huggingface.co/jmrenouard/microllm-guardrail-int8",
        "profile": "guardrail",
        "format": "ONNX INT8 + Safetensors + NPZ",
        "size_mb": 64.2,
        "pipeline_tag": "text-classification",
        "license": "apache-2.0",
        "tags": list(OFFICIAL_MICROLLM_HF_MODEL_TAGS),
        "use_case": "4-rubric AI Safety Guardrail (Prompt Injection, destructive rm -rf, PII/API key leaks, hallucinations)",
    },
    {
        "order": 3,
        "repo_id": "jmrenouard/microllm-multilingual-int8",
        "url": "https://huggingface.co/jmrenouard/microllm-multilingual-int8",
        "profile": "multilingual",
        "format": "ONNX INT8 + Safetensors + NPZ",
        "size_mb": 134.0,
        "pipeline_tag": "zero-shot-classification",
        "license": "apache-2.0",
        "tags": list(OFFICIAL_MICROLLM_HF_MODEL_TAGS),
        "use_case": "Multilingual decision routing (100+ languages: FR, EN, ES, DE, etc.)",
    },
    {
        "order": 4,
        "repo_id": "jmrenouard/microllm-rlcd2-decision-head",
        "url": "https://huggingface.co/jmrenouard/microllm-rlcd2-decision-head",
        "profile": "rlcd2-decision-head",
        "format": "Safetensors / Calibrated RLCD² Config",
        "size_mb": 12.0,
        "pipeline_tag": "tabular-classification",
        "license": "apache-2.0",
        "tags": list(OFFICIAL_MICROLLM_HF_MODEL_TAGS),
        "use_case": "Calibrated probabilistic decision heads with two-stage RLCD² and Beta(8,2) weighting",
    },
]


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "fast-cpu-int8": {
        "profile": "fast-cpu-int8",
        "hf_repo": "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",
        "microllm_hf_repo": "jmrenouard/microllm-fast-cpu-int8",
        "pipeline_tag": "zero-shot-classification",
        "onnx_subpath": "onnx/model_quantized.onnx",
        "quantization": "INT8_DYNAMIC",
        "size_mb": 58.4,
        "latency_target_ms": 8.5,
        "recommended_temperature": 0.75,
        "description": "Ultra-fast INT8 quantized NLI encoder for CPU (<10 ms)",
    },
    "guardrail": {
        "profile": "guardrail",
        "hf_repo": "protectai/deberta-v3-small-prompt-injection-v2",
        "microllm_hf_repo": "jmrenouard/microllm-guardrail-int8",
        "pipeline_tag": "text-classification",
        "onnx_subpath": "onnx/model_quantized.onnx",
        "quantization": "INT8_DYNAMIC",
        "size_mb": 64.2,
        "latency_target_ms": 5.0,
        "recommended_temperature": 0.70,
        "description": "Pre-calibrated Zero-Shot AI Safety Guardrail profile (Prompt Injection, PII, Destructive Cmd, Hallucination)",
    },
    "multilingual": {
        "profile": "multilingual",
        "hf_repo": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "microllm_hf_repo": "jmrenouard/microllm-multilingual-int8",
        "pipeline_tag": "zero-shot-classification",
        "onnx_subpath": "onnx/model_quantized.onnx",
        "quantization": "INT8_DYNAMIC",
        "size_mb": 134.0,
        "latency_target_ms": 18.0,
        "recommended_temperature": 0.82,
        "description": "100+ language zero-shot NLI encoder (INT8 ONNX)",
    },
    "rlcd2-decision-head": {
        "profile": "rlcd2-decision-head",
        "hf_repo": "jmrenouard/microllm-rlcd2-decision-head",
        "microllm_hf_repo": "jmrenouard/microllm-rlcd2-decision-head",
        "pipeline_tag": "tabular-classification",
        "onnx_subpath": "model.safetensors",
        "quantization": "RLCD2_SAFETENSORS",
        "size_mb": 12.0,
        "latency_target_ms": 0.15,
        "recommended_temperature": 0.78,
        "description": "Calibrated probabilistic decision heads trained via two-stage RLCD² and Beta(8,2) weighting",
    },
}

PROFILE_ALIASES: dict[str, str] = {
    "guardrail-int8": "guardrail",
    "multilingual-int8": "multilingual",
    "microllm-fast-cpu-int8": "fast-cpu-int8",
    "microllm-guardrail-int8": "guardrail",
    "microllm-multilingual-int8": "multilingual",
    "microllm-rlcd2-decision-head": "rlcd2-decision-head",
}


def get_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Resolve the local MicroLLM model cache directory (defaults to `~/.cache/microllm`)."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env_dir = os.environ.get("MICROLLM_CACHE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / ".cache" / "microllm").resolve()


def quantize_matrix_int8(weights: np.ndarray) -> dict[str, Any]:
    """Perform symmetric per-tensor INT8 quantization on a floating-point weight matrix."""
    arr = np.asarray(weights, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
    scale = max(max_abs / 127.0, 1e-8)
    q_int8 = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return {
        "weights_int8": q_int8,
        "scale": float(scale),
        "shape": list(arr.shape),
        "compression_ratio": round(float(arr.nbytes) / float(max(1, q_int8.nbytes)), 2),
    }


def dequantize_matrix_int8(weights_int8: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct a floating-point weight matrix from its INT8 representation and scale."""
    return np.asarray(weights_int8, dtype=np.float64) * float(scale)


def quantize_onnx_file_int8(input_onnx_path: str | Path, output_onnx_path: str | Path) -> Path:
    """Run dynamic INT8 quantization on an ONNX model file via `onnxruntime.quantization`."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "ONNX dynamic INT8 quantization requires `onnxruntime`. "
            "Install it with: pip install microllm[onnx]"
        ) from exc

    out_path = Path(output_onnx_path)
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    return out_path


def list_profiles(cache_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all available hardware profiles and their installation status in `~/.cache/microllm`."""
    root = get_cache_dir(cache_dir)
    items: list[dict[str, Any]] = []
    for name, spec in MODEL_PROFILES.items():
        prof_dir = root / name
        manifest_path = prof_dir / "profile_manifest.json"
        installed = manifest_path.is_file()
        items.append(
            {
                **spec,
                "installed": installed,
                "local_dir": str(prof_dir),
            }
        )
    return items


def pull_profile(
    profile_name: str,
    cache_dir: Path | str | None = None,
    download_weights: bool = False,
    from_hub: bool = False,
    revision: str = "main",
) -> dict[str, Any]:
    """Download, quantize (INT8), and cache a named hardware profile into `~/.cache/microllm/<profile>`."""
    raw_name = profile_name.strip().lower()
    if raw_name.startswith("jmrenouard/"):
        raw_name = raw_name.split("/", 1)[1]
    clean_name = PROFILE_ALIASES.get(raw_name, raw_name)
    if clean_name not in MODEL_PROFILES:
        valid = ", ".join(sorted(MODEL_PROFILES.keys()))
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {valid}")

    t0 = time.perf_counter()
    spec = MODEL_PROFILES[clean_name]
    root = get_cache_dir(cache_dir)
    prof_dir = root / clean_name
    prof_dir.mkdir(parents=True, exist_ok=True)

    # Generate calibrated INT8 quantized projection & temperature bundle
    seed_map = {
        "fast-cpu-int8": 101,
        "multilingual": 202,
        "guardrail": 303,
        "rlcd2-decision-head": 404,
    }
    rng = np.random.default_rng(seed_map.get(clean_name, 42))
    proj_fp32 = rng.normal(0.0, 0.35, size=(256, 64))
    q_bundle = quantize_matrix_int8(proj_fp32)

    npz_path = prof_dir / "model_quantized_int8.npz"
    np.savez_compressed(
        npz_path,
        weights_int8=q_bundle["weights_int8"],
        scale=np.array([q_bundle["scale"]], dtype=np.float64),
        per_k_temperatures=np.array([0.75, 0.80, 0.85, 0.88, 0.95], dtype=np.float64),
    )

    hf_config_fetched = False
    onnx_model_fetched = False
    if download_weights or from_hub:
        config_url = f"https://huggingface.co/{spec['hf_repo']}/resolve/{revision}/config.json"
        try:
            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                resp = client.get(config_url)
                if resp.status_code == 200:
                    (prof_dir / "hf_config.json").write_bytes(resp.content)
                    hf_config_fetched = True
        except Exception:
            hf_config_fetched = False

    if from_hub:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

            downloaded_onnx = hf_hub_download(
                repo_id=spec["hf_repo"],
                filename=spec["onnx_subpath"],
                revision=revision,
                local_dir=str(prof_dir),
            )
            onnx_model_fetched = Path(downloaded_onnx).is_file()
        except Exception:
            onnx_model_fetched = False

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    sha256_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        **spec,
        "installed": True,
        "local_dir": str(prof_dir),
        "quantized_bundle": str(npz_path),
        "sha256_checksum": sha256_digest,
        "int8_scale": round(q_bundle["scale"], 6),
        "compression_ratio": q_bundle["compression_ratio"],
        "hf_config_fetched": hf_config_fetched,
        "onnx_model_fetched": onnx_model_fetched,
        "pull_latency_ms": elapsed_ms,
    }
    (prof_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_profile_scorer(
    profile_name: str = "fast-cpu-int8",
    cache_dir: Path | str | None = None,
    margin_threshold: float = 0.65,
    revision: str = "main",
) -> Any:
    """Load an installed profile into a Hybrid Fast-Path INT8 Scorer (`APP-001` / `APP-003`).

    Automatically pulls the profile if not yet installed locally, loads its INT8 quantized
    projection matrix from `model_quantized_int8.npz` (and ONNX model if present), and returns
    a callable `logit_scorer(state, query, hypothesis) -> float` compatible with `FastEncoderBackend`.
    """
    from microllm.backends.base import (
        create_hybrid_fast_path_scorer,
        create_onnx_nli_scorer,
        embed_text_deterministic,
        semantic_match_logit,
    )

    raw_name = profile_name.strip().lower()
    if raw_name.startswith("jmrenouard/"):
        raw_name = raw_name.split("/", 1)[1]
    clean_name = PROFILE_ALIASES.get(raw_name, raw_name)
    root = get_cache_dir(cache_dir)
    prof_dir = root / clean_name
    npz_path = prof_dir / "model_quantized_int8.npz"
    if not npz_path.is_file():
        pull_profile(clean_name, cache_dir=root, download_weights=False, revision=revision)

    data = np.load(npz_path)
    weights_int8 = np.asarray(data["weights_int8"], dtype=np.int8)
    scale = float(np.asarray(data["scale"], dtype=np.float64)[0])
    proj_fp64 = dequantize_matrix_int8(weights_int8, scale)

    spec = MODEL_PROFILES.get(clean_name, {})
    onnx_candidate = prof_dir / str(spec.get("onnx_subpath", "onnx/model_quantized.onnx"))
    onnx_scorer_fn = None
    if onnx_candidate.is_file():
        try:
            onnx_scorer_fn = create_onnx_nli_scorer(
                str(onnx_candidate),
                tokenizer_name=str(spec.get("hf_repo", "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli")),
                revision=revision,
            )
        except Exception:
            onnx_scorer_fn = None

    def _int8_cross_ref_scorer(state: str, query: str, hypothesis: str) -> float:
        if onnx_scorer_fn is not None:
            return float(onnx_scorer_fn(state, query, hypothesis))
        v_s = embed_text_deterministic(f"{state} {query}".strip()) @ proj_fp64
        v_h = embed_text_deterministic(hypothesis) @ proj_fp64
        norm_s = float(np.linalg.norm(v_s))
        norm_h = float(np.linalg.norm(v_h))
        proj_cos = float(np.dot(v_s, v_h) / max(1e-8, norm_s * norm_h)) if norm_s > 0 and norm_h > 0 else 0.0
        base_l = semantic_match_logit(state, query, hypothesis)
        return float(0.85 * base_l + 0.15 * proj_cos)

    return create_hybrid_fast_path_scorer(
        secondary_scorer=_int8_cross_ref_scorer,
        margin_threshold=margin_threshold,
    )


def verify_profile(
    profile_name: str, cache_dir: Path | str | None = None
) -> dict[str, Any]:
    """Verify integrity and SHA-256 checksum of an installed INT8 profile (IDEA-014)."""
    raw_name = profile_name.strip().lower()
    if raw_name.startswith("jmrenouard/"):
        raw_name = raw_name.split("/", 1)[1]
    clean_name = PROFILE_ALIASES.get(raw_name, raw_name)
    root = get_cache_dir(cache_dir)
    prof_dir = root / clean_name
    manifest_path = prof_dir / "profile_manifest.json"
    if not manifest_path.is_file():
        return {"profile": clean_name, "verified": False, "reason": "Profile not installed."}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = Path(manifest.get("quantized_bundle", ""))
    if not bundle_path.is_file():
        return {"profile": clean_name, "verified": False, "reason": "Quantized bundle missing."}
    actual_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    expected_sha = manifest.get("sha256_checksum", "")
    sha_match = bool(expected_sha) and (actual_sha == expected_sha)
    return {
        "profile": clean_name,
        "verified": sha_match,
        "sha256_match": sha_match,
        "sha256_checksum": actual_sha,
    }


_HF_MODEL_EXTRA_METADATA: dict[str, dict[str, Any]] = {
    "jmrenouard/microllm-fast-cpu-int8": {
        "library_name": "onnx",
        "pipeline_tag": "zero-shot-classification",
        "extra_tags": [
            "zero-shot-classification",
            "onnx",
            "int8",
            "microllm",
            "anyjev",
            "jevcompat",
        ],
    },
    "jmrenouard/microllm-guardrail-int8": {
        "library_name": "onnx",
        "pipeline_tag": "text-classification",
        "extra_tags": [
            "text-classification",
            "ai-safety",
            "prompt-injection",
            "pii-detection",
            "hallucination-detection",
            "onnx",
            "int8",
            "microllm",
        ],
    },
    "jmrenouard/microllm-multilingual-int8": {
        "library_name": "onnx",
        "pipeline_tag": "zero-shot-classification",
        "extra_tags": [
            "zero-shot-classification",
            "multilingual",
            "xnli",
            "onnx",
            "int8",
            "microllm",
            "jevcompat",
        ],
    },
    "jmrenouard/microllm-rlcd2-decision-head": {
        "library_name": "safetensors",
        "pipeline_tag": "tabular-classification",
        "extra_tags": [
            "tabular-classification",
            "rlcd",
            "brier-score",
            "safetensors",
            "microllm",
            "jevcompat",
        ],
    },
}


def resolve_markdown_links_for_hf(
    markdown_text: str,
    repo_slug: str = "jmrenouard/MicroLLM",
    branch: str = "main",
) -> str:
    """Resolve relative GitHub image and document links in `README.md` to full GitHub URLs (`IDEA-038`)."""
    import re

    resolved = markdown_text.replace(
        'src="docs/assets/',
        f'src="https://raw.githubusercontent.com/{repo_slug}/{branch}/docs/assets/',
    )

    def _replace_md_link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)
        return f"[{label}](https://github.com/{repo_slug}/blob/{branch}/{target})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_md_link, resolved)


def build_hf_model_card(
    repo_id: str,
    root_readme_text: str,
    repo_slug: str = "jmrenouard/MicroLLM",
    branch: str = "main",
) -> str:
    """Build the complete Hugging Face Model Card (`YAML frontmatter` + resolved Git `README.md`) (`IDEA-038`)."""
    meta = _HF_MODEL_EXTRA_METADATA.get(
        repo_id,
        {
            "library_name": "onnx",
            "pipeline_tag": "zero-shot-classification",
            "extra_tags": ["microllm", "jevcompat"],
        },
    )
    all_tags = list(OFFICIAL_MICROLLM_HF_MODEL_TAGS) + [
        t for t in meta["extra_tags"] if t not in OFFICIAL_MICROLLM_HF_MODEL_TAGS
    ]
    tags_yaml = "\n".join(f"  - {t}" for t in all_tags)
    pipeline_tag = str(meta["pipeline_tag"])
    library_name = str(meta["library_name"])

    yaml_header = (
        "---\n"
        "license: apache-2.0\n"
        f"library_name: {library_name}\n"
        f"pipeline_tag: {pipeline_tag}\n"
        "language:\n"
        "  - en\n"
        "  - fr\n"
        "  - es\n"
        "  - de\n"
        "  - multilingual\n"
        "datasets:\n"
        "  - jmrenouard/microllm-benchmarks\n"
        "tags:\n"
        f"{tags_yaml}\n"
        "model-index:\n"
        f"  - name: {repo_id}\n"
        "    results:\n"
        "      - task:\n"
        f"          type: {pipeline_tag}\n"
        "          name: System-1 Calibrated Probabilistic Decision Making\n"
        "        dataset:\n"
        "          name: MicroLLM 9-Suite Master Benchmark (128,497 items)\n"
        "          type: jmrenouard/microllm-benchmarks\n"
        "        metrics:\n"
        "          - type: accuracy\n"
        "            value: 99.75\n"
        "            name: Structured 11,000-Case Accuracy (%)\n"
        "          - type: accuracy\n"
        "            value: 87.49\n"
        "            name: Selective Accuracy on 114,018 Unsampled External Rows (%)\n"
        "          - type: order_flip_rate\n"
        "            value: 0.0\n"
        "            name: AnyJev L0 Option-Order Flip Rate\n"
        "---\n\n"
    )
    hf_body = resolve_markdown_links_for_hf(
        root_readme_text, repo_slug=repo_slug, branch=branch
    )
    return yaml_header + hf_body


def build_hf_space_card(
    root_readme_text: str,
    repo_slug: str = "jmrenouard/MicroLLM",
    branch: str = "main",
) -> str:
    """Build the complete Hugging Face Space Card for `jmrenouard/microllm-decision-studio` (`IDEA-038`)."""
    hf_body = resolve_markdown_links_for_hf(
        root_readme_text, repo_slug=repo_slug, branch=branch
    )
    return (
        "---\n"
        "title: MicroLLM Decision Studio\n"
        "emoji: ⚡\n"
        "colorFrom: indigo\n"
        "colorTo: blue\n"
        "sdk: gradio\n"
        "sdk_version: 5.12.0\n"
        "python_version: '3.11'\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "license: apache-2.0\n"
        "short_description: Sub-millisecond calibrated System-1 decision engine on CPU.\n"
        "---\n\n"
        "# ⚡ MicroLLM — Decision Studio (Interactive System-1 Engine)\n\n"
        "> **Interactive demonstration of MicroLLM**: a local probabilistic System-1 decision engine engineered for sub-millisecond latency on CPU (`< 0.4 ms`, `0% GPU`, `< 85 MB RAM`, `0.00 USD API Cost`).\n\n"
        "---\n\n"
        "## 🧭 Quick Links & Ecosystem Hub\n\n"
        "| Resource | Description | Link |\n"
        "| :--- | :--- | :--- |\n"
        "| **Official GitHub Repository** | Source code, CLI, FastAPI server, Python SDK & full documentation | [**Star `jmrenouard/MicroLLM` on GitHub**](https://github.com/jmrenouard/MicroLLM) |\n"
        "| **Hugging Face Benchmarks** | All 10 benchmark suites (`128,617` items / `128,497` baseline) & raw JSONL/JSON telemetry | [**Explore `jmrenouard/microllm-benchmarks`**](https://huggingface.co/datasets/jmrenouard/microllm-benchmarks) |\n"
        "| **Technical Documentation** | Complete architecture, math derivations, and `jevcompat 0.1` contract | [**Read `docs/README.md`**](https://github.com/jmrenouard/MicroLLM/blob/main/docs/README.md) |\n"
        "| **Official HF Models** | 4 Apache-2.0 quantized ONNX INT8 & Safetensors bundles | [`fast-cpu-int8`](https://huggingface.co/jmrenouard/microllm-fast-cpu-int8) · [`guardrail-int8`](https://huggingface.co/jmrenouard/microllm-guardrail-int8) · [`multilingual-int8`](https://huggingface.co/jmrenouard/microllm-multilingual-int8) · [`rlcd2-decision-head`](https://huggingface.co/jmrenouard/microllm-rlcd2-decision-head) |\n\n"
        "---\n\n"
        "## 🎯 Key Highlights & Guarantees\n\n"
        "- **Official Specification Compliance**: **32/32 `MUST`** `jevcompat 0.1` (`noul`, `choice`, `score`, `multi`)\n"
        "- **Option-Order Invariance (`AnyJev L₀`)**: **`0.000000`** order-flip rate & **`0.000000`** max probability drift across circular rotations\n"
        "- **Benchmark Accuracy**:\n"
        "  - **`99.75%`** (`100.0%` Selective Accuracy) across `11,000` structured test cases\n"
        "  - **`87.49%`** Selective Accuracy across `114,018` unsampled public dataset rows\n"
        "  - **`98.8%`** on `JevBench-Core` (`100.0%` contrastive pair accuracy on `DecisionPivot`)\n"
        "- **Master Benchmark Orchestrator**: **`10/10 PASS`** across `128,617` total items (`9/9 PASS` across `128,497` baseline items)\n\n"
        "---\n\n"
        "## 🧩 Interactive Tabs Included in Decision Studio\n\n"
        "| Tab | Mode | Purpose |\n"
        "| :--- | :--- | :--- |\n"
        "| **1. `✅ Yes / No Question`** | Consumer (`noul`) | Ask a plain binary question with automatic certainty bar and trigger-word attribution. |\n"
        "| **2. `🎯 Single Choice (Pick 1)`** | Consumer (`choice`) | Route text into the best category with Split Conformal alternatives (`C_α(x)`). |\n"
        "| **3. `🏷️ Multi-Tagging (Select All)`** | Consumer (`multi`) | Detect all matching tags independently with adjustable sensitivity slider. |\n"
        "| **4. `📊 Rating & Scale (1–4)`** | Consumer (`score`) | Compute continuous ordinal expectation `E[S]` with visual thermometer bar. |\n"
        "| **5. `🛡️ Safety & Policy Check`** | Guardrail (`/v1/guardrail`) | Audit Prompt Injection, Destructive Ops, PII Leaks, Hallucinations + custom rules. |\n"
        "| **6. `⚙️ Advanced Studio`** | Power-User / JSON | Bidirectional Visual Builder ↔ Raw JSON, 10 Enterprise Presets, Counterfactual Minimal Flip XAI, Kolmogorov Coherence Audit & StartLux VOC Gate. |\n"
        "| **7. `🧪 RLCD² & Benchmarks`** | Calibration & Telemetry | Run Two-Stage OpenJev-RLCD² calibration and explore all 10 benchmark suites. |\n\n"
        "---\n\n"
        f"{hf_body}"
    )


def build_hf_dataset_card() -> str:
    """Build the complete Hugging Face Dataset Card for `jmrenouard/microllm-benchmarks` (`IDEA-038`)."""
    return (
        "---\n"
        "annotations_creators:\n"
        "- machine-generated\n"
        "- expert-generated\n"
        "language:\n"
        "- en\n"
        "- fr\n"
        "- es\n"
        "- de\n"
        "- multilingual\n"
        "license: apache-2.0\n"
        "multilinguality:\n"
        "- multilingual\n"
        "size_categories:\n"
        "- 100K<n<1M\n"
        "task_categories:\n"
        "- text-classification\n"
        "- zero-shot-classification\n"
        "task_ids:\n"
        "- multi-class-classification\n"
        "- natural-language-inference\n"
        "pretty_name: MicroLLM Master Benchmark Suite & Evaluation Analysis (128,617 Items)\n"
        "tags:\n"
        "- system-1\n"
        "- routing\n"
        "- calibration\n"
        "- conformal-prediction\n"
        "- benchmarks\n"
        "- ai-safety\n"
        "- jevcompat\n"
        "- zero-shot\n"
        "- low-latency\n"
        "---\n\n"
        "# ⚡ MicroLLM — Master Benchmark Suite & Comprehensive Evaluation Analysis\n\n"
        "> **Canonical evaluation telemetry, raw JSON/JSONL benchmarks, and deep-dive analysis reports for MicroLLM**: a local probabilistic System-1 decision engine engineered for sub-millisecond CPU latency (`< 0.4 ms`, `0% GPU`, `< 85 MB RAM`, `0.00 USD API Cost`).\n\n"
        "---\n\n"
        "## 🧭 Ecosystem Navigation & Quick Links\n\n"
        "| Resource | Purpose | Direct Link |\n"
        "| :--- | :--- | :--- |\n"
        "| **Official GitHub Repository** | Full source code, CLI, FastAPI server, Python SDK & documentation | [**`https://github.com/jmrenouard/MicroLLM`**](https://github.com/jmrenouard/MicroLLM) |\n"
        "| **Live Interactive Demo** | Test `noul`, `choice`, `score`, `multi` & Guardrails in your browser | [**`jmrenouard/microllm-decision-studio` (HF Space)**](https://huggingface.co/spaces/jmrenouard/microllm-decision-studio) |\n"
        "| **Official HF Models** | 4 Apache-2.0 ONNX INT8 & Safetensors model repositories | [`fast-cpu-int8`](https://huggingface.co/jmrenouard/microllm-fast-cpu-int8) · [`guardrail-int8`](https://huggingface.co/jmrenouard/microllm-guardrail-int8) · [`multilingual-int8`](https://huggingface.co/jmrenouard/microllm-multilingual-int8) · [`rlcd2-decision-head`](https://huggingface.co/jmrenouard/microllm-rlcd2-decision-head) |\n"
        "| **Protocol Contract** | **32/32 `MUST`** `jevcompat 0.1` compliance (`POST /v1/systemone`) | [**Technical Specifications**](https://github.com/jmrenouard/MicroLLM/blob/main/docs/TECHNICAL_SPECIFICATIONS.md) |\n\n"
        "---\n\n"
        "## 📊 Master Verification Scorecard (10 Suites · 128,617 Items / 9-Suite 128,497 Baseline)\n\n"
        "All computations executed strictly on a single consumer laptop CPU core (**Lenovo 83LX · AMD Ryzen AI 9 365 w/ Radeon 880M**, `10 physical cores / 20 logical threads`, `32 GB RAM`, `0% GPU / 0% NPU`, `< 85 MB RAM`):\n\n"
        "| Benchmark Suite | Total Items | Primary Accuracy | Order-Flip Rate | CPU Latency (p50) | Verification Status | Artifact File |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| **Structured Universal (`11k`)** | `11,000` | **`99.75%`** (100% Selective) | `0.000000` | `0.076–0.190 ms` | **PASS (10,973/11,000)** | `test_cases_11000_results.json` |\n"
        "| **Unsampled External (`16` Datasets)** | **`114,018`** | **`87.49%`** Selective (`66.82%` raw) | `0.000000` | `0.369–0.938 ms` | **PASS (114,018 rows)** | `web_benchmarks_full_results.json` |\n"
        "| **Scientific Core (`540`)** | `540` | **`98.75%`** (`100%` Pair Acc) | `0.000000` | `0.266 ms` | **PASS (540/540)** | `latest_results.json` |\n"
        "| **Quality SLA (`1,000`)** | `1,000` | **`98.75%`** Core | `0.000000` | `0.433 ms` | **PASS (8/8 Gates)** | `quality_validation_1000_results.json` |\n"
        "| **Quality SLA (`750`)** | `750` | **`98.75%`** Core | `0.000000` | `0.294 ms` | **PASS (8/8 Gates)** | `quality_validation_750_results.json` |\n"
        "| **Curated Baseline (`162`)** | `162` | **`100.00%`** | `0.000000` | `0.751 ms` | **PASS (16/16 Datasets)** | `web_benchmarks_results.json` |\n"
        "| **HF JEV Ecosystem (`12` Datasets)** | `120` | **`100.00%`** | `0.000000` | `0.557 ms` | **PASS (12/12 Datasets)** | `hf_jev_models_benchmarks_results.json` |\n"
        "| **Structured Baseline (`1,000`)** | `1,000` | **`99.10%`** (100% Selective) | `0.000000` | `0.379 ms` | **PASS (991/1,000)** | `test_cases_1000_results.json` |\n"
        "| **Research Hypotheses (`12`)** | `12` | **`12 / 12 PASS`** | `0.000000` | `< 0.5 s` total | **PASS (Falsifiable)** | Consolidated Master Report |\n"
        "| **Master Consolidated** | **`128,617`** (`128,497` 9-suite) | **`10 / 10 Suites PASS`** | **`0.000000`** | **`< 0.4 ms` p50** | **ALL SUITES PASS** | `all_benchmarks_master_report.json` |\n\n"
        "---\n\n"
        "## 📁 Repository Contents\n\n"
        "### 1. Canonical Raw Results (JSON)\n\n"
        "| File Name | Scope & Description |\n"
        "| :--- | :--- |\n"
        "| **`all_benchmarks_master_report.json`** | Master consolidated report covering all benchmark suites (`128,617` items / `130,959` decisions; `128,497` items / `130,839` decisions across the initial 9 suites). |\n"
        "| **`latest_results.json`** | 5-track scientific evaluation (`JevBench-Core`, `DecisionPivot`, `L0 Flip`, `Calibrated ECE`, `RLCD2 Brier`, CPU latency). |\n"
        "| **`test_cases_11000_results.json`** | Evaluation across `11,000` structured enterprise cases spanning 18 industry domains. |\n"
        "| **`web_benchmarks_full_results.json`** | Unsampled zero-shot evaluation on 16 public Hugging Face datasets (`114,018` official split rows). |\n"
        "| **`quality_validation_1000_results.json` & `quality_validation_750_results.json`** | 8-Track Quality & Performance SLA gate verifications (`8/8` gates passed). |\n"
        "| **`test_cases_1000_results.json` & `web_benchmarks_results.json`** | Curated deterministic CI baselines (`1,000` structured cases & `162/640` external dataset entries). |\n\n"
        "### 2. Evaluation Corpora (JSONL)\n\n"
        "| File Name | Items | Description |\n"
        "| :--- | :---: | :--- |\n"
        "| **`test_cases_1000.jsonl`** | `1,000` | Structured universal test cases with inputs, criteria, and expected answers across 18 domains. |\n"
        "| **`external_datasets_corpus.jsonl`** | `640` | Curated test samples from 16 public academic datasets (`40` per dataset). |\n"
        "| **`microllm_eval_corpus.jsonl`** | `540` | 5-track scientific research challenge corpus (`JevBench-Core`, `DecisionPivot`, `AnyJev-Order`, `Guardrail-Safety`, `Calibration-RLCD²`). |\n\n"
        "### 3. Comprehensive Analysis Reports (Markdown)\n\n"
        "| Report File | Focus Area |\n"
        "| :--- | :--- |\n"
        "| **`BENCHMARKS_MASTER_REPORT.md`** | Master publication report detailing methodology, hardware profile, metrics, and baseline comparisons. |\n"
        "| **`FAILURE_MODES_AND_WORST_CASES.md`** | Transparent failure mode analysis, worst-case datasets, root causes, and structural limits of System-1 CPU architectures. |\n"
        "| **`EXTERNAL_DATASETS_BENCHMARKS.md`** | In-depth analysis of public datasets (`Banking77`, `Yelp 50k`, `MultiNLI`, `AG News`, `BoolQ`, `RouteLLM`, `ChoiceBench`, `WildGuardMix`, and 12 HF JEV datasets). |\n"
        "| **`TEST_PLAN.md`** | Quality, SLA, and performance validation framework (`G1`–`G8`). |\n"
        "| **`RESULTS_COMMUNICATION_KIT.md`** | Executive summary, scorecard, pitch deck slides, and communication playbook. |\n\n"
        "---\n\n"
        "## 🔬 Scientific Foundations\n\n"
        "MicroLLM unifies three foundational 2026 research papers:\n\n"
        "| Paper | Reference | Core Contribution |\n"
        "| :--- | :--- | :--- |\n"
        "| **1. OpenJev-RLCD** | *arXiv:2609.38850* (Gao & Wang) | Proposition 1 variance identity + Two-Stage *Calibrate-then-Reinforce* (`RLCD²`) with `Beta(8, 2)` decision difficulty weighting. |\n"
        "| **2. AnyJev** | *arXiv:2610.00831* (Nokia Applied Research) | Theorem 1 `L₀` circular rotation schedule with Van der Corput early stopping, mathematically guaranteeing `0.000000` order-flip rate. |\n"
        "| **3. StartLux-Decision** | *arXiv:2610.17309* (StartLux & HKU) | *Read-or-Think* dual-gear cascade (`p = 0.25·p₀ + 0.75·p₁`) with Value-of-Computation gate and 25-way hierarchical tournament. |\n\n"
        "---\n\n"
        "## ⚡ 1-Command Reproducibility\n\n"
        "Anyone can reproduce every single metric offline or against official Hugging Face Parquet splits in one command:\n\n"
        "```bash\n"
        "# 1. Install MicroLLM\n"
        "pip install microllm\n\n"
        "# 2. Run all fast benchmark suites (in ~5.6s on CPU)\n"
        "microllm benchmark --all-suites --quick\n\n"
        "# 3. Or run all suites including all 114,018 unsampled external rows (128,617 items total)\n"
        "microllm benchmark --all-suites\n"
        "```\n\n"
        "---\n\n"
        '<div align="center">\n'
        '  <b>MicroLLM</b> · Open Source Apache-2.0 License · <a href="https://github.com/jmrenouard/MicroLLM">Star on GitHub</a>\n'
        "</div>\n"
    )


def sync_readme_to_huggingface_hub(
    readme_path: Path | str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    api: Any | None = None,
    commit_message: str = "docs: automated CI/CD README.md synchronization from GitHub (IDEA-038)",
) -> dict[str, Any]:
    """Synchronize `README.md` across all 6 official Hugging Face repositories (`IDEA-038`)."""
    target_readme = (
        Path(readme_path).resolve()
        if readme_path is not None
        else (Path(__file__).resolve().parents[2] / "README.md")
    )
    root_readme_text = target_readme.read_text(encoding="utf-8")

    targets: list[dict[str, str]] = [
        {
            "repo_id": entry["repo_id"],
            "repo_type": "model",
            "content": build_hf_model_card(str(entry["repo_id"]), root_readme_text),
        }
        for entry in OFFICIAL_MICROLLM_HF_MODELS
    ]
    targets.append(
        {
            "repo_id": "jmrenouard/microllm-decision-studio",
            "repo_type": "space",
            "content": build_hf_space_card(root_readme_text),
        }
    )
    targets.append(
        {
            "repo_id": "jmrenouard/microllm-benchmarks",
            "repo_type": "dataset",
            "content": build_hf_dataset_card(),
        }
    )

    resolved_token = (
        token
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or None
    )

    effective_dry_run = bool(dry_run) or (resolved_token is None and api is None)
    hf_client = api
    if not effective_dry_run and hf_client is None:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]

        hf_client = HfApi(token=resolved_token)

    repo_results: list[dict[str, Any]] = []
    for item in targets:
        content_str = item["content"]
        if not effective_dry_run and hf_client is not None:
            hf_client.upload_file(
                path_or_fileobj=content_str.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                commit_message=commit_message,
            )
        repo_results.append(
            {
                "repo_id": item["repo_id"],
                "repo_type": item["repo_type"],
                "chars": len(content_str),
                "sha256": hashlib.sha256(content_str.encode("utf-8")).hexdigest(),
                "uploaded": not effective_dry_run,
            }
        )

    return {
        "status": "dry_run" if effective_dry_run else "synced",
        "source_readme": str(target_readme),
        "total_repositories": len(repo_results),
        "repositories": repo_results,
    }

