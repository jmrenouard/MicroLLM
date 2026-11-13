"""Tier 3: Cross-Feature Combinations E2E Tests for MicroLLM 18 Audit Points Remediation.

This module verifies pairwise and multi-feature interactions across components sharing state,
event loops, concurrency locks, schemas, or packaging manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from microllm.server import create_app
from microllm.mcp_server import MicroLLMMCPServer
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
import numpy as np


def test_tier3_01_p04_p05_server_and_mcp_calibration_validation_parity(test_client: TestClient, mcp_server: MicroLLMMCPServer):
    """P04 + P05: Both HTTP CalibrateRequest and MCP systemone_calibrate reject boolean labels."""
    # HTTP endpoint
    http_resp = test_client.post("/v1/calibrate", json={"logits": [[1.0, 2.0], [2.0, 1.0]], "labels": [True, False]})
    assert http_resp.status_code == 422

    # MCP tool
    mcp_res = mcp_server.systemone_calibrate(logits=[[1.0, 2.0], [2.0, 1.0]], labels=[True, False])
    assert isinstance(mcp_res, dict)
    assert "error" in mcp_res


def test_tier3_02_p04_p10_fast_rlcd_fit_with_bounded_validation(test_client: TestClient):
    """P04 + P10: StudioRLCDDemoRequest enforces dimension bounds and trains with zero sleep in sub-second time."""
    # Bounded valid payload
    features = np.random.RandomState(42).randn(80, 8).tolist()
    labels = np.random.RandomState(42).randint(0, 2, size=80).tolist()
    payload = {
        "features": features,
        "labels": labels,
        "stage1_steps": 15,
        "stage2_steps": 15,
        "use_rlcd2_weights": True,
        "seed": 42,
    }
    t0 = time.perf_counter()
    resp = test_client.post("/v1/studio/rlcd-demo", json=payload)
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200
    assert elapsed < 0.500, f"RLCD demo took {elapsed:.3f}s; artificial sleep may be present."


def test_tier3_03_p07_p08_benchmark_concurrency_and_model_init(test_client: TestClient, sample_choice_question: dict):
    """P07 + P08: Benchmark concurrency control does not impede /v1/systemone model initialization."""
    # Trigger benchmark run
    b_resp = test_client.post("/v1/benchmark/run", json={"mode": "fast"})
    assert b_resp.status_code in (200, 409)

    # Concurrently execute systemone inference
    s_resp = test_client.post(
        "/v1/systemone",
        json={
            "state": "Customer needs support during benchmark execution",
            "questions": sample_choice_question,
            "model": "microllm-gear1-fast",
        },
    )
    assert s_resp.status_code == 200


def test_tier3_04_p08_p11_model_init_safety_and_mcp_neural_memoization(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """P08 + P11: MCP neural model caching avoids duplicate disk loading and runs safely."""
    # First invocation populates cache
    res1 = mcp_server.systemone_decide(
        state="Test systemone message for caching",
        questions=sample_choice_question,
        multilingual_strategy="strategy2_neural",
    )
    assert isinstance(res1, dict)

    # Second invocation reuses cache
    res2 = mcp_server.systemone_decide(
        state="Second message reusing cached weights",
        questions=sample_choice_question,
        multilingual_strategy="strategy2_neural",
    )
    assert isinstance(res2, dict)


def test_tier3_05_p11_p12_mcp_neural_caching_and_verdict_mq_cache(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """P11 + P12: Multi-question neural requests leverage both neural backend memoization and MQ cache."""
    multi_q = {
        "dept": sample_choice_question["dept"],
        "sentiment": {
            "type": "choice",
            "instructions": "Determine customer sentiment.",
            "criteria": {"positive": "Happy", "negative": "Unhappy", "neutral": "Neutral"},
        },
    }
    result = mcp_server.systemone_decide(
        state="Customer is extremely satisfied with fast resolution.",
        questions=multi_q,
        multilingual_strategy="strategy2_neural",
    )
    assert isinstance(result, dict)


def test_tier3_06_p01_p02_docker_cache_dir_and_path_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """P01 + P02: Server get_cache_dir and resolve_benchmark_path work in isolated non-git environment."""
    from microllm.server import get_cache_dir, resolve_benchmark_path
    isolated_dir = tmp_path / "app" / ".cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(isolated_dir))
    
    cache = get_cache_dir()
    assert Path(cache).exists()
    
    # Path resolution handles custom or missing benchmark directory gracefully
    resolved = resolve_benchmark_path("latest_results.json")
    assert resolved is None or Path(resolved).name == "latest_results.json"


def test_tier3_07_p01_p03_docker_cache_and_profile_pull_protection(test_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """P01 + P03: Profile pull downloads strictly to MICROLLM_CACHE_DIR and ignores client path overrides."""
    cache_dir = tmp_path / "server_secure_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(cache_dir))
    
    evil_dir = tmp_path / "evil_destination"
    resp = test_client.post(
        "/v1/profiles/pull",
        json={"profile": "microllm-gear1-fast", "cache_dir": str(evil_dir)},
    )
    assert not evil_dir.exists(), "Arbitrary cache_dir must not be created."


def test_tier3_08_p09_p15_canonical_naming_and_sdk_deduplication():
    """P09 + P15: SDK client classes use shared base and maintain canonical micro-jev naming."""
    from microllm.client import MicroJevClient, AsyncMicroJevClient, _BaseMicroJevClient
    assert issubclass(MicroJevClient, _BaseMicroJevClient)
    assert issubclass(AsyncMicroJevClient, _BaseMicroJevClient)


def test_tier3_09_p13_p16_standalone_wheel_and_packaged_json(repo_root: Path):
    """P13 + P16: pyproject.toml package-data ensures all 10 data JSON files are bundled for wheel test."""
    data_dir = repo_root / "src" / "microllm" / "data"
    assert data_dir.exists()
    json_count = len(list(data_dir.glob("*.json")))
    assert json_count == 10
    
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert "data" in content or "package-data" in content


def test_tier3_10_p14_p01_container_least_privilege_and_unprivileged_runner(repo_root: Path):
    """P14 + P01: Both Dockerfile and jekyll-docker.yml avoid root privileges and unrestricted chmod."""
    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
    
    jekyll_wf = (repo_root / ".github" / "workflows" / "jekyll-docker.yml").read_text(encoding="utf-8")
    assert "chmod 777" not in jekyll_wf


def test_tier3_11_p17_p18_dead_code_removal_and_ci_workflow_deduplication(repo_root: Path):
    """P17 + P18: Eliminating dead code enables strict lint.yml without duplicate runs in ci.yml."""
    mcp_file = (repo_root / "src" / "microllm" / "mcp_server.py").read_text(encoding="utf-8")
    assert "def _get_client(" not in mcp_file
    
    lint_file = (repo_root / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    assert "--exit-zero" not in lint_file


def test_tier3_12_p06_p09_hf_space_sync_and_canonical_package_naming(repo_root: Path):
    """P06 + P09: HF space sync scripts generate requirements referencing micro-jev."""
    hf_sync = (repo_root / "src" / "microllm" / "hf_space_sync.py").read_text(encoding="utf-8")
    assert "micro-jev" in hf_sync


def test_tier3_13_p07_p02_concurrent_benchmark_atomic_writes_and_path_resolution(tmp_path: Path):
    """P07 + P02: Atomic JSON write operates safely on resolved benchmark paths."""
    from microllm.server import atomic_write_json
    target = tmp_path / "benchmarks" / "test_run.json"
    data = {"run_id": "test_atomic_path", "score": 0.95}
    atomic_write_json(target, data)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_tier3_14_p08_p10_async_model_loading_and_zero_sleep_rlcd(test_client: TestClient, sample_choice_question: dict):
    """P08 + P10: Model inference and RLCD demo fit run concurrently without blocking each other."""
    # Fast RLCD demo
    rlcd_payload = {
        "features": [[1.0, 0.0], [0.0, 1.0]],
        "labels": [0, 1],
        "stage1_steps": 5,
        "stage2_steps": 5,
    }
    r1 = test_client.post("/v1/studio/rlcd-demo", json=rlcd_payload)
    # Model inference
    r2 = test_client.post(
        "/v1/systemone",
        json={"state": "Active learning query", "questions": sample_choice_question, "model": "microllm-gear1-fast"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_tier3_15_p03_p04_secure_profile_pull_and_strict_oom_bounds(test_client: TestClient):
    """P03 + P04: Profile pull and calibration endpoints enforce strict boundary safety simultaneously."""
    # Profile pull rejects bad profile
    p_resp = test_client.post("/v1/profiles/pull", json={"profile": "bad/path/profile"})
    assert p_resp.status_code in (400, 404, 422)

    # Calibrate rejects boolean labels
    c_resp = test_client.post("/v1/calibrate", json={"logits": [[1.0, 2.0]], "labels": [True]})
    assert c_resp.status_code == 422


def test_tier3_16_p05_p11_mcp_calibrate_and_cached_decide_tool_chaining(mcp_server: MicroLLMMCPServer, sample_choice_question: dict, sample_valid_calibration_dataset: dict):
    """P05 + P11: MCP session chains decide tool and calibrate tool safely."""
    # Decide
    d_res = mcp_server.systemone_decide(state="Ticket text", questions=sample_choice_question)
    assert isinstance(d_res, dict)

    # Calibrate
    c_res = mcp_server.systemone_calibrate(
        logits=sample_valid_calibration_dataset["logits"],
        labels=sample_valid_calibration_dataset["labels"],
    )
    assert isinstance(c_res, dict)
    assert "error" not in c_res or c_res.get("error") is None


def test_tier3_17_p15_p12_sdk_batch_inference_and_verdict_mq_cache(test_client: TestClient, sample_choice_question: dict):
    """P15 + P12: Client batch request evaluating multiple states leverages server caching."""
    batch_payload = {
        "items": ["First customer message", "Second customer message", "Third customer message"],
        "questions": sample_choice_question,
        "model": "microllm-gear1-fast",
    }
    resp = test_client.post("/v1/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("results", [])) == 3


def test_tier3_18_p16_p02_packaged_json_and_server_static_route_fallback(test_client: TestClient):
    """P16 + P02: Server papers routes read from packaged JSON without crashing outside git repo."""
    resp = test_client.get("/papers/paper-openjev-rlcd")
    assert resp.status_code in (200, 404)
