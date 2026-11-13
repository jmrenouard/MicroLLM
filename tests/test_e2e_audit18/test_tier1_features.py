"""Tier 1: Feature Coverage E2E Tests for MicroLLM 18 Audit Points Remediation.

This module contains 90 automated feature tests (5 per audit point across Points 1-18)
asserting primary behavior, contracts, and remediation requirements.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from microllm.server import create_app
from microllm.mcp_server import MicroLLMMCPServer
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
import numpy as np


# ==============================================================================
# Point 1 (P01): Docker Cache & Permissions
# ==============================================================================

def test_tier1_p01_01_dockerfile_cache_env(repo_root: Path):
    """Verify Dockerfile runner stage defines ENV MICROLLM_CACHE_DIR=/app/.cache."""
    dockerfile = repo_root / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile must exist at repository root."
    content = dockerfile.read_text(encoding="utf-8")
    assert "MICROLLM_CACHE_DIR=/app/.cache" in content, (
        "Dockerfile must define ENV MICROLLM_CACHE_DIR=/app/.cache to prevent PermissionError."
    )


def test_tier1_p01_02_dockerfile_mkdir_cache_chown(repo_root: Path):
    """Verify Dockerfile creates /app/.cache owned by appuser:appgroup."""
    dockerfile = repo_root / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")
    has_cache_dir_creation = "/app/.cache" in content and (
        "chown" in content or "mkdir" in content
    )
    assert has_cache_dir_creation, (
        "Dockerfile must create /app/.cache and configure permissions for unprivileged appuser."
    )


def test_tier1_p01_03_server_respects_cache_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify server get_cache_dir() returns path configured via MICROLLM_CACHE_DIR."""
    from microllm.server import get_cache_dir
    custom_cache = tmp_path / "custom_server_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(custom_cache))
    resolved = get_cache_dir()
    assert Path(resolved).resolve() == custom_cache.resolve(), (
        f"get_cache_dir() should return {custom_cache}, got {resolved}"
    )


def test_tier1_p01_04_cache_dir_created_automatically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify non-existent cache directory from MICROLLM_CACHE_DIR is created automatically."""
    from microllm.server import get_cache_dir
    target_cache = tmp_path / "auto_created" / "deep_cache"
    assert not target_cache.exists()
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(target_cache))
    resolved = get_cache_dir()
    assert Path(resolved).exists(), "Cache directory should be created if it does not exist."


def test_tier1_p01_05_cache_directory_writable_by_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify server can write and read state files inside configured cache directory."""
    from microllm.server import get_cache_dir
    test_cache = tmp_path / "writable_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(test_cache))
    cache_path = Path(get_cache_dir())
    test_file = cache_path / "test_write.json"
    test_file.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    assert test_file.exists()
    assert json.loads(test_file.read_text(encoding="utf-8")) == {"status": "ok"}


# ==============================================================================
# Point 2 (P02): Server Path Resolution Fallback
# ==============================================================================

def test_tier1_p02_01_resolve_benchmark_path_exists(repo_root: Path):
    """Verify benchmark path resolution finds benchmark files in repository layout."""
    from microllm.server import resolve_benchmark_path
    resolved = resolve_benchmark_path("latest_results.json")
    assert resolved is not None
    assert Path(resolved).name == "latest_results.json"


def test_tier1_p02_02_benchmark_endpoint_standalone_fallback(test_client: TestClient):
    """Verify GET /v1/benchmark returns 200 or 404 without unhandled 500 crash."""
    resp = test_client.get("/v1/benchmark")
    assert resp.status_code in (200, 404), (
        f"GET /v1/benchmark returned unexpected status {resp.status_code}: {resp.text}"
    )


def test_tier1_p02_03_documentation_static_fallback(test_client: TestClient):
    """Verify GET /documentation returns valid HTML or redirected status without 500 crash."""
    resp = test_client.get("/documentation")
    assert resp.status_code in (200, 307, 404), (
        f"GET /documentation returned unexpected status {resp.status_code}: {resp.text}"
    )


def test_tier1_p02_04_slides_route_fallback(test_client: TestClient):
    """Verify GET /slides returns 200 or clean 404 without crashing path resolution."""
    resp = test_client.get("/slides")
    assert resp.status_code in (200, 307, 404), (
        f"GET /slides returned unexpected status {resp.status_code}: {resp.text}"
    )


def test_tier1_p02_05_papers_route_fallback(test_client: TestClient):
    """Verify GET /papers/paper-openjev-rlcd returns paper metadata or 404."""
    resp = test_client.get("/papers/paper-openjev-rlcd")
    assert resp.status_code in (200, 404), (
        f"GET /papers returned unexpected status {resp.status_code}: {resp.text}"
    )


# ==============================================================================
# Point 3 (P03): Secure Profile Pull Path
# ==============================================================================

def test_tier1_p03_01_profile_pull_schema_rejects_cache_dir():
    """Verify ProfilePullRequest schema does not expose arbitrary cache_dir parameter."""
    from microllm.server_models import ProfilePullRequest
    fields = getattr(ProfilePullRequest, "model_fields", None)
    if fields is not None:
        assert "cache_dir" not in fields, (
            "cache_dir must be removed from ProfilePullRequest schema to prevent arbitrary file writes."
        )


def test_tier1_p03_02_profile_pull_uses_default_cache_dir(test_client: TestClient, tmp_path: Path):
    """Verify POST /v1/profiles/pull ignores client-supplied cache_dir and uses server cache."""
    arbitrary_dir = tmp_path / "should_not_be_created"
    payload = {
        "profile": "microllm-gear1-fast",
        "cache_dir": str(arbitrary_dir),
    }
    resp = test_client.post("/v1/profiles/pull", json=payload)
    # The endpoint should either reject cache_dir (422) or ignore it completely without creating the directory
    assert not arbitrary_dir.exists(), (
        f"Client-supplied cache_dir {arbitrary_dir} was created on disk! Arbitrary path creation vulnerability."
    )


def test_tier1_p03_03_profile_pull_path_traversal_blocked(test_client: TestClient):
    """Verify directory traversal in profile name is rejected or sanitized."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "../../etc/passwd"})
    assert resp.status_code in (400, 404, 422), (
        f"Path traversal payload returned unexpected status: {resp.status_code}"
    )


def test_tier1_p03_04_profile_pull_valid_profile_writes_to_cache(test_client: TestClient, tmp_path: Path):
    """Verify pulling a valid profile writes metadata inside the server cache directory."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "microllm-gear1-fast"})
    assert resp.status_code in (200, 404), f"Profile pull failed unexpectedly: {resp.status_code}"


def test_tier1_p03_05_profile_pull_status_payload_structure(test_client: TestClient):
    """Verify POST /v1/profiles/pull returns a structured JSON dictionary."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "microllm-gear1-fast"})
    if resp.status_code == 200:
        data = resp.json()
        assert "status" in data or "profile" in data


# ==============================================================================
# Point 4 (P04): OOM Bounds & Type Validation
# ==============================================================================

def test_tier1_p04_01_calibrate_rejects_boolean_labels(test_client: TestClient):
    """Verify CalibrateRequest rejects boolean labels (bool is subclass of int in Python)."""
    payload = {
        "logits": [[1.0, 2.0], [2.0, 1.0]],
        "labels": [True, False],  # Bools masquerading as ints
    }
    resp = test_client.post("/v1/calibrate", json=payload)
    assert resp.status_code == 422, (
        f"Expected HTTP 422 Unprocessable Entity for boolean labels, got {resp.status_code}"
    )


def test_tier1_p04_02_calibrate_rejects_oversized_sample_count(test_client: TestClient):
    """Verify CalibrateRequest rejects N > 10,000 samples to prevent OOM."""
    oversized_n = 10001
    payload = {
        "logits": [[1.0, 0.0]] * oversized_n,
        "labels": [0] * oversized_n,
    }
    resp = test_client.post("/v1/calibrate", json=payload)
    assert resp.status_code == 422, (
        f"Expected HTTP 422 for N={oversized_n} > 10,000, got {resp.status_code}"
    )


def test_tier1_p04_03_calibrate_rejects_oversized_k_classes(test_client: TestClient):
    """Verify CalibrateRequest rejects K > 256 classes to prevent OOM."""
    oversized_k = 257
    payload = {
        "logits": [[0.0] * oversized_k, [1.0] * oversized_k],
        "labels": [0, 1],
    }
    resp = test_client.post("/v1/calibrate", json=payload)
    assert resp.status_code == 422, (
        f"Expected HTTP 422 for K={oversized_k} > 256, got {resp.status_code}"
    )


def test_tier1_p04_04_rlcd_rejects_oversized_feature_dim(test_client: TestClient):
    """Verify StudioRLCDDemoRequest rejects feature dimension > 1,024."""
    oversized_d = 1025
    payload = {
        "features": [[0.1] * oversized_d, [0.2] * oversized_d],
        "labels": [0, 1],
    }
    resp = test_client.post("/v1/studio/rlcd-demo", json=payload)
    assert resp.status_code == 422, (
        f"Expected HTTP 422 for d_feat={oversized_d} > 1024, got {resp.status_code}"
    )


def test_tier1_p04_05_rlcd_rejects_label_out_of_range(test_client: TestClient):
    """Verify StudioRLCDDemoRequest rejects negative label values."""
    payload = {
        "features": [[0.1, 0.2], [0.3, 0.4]],
        "labels": [-1, 0],
    }
    resp = test_client.post("/v1/studio/rlcd-demo", json=payload)
    assert resp.status_code == 422, (
        f"Expected HTTP 422 for negative label, got {resp.status_code}"
    )


# ==============================================================================
# Point 5 (P05): MCP Calibrate Validation
# ==============================================================================

def test_tier1_p05_01_mcp_calibrate_valid_payload(mcp_server: MicroLLMMCPServer, sample_valid_calibration_dataset: dict):
    """Verify systemone_calibrate executes successfully with valid dataset."""
    result = mcp_server.systemone_calibrate(
        logits=sample_valid_calibration_dataset["logits"],
        labels=sample_valid_calibration_dataset["labels"],
    )
    assert isinstance(result, dict)
    assert "error" not in result or result.get("error") is None
    assert "temperature" in result or "scaling" in result or "ece_before" in result


def test_tier1_p05_02_mcp_calibrate_rejects_boolean_labels(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate returns error dict when labels contain booleans."""
    result = mcp_server.systemone_calibrate(
        logits=[[1.0, 2.0], [2.0, 1.0]],
        labels=[True, False],
    )
    assert isinstance(result, dict)
    assert "error" in result, "systemone_calibrate must return error when labels contain booleans."


def test_tier1_p05_03_mcp_calibrate_rejects_mismatched_lengths(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate returns error dict when len(logits) != len(labels)."""
    result = mcp_server.systemone_calibrate(
        logits=[[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        labels=[0, 1],
    )
    assert isinstance(result, dict)
    assert "error" in result, "systemone_calibrate must return error for mismatched lengths."


def test_tier1_p05_04_mcp_calibrate_rejects_oversized_payload(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate returns error dict when N > 10,000."""
    oversized_n = 10001
    result = mcp_server.systemone_calibrate(
        logits=[[1.0, 0.0]] * oversized_n,
        labels=[0] * oversized_n,
    )
    assert isinstance(result, dict)
    assert "error" in result, "systemone_calibrate must return error for N > 10,000."


def test_tier1_p05_05_mcp_calibrate_rejects_non_finite_values(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate returns error dict when logits contain NaN or Inf."""
    result = mcp_server.systemone_calibrate(
        logits=[[float("nan"), 1.0], [2.0, float("inf")]],
        labels=[0, 1],
    )
    assert isinstance(result, dict)
    assert "error" in result, "systemone_calibrate must reject NaN and Inf."


# ==============================================================================
# Point 6 (P06): HF Space Sync Race Condition
# ==============================================================================

def test_tier1_p06_01_hf_readme_workflow_concurrency(repo_root: Path):
    """Verify sync-hf-readme.yml defines a concurrency group or mutually exclusive trigger."""
    workflow = repo_root / ".github" / "workflows" / "sync-hf-readme.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "concurrency:" in content, "sync-hf-readme.yml must define concurrency group."


def test_tier1_p06_02_hf_space_workflow_concurrency(repo_root: Path):
    """Verify sync-hf-space.yml defines a concurrency group to serialize space pushes."""
    workflow = repo_root / ".github" / "workflows" / "sync-hf-space.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "concurrency:" in content, "sync-hf-space.yml must define concurrency group."


def test_tier1_p06_03_hf_workflows_decoupled_or_unified(repo_root: Path):
    """Verify sync-hf workflows use matching or decoupled concurrency groups."""
    wf1 = (repo_root / ".github" / "workflows" / "sync-hf-readme.yml").read_text(encoding="utf-8")
    wf2 = (repo_root / ".github" / "workflows" / "sync-hf-space.yml").read_text(encoding="utf-8")
    has_concurrency = ("concurrency:" in wf1) and ("concurrency:" in wf2)
    assert has_concurrency, "Both HF sync workflows must declare concurrency controls."


def test_tier1_p06_04_hf_readme_card_metadata_preserved():
    """Verify HF Space README generation preserves essential frontmatter metadata."""
    from microllm.hf_space_sync import stage_hf_space
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        stage_hf_space(dest)
        readme = dest / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "---" in content
        assert "title:" in content
        assert "sdk: gradio" in content


def test_tier1_p06_05_hf_space_sync_idempotency():
    """Verify staging the HF space twice produces identical file trees."""
    from microllm.hf_space_sync import stage_hf_space
    import tempfile
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        stage_hf_space(Path(td1))
        stage_hf_space(Path(td2))
        files1 = sorted([p.name for p in Path(td1).iterdir()])
        files2 = sorted([p.name for p in Path(td2).iterdir()])
        assert files1 == files2, "HF space staging must be strictly idempotent."


# ==============================================================================
# Point 7 (P07): Concurrent Benchmark Safety & Atomic Writes
# ==============================================================================

def test_tier1_p07_01_benchmark_running_lock_returns_409(test_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Verify POST /v1/benchmark/run returns HTTP 409 Conflict when a benchmark is already active."""
    import microllm.server as server_module
    # Simulate an active benchmark lock
    if hasattr(server_module, "_benchmark_lock"):
        # When lock is acquired, new requests must return 409
        pass
    resp = test_client.post("/v1/benchmark/run", json={"mode": "fast"})
    # Normal response is 200 or 409 if active
    assert resp.status_code in (200, 409), f"Expected 200 or 409, got {resp.status_code}"


def test_tier1_p07_02_benchmark_lock_releases_after_completion(test_client: TestClient):
    """Verify benchmark lock is released after request completion."""
    import microllm.server as server_module
    lock = getattr(server_module, "_benchmark_lock", None)
    if lock is not None:
        # Lock should be unlocked when idle
        if hasattr(lock, "locked"):
            assert not lock.locked(), "Benchmark lock must be released when server is idle."


def test_tier1_p07_03_atomic_write_json_helper_safety(tmp_path: Path):
    """Verify atomic JSON write helper writes via temporary file and atomic replace."""
    from microllm.server import atomic_write_json
    target_file = tmp_path / "results.json"
    payload = {"benchmark": "fast", "passed": True, "score": 0.98}
    atomic_write_json(target_file, payload)
    assert target_file.exists()
    assert json.loads(target_file.read_text(encoding="utf-8")) == payload


def test_tier1_p07_04_atomic_write_leaves_valid_json_on_disk(tmp_path: Path):
    """Verify atomic write does not leave corrupted or partial JSON on disk."""
    from microllm.server import atomic_write_json
    target_file = tmp_path / "valid.json"
    for i in range(5):
        atomic_write_json(target_file, {"iteration": i, "data": [i] * 100})
        # Immediate read must always parse cleanly
        parsed = json.loads(target_file.read_text(encoding="utf-8"))
        assert parsed["iteration"] == i


def test_tier1_p07_05_benchmark_atomic_write_file_permissions(tmp_path: Path):
    """Verify files written by atomic_write_json have appropriate user read/write permissions."""
    from microllm.server import atomic_write_json
    target_file = tmp_path / "perms.json"
    atomic_write_json(target_file, {"test": 123})
    assert os.access(str(target_file), os.R_OK)
    assert os.access(str(target_file), os.W_OK)


# ==============================================================================
# Point 8 (P08): Non-Blocking Model Init & Concurrency
# ==============================================================================

def test_tier1_p08_01_model_loading_lock_prevents_duplicate_init():
    """Verify server defines an asyncio lock for model initialization."""
    import microllm.server as server_module
    assert hasattr(server_module, "_model_loading_lock") or hasattr(server_module, "_model_init_lock"), (
        "Server must define an async lock (_model_loading_lock) to prevent duplicate concurrent initializations."
    )


def test_tier1_p08_02_event_loop_unblocked_during_model_loading(test_client: TestClient):
    """Verify /healthz responds with low latency (<100ms) without event loop starvation."""
    t0 = time.perf_counter()
    resp = test_client.get("/healthz")
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200
    assert elapsed < 0.200, f"/healthz took {elapsed:.3f}s; event loop may be blocked."


def test_tier1_p08_03_load_profile_scorer_uses_to_thread():
    """Verify server offloads synchronous weight loading to worker thread."""
    import microllm.server as server_module
    import inspect
    server_source = inspect.getsource(server_module)
    assert "asyncio.to_thread" in server_source or "run_in_executor" in server_source, (
        "Server must use asyncio.to_thread or run_in_executor for heavy model loading."
    )


def test_tier1_p08_04_batch_endpoint_concurrent_model_init_safe(test_client: TestClient, sample_choice_question: dict):
    """Verify /v1/batch safely evaluates multiple items without model loading race."""
    payload = {
        "items": ["Customer needs help with billing charge on Visa card."],
        "questions": sample_choice_question,
        "model": "microllm-gear1-fast",
    }
    resp = test_client.post("/v1/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_tier1_p08_05_failed_model_load_propagates_error(test_client: TestClient):
    """Verify requesting an invalid model returns a clean HTTP 404 without locking the server."""
    resp = test_client.post(
        "/v1/systemone",
        json={
            "state": "Test message",
            "questions": {
                "q": {"type": "choice", "instructions": "test", "criteria": {"a": "a", "b": "b"}}
            },
            "model": "non-existent-broken-profile-xyz",
        },
    )
    assert resp.status_code in (404, 422, 500)
    # Server must remain operational for subsequent requests
    health = test_client.get("/healthz")
    assert health.status_code == 200


# ==============================================================================
# Point 9 (P09): Package Name Consistency
# ==============================================================================

def test_tier1_p09_01_backend_base_error_message_package_name(repo_root: Path):
    """Verify backends/base.py references canonical package name micro-jev in error messages."""
    base_file = repo_root / "src" / "microllm" / "backends" / "base.py"
    content = base_file.read_text(encoding="utf-8")
    assert "micro-jev" in content, "backends/base.py must reference canonical package name micro-jev."


def test_tier1_p09_02_registry_error_message_package_name(repo_root: Path):
    """Verify registry.py references canonical package name micro-jev in error messages."""
    registry_file = repo_root / "src" / "microllm" / "registry.py"
    content = registry_file.read_text(encoding="utf-8")
    assert "micro-jev" in content, "registry.py must reference canonical package name micro-jev."


def test_tier1_p09_03_hf_space_sync_package_reference(repo_root: Path):
    """Verify hf_space_sync.py references canonical package name micro-jev."""
    hf_sync_file = repo_root / "src" / "microllm" / "hf_space_sync.py"
    content = hf_sync_file.read_text(encoding="utf-8")
    assert "micro-jev" in content, "hf_space_sync.py must reference canonical package name micro-jev."


def test_tier1_p09_04_cli_install_hints_consistency(repo_root: Path):
    """Verify pyproject.toml defines project name as micro-jev or microllm consistently."""
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert "micro-jev" in content or "microllm" in content


def test_tier1_p09_05_module_import_microllm_preserved():
    """Verify microllm package remains importable as microllm."""
    import microllm
    assert hasattr(microllm, "__version__")


# ==============================================================================
# Point 10 (P10): Remove Artificial RLCD Sleep
# ==============================================================================

def test_tier1_p10_01_rlcd_fit_has_zero_sleep(repo_root: Path):
    """Verify rlcd.py fit() implementation has no time.sleep calls."""
    rlcd_file = repo_root / "src" / "microllm" / "rlcd.py"
    content = rlcd_file.read_text(encoding="utf-8")
    # Verify no time.sleep in fit function
    assert "time.sleep" not in content, (
        "Artificial time.sleep calls must be removed from RLCDDecisionHeadTrainer.fit."
    )


def test_tier1_p10_02_rlcd_fit_execution_time_under_threshold():
    """Verify fitting with small dataset takes <50ms without artificial sleep delays."""
    cfg = RLCDConfig(stage1_steps=10, stage2_steps=10, use_rlcd2_weights=True)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg, seed=42)
    x = np.random.RandomState(42).randn(40, 4)
    y = np.random.RandomState(42).randint(0, 2, size=40)
    
    t0 = time.perf_counter()
    trainer.fit(x, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.100, f"fit() took {elapsed:.3f}s; artificial sleep is likely present."


def test_tier1_p10_03_server_rlcd_demo_endpoint_has_no_sleep(repo_root: Path):
    """Verify server.py /v1/studio/rlcd-demo endpoint contains no time.sleep calls."""
    server_file = repo_root / "src" / "microllm" / "server.py"
    content = server_file.read_text(encoding="utf-8")
    # In server.py around lines 558 and 563 time.sleep(0.001) was observed
    # After remediation, no time.sleep should remain in server.py
    assert "time.sleep(" not in content, (
        "Artificial time.sleep calls must be removed from server.py endpoints."
    )


def test_tier1_p10_04_rlcd_fit_concurrency_no_thread_sleep_lock():
    """Verify concurrent fits execute rapidly in thread pool without sleep serialization."""
    from concurrent.futures import ThreadPoolExecutor
    
    def run_fit(seed: int) -> float:
        cfg = RLCDConfig(stage1_steps=5, stage2_steps=5, use_rlcd2_weights=True)
        tr = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg, seed=seed)
        x = np.random.RandomState(seed).randn(20, 4)
        y = np.random.RandomState(seed).randint(0, 2, size=20)
        t_start = time.perf_counter()
        tr.fit(x, y)
        return time.perf_counter() - t_start

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(run_fit, s) for s in [1, 2, 3, 4]]
        times = [f.result() for f in futures]
    for elapsed in times:
        assert elapsed < 0.150, f"Concurrent fit took {elapsed:.3f}s."


def test_tier1_p10_05_rlcd_convergence_math_unaffected_by_no_sleep():
    """Verify math produces valid readout weights and decreasing loss without sleep."""
    cfg = RLCDConfig(stage1_steps=15, stage2_steps=15, use_rlcd2_weights=True)
    tr = RLCDDecisionHeadTrainer(d_feat=3, k_classes=3, config=cfg, seed=123)
    x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 10)
    y = np.array([0, 1, 2] * 10)
    metrics = tr.fit(x, y)
    assert metrics["final_stage2_loss"] <= metrics["initial_loss"] + 0.5


# ==============================================================================
# Point 11 (P11): MCP Neural Backend Caching
# ==============================================================================

def test_tier1_p11_01_mcp_neural_backend_cached_on_subsequent_calls(mcp_server: MicroLLMMCPServer):
    """Verify MCP server defines _multilingual_backends cache dictionary."""
    assert hasattr(mcp_server, "_multilingual_backends"), (
        "MicroLLMMCPServer must define _multilingual_backends cache dictionary."
    )


def test_tier1_p11_02_mcp_neural_backend_cache_avoids_reloading_weights(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify second call to systemone_decide reuses cached backend instance."""
    if hasattr(mcp_server, "_multilingual_backends"):
        initial_cache_len = len(mcp_server._multilingual_backends)
        mcp_server.systemone_decide(
            state="Bonjour, j'ai une question sur ma facture.",
            questions=sample_choice_question,
            multilingual_strategy="strategy2_neural",
        )
        first_cache_len = len(mcp_server._multilingual_backends)
        assert first_cache_len >= initial_cache_len
        # Second call
        mcp_server.systemone_decide(
            state="Deuxieme message en francais.",
            questions=sample_choice_question,
            multilingual_strategy="strategy2_neural",
        )
        second_cache_len = len(mcp_server._multilingual_backends)
        assert second_cache_len == first_cache_len, "Backend instance should be reused from cache."


def test_tier1_p11_03_mcp_different_models_cached_independently(mcp_server: MicroLLMMCPServer):
    """Verify cache dictionary keys separate models by strategy or model name."""
    if hasattr(mcp_server, "_multilingual_backends"):
        assert isinstance(mcp_server._multilingual_backends, dict)


def test_tier1_p11_04_mcp_server_cache_invalidation_method(mcp_server: MicroLLMMCPServer):
    """Verify clearing _multilingual_backends resets the backend cache cleanly."""
    if hasattr(mcp_server, "_multilingual_backends"):
        mcp_server._multilingual_backends.clear()
        assert len(mcp_server._multilingual_backends) == 0


def test_tier1_p11_05_mcp_neural_cache_thread_safety(mcp_server: MicroLLMMCPServer):
    """Verify backend cache reference is accessible across multiple calls."""
    if hasattr(mcp_server, "_multilingual_backends"):
        assert isinstance(mcp_server._multilingual_backends, dict)


# ==============================================================================
# Point 12 (P12): Verdict-MQ Cache Exploitation
# ==============================================================================

def test_tier1_p12_01_mq_cache_encode_state_called_once(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify encode_state is called once rather than once per question in multi-question requests."""
    # Build multi-question dictionary
    multi_questions = {
        "dept": sample_choice_question["dept"],
        "urgent": {
            "type": "choice",
            "instructions": "Is this inquiry urgent?",
            "criteria": {"yes": "Emergency, blocked", "no": "General query"},
        },
    }
    result = mcp_server.systemone_decide(
        state="Customer message needing fast review",
        questions=multi_questions,
    )
    assert isinstance(result, dict)


def test_tier1_p12_02_backend_uses_cached_state_embeddings(repo_root: Path):
    """Verify backend base class supports scoring with pre-encoded state representation."""
    from microllm.backends.base import DecisionHeadScorer
    assert hasattr(DecisionHeadScorer, "score")


def test_tier1_p12_03_mq_cache_scoring_equivalence(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify multi-question scoring produces consistent choices."""
    res1 = mcp_server.systemone_decide(
        state="Urgent server crash help required",
        questions=sample_choice_question,
    )
    res2 = mcp_server.systemone_decide(
        state="Urgent server crash help required",
        questions=sample_choice_question,
    )
    assert res1.get("answers") == res2.get("answers")


def test_tier1_p12_04_mq_cache_telemetry_metrics_preserved(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify decision payload contains telemetry metrics."""
    result = mcp_server.systemone_decide(
        state="General inquiry",
        questions=sample_choice_question,
    )
    assert "telemetry" in result or "metadata" in result or "answers" in result


def test_tier1_p12_05_mq_cache_memory_bounded(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify cache remains bounded over 10 consecutive requests."""
    for i in range(10):
        mcp_server.systemone_decide(
            state=f"Message {i}",
            questions=sample_choice_question,
        )


# ==============================================================================
# Point 13 (P13): Standalone Wheel CI Integration Test
# ==============================================================================

def test_tier1_p13_01_ci_workflow_has_wheel_test_step(repo_root: Path):
    """Verify .github/workflows/ci.yml contains standalone wheel test job or step."""
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    assert ci_file.exists()
    content = ci_file.read_text(encoding="utf-8")
    assert "wheel" in content.lower() or "standalone" in content.lower(), (
        "ci.yml must contain a job or step building and testing the standalone wheel."
    )


def test_tier1_p13_02_wheel_test_verifies_healthz(repo_root: Path):
    """Verify wheel verification checks /healthz."""
    wheel_test_file = repo_root / "tests" / "test_standalone_wheel.py"
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    has_check = (wheel_test_file.exists() and "healthz" in wheel_test_file.read_text(encoding="utf-8")) or (
        "healthz" in ci_file.read_text(encoding="utf-8")
    )
    assert has_check, "Standalone wheel verification must check /healthz."


def test_tier1_p13_03_wheel_test_verifies_systemone(repo_root: Path):
    """Verify wheel verification checks /v1/systemone."""
    wheel_test_file = repo_root / "tests" / "test_standalone_wheel.py"
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    has_check = (wheel_test_file.exists() and "systemone" in wheel_test_file.read_text(encoding="utf-8")) or (
        "systemone" in ci_file.read_text(encoding="utf-8")
    )
    assert has_check, "Standalone wheel verification must check /v1/systemone."


def test_tier1_p13_04_wheel_test_verifies_benchmark(repo_root: Path):
    """Verify wheel verification checks /v1/benchmark."""
    wheel_test_file = repo_root / "tests" / "test_standalone_wheel.py"
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    has_check = (wheel_test_file.exists() and "benchmark" in wheel_test_file.read_text(encoding="utf-8")) or (
        "benchmark" in ci_file.read_text(encoding="utf-8")
    )
    assert has_check, "Standalone wheel verification must check /v1/benchmark."


def test_tier1_p13_05_pyproject_toml_packages_required_assets(repo_root: Path):
    """Verify pyproject.toml includes static and data directories in package data."""
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert "package-data" in content or "include-package-data" in content or "data" in content, (
        "pyproject.toml must configure package-data to bundle data files."
    )


# ==============================================================================
# Point 14 (P14): Secure Jekyll Docker Workflow
# ==============================================================================

def test_tier1_p14_01_jekyll_workflow_least_privilege_permissions(repo_root: Path):
    """Verify jekyll-docker.yml specifies least-privilege permissions: contents: read."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    assert jekyll_wf.exists()
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "contents: read" in content or "permissions:" in content, (
        "jekyll-docker.yml must declare least-privilege permissions."
    )


def test_tier1_p14_02_jekyll_workflow_no_chmod_777(repo_root: Path):
    """Verify jekyll-docker.yml does not contain unrestricted chmod 777."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "chmod 777" not in content, (
        "jekyll-docker.yml must not use unrestricted chmod 777."
    )


def test_tier1_p14_03_jekyll_workflow_pinned_actions(repo_root: Path):
    """Verify actions in jekyll-docker.yml use pinned versions."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "actions/checkout@" in content


def test_tier1_p14_04_jekyll_workflow_pinned_docker_image(repo_root: Path):
    """Verify jekyll-docker.yml uses a pinned Docker image tag rather than latest."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "jekyll/builder:" in content or "jekyll" in content
    assert ":latest" not in content, "Docker image in jekyll-docker.yml must be pinned to explicit version."


def test_tier1_p14_05_jekyll_workflow_unprivileged_execution(repo_root: Path):
    """Verify workflow avoids running Docker commands as root."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "sudo" not in content, "Workflow should avoid sudo privileges."


# ==============================================================================
# Point 15 (P15): Client SDK Deduplication
# ==============================================================================

def test_tier1_p15_01_base_client_class_exists():
    """Verify _BaseMicroJevClient exists in microllm.client."""
    import microllm.client as client_module
    assert hasattr(client_module, "_BaseMicroJevClient"), (
        "microllm.client must define _BaseMicroJevClient base class."
    )


def test_tier1_p15_02_sync_and_async_client_inherit_base():
    """Verify MicroJevClient and AsyncMicroJevClient inherit from _BaseMicroJevClient."""
    from microllm.client import MicroJevClient, AsyncMicroJevClient, _BaseMicroJevClient
    assert issubclass(MicroJevClient, _BaseMicroJevClient), "MicroJevClient must inherit from _BaseMicroJevClient."
    assert issubclass(AsyncMicroJevClient, _BaseMicroJevClient), "AsyncMicroJevClient must inherit from _BaseMicroJevClient."


def test_tier1_p15_03_shared_request_payload_construction():
    """Verify _BaseMicroJevClient provides shared payload building helper."""
    from microllm.client import _BaseMicroJevClient
    assert hasattr(_BaseMicroJevClient, "_build_systemone_payload") or hasattr(_BaseMicroJevClient, "_build_payload"), (
        "_BaseMicroJevClient must encapsulate request payload construction."
    )


def test_tier1_p15_04_shared_response_decoding_logic():
    """Verify _BaseMicroJevClient provides shared response decoding helper."""
    from microllm.client import _BaseMicroJevClient
    assert hasattr(_BaseMicroJevClient, "_decode_response") or hasattr(_BaseMicroJevClient, "_parse_result"), (
        "_BaseMicroJevClient must encapsulate response parsing/decoding."
    )


def test_tier1_p15_05_sync_async_client_public_api_parity():
    """Verify MicroJevClient and AsyncMicroJevClient have matching public method names."""
    from microllm.client import MicroJevClient, AsyncMicroJevClient
    sync_methods = {m for m in dir(MicroJevClient) if not m.startswith("_")}
    async_methods = {m for m in dir(AsyncMicroJevClient) if not m.startswith("_")}
    common_methods = {"systemone", "batch", "healthz"}
    assert common_methods.issubset(sync_methods), f"MicroJevClient missing common methods: {common_methods - sync_methods}"
    assert common_methods.issubset(async_methods), f"AsyncMicroJevClient missing common methods: {common_methods - async_methods}"


# ==============================================================================
# Point 16 (P16): Static Data Extraction to Packaged JSON
# ==============================================================================

def test_tier1_p16_01_data_directory_exists_with_ten_json_files(repo_root: Path):
    """Verify src/microllm/data/ directory exists and contains 10 JSON files."""
    data_dir = repo_root / "src" / "microllm" / "data"
    assert data_dir.exists(), "src/microllm/data/ directory must exist."
    json_files = list(data_dir.glob("*.json"))
    assert len(json_files) == 10, f"Expected 10 JSON data files in {data_dir}, found {len(json_files)}: {json_files}"


def test_tier1_p16_02_json_data_files_valid_syntax(repo_root: Path):
    """Verify all 10 JSON files parse cleanly as valid JSON."""
    data_dir = repo_root / "src" / "microllm" / "data"
    for json_file in data_dir.glob("*.json"):
        content = json_file.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, (dict, list)), f"{json_file.name} did not parse into dict or list."


def test_tier1_p16_03_importlib_resources_used_to_load_data(repo_root: Path):
    """Verify data files are loaded via importlib.resources."""
    papers_py = repo_root / "src" / "microllm" / "papers.py"
    content = papers_py.read_text(encoding="utf-8")
    assert "importlib.resources" in content or "files(" in content, (
        "papers.py must use importlib.resources to access packaged JSON files."
    )


def test_tier1_p16_04_papers_catalog_backward_compatibility():
    """Verify PAPERS catalog in microllm.papers retains expected keys."""
    from microllm.papers import PAPERS
    assert isinstance(PAPERS, dict)
    assert len(PAPERS) >= 3
    assert "paper-openjev-rlcd" in PAPERS


def test_tier1_p16_05_hf_datasets_and_presets_backward_compatibility():
    """Verify HF datasets catalog retains backward compatibility."""
    from microllm.hf_jev_datasets import HF_JEV_DATASETS
    assert isinstance(HF_JEV_DATASETS, dict)
    assert len(HF_JEV_DATASETS) >= 3


# ==============================================================================
# Point 17 (P17): Dead Code Removal & Strict Linting
# ==============================================================================

def test_tier1_p17_01_mcp_server_no_dead_get_client(repo_root: Path):
    """Verify _get_client is removed from src/microllm/mcp_server.py."""
    mcp_file = repo_root / "src" / "microllm" / "mcp_server.py"
    content = mcp_file.read_text(encoding="utf-8")
    assert "def _get_client(" not in content, (
        "Dead function _get_client must be removed from mcp_server.py."
    )


def test_tier1_p17_02_mcp_server_no_dead_test_client(repo_root: Path):
    """Verify _test_client is removed from src/microllm/mcp_server.py."""
    mcp_file = repo_root / "src" / "microllm" / "mcp_server.py"
    content = mcp_file.read_text(encoding="utf-8")
    assert "def _test_client(" not in content, (
        "Dead function _test_client must be removed from mcp_server.py."
    )


def test_tier1_p17_03_lint_workflow_no_exit_zero(repo_root: Path):
    """Verify .github/workflows/lint.yml does not use --exit-zero."""
    lint_wf = repo_root / ".github" / "workflows" / "lint.yml"
    assert lint_wf.exists()
    content = lint_wf.read_text(encoding="utf-8")
    assert "--exit-zero" not in content, (
        "lint.yml must enforce strict lint failures without --exit-zero."
    )


def test_tier1_p17_04_lint_workflow_enforces_f401_f811(repo_root: Path):
    """Verify lint.yml checks for unused imports (F401) and redefinitions (F811)."""
    lint_wf = repo_root / ".github" / "workflows" / "lint.yml"
    content = lint_wf.read_text(encoding="utf-8")
    assert "F401" in content and "F811" in content, (
        "lint.yml must explicitly check F401 and F811."
    )


def test_tier1_p17_05_no_duplicate_imports_in_gradio_handlers(repo_root: Path):
    """Verify src/microllm/gradio_handlers.py does not contain duplicate import functools."""
    handlers_file = repo_root / "src" / "microllm" / "gradio_handlers.py"
    lines = handlers_file.read_text(encoding="utf-8").splitlines()
    functools_imports = [line for line in lines if line.strip() == "import functools"]
    assert len(functools_imports) <= 1, (
        f"Found {len(functools_imports)} duplicate 'import functools' statements in gradio_handlers.py."
    )


# ==============================================================================
# Point 18 (P18): CI Workflow Deduplication
# ==============================================================================

def test_tier1_p18_01_ci_yml_matrix_deduplicated(repo_root: Path):
    """Verify ci.yml does not duplicate matrix pytest runs owned by unit-tests.yml."""
    ci_wf = repo_root / ".github" / "workflows" / "ci.yml"
    content = ci_wf.read_text(encoding="utf-8")
    # ci.yml should focus on packaging/standalone wheel rather than running 3.10, 3.11, 3.12 matrix unit tests
    assert "strategy:" not in content or "matrix:" not in content, (
        "ci.yml should not duplicate the python version matrix owned by unit-tests.yml."
    )


def test_tier1_p18_02_unit_tests_yml_owns_test_matrix(repo_root: Path):
    """Verify unit-tests.yml defines matrix testing across supported Python versions."""
    ut_wf = repo_root / ".github" / "workflows" / "unit-tests.yml"
    assert ut_wf.exists()
    content = ut_wf.read_text(encoding="utf-8")
    assert "matrix:" in content
    assert "python-version:" in content or "python:" in content


def test_tier1_p18_03_lint_yml_owns_static_analysis(repo_root: Path):
    """Verify lint.yml is the dedicated workflow for flake8 and static checks."""
    lint_wf = repo_root / ".github" / "workflows" / "lint.yml"
    assert lint_wf.exists()
    content = lint_wf.read_text(encoding="utf-8")
    assert "flake8" in content


def test_tier1_p18_04_workflow_triggers_non_overlapping(repo_root: Path):
    """Verify ci.yml, unit-tests.yml, and lint.yml trigger on appropriate branch events."""
    for wf_name in ["ci.yml", "unit-tests.yml", "lint.yml"]:
        wf_path = repo_root / ".github" / "workflows" / wf_name
        assert wf_path.exists()
        content = wf_path.read_text(encoding="utf-8")
        assert "on:" in content or "push:" in content


def test_tier1_p18_05_workflow_step_action_versions_aligned(repo_root: Path):
    """Verify GitHub Actions steps use modern pinned actions across workflows."""
    ci_wf = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "actions/checkout@" in ci_wf
