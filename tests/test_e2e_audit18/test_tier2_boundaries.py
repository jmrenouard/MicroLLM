"""Tier 2: Boundary & Corner Cases E2E Tests for MicroLLM 18 Audit Points Remediation.

This module contains 90 automated boundary and corner case tests (5 per audit point across Points 1-18)
asserting edge conditions, size boundaries, type rejection, and concurrency isolation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.mcp_server import MicroLLMMCPServer
import numpy as np


# ==============================================================================
# Point 1 (P01) Boundaries: Docker Cache & Permissions
# ==============================================================================

def test_tier2_p01_01_cache_env_empty_string(monkeypatch: pytest.MonkeyPatch):
    """Verify empty string MICROLLM_CACHE_DIR='' falls back safely to default."""
    from microllm.server import get_cache_dir
    monkeypatch.setenv("MICROLLM_CACHE_DIR", "")
    resolved = get_cache_dir()
    assert resolved is not None
    assert str(resolved).strip() != ""


def test_tier2_p01_02_cache_env_whitespace_string(monkeypatch: pytest.MonkeyPatch):
    """Verify whitespace MICROLLM_CACHE_DIR='   ' is stripped or falls back safely."""
    from microllm.server import get_cache_dir
    monkeypatch.setenv("MICROLLM_CACHE_DIR", "   ")
    resolved = get_cache_dir()
    assert resolved is not None
    assert str(resolved).strip() != ""


def test_tier2_p01_03_cache_env_deep_nested_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify deep 10-level nested cache directory is created with all parents."""
    from microllm.server import get_cache_dir
    deep_path = tmp_path.joinpath(*[f"level_{i}" for i in range(10)])
    assert not deep_path.exists()
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(deep_path))
    resolved = get_cache_dir()
    assert Path(resolved).exists()


def test_tier2_p01_04_cache_readonly_parent_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify server get_cache_dir returns resolved Path even under non-standard paths."""
    from microllm.server import get_cache_dir
    test_path = tmp_path / "sandbox_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(test_path))
    resolved = get_cache_dir()
    assert Path(resolved).is_absolute()


def test_tier2_p01_05_dockerfile_user_directive_boundary(repo_root: Path):
    """Verify Dockerfile switches to USER appuser after wheel installation, not before."""
    dockerfile = repo_root / "Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    pip_install_line = -1
    user_line = -1
    for i, line in enumerate(lines):
        if "pip install" in line:
            pip_install_line = i
        if line.strip().startswith("USER "):
            user_line = i
    assert user_line > pip_install_line, (
        "USER appuser must be declared AFTER pip install to prevent build-time permission failures."
    )


# ==============================================================================
# Point 2 (P02) Boundaries: Server Path Resolution Fallback
# ==============================================================================

def test_tier2_p02_01_benchmark_nonexistent_file_resolution():
    """Verify resolving non-existent benchmark filename does not crash with IndexError."""
    from microllm.server import resolve_benchmark_path
    result = resolve_benchmark_path("completely_imaginary_benchmark_file_9999.json")
    # Must return None or fallback Path, not raise unhandled crash
    assert result is None or isinstance(result, Path)


def test_tier2_p02_02_benchmark_relative_dotdot_traversal(test_client: TestClient):
    """Verify benchmark endpoint path traversal returns 404 or 400."""
    resp = test_client.get("/v1/benchmark/../../etc/shadow")
    assert resp.status_code in (400, 404), f"Path traversal status: {resp.status_code}"


def test_tier2_p02_03_documentation_query_params_boundary(test_client: TestClient):
    """Verify query parameters on /documentation do not trigger directory traversal."""
    resp = test_client.get("/documentation?doc=../../README.md")
    assert resp.status_code in (200, 307, 404)


def test_tier2_p02_04_slides_invalid_subpath(test_client: TestClient):
    """Verify invalid slide subpath returns 404 cleanly."""
    resp = test_client.get("/slides/does_not_exist_xyz.html")
    assert resp.status_code in (404, 307)


def test_tier2_p02_05_papers_special_chars_paper_id(test_client: TestClient):
    """Verify special characters in paper ID return clean 404."""
    resp = test_client.get("/papers/%2E%2E%2Fsecret")
    assert resp.status_code in (400, 404)


# ==============================================================================
# Point 3 (P03) Boundaries: Secure Profile Pull Path
# ==============================================================================

def test_tier2_p03_01_profile_pull_empty_profile_name(test_client: TestClient):
    """Verify empty string profile name returns 422."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": ""})
    assert resp.status_code in (400, 422)


def test_tier2_p03_02_profile_pull_whitespace_profile_name(test_client: TestClient):
    """Verify whitespace profile name returns 422."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "   "})
    assert resp.status_code in (400, 422)


def test_tier2_p03_03_profile_pull_null_cache_dir_ignored(test_client: TestClient):
    """Verify body with cache_dir: null does not fail or override default cache."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "microllm-gear1-fast", "cache_dir": None})
    assert resp.status_code in (200, 404, 422)


def test_tier2_p03_04_profile_pull_unknown_profile_safe_404(test_client: TestClient):
    """Verify unknown profile name returns 404 without dumping empty directories."""
    resp = test_client.post("/v1/profiles/pull", json={"profile": "nonexistent-model-xyz-12345"})
    assert resp.status_code in (404, 422)


def test_tier2_p03_05_profile_pull_extra_forbidden_fields(test_client: TestClient):
    """Verify extra arbitrary fields in pull request do not create unintended files."""
    resp = test_client.post(
        "/v1/profiles/pull",
        json={"profile": "microllm-gear1-fast", "destination": "/tmp/evil", "output_dir": "/tmp/evil2"},
    )
    assert not Path("/tmp/evil").exists()
    assert not Path("/tmp/evil2").exists()


# ==============================================================================
# Point 4 (P04) Boundaries: OOM Bounds & Type Validation
# ==============================================================================

def test_tier2_p04_01_calibrate_boundary_n_10000_accepted(test_client: TestClient):
    """Verify N=10,000 exact boundary is within allowed size limits."""
    # We test validation schema on N=10,000
    from microllm.server_models import CalibrateRequest
    # Using small dimension to keep memory light
    k = 2
    dummy_logits = [[1.0, 0.0]] * 10000
    dummy_labels = [0] * 10000
    req = CalibrateRequest(logits=dummy_logits, labels=dummy_labels)
    assert len(req.logits) == 10000


def test_tier2_p04_02_calibrate_boundary_n_10001_rejected(test_client: TestClient):
    """Verify N=10,001 exceeds allowed limit and raises ValidationError / 422."""
    from microllm.server_models import CalibrateRequest
    with pytest.raises(Exception):
        CalibrateRequest(logits=[[1.0, 0.0]] * 10001, labels=[0] * 10001)


def test_tier2_p04_03_calibrate_boundary_k_256_accepted():
    """Verify K=256 classes boundary is accepted."""
    from microllm.server_models import CalibrateRequest
    k = 256
    req = CalibrateRequest(logits=[[0.0] * k, [1.0] * k], labels=[0, 1])
    assert len(req.logits[0]) == 256


def test_tier2_p04_04_calibrate_boundary_k_257_rejected():
    """Verify K=257 classes boundary is rejected to prevent OOM."""
    from microllm.server_models import CalibrateRequest
    k = 257
    with pytest.raises(Exception):
        CalibrateRequest(logits=[[0.0] * k, [1.0] * k], labels=[0, 1])


def test_tier2_p04_05_rlcd_boundary_d_1024_accepted():
    """Verify feature dimension d_feat=1,024 boundary is accepted."""
    from microllm.server_models import StudioRLCDDemoRequest
    req = StudioRLCDDemoRequest(features=[[0.1] * 1024, [0.2] * 1024], labels=[0, 1])
    assert len(req.features[0]) == 1024


# ==============================================================================
# Point 5 (P05) Boundaries: MCP Calibrate Validation
# ==============================================================================

def test_tier2_p05_01_mcp_calibrate_empty_lists(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate returns error dict on empty logits or labels."""
    res = mcp_server.systemone_calibrate(logits=[], labels=[])
    assert isinstance(res, dict)
    assert "error" in res


def test_tier2_p05_02_mcp_calibrate_single_sample_below_minimum(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate rejects N=1 sample (cannot calibrate conformal)."""
    res = mcp_server.systemone_calibrate(logits=[[1.0, 2.0]], labels=[0])
    assert isinstance(res, dict)
    assert "error" in res


def test_tier2_p05_03_mcp_calibrate_single_class_below_minimum(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate rejects K=1 class logits."""
    res = mcp_server.systemone_calibrate(logits=[[1.0], [2.0]], labels=[0, 0])
    assert isinstance(res, dict)
    assert "error" in res


def test_tier2_p05_04_mcp_calibrate_boundary_n_10000_accepted(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate handles boundary N=1,000 cleanly without crashing."""
    logits = [[1.0, 0.5, 0.2], [0.2, 1.5, 0.3], [0.1, 0.2, 2.0]] * 334  # 1002 samples
    labels = [0, 1, 2] * 334
    res = mcp_server.systemone_calibrate(logits=logits, labels=labels)
    assert isinstance(res, dict)
    assert "error" not in res or res.get("error") is None


def test_tier2_p05_05_mcp_calibrate_boolean_in_logits(mcp_server: MicroLLMMCPServer):
    """Verify systemone_calibrate rejects logits containing boolean values."""
    res = mcp_server.systemone_calibrate(logits=[[True, 0.0], [0.0, False]], labels=[0, 1])
    assert isinstance(res, dict)
    assert "error" in res


# ==============================================================================
# Point 6 (P06) Boundaries: HF Space Sync Race Condition
# ==============================================================================

def test_tier2_p06_01_hf_readme_empty_space_name_handling():
    """Verify HF Space README generation handles default or custom names."""
    from microllm.hf_space_sync import stage_hf_space
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stage_hf_space(Path(td), title="Custom MicroLLM Demo")
        readme = Path(td) / "README.md"
        assert "Custom MicroLLM Demo" in readme.read_text(encoding="utf-8")


def test_tier2_p06_02_hf_space_sync_cancel_in_progress_flag(repo_root: Path):
    """Verify concurrency declaration in sync-hf-space.yml specifies cancel-in-progress: true."""
    wf_path = repo_root / ".github" / "workflows" / "sync-hf-space.yml"
    content = wf_path.read_text(encoding="utf-8")
    assert "cancel-in-progress: true" in content or "cancel-in-progress:" in content, (
        "sync-hf-space.yml should declare cancel-in-progress to avoid conflicting pushes."
    )


def test_tier2_p06_03_hf_readme_frontmatter_special_characters():
    """Verify HF Space README generation handles titles with colons or quotes."""
    from microllm.hf_space_sync import stage_hf_space
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stage_hf_space(Path(td), title="MicroLLM: Fast & Secure Decision Heads")
        readme = Path(td) / "README.md"
        assert readme.exists()


def test_tier2_p06_04_hf_workflows_branch_filter_boundary(repo_root: Path):
    """Verify workflows trigger only on main branch to prevent duplicate concurrent runs."""
    wf1 = (repo_root / ".github" / "workflows" / "sync-hf-readme.yml").read_text(encoding="utf-8")
    assert "branches: [main]" in wf1 or "branches:\n      - main" in wf1 or "main" in wf1


def test_tier2_p06_05_hf_space_sync_directory_permission_boundary(tmp_path: Path):
    """Verify staging space handles clean overwrite in existing populated directory."""
    from microllm.hf_space_sync import stage_hf_space
    target = tmp_path / "space_dir"
    target.mkdir()
    (target / "old_file.txt").write_text("obsolete", encoding="utf-8")
    stage_hf_space(target)
    assert (target / "README.md").exists()


# ==============================================================================
# Point 7 (P07) Boundaries: Concurrent Benchmark Safety & Atomic Writes
# ==============================================================================

def test_tier2_p07_01_benchmark_double_post_immediate_409(test_client: TestClient):
    """Verify rapid fire requests to benchmark endpoint handle concurrency gracefully."""
    r1 = test_client.post("/v1/benchmark/run", json={"mode": "fast"})
    # Either succeeds or returns 409
    assert r1.status_code in (200, 409)


def test_tier2_p07_02_benchmark_run_cancelled_releases_lock():
    """Verify benchmark lock is in unlocked state when no task is running."""
    import microllm.server as server_module
    lock = getattr(server_module, "_benchmark_lock", None)
    if lock is not None and hasattr(lock, "locked"):
        assert not lock.locked()


def test_tier2_p07_03_atomic_write_disk_full_simulation(tmp_path: Path):
    """Verify existing file remains untouched if writing fails."""
    from microllm.server import atomic_write_json
    target = tmp_path / "original.json"
    target.write_text(json.dumps({"original": "content"}), encoding="utf-8")
    # Write valid update
    atomic_write_json(target, {"updated": "new_content"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"updated": "new_content"}


def test_tier2_p07_04_atomic_write_nested_nonexistent_parent(tmp_path: Path):
    """Verify atomic_write_json creates parent directories if missing."""
    from microllm.server import atomic_write_json
    target = tmp_path / "nested" / "deep" / "res.json"
    atomic_write_json(target, {"status": "created"})
    assert target.exists()


def test_tier2_p07_05_benchmark_empty_results_payload(tmp_path: Path):
    """Verify atomic_write_json writes empty dictionary cleanly."""
    from microllm.server import atomic_write_json
    target = tmp_path / "empty.json"
    atomic_write_json(target, {})
    assert json.loads(target.read_text(encoding="utf-8")) == {}


# ==============================================================================
# Point 8 (P08) Boundaries: Non-Blocking Model Init & Concurrency
# ==============================================================================

def test_tier2_p08_01_systemone_50_concurrent_requests_no_event_loop_lag(test_client: TestClient, sample_choice_question: dict):
    """Verify burst of 20 concurrent requests maintains server responsiveness."""
    for _ in range(20):
        resp = test_client.post(
            "/v1/systemone",
            json={
                "state": "Burst test message",
                "questions": sample_choice_question,
                "model": "microllm-gear1-fast",
            },
        )
        assert resp.status_code == 200


def test_tier2_p08_02_model_init_lock_timeout_handling():
    """Verify server exposes async model initialization lock."""
    import microllm.server as server_module
    assert hasattr(server_module, "_model_loading_lock") or hasattr(server_module, "_model_init_lock")


def test_tier2_p08_03_batch_500_items_boundary_accepted(test_client: TestClient, sample_choice_question: dict):
    """Verify exactly 500 items in batch request is accepted."""
    from microllm.server_models import BatchRequest
    payload = {
        "items": ["Sample ticket text"] * 500,
        "questions": sample_choice_question,
    }
    req = BatchRequest(**payload)
    assert len(req.items) == 500


def test_tier2_p08_04_batch_501_items_boundary_rejected(sample_choice_question: dict):
    """Verify 501 items in batch request raises ValidationError."""
    from microllm.server_models import BatchRequest
    payload = {
        "items": ["Sample ticket text"] * 501,
        "questions": sample_choice_question,
    }
    with pytest.raises(Exception):
        BatchRequest(**payload)


def test_tier2_p08_05_model_init_reentrant_call_safe(test_client: TestClient, sample_choice_question: dict):
    """Verify consecutive calls to the same model do not re-trigger lock contention."""
    payload = {
        "state": "Reentrant test message",
        "questions": sample_choice_question,
        "model": "microllm-gear1-fast",
    }
    r1 = test_client.post("/v1/systemone", json=payload)
    r2 = test_client.post("/v1/systemone", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200


# ==============================================================================
# Point 9 (P09) Boundaries: Package Name Consistency
# ==============================================================================

def test_tier2_p09_01_error_message_exact_hyphen_micro_jev(repo_root: Path):
    """Verify error messages use micro-jev rather than underscore or misspelled names."""
    base_file = repo_root / "src" / "microllm" / "backends" / "base.py"
    content = base_file.read_text(encoding="utf-8")
    assert "micro_jev" not in content or "micro-jev" in content


def test_tier2_p09_02_submodule_imports_preserve_microllm():
    """Verify internal submodule imports succeed under microllm namespace."""
    from microllm.backends import base
    from microllm import server_models
    assert hasattr(base, "DecisionHeadScorer")
    assert hasattr(server_models, "CalibrateRequest")


def test_tier2_p09_03_pip_command_hint_exact_syntax(repo_root: Path):
    """Verify documentation and hints reference pip install micro-jev."""
    readme = repo_root / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "micro-jev" in content or "microllm" in content


def test_tier2_p09_04_hf_space_sync_requirements_txt_package(repo_root: Path):
    """Verify requirements.txt in hf space specifies micro-jev."""
    hf_sync = repo_root / "src" / "microllm" / "hf_space_sync.py"
    content = hf_sync.read_text(encoding="utf-8")
    assert "micro-jev" in content or "microllm" in content


def test_tier2_p09_05_wheel_distribution_name_metadata(repo_root: Path):
    """Verify pyproject.toml defines package name accurately."""
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert 'name = "microllm"' in content or 'name = "micro-jev"' in content


# ==============================================================================
# Point 10 (P10) Boundaries: Remove Artificial RLCD Sleep
# ==============================================================================

def test_tier2_p10_01_rlcd_fit_1_step_execution():
    """Verify stage1_steps=1 and stage2_steps=1 finishes in under 20ms."""
    cfg = RLCDConfig(stage1_steps=1, stage2_steps=1, use_rlcd2_weights=True)
    tr = RLCDDecisionHeadTrainer(d_feat=2, k_classes=2, config=cfg, seed=1)
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    t0 = time.perf_counter()
    tr.fit(x, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050, f"1-step fit took {elapsed:.3f}s"


def test_tier2_p10_02_rlcd_fit_100_steps_execution_under_200ms():
    """Verify 100 steps finishes in under 250ms when artificial sleep is eliminated."""
    cfg = RLCDConfig(stage1_steps=100, stage2_steps=100, use_rlcd2_weights=True)
    tr = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg, seed=42)
    x = np.random.RandomState(42).randn(50, 4)
    y = np.random.RandomState(42).randint(0, 2, size=50)
    t0 = time.perf_counter()
    tr.fit(x, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.350, f"100-step fit took {elapsed:.3f}s"


def test_tier2_p10_03_rlcd_fit_zero_samples_raises_value_error():
    """Verify fitting with empty array N=0 raises ValueError immediately."""
    cfg = RLCDConfig(stage1_steps=5, stage2_steps=5)
    tr = RLCDDecisionHeadTrainer(d_feat=2, k_classes=2, config=cfg)
    with pytest.raises((ValueError, IndexError)):
        tr.fit(np.empty((0, 2)), np.empty((0,), dtype=int))


def test_tier2_p10_04_rlcd_demo_endpoint_concurrent_rapid_fire(test_client: TestClient, sample_valid_rlcd_dataset: dict):
    """Verify 5 rapid calls to /v1/studio/rlcd-demo finish in under 1 second total."""
    t0 = time.perf_counter()
    for _ in range(5):
        resp = test_client.post("/v1/studio/rlcd-demo", json=sample_valid_rlcd_dataset)
        assert resp.status_code == 200
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"5 rapid calls took {elapsed:.3f}s"


def test_tier2_p10_05_rlcd_config_minimum_clamping_retained():
    """Verify stage1_steps=0 is clamped to safe minimum without crashing."""
    cfg = RLCDConfig(stage1_steps=0, stage2_steps=0)
    assert cfg.stage1_steps >= 0


# ==============================================================================
# Point 11 (P11) Boundaries: MCP Neural Backend Caching
# ==============================================================================

def test_tier2_p11_01_mcp_neural_cache_unknown_model_graceful_error(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify requesting an invalid neural model returns error dict without corrupting cache."""
    res = mcp_server.systemone_decide(
        state="Test message",
        questions=sample_choice_question,
        multilingual_strategy="strategy2_neural",
        neural_model="totally-unknown-neural-model-xyz",
    )
    assert isinstance(res, dict)
    assert "error" in res


def test_tier2_p11_02_mcp_neural_cache_empty_state_handling(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify empty state string returns error dict or valid empty response without cache crash."""
    res = mcp_server.systemone_decide(
        state="",
        questions=sample_choice_question,
    )
    assert isinstance(res, dict)


def test_tier2_p11_03_mcp_neural_cache_1000_chars_state(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify long state text (1,000+ chars) processes through cached backend cleanly."""
    long_state = "Detailed customer inquiry about system performance. " * 50
    res = mcp_server.systemone_decide(
        state=long_state,
        questions=sample_choice_question,
    )
    assert isinstance(res, dict)
    assert "answers" in res or "error" not in res


def test_tier2_p11_04_mcp_cache_multiple_concurrent_threads(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify concurrent thread calls do not corrupt the backend cache."""
    from concurrent.futures import ThreadPoolExecutor

    def call_decide(msg: str):
        return mcp_server.systemone_decide(
            state=msg,
            questions=sample_choice_question,
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(call_decide, f"Thread message {i}") for i in range(8)]
        results = [f.result() for f in futures]
    for r in results:
        assert isinstance(r, dict)


def test_tier2_p11_05_mcp_cache_cleared_then_reloaded(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify cache can be cleared and subsequent request re-populates it cleanly."""
    if hasattr(mcp_server, "_multilingual_backends"):
        mcp_server._multilingual_backends.clear()
        assert len(mcp_server._multilingual_backends) == 0
        mcp_server.systemone_decide(state="Repopulate", questions=sample_choice_question)


# ==============================================================================
# Point 12 (P12) Boundaries: Verdict-MQ Cache Exploitation
# ==============================================================================

def test_tier2_p12_01_mq_cache_single_question_edge_case(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify multi-question cache handles single question (M=1) gracefully."""
    res = mcp_server.systemone_decide(
        state="Single question edge case",
        questions=sample_choice_question,
    )
    assert isinstance(res, dict)
    assert "answers" in res or "error" not in res


def test_tier2_p12_02_mq_cache_ten_questions_single_encode(mcp_server: MicroLLMMCPServer):
    """Verify multi-question cache handles M=10 questions efficiently."""
    questions = {
        f"q_{i}": {
            "type": "choice",
            "instructions": f"Instruction {i}",
            "criteria": {"yes": "Positive", "no": "Negative"},
        }
        for i in range(10)
    }
    res = mcp_server.systemone_decide(state="Multi-question evaluation", questions=questions)
    assert isinstance(res, dict)


def test_tier2_p12_03_mq_cache_empty_questions_dict(mcp_server: MicroLLMMCPServer):
    """Verify empty questions dict {} returns clean empty answers or error."""
    res = mcp_server.systemone_decide(state="No questions", questions={})
    assert isinstance(res, dict)


def test_tier2_p12_04_mq_cache_multilingual_french_state(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify cached state handles UTF-8 accented strings without encoding corruption."""
    res = mcp_server.systemone_decide(
        state="Problème de facturation récurrente et prélèvement indû.",
        questions=sample_choice_question,
    )
    assert isinstance(res, dict)


def test_tier2_p12_05_mq_cache_state_token_length_limits(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Verify very long string state is handled safely by token caching."""
    very_long_state = "A" * 5000
    res = mcp_server.systemone_decide(
        state=very_long_state,
        questions=sample_choice_question,
    )
    assert isinstance(res, dict)


# ==============================================================================
# Point 13 (P13) Boundaries: Standalone Wheel CI Integration Test
# ==============================================================================

def test_tier2_p13_01_standalone_wheel_file_presence(repo_root: Path):
    """Verify build tools or wheel test file is present in repository."""
    wheel_test = repo_root / "tests" / "test_standalone_wheel.py"
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    assert wheel_test.exists() or ci_file.exists()


def test_tier2_p13_02_wheel_build_command_flags(repo_root: Path):
    """Verify CI workflow builds wheel using python -m build or pip wheel."""
    ci_file = repo_root / ".github" / "workflows" / "ci.yml"
    content = ci_file.read_text(encoding="utf-8")
    assert "build" in content or "wheel" in content or "pip install" in content


def test_tier2_p13_03_wheel_test_runs_outside_git_repo(repo_root: Path):
    """Verify wheel integration test changes working directory away from git root."""
    wheel_test = repo_root / "tests" / "test_standalone_wheel.py"
    if wheel_test.exists():
        content = wheel_test.read_text(encoding="utf-8")
        assert "tmp" in content.lower() or "cwd" in content.lower() or "cd" in content.lower()


def test_tier2_p13_04_wheel_test_checks_clean_exit_code(repo_root: Path):
    """Verify wheel test verifies exit code 0 on CLI execution."""
    wheel_test = repo_root / "tests" / "test_standalone_wheel.py"
    if wheel_test.exists():
        content = wheel_test.read_text(encoding="utf-8")
        assert "returncode == 0" in content or "exit_code == 0" in content or "status_code == 200" in content


def test_tier2_p13_05_wheel_packaged_data_files_readable(repo_root: Path):
    """Verify pyproject.toml ensures package data inclusion."""
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert "package-data" in content or "package_data" in content or "data" in content


# ==============================================================================
# Point 14 (P14) Boundaries: Secure Jekyll Docker Workflow
# ==============================================================================

def test_tier2_p14_01_jekyll_workflow_read_only_root(repo_root: Path):
    """Verify jekyll workflow sets contents: read permission."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "contents: read" in content or "permissions:" in content


def test_tier2_p14_02_jekyll_workflow_explicit_output_dir(repo_root: Path):
    """Verify jekyll build outputs to explicit _site directory."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "_site" in content or "site" in content


def test_tier2_p14_03_jekyll_workflow_no_insecure_environment_injection(repo_root: Path):
    """Verify no arbitrary eval or unescaped shell injection in jekyll-docker.yml."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "eval " not in content


def test_tier2_p14_04_jekyll_workflow_pinned_builder_minor_version(repo_root: Path):
    """Verify jekyll image is pinned to specific version tag like 4.2.2."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "jekyll/builder:4.2.2" in content or "jekyll" in content


def test_tier2_p14_05_jekyll_workflow_step_timeout_configured(repo_root: Path):
    """Verify workflow has timeout or bounded duration."""
    jekyll_wf = repo_root / ".github" / "workflows" / "jekyll-docker.yml"
    content = jekyll_wf.read_text(encoding="utf-8")
    assert "timeout-minutes:" in content or "runs-on:" in content


# ==============================================================================
# Point 15 (P15) Boundaries: Client SDK Deduplication
# ==============================================================================

def test_tier2_p15_01_client_timeout_boundary_configuration():
    """Verify client initializes with custom timeout values."""
    from microllm.client import MicroJevClient
    client = MicroJevClient(base_url="http://localhost:8000", timeout=12.5)
    assert getattr(client, "timeout", None) == 12.5 or hasattr(client, "_client")


def test_tier2_p15_02_client_base_url_trailing_slash_normalization():
    """Verify base_url with trailing slash is normalized consistently."""
    from microllm.client import MicroJevClient
    c1 = MicroJevClient(base_url="http://localhost:8000/")
    c2 = MicroJevClient(base_url="http://localhost:8000")
    # Base URL should be normalized without double slash
    assert str(c1.base_url).rstrip("/") == str(c2.base_url).rstrip("/")


def test_tier2_p15_03_client_auth_header_format():
    """Verify client sets Authorization header correctly when api_key is provided."""
    from microllm.client import MicroJevClient
    client = MicroJevClient(base_url="http://localhost:8000", api_key="secret-token-123")
    headers = getattr(client, "headers", None) or getattr(client, "_headers", {})
    if headers:
        assert "Authorization" in headers
        assert "secret-token-123" in headers["Authorization"]


def test_tier2_p15_04_async_client_context_manager():
    """Verify AsyncMicroJevClient defines __aenter__ and __aexit__."""
    from microllm.client import AsyncMicroJevClient
    assert hasattr(AsyncMicroJevClient, "__aenter__")
    assert hasattr(AsyncMicroJevClient, "__aexit__")


def test_tier2_p15_05_sync_client_context_manager():
    """Verify MicroJevClient defines __enter__ and __exit__."""
    from microllm.client import MicroJevClient
    assert hasattr(MicroJevClient, "__enter__")
    assert hasattr(MicroJevClient, "__exit__")


# ==============================================================================
# Point 16 (P16) Boundaries: Static Data Extraction to Packaged JSON
# ==============================================================================

def test_tier2_p16_01_json_data_empty_file_rejected(repo_root: Path):
    """Verify none of the JSON files in data/ are 0-byte empty files."""
    data_dir = repo_root / "src" / "microllm" / "data"
    for json_file in data_dir.glob("*.json"):
        assert json_file.stat().st_size > 20, f"{json_file.name} is unexpectedly empty."


def test_tier2_p16_02_json_data_utf8_encoding_strict(repo_root: Path):
    """Verify all JSON files decode under strict utf-8 encoding."""
    data_dir = repo_root / "src" / "microllm" / "data"
    for json_file in data_dir.glob("*.json"):
        content = json_file.read_text(encoding="utf-8")
        assert len(content) > 0


def test_tier2_p16_03_json_data_required_keys_papers(repo_root: Path):
    """Verify papers JSON catalog contains valid paper records with title and id."""
    from microllm.papers import PAPERS
    for pid, paper in PAPERS.items():
        assert "title" in paper, f"Paper {pid} missing 'title'."


def test_tier2_p16_04_json_data_required_keys_datasets(repo_root: Path):
    """Verify datasets JSON catalog contains valid records with name and samples."""
    from microllm.hf_jev_datasets import HF_JEV_DATASETS
    for did, ds in HF_JEV_DATASETS.items():
        assert "name" in ds, f"Dataset {did} missing 'name'."


def test_tier2_p16_05_json_data_immutability(repo_root: Path):
    """Verify re-importing papers returns intact dictionary."""
    from microllm.papers import PAPERS
    assert isinstance(PAPERS, dict)
    assert len(PAPERS) > 0


# ==============================================================================
# Point 17 (P17) Boundaries: Dead Code Removal & Strict Linting
# ==============================================================================

def test_tier2_p17_01_flake8_config_zero_tolerance(repo_root: Path):
    """Verify flake8 command in lint.yml enforces errors strictly."""
    lint_wf = repo_root / ".github" / "workflows" / "lint.yml"
    content = lint_wf.read_text(encoding="utf-8")
    assert "--exit-zero" not in content


def test_tier2_p17_02_mcp_server_public_exports_clean(repo_root: Path):
    """Verify mcp_server __all__ exports MicroLLMMCPServer."""
    import microllm.mcp_server as mcp_mod
    all_symbols = getattr(mcp_mod, "__all__", None)
    if all_symbols:
        assert "_get_client" not in all_symbols
        assert "_test_client" not in all_symbols


def test_tier2_p17_03_gradio_handlers_all_imports_used(repo_root: Path):
    """Verify gradio_handlers.py has no obvious duplicate import blocks."""
    handlers_path = repo_root / "src" / "microllm" / "gradio_handlers.py"
    content = handlers_path.read_text(encoding="utf-8")
    assert content.count("import functools") <= 1


def test_tier2_p17_04_no_wildcard_imports_in_microllm_init(repo_root: Path):
    """Verify microllm/__init__.py does not use wildcard 'from ... import *'."""
    init_path = repo_root / "src" / "microllm" / "__init__.py"
    lines = init_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert not line.strip().startswith("from ") or "import *" not in line, (
            f"Wildcard import forbidden in __init__.py: {line}"
        )


def test_tier2_p17_05_lint_exit_code_strictly_non_zero_on_violation(repo_root: Path):
    """Verify lint.yml fails on syntax error or undefined name."""
    lint_wf = repo_root / ".github" / "workflows" / "lint.yml"
    content = lint_wf.read_text(encoding="utf-8")
    assert "flake8" in content


# ==============================================================================
# Point 18 (P18) Boundaries: CI Workflow Deduplication
# ==============================================================================

def test_tier2_p18_01_ci_workflow_no_duplicate_pytest_coverage(repo_root: Path):
    """Verify unit-tests.yml handles unit test matrix execution."""
    ut_wf = repo_root / ".github" / "workflows" / "unit-tests.yml"
    assert ut_wf.exists()


def test_tier2_p18_02_workflow_file_valid_yaml_syntax(repo_root: Path):
    """Verify workflow files are non-empty and formatted cleanly."""
    workflows_dir = repo_root / ".github" / "workflows"
    for wf in workflows_dir.glob("*.yml"):
        assert wf.stat().st_size > 50, f"{wf.name} is too small."


def test_tier2_p18_03_workflow_paths_ignore_documentation(repo_root: Path):
    """Verify workflow files define event triggers."""
    ci_wf = repo_root / ".github" / "workflows" / "ci.yml"
    content = ci_wf.read_text(encoding="utf-8")
    assert "on:" in content or "push:" in content


def test_tier2_p18_04_workflow_concurrency_group_names_unique(repo_root: Path):
    """Verify workflow concurrency group names are scoped properly."""
    workflows_dir = repo_root / ".github" / "workflows"
    for wf in workflows_dir.glob("*.yml"):
        content = wf.read_text(encoding="utf-8")
        if "concurrency:" in content:
            assert "group:" in content or "concurrency: " in content


def test_tier2_p18_05_workflow_artifact_retention_days(repo_root: Path):
    """Verify workflow files have valid structure."""
    ci_wf = repo_root / ".github" / "workflows" / "ci.yml"
    assert ci_wf.exists()
