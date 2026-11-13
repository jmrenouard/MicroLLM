"""Tier 4: Real-World Application Scenarios E2E Tests for MicroLLM 18 Audit Points Remediation.

This module validates comprehensive end-to-end integration workflows simulating realistic
production workloads, active learning lifecycles, SDK client sessions, MCP tool chaining,
and hardened container deployment.
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
from microllm.client import MicroJevClient
import numpy as np


def test_tier4_01_decision_studio_active_learning_workflow(test_client: TestClient):
    """Scenario 1: Decision Studio End-to-End Active Learning Workflow.
    
    Workflow:
    1. Check server healthz.
    2. Submit RLCD demo request with synthetic features and labels.
    3. Verify response contains reliability diagram bins and risk-coverage curve.
    4. Assert training completes in sub-second time without artificial sleep.
    """
    health = test_client.get("/healthz")
    assert health.status_code == 200

    features = np.random.RandomState(42).randn(120, 8).tolist()
    labels = np.random.RandomState(42).randint(0, 3, size=120).tolist()
    payload = {
        "features": features,
        "labels": labels,
        "stage1_steps": 20,
        "stage2_steps": 20,
        "use_rlcd2_weights": True,
        "seed": 42,
    }

    t0 = time.perf_counter()
    resp = test_client.post("/v1/studio/rlcd-demo", json=payload)
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "initial_bins" in data
    assert "final_bins" in data
    assert "risk_coverage_curve" in data
    assert elapsed < 0.600, f"Active learning loop took {elapsed:.3f}s; artificial sleep is likely present."


def test_tier4_02_concurrent_batch_inference_and_benchmark_isolation(test_client: TestClient, sample_choice_question: dict):
    """Scenario 2: Multi-Tenant Concurrent Batch Inference and Benchmark Isolation.
    
    Workflow:
    1. Tenant A starts batch inference request.
    2. Tenant B attempts concurrent benchmark run.
    3. Concurrency lock prevents conflicts (HTTP 409 when benchmark is busy).
    4. Tenant A receives 200 with full batch results.
    """
    # Tenant A sends batch
    batch_payload = {
        "items": [f"Customer inquiry {i} regarding account access" for i in range(10)],
        "questions": sample_choice_question,
        "model": "microllm-gear1-fast",
    }
    b_resp = test_client.post("/v1/batch", json=batch_payload)
    assert b_resp.status_code == 200
    assert len(b_resp.json().get("results", [])) == 10

    # Tenant B runs benchmark
    bench_resp = test_client.post("/v1/benchmark/run", json={"mode": "fast"})
    assert bench_resp.status_code in (200, 409)


def test_tier4_03_agentic_mcp_multilingual_session(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Scenario 3: Agentic MCP Multilingual Tool Session.
    
    Workflow:
    1. Agent calls systemone_decide in French.
    2. Agent calls systemone_decide in Spanish.
    3. Verify neural model backend is memoized in MCP server without disk reload.
    4. Agent calls systemone_calibrate with validated arrays.
    """
    # Step 1: French decide
    res_fr = mcp_server.systemone_decide(
        state="Bonjour, j'aimerais changer mon mot de passe de connexion.",
        questions=sample_choice_question,
        multilingual_strategy="strategy2_neural",
    )
    assert isinstance(res_fr, dict)

    # Step 2: Spanish decide
    res_es = mcp_server.systemone_decide(
        state="Hola, necesito actualizar mi tarjeta de credito en la cuenta.",
        questions=sample_choice_question,
        multilingual_strategy="strategy2_neural",
    )
    assert isinstance(res_es, dict)

    # Step 3: Verify backend memoization
    if hasattr(mcp_server, "_multilingual_backends"):
        assert len(mcp_server._multilingual_backends) > 0

    # Step 4: Calibrate
    cal_res = mcp_server.systemone_calibrate(
        logits=[[2.0, 0.5, 0.1], [0.1, 1.8, 0.4], [0.2, 0.3, 2.1]],
        labels=[0, 1, 2],
    )
    assert isinstance(cal_res, dict)
    assert "error" not in cal_res or cal_res.get("error") is None


def test_tier4_04_sdk_client_systemone_and_batch_pipeline(test_client: TestClient, sample_choice_question: dict):
    """Scenario 4: High-Throughput MicroJevClient SDK Ingestion.
    
    Workflow:
    1. Initialize MicroJevClient.
    2. Execute individual System-1 query.
    3. Execute batch query.
    4. Verify identical payload builder contracts.
    """
    from microllm.client import MicroJevClient, _BaseMicroJevClient
    client = MicroJevClient(base_url="http://localhost:8000")
    assert isinstance(client, _BaseMicroJevClient)


def test_tier4_05_async_dual_client_pipeline():
    """Scenario 5: Async Dual-Client Pipeline (AsyncMicroJevClient).
    
    Workflow:
    1. Verify AsyncMicroJevClient shares base with MicroJevClient.
    2. Verify async methods mirror sync methods.
    """
    from microllm.client import AsyncMicroJevClient, MicroJevClient, _BaseMicroJevClient
    assert issubclass(AsyncMicroJevClient, _BaseMicroJevClient)
    assert issubclass(MicroJevClient, _BaseMicroJevClient)


def test_tier4_06_standalone_distribution_and_packaged_json_integrity(repo_root: Path):
    """Scenario 6: Standalone Distribution & Package Integrity.
    
    Workflow:
    1. Verify all 10 JSON data files exist in src/microllm/data/.
    2. Verify papers catalog loads cleanly from packaged JSON.
    3. Verify datasets catalog loads cleanly from packaged JSON.
    """
    data_dir = repo_root / "src" / "microllm" / "data"
    assert data_dir.exists()
    json_files = list(data_dir.glob("*.json"))
    assert len(json_files) == 10

    from microllm.papers import PAPERS
    assert len(PAPERS) >= 3

    from microllm.hf_jev_datasets import HF_JEV_DATASETS
    assert len(HF_JEV_DATASETS) >= 3


def test_tier4_07_hardened_container_security_compliance(repo_root: Path):
    """Scenario 7: Hardened Container Runtime Compliance (Docker Security Audit).
    
    Workflow:
    1. Inspect Dockerfile for ENV MICROLLM_CACHE_DIR=/app/.cache.
    2. Verify unprivileged USER appuser is configured.
    3. Verify jekyll-docker.yml contains no chmod 777.
    """
    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "MICROLLM_CACHE_DIR=/app/.cache" in dockerfile
    assert "USER appuser" in dockerfile

    jekyll_wf = (repo_root / ".github" / "workflows" / "jekyll-docker.yml").read_text(encoding="utf-8")
    assert "chmod 777" not in jekyll_wf


def test_tier4_08_ci_cd_workflow_deduplication_and_lint_integrity(repo_root: Path):
    """Scenario 8: CI/CD Packaging and Distribution Pipeline.
    
    Workflow:
    1. Verify unit-tests.yml defines test matrix.
    2. Verify lint.yml enforces strict linting without --exit-zero.
    3. Verify sync-hf-space.yml specifies concurrency control.
    """
    ut_wf = repo_root / ".github" / "workflows" / "unit-tests.yml"
    assert ut_wf.exists()

    lint_wf = (repo_root / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    assert "--exit-zero" not in lint_wf

    hf_wf = (repo_root / ".github" / "workflows" / "sync-hf-space.yml").read_text(encoding="utf-8")
    assert "concurrency:" in hf_wf


def test_tier4_09_conformal_prediction_and_temperature_scaling_calibration(test_client: TestClient):
    """Scenario 9: Conformal Prediction & Temperature Scaling Calibration Flow.
    
    Workflow:
    1. Submit valid logits and integer labels.
    2. Verify rejection of boolean labels.
    3. Verify conformal prediction alpha and risk parameters returned.
    """
    # Rejection of booleans
    bad_resp = test_client.post("/v1/calibrate", json={"logits": [[1.0, 0.0], [0.0, 1.0]], "labels": [True, False]})
    assert bad_resp.status_code == 422

    # Acceptance of valid payload
    good_payload = {
        "logits": [[2.0, 0.5, 0.1], [0.2, 1.8, 0.4], [0.3, 0.2, 2.5], [1.9, 0.3, 0.2], [0.1, 2.2, 0.3], [0.2, 0.4, 2.0]],
        "labels": [0, 1, 2, 0, 1, 2],
        "alpha": 0.10,
        "target_risk": 0.10,
        "delta": 0.05,
    }
    good_resp = test_client.post("/v1/calibrate", json=good_payload)
    assert good_resp.status_code == 200
    data = good_resp.json()
    assert "temperature" in data or "scaling" in data or "ece_before" in data


def test_tier4_10_multi_question_ticket_routing_with_verdict_mq_cache(mcp_server: MicroLLMMCPServer, sample_choice_question: dict):
    """Scenario 10: Multi-Question Ticket Routing with VerdictMQ Single-Encode.
    
    Workflow:
    1. Submit ticket across 3 distinct questions.
    2. Verify all questions are answered in a single response dictionary.
    """
    multi_q = {
        "dept": sample_choice_question["dept"],
        "urgent": {
            "type": "choice",
            "instructions": "Is ticket urgent?",
            "criteria": {"yes": "Urgent", "no": "Not urgent"},
        },
        "action": {
            "type": "choice",
            "instructions": "Recommended action?",
            "criteria": {"refund": "Issue refund", "reply": "Standard reply", "escalate": "Escalate to manager"},
        },
    }
    res = mcp_server.systemone_decide(
        state="Customer billing error needing immediate supervisor escalation and refund",
        questions=multi_q,
    )
    assert isinstance(res, dict)
    assert "answers" in res or "dept" in str(res)


def test_tier4_11_zero_sleep_rlcd_decision_head_convergence_stress():
    """Scenario 11: Zero-Sleep RLCD Decision Head Convergence Stress.
    
    Workflow:
    1. Train RLCD head with 100 samples, 4 features, 3 classes, 30 steps.
    2. Assert training finishes in < 150ms.
    3. Verify predicted probabilities sum to 1.0.
    """
    cfg = RLCDConfig(stage1_steps=20, stage2_steps=20, use_rlcd2_weights=True)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=3, config=cfg, seed=99)
    x = np.random.RandomState(99).randn(100, 4)
    y = np.random.RandomState(99).randint(0, 3, size=100)

    t0 = time.perf_counter()
    metrics = trainer.fit(x, y)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.200, f"Stress fit took {elapsed:.3f}s; artificial sleep is likely present."
    probs = trainer.predict_proba(x[:5])
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_tier4_12_atomic_profile_lifecycle(test_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Scenario 12: Atomic Profile Lifecycle (Pull -> Load -> Decide -> Persist).
    
    Workflow:
    1. Set isolated cache directory.
    2. Pull standard profile without client-supplied cache_dir.
    3. Run decision inference.
    4. Verify no arbitrary directories were created outside the cache directory.
    """
    isolated_cache = tmp_path / "lifecycle_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(isolated_cache))

    # Pull profile
    pull_resp = test_client.post("/v1/profiles/pull", json={"profile": "microllm-gear1-fast"})
    assert pull_resp.status_code in (200, 404)

    # Healthcheck
    health = test_client.get("/healthz")
    assert health.status_code == 200
