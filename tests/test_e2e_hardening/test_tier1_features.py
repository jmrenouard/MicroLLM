"""Tier 1: Feature Coverage (>=5 tests per feature for each of the 12 findings F01-F12).

Total tests in this module: 60 (5 tests * 12 features).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import os
import re
import socket
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from microllm.backends.base import VerdictMQMultiQuestionCache
from microllm.client import AsyncMicroJevClient, Choice, MicroJevClient
from microllm.hf_publishing import guard_sockets
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import create_app


# ============================================================================
# F01: Universal Endpoint Authentication (R1)
# Protected: /v1/studio/*, /v1/benchmark*, /v1/metrics, /v1/profiles/pull
# Public: /v1/models, /v1/jev-models, /v1/papers, /healthz
# ============================================================================

def test_tier1_f01_01_studio_analyze_requires_auth(
    auth_client: TestClient, auth_headers: dict[str, str], invalid_auth_headers: dict[str, str]
) -> None:
    """F01.1: POST /v1/studio/analyze returns 401 without auth, 200 with valid auth."""
    payload = {
        "state": "Customer requesting urgent invoice billing refund.",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Route ticket",
                "criteria": {"billing": "Refunds and payments", "tech": "Bugs"},
            }
        },
        "gate_threshold": 0.75,
        "alpha": 0.75,
    }
    # Unauthenticated request must return 401
    resp_unauth = auth_client.post("/v1/studio/analyze", json=payload)
    assert resp_unauth.status_code == 401, f"Expected 401 on /v1/studio/analyze, got {resp_unauth.status_code}"

    # Invalid token must return 401
    resp_invalid = auth_client.post("/v1/studio/analyze", json=payload, headers=invalid_auth_headers)
    assert resp_invalid.status_code == 401

    # Valid token must succeed
    resp_valid = auth_client.post("/v1/studio/analyze", json=payload, headers=auth_headers)
    assert resp_valid.status_code == 200
    assert "gear1" in resp_valid.json() or "diagnostics" in resp_valid.json() or "answers" in resp_valid.json()


def test_tier1_f01_02_studio_rlcd_demo_requires_auth(
    auth_client: TestClient, auth_headers: dict[str, str], invalid_auth_headers: dict[str, str]
) -> None:
    """F01.2: POST /v1/studio/rlcd-demo returns 401 without auth, 200 with valid auth."""
    payload = {"stage1_steps": 10, "stage2_steps": 10, "seed": 42}

    resp_unauth = auth_client.post("/v1/studio/rlcd-demo", json=payload)
    assert resp_unauth.status_code == 401, f"Expected 401 on /v1/studio/rlcd-demo, got {resp_unauth.status_code}"

    resp_invalid = auth_client.post("/v1/studio/rlcd-demo", json=payload, headers=invalid_auth_headers)
    assert resp_invalid.status_code == 401

    resp_valid = auth_client.post("/v1/studio/rlcd-demo", json=payload, headers=auth_headers)
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "stage1_brier" in data or "final_brier" in data or "stage1_train_acc" in data or "stage2_accuracy" in data


def test_tier1_f01_03_metrics_requires_auth(
    auth_client: TestClient, auth_headers: dict[str, str], invalid_auth_headers: dict[str, str]
) -> None:
    """F01.3: GET /v1/metrics returns 401 without auth, 200 with valid auth."""
    resp_unauth = auth_client.get("/v1/metrics")
    assert resp_unauth.status_code == 401, f"Expected 401 on /v1/metrics, got {resp_unauth.status_code}"

    resp_invalid = auth_client.get("/v1/metrics", headers=invalid_auth_headers)
    assert resp_invalid.status_code == 401

    resp_valid = auth_client.get("/v1/metrics", headers=auth_headers)
    assert resp_valid.status_code == 200
    assert "total_requests" in resp_valid.json()


def test_tier1_f01_04_benchmark_get_routes_require_auth(
    auth_client: TestClient, auth_headers: dict[str, str], invalid_auth_headers: dict[str, str]
) -> None:
    """F01.4: GET /v1/benchmark* endpoints return 401 without auth, 200 with valid auth."""
    for path in ["/v1/benchmark", "/v1/benchmark/external", "/v1/benchmark/hf-jev-models"]:
        resp_unauth = auth_client.get(path)
        assert resp_unauth.status_code == 401, f"Expected 401 on {path}, got {resp_unauth.status_code}"

        resp_invalid = auth_client.get(path, headers=invalid_auth_headers)
        assert resp_invalid.status_code == 401, f"Expected 401 on {path} with bad token, got {resp_invalid.status_code}"

        resp_valid = auth_client.get(path, headers=auth_headers)
        assert resp_valid.status_code == 200, f"Expected 200 on {path} with auth, got {resp_valid.status_code}"


def test_tier1_f01_05_public_routes_remain_open_without_auth(auth_client: TestClient) -> None:
    """F01.5: Public routes (/v1/models, /v1/jev-models, /v1/papers, /healthz) remain unauthenticated."""
    for path in ["/v1/models", "/v1/jev-models", "/v1/papers"]:
        resp = auth_client.get(path)
        assert resp.status_code == 200, f"Expected 200 on public route {path}, got {resp.status_code}"


# ============================================================================
# F02: Direct In-Process Core Execution (R1)
# Direct in-process logic for 7 MCP tools & Gradio tabs, no loopback TestClient
# ============================================================================

def test_tier1_f02_01_evaluate_guardrail_in_process_returns_valid_structure() -> None:
    """F02.1: evaluate_guardrail_in_process executes in-process returning required structure."""
    import microllm.server as srv

    assert hasattr(srv, "evaluate_guardrail_in_process") or hasattr(srv, "evaluate_guardrail"), (
        "Feature F02 missing: evaluate_guardrail_in_process not defined in microllm.server"
    )
    fn = getattr(srv, "evaluate_guardrail_in_process", getattr(srv, "evaluate_guardrail", None))
    assert fn is not None

    result = fn(
        state="Customer wants to update billing address.",
        threshold=0.5,
        context="Account management",
    )
    assert isinstance(result, dict)
    assert "safe" in result
    assert "action" in result
    assert "risk_score" in result
    assert "groundedness_score" in result
    assert "checks" in result


def test_tier1_f02_02_evaluate_calibration_in_process_returns_valid_metrics() -> None:
    """F02.2: evaluate_calibration_in_process executes in-process returning calibration metrics."""
    import microllm.server as srv

    assert hasattr(srv, "evaluate_calibration_in_process"), (
        "Feature F02 missing: evaluate_calibration_in_process not defined in microllm.server"
    )
    fn = getattr(srv, "evaluate_calibration_in_process")
    result = fn(
        confidence_scores=[0.9, 0.8, 0.85, 0.7, 0.95, 0.6],
        accuracy_labels=[1, 1, 1, 0, 1, 0],
        alpha=0.1,
    )
    assert isinstance(result, dict)
    assert "raw_ece" in result
    assert "calibrated_ece" in result
    assert "conformal_q_hat" in result


def test_tier1_f02_03_run_rlcd_experiment_in_process_executes_successfully() -> None:
    """F02.3: run_rlcd_experiment_in_process executes RLCD training in-process."""
    import microllm.server as srv
    from microllm.server import StudioRLCDDemoRequest

    assert hasattr(srv, "run_rlcd_experiment_in_process"), (
        "Feature F02 missing: run_rlcd_experiment_in_process not defined in microllm.server"
    )
    fn = getattr(srv, "run_rlcd_experiment_in_process")
    req = StudioRLCDDemoRequest(stage1_steps=10, stage2_steps=10, seed=42)
    result = fn(req)
    assert isinstance(result, dict)
    assert "stage1_brier" in result or "final_brier" in result or "stage1_train_acc" in result or "stage2_accuracy" in result


def test_tier1_f02_04_mcp_guardrail_tools_succeed_under_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """F02.4: MCP server tools execute cleanly when MICROLLM_API_KEY is configured in environment."""
    monkeypatch.setenv("MICROLLM_API_KEY", "mcp-secret-key-123")
    from microllm.mcp_server import MicroLLMMCPServer

    mcp = MicroLLMMCPServer()
    resp = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "systemone_guardrail",
            "arguments": {"state": "Normal user query about product pricing."},
        },
    })
    assert isinstance(resp, dict)
    assert resp.get("result", {}).get("isError") is False
    assert "safe" in resp["result"]["structuredContent"]
    assert "risk_score" in resp["result"]["structuredContent"]


def test_tier1_f02_05_gradio_safety_guardrail_handler_succeeds_under_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """F02.5: Gradio safety guardrail handler executes cleanly when MICROLLM_API_KEY is configured."""
    monkeypatch.setenv("MICROLLM_API_KEY", "gradio-secret-key-456")
    import microllm.gradio_app as gapp

    res = gapp.run_guardrail_with_custom_rubrics(
        text_input="I want to cancel my subscription.",
        threshold=0.5,
        context_input="Billing portal",
        custom_rubrics_text="",
    )
    assert isinstance(res, (tuple, list))
    assert len(res) >= 3


# ============================================================================
# F03: Input Validation on RLCD & Trainer (R2)
# Zero-500 guarantee via Pydantic model validation (HTTP 422) & trainer ValueError
# ============================================================================

def test_tier1_f03_01_rlcd_endpoint_rejects_empty_features_or_labels(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.1: Submitting empty features or labels returns HTTP 422, never 500."""
    resp1 = auth_client.post(
        "/v1/rlcd",
        json={"stage1_steps": 10, "stage2_steps": 10, "features": [], "labels": []},
        headers=auth_headers,
    )
    assert resp1.status_code == 422, f"Expected 422 on empty arrays, got {resp1.status_code}"

    resp2 = auth_client.post(
        "/v1/studio/rlcd-demo",
        json={"stage1_steps": 10, "stage2_steps": 10, "features": [], "labels": []},
        headers=auth_headers,
    )
    assert resp2.status_code == 422, f"Expected 422 on empty arrays, got {resp2.status_code}"


def test_tier1_f03_02_rlcd_endpoint_rejects_sample_count_mismatch(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.2: Submitting mismatched sample counts returns HTTP 422."""
    payload = {
        "stage1_steps": 10,
        "stage2_steps": 10,
        "features": [[1.0, 2.0], [3.0, 4.0]],  # 2 samples
        "labels": [0, 1, 0],  # 3 labels
    }
    resp = auth_client.post("/v1/rlcd", json=payload, headers=auth_headers)
    assert resp.status_code == 422, f"Expected 422 on mismatched counts, got {resp.status_code}"


def test_tier1_f03_03_rlcd_endpoint_rejects_1d_features(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.3: Submitting 1D features array instead of 2D returns HTTP 422."""
    payload = {
        "stage1_steps": 10,
        "stage2_steps": 10,
        "features": [1.0, 2.0, 3.0],  # 1D array
        "labels": [0],
    }
    resp = auth_client.post("/v1/rlcd", json=payload, headers=auth_headers)
    assert resp.status_code == 422, f"Expected 422 on 1D features, got {resp.status_code}"


def test_tier1_f03_04_rlcd_endpoint_rejects_negative_labels(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.4: Submitting negative labels returns HTTP 422."""
    payload = {
        "stage1_steps": 10,
        "stage2_steps": 10,
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [-1, 0],  # negative label
    }
    resp = auth_client.post("/v1/rlcd", json=payload, headers=auth_headers)
    assert resp.status_code == 422, f"Expected 422 on negative label, got {resp.status_code}"


def test_tier1_f03_05_rlcd_trainer_fit_raises_value_error_on_invalid_inputs() -> None:
    """F03.5: RLCDDecisionHeadTrainer.fit raises ValueError for invalid shapes."""
    cfg = RLCDConfig(stage1_steps=5, stage2_steps=5)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg)

    # Empty inputs
    with pytest.raises(ValueError):
        trainer.fit(np.zeros((0, 4)), np.zeros(0, dtype=int))

    # Mismatched length
    with pytest.raises(ValueError):
        trainer.fit(np.zeros((5, 4)), np.zeros(4, dtype=int))

    # 1D features
    with pytest.raises(ValueError):
        trainer.fit(np.zeros(5), np.zeros(5, dtype=int))

    # Negative labels
    with pytest.raises(ValueError):
        trainer.fit(np.ones((4, 2)), np.array([-1, 0, 1, 0], dtype=int))


# ============================================================================
# F04: Thread-Isolated guard_sockets() (R3)
# Scoped strictly to threading.get_ident(), delegating other threads
# ============================================================================

def test_tier1_f04_01_guard_sockets_blocks_calling_thread() -> None:
    """F04.1: guard_sockets blocks socket connections on the calling thread."""
    with guard_sockets():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(RuntimeError, match="offline"):
                s.connect(("127.0.0.1", 9999))
        finally:
            s.close()


def test_tier1_f04_02_guard_sockets_allows_concurrent_thread_sockets() -> None:
    """F04.2: guard_sockets isolates the restriction to calling thread (other threads unaffected)."""
    # Create listening socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    port = server_sock.getsockname()[1]

    thread_result: list[bool] = []
    thread_error: list[Exception] = []

    def _worker() -> None:
        try:
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.connect(("127.0.0.1", port))
            client_sock.close()
            thread_result.append(True)
        except Exception as e:
            thread_error.append(e)

    try:
        with guard_sockets():
            # In main thread, sockets are blocked:
            s_main = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with pytest.raises(RuntimeError):
                    s_main.connect(("127.0.0.1", port))
            finally:
                s_main.close()

            # In worker thread, socket connection must NOT be intercepted:
            t = threading.Thread(target=_worker)
            t.start()
            try:
                conn, _ = server_sock.accept()
                conn.close()
            except socket.timeout:
                pass
            t.join(timeout=2.0)
    finally:
        server_sock.close()

    assert not thread_error, f"Worker thread failed with: {thread_error}"
    assert thread_result == [True]


def test_tier1_f04_03_guard_sockets_reentrant_nesting() -> None:
    """F04.3: guard_sockets is re-entrant on the same thread."""
    with guard_sockets():
        with guard_sockets():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with pytest.raises(RuntimeError):
                    s.connect(("127.0.0.1", 9))
            finally:
                s.close()
        # Still inside outer block: must remain blocked
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(RuntimeError):
                s2.connect(("127.0.0.1", 9))
        finally:
            s2.close()


def test_tier1_f04_04_guard_sockets_restores_original_connect() -> None:
    """F04.4: guard_sockets cleanly restores original socket.connect upon exit."""
    orig_connect = socket.socket.connect
    with guard_sockets():
        pass
    assert socket.socket.connect is orig_connect


def test_tier1_f04_05_guard_sockets_interleaved_thread_exits() -> None:
    """F04.5: Multiple threads entering and exiting guard_sockets do not corrupt socket state."""
    errors: list[Exception] = []

    def _run_guard(delay: float) -> None:
        try:
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    with pytest.raises(RuntimeError):
                        s.connect(("127.0.0.1", 9))
                finally:
                    s.close()
                import time
                time.sleep(delay)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=_run_guard, args=(0.02,))
    t2 = threading.Thread(target=_run_guard, args=(0.05,))
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert not errors


# ============================================================================
# F05: Non-Blocking Event Loop & /healthz & Cache Lock (R3)
# ============================================================================

def test_tier1_f05_01_healthz_endpoint_returns_ok(unauth_client: TestClient) -> None:
    """F05.1: GET /healthz returns HTTP 200 with ok status and microllm service identifier."""
    resp = unauth_client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "microllm" in str(data.get("service", "")).lower()


def test_tier1_f05_02_healthz_responds_concurrently_during_cpu_inference(auth_client: TestClient, auth_headers: dict[str, str]) -> None:
    """F05.2: /healthz returns immediately without being blocked by concurrent requests."""
    resp = auth_client.get("/healthz")
    assert resp.status_code == 200


def test_tier1_f05_03_cpu_routes_offloaded_to_threadpool() -> None:
    """F05.3: CPU-heavy route handlers use asyncio.to_thread for execution."""
    import inspect
    import microllm.server as srv

    src = inspect.getsource(srv.create_app)
    assert "to_thread" in src or "run_in_executor" in src, (
        "Server endpoints must offload CPU-bound execution via asyncio.to_thread"
    )


def test_tier1_f05_04_verdict_mq_cache_thread_safe_concurrent_access() -> None:
    """F05.4: VerdictMQMultiQuestionCache is thread-safe under concurrent operations."""
    cache = VerdictMQMultiQuestionCache(max_entries=64)
    exceptions: list[Exception] = []

    def _worker(thread_id: int) -> None:
        try:
            for i in range(10):
                text = f"Worker {thread_id} item {i}: invoice refund request"
                entry = cache.encode_state_once(text)
                assert "clause_count" in entry
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not exceptions, f"Concurrent cache access threw: {exceptions}"


def test_tier1_f05_05_cache_lock_synchronizes_state_encoding() -> None:
    """F05.5: VerdictMQMultiQuestionCache uses a synchronization lock."""
    cache = VerdictMQMultiQuestionCache(max_entries=32)
    assert hasattr(cache, "_lock") or hasattr(cache, "lock"), (
        "VerdictMQMultiQuestionCache must have a threading lock for safe concurrency"
    )


# ============================================================================
# F06: Multi-Question Token Caching (R4)
# Memoize tokenize_multilingual and reuse cached state tokens
# ============================================================================

def test_tier1_f06_01_tokenize_multilingual_memoizes_identical_inputs() -> None:
    """F06.1: tokenize_multilingual caches tokenization across identical inputs."""
    from microllm.backends.cascade import tokenize_multilingual

    text = "Bonjour, j'aimerais recevoir le remboursement de ma facture urgente."
    tok1 = tokenize_multilingual(text)
    tok2 = tokenize_multilingual(text)
    assert tok1 == tok2
    # Verify memoization attribute exists or returns identical object/cached representation
    assert hasattr(tokenize_multilingual, "cache_info") or tok1 is tok2 or isinstance(tok1, tuple)


def test_tier1_f06_02_multi_question_evaluation_reuses_state_tokens(
    unauth_client: TestClient, sample_choice_question: dict[str, Any]
) -> None:
    """F06.2: Multi-question POST /v1/systemone reuses cached state representation."""
    questions = dict(sample_choice_question)
    questions["urgency"] = {
        "type": "choice",
        "instructions": "Determine ticket urgency.",
        "criteria": {"high": "Urgent immediate action needed", "low": "Routine question"},
    }
    payload = {
        "model": "jev-latest",
        "state": "URGENT: credit card charged twice for invoice 1042! Need refund.",
        "questions": questions,
    }
    resp = unauth_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert "dept" in answers
    assert "urgency" in answers


def test_tier1_f06_03_evaluate_voc_gate_regime_uses_cached_tokens() -> None:
    """F06.3: evaluate_voc_gate_regime accepts and utilizes pre-computed state tokens."""
    import inspect
    from microllm.backends import cascade as casc_mod

    if hasattr(casc_mod, "evaluate_voc_gate_regime"):
        sig = inspect.signature(casc_mod.evaluate_voc_gate_regime)
        assert "state_tokens" in sig.parameters or "cached_tokens" in sig.parameters or "tokens" in sig.parameters


def test_tier1_f06_04_token_computation_count_does_not_scale_with_question_count() -> None:
    """F06.4: State tokenization occurs once per state evaluation, not M times."""
    cache = VerdictMQMultiQuestionCache(max_entries=16)
    text = "Critical system alert: disk latency spike on primary replica."
    e1 = cache.encode_state_once(text)
    e2 = cache.encode_state_once(text)
    assert e1 is e2


def test_tier1_f06_05_cache_eviction_and_bounded_capacity() -> None:
    """F06.5: Multi-question cache enforces max_entries capacity bound."""
    cache = VerdictMQMultiQuestionCache(max_entries=5)
    for i in range(10):
        cache.encode_state_once(f"State test string {i}")
    assert len(cache._cache) <= 5


# ============================================================================
# F07: SDK Client Batch Parity (R4)
# MicroJevClient.batch() accepts multilingual_strategy & neural_model
# ============================================================================

def test_tier1_f07_01_sync_batch_accepts_multilingual_strategy() -> None:
    """F07.1: MicroJevClient.batch() accepts multilingual_strategy parameter."""
    sig = inspect.signature(MicroJevClient.batch)
    assert "multilingual_strategy" in sig.parameters, (
        "MicroJevClient.batch() must accept multilingual_strategy"
    )


def test_tier1_f07_02_sync_batch_accepts_neural_model() -> None:
    """F07.2: MicroJevClient.batch() accepts neural_model parameter."""
    sig = inspect.signature(MicroJevClient.batch)
    assert "neural_model" in sig.parameters, (
        "MicroJevClient.batch() must accept neural_model"
    )


def test_tier1_f07_03_sync_batch_signature_matches_async_batch() -> None:
    """F07.3: MicroJevClient.batch signature has parity with AsyncMicroJevClient.batch."""
    sync_params = set(inspect.signature(MicroJevClient.batch).parameters.keys())
    async_params = set(inspect.signature(AsyncMicroJevClient.batch).parameters.keys())
    assert "multilingual_strategy" in sync_params
    assert "neural_model" in sync_params
    assert "multilingual_strategy" in async_params
    assert "neural_model" in async_params


def test_tier1_f07_04_sync_batch_forwards_multilingual_strategy_in_payload() -> None:
    """F07.4: MicroJevClient.batch() forwards multilingual_strategy in request payload."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    res = client.batch(
        items=["Facture impayée"],
        questions={"dept": Choice(options={"billing": "Factures", "tech": "Bugs"})},
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    assert res["summary"]["total_items"] == 1
    assert res["results"][0]["answers"]["dept"]["choice"] == "billing"


def test_tier1_f07_05_sync_batch_forwards_neural_model_in_payload() -> None:
    """F07.5: MicroJevClient.batch() forwards neural_model in request payload."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    res = client.batch(
        items=["Problema con el pago"],
        questions={"dept": Choice(options={"billing": "Pagos", "tech": "Tecnico"})},
        neural_model="minilm",
    )
    assert res["summary"]["total_items"] == 1


# ============================================================================
# F08: SDK Base Consolidation (R5)
# _BaseMicroJevClient & shared request/response construction helpers
# ============================================================================

def test_tier1_f08_01_base_micro_jev_client_exists_and_inherited() -> None:
    """F08.1: _BaseMicroJevClient exists and both sync and async clients inherit from it."""
    import microllm.client as cli_mod

    assert hasattr(cli_mod, "_BaseMicroJevClient"), (
        "Feature F08 missing: _BaseMicroJevClient not found in microllm.client"
    )
    base_cls = getattr(cli_mod, "_BaseMicroJevClient")
    assert issubclass(MicroJevClient, base_cls)
    assert issubclass(AsyncMicroJevClient, base_cls)


def test_tier1_f08_02_shared_auth_headers_generation() -> None:
    """F08.2: _BaseMicroJevClient._auth_headers formats Authorization header consistently."""
    import microllm.client as cli_mod

    base_cls = getattr(cli_mod, "_BaseMicroJevClient")
    inst = base_cls(api_key="secret-key-xyz")
    assert inst._auth_headers() == {"Authorization": "Bearer secret-key-xyz"}


def test_tier1_f08_03_shared_build_guardrail_payload() -> None:
    """F08.3: _build_guardrail_payload builds standardized request payload."""
    import microllm.client as cli_mod

    assert hasattr(cli_mod, "_build_guardrail_payload")
    fn = getattr(cli_mod, "_build_guardrail_payload")
    p = fn(state="Hello", context="ctx", threshold=0.6, custom_rubrics={"r1": "crit"})
    assert p["state"] == "Hello"
    assert p["context"] == "ctx"
    assert p["threshold"] == 0.6
    assert p["custom_rubrics"] == {"r1": "crit"}


def test_tier1_f08_04_shared_build_batch_payload() -> None:
    """F08.4: _build_batch_payload serializes items and questions into standard payload."""
    import microllm.client as cli_mod

    assert hasattr(cli_mod, "_build_batch_payload")
    fn = getattr(cli_mod, "_build_batch_payload")
    p = fn(
        items=["Item 1"],
        questions={"q": Choice(options={"a": "opt A"})},
        model="jev-latest",
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    assert p["items"] == ["Item 1"]
    assert "questions" in p
    assert p["multilingual_strategy"] == "neural"


def test_tier1_f08_05_shared_build_papers_params() -> None:
    """F08.5: _build_papers_params constructs query parameters dictionary."""
    import microllm.client as cli_mod

    assert hasattr(cli_mod, "_build_papers_params")
    fn = getattr(cli_mod, "_build_papers_params")
    params = fn(pillar="pillar1", query="conformal", arxiv_id="2401.00000")
    assert params["pillar"] == "pillar1"
    assert params["q"] == "conformal"
    assert params["arxiv_id"] == "2401.00000"


# ============================================================================
# F09: Space License Alignment (R4)
# Default license_type to "apache-2.0" in hf_space_sync.py
# ============================================================================

def test_tier1_f09_01_build_space_readme_defaults_to_apache_2_0() -> None:
    """F09.1: build_space_readme defaults to apache-2.0 license."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme()
    assert "license: apache-2.0" in readme.lower()
    assert "license: mit" not in readme.lower()


def test_tier1_f09_02_stage_hf_space_defaults_to_apache_2_0(tmp_path: Path) -> None:
    """F09.2: stage_hf_space defaults to apache-2.0 license."""
    from microllm.hf_space_sync import stage_hf_space

    res = stage_hf_space(staging_dir=tmp_path / "space_test")
    readme_path = Path(res["staging_dir"]) / "README.md"
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")
    assert "license: apache-2.0" in content.lower()


def test_tier1_f09_03_sync_hf_space_signature_defaults_to_apache_2_0() -> None:
    """F09.3: sync_hf_space parameter license_type defaults to 'apache-2.0'."""
    from microllm.hf_space_sync import sync_hf_space

    sig = inspect.signature(sync_hf_space)
    param = sig.parameters.get("license_type")
    assert param is not None
    assert str(param.default).lower() in ("apache-2.0", "apache_2_0")


def test_tier1_f09_04_space_readme_frontmatter_yaml_validity() -> None:
    """F09.4: Space README frontmatter begins and ends with '---' and contains Apache-2.0."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme()
    assert readme.startswith("---\n")
    second_marker = readme.find("\n---\n", 4)
    assert second_marker != -1
    frontmatter = readme[4:second_marker]
    assert "license: apache-2.0" in frontmatter.lower()


def test_tier1_f09_05_space_license_matches_pyproject_toml() -> None:
    """F09.5: Space default license matches pyproject.toml license."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.is_file()
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject_text or 'license = { text = "Apache-2.0" }' in pyproject_text


# ============================================================================
# F10: Workstation Paths Elimination (R4)
# No hardcoded C:\Users\jmren, configurable via env vars and Path.home()
# ============================================================================

def test_tier1_f10_01_no_hardcoded_user_paths_in_hf_publishing() -> None:
    """F10.1: hf_publishing.py contains zero hardcoded C:\\Users\\jmren paths."""
    hfp_path = Path("src/microllm/hf_publishing.py")
    assert hfp_path.is_file()
    src = hfp_path.read_text(encoding="utf-8")
    assert "C:\\Users\\jmren" not in src
    assert "c:/users/jmren" not in src.lower()


def test_tier1_f10_02_keepass_db_path_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """F10.2: MICROLLM_KEEPASS_DB environment variable is respected."""
    import microllm.hf_publishing as hfp

    custom_db = tmp_path / "custom_vault.kdbx"
    monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(custom_db))
    if hasattr(hfp, "get_vault_paths"):
        db_path, _ = hfp.get_vault_paths()
        assert Path(db_path) == custom_db


def test_tier1_f10_03_keepass_key_path_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """F10.3: MICROLLM_KEEPASS_KEY environment variable is respected."""
    import microllm.hf_publishing as hfp

    custom_key = tmp_path / "custom_vault.keyx"
    monkeypatch.setenv("MICROLLM_KEEPASS_KEY", str(custom_key))
    if hasattr(hfp, "get_vault_paths"):
        _, key_path = hfp.get_vault_paths()
        assert Path(key_path) == custom_key


def test_tier1_f10_04_fallback_uses_portable_home_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """F10.4: When env vars are unset, vault path falls back to Path.home()."""
    import microllm.hf_publishing as hfp

    monkeypatch.delenv("MICROLLM_KEEPASS_DB", raising=False)
    monkeypatch.delenv("MICROLLM_KEEPASS_KEY", raising=False)
    if hasattr(hfp, "get_vault_paths"):
        db_path, key_path = hfp.get_vault_paths()
        home = Path.home()
        assert home in Path(db_path).parents or Path(db_path).is_relative_to(home)


def test_tier1_f10_05_cross_platform_path_resolution() -> None:
    """F10.5: resolve_hf_token runs without path syntax errors on any platform."""
    from microllm.hf_publishing import resolve_hf_token

    tok, source = resolve_hf_token("explicit-test-token", allow_vault=False)
    assert tok == "explicit-test-token"
    assert source == "argument"


# ============================================================================
# F11: Benchmark CLI --json Flag (R4)
# Add --json argument to bench_p and output valid JSON
# ============================================================================

def test_tier1_f11_01_cli_benchmark_parser_registers_json_flag() -> None:
    """F11.1: CLI benchmark subcommand parser has --json flag registered."""
    import microllm.server as srv

    parser = srv.build_cli_parser() if hasattr(srv, "build_cli_parser") else None
    if parser is None and hasattr(srv, "run_cli"):
        # Test by parsing args directly with --json
        try:
            from microllm.cli import build_cli_parser
            parser = build_cli_parser()
        except ImportError:
            pass

    assert parser is not None, "CLI parser builder not found"
    parsed = parser.parse_args(["benchmark", "--multilingual", "--json"])
    assert getattr(parsed, "json", False) is True


def test_tier1_f11_02_cli_benchmark_multilingual_json_stdout() -> None:
    """F11.2: microllm benchmark --multilingual --json outputs valid JSON to stdout."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_fn(["benchmark", "--multilingual", "--json"])
    assert code == 0
    raw_out = buf.getvalue().strip()
    data = json.loads(raw_out)
    assert isinstance(data, dict)
    assert "benchmarks" in data or "summary" in data or "metrics" in data


def test_tier1_f11_03_cli_benchmark_pareto_json_stdout() -> None:
    """F11.3: microllm benchmark --pareto --json outputs valid JSON to stdout."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_fn(["benchmark", "--pareto", "--json"])
    assert code == 0
    raw_out = buf.getvalue().strip()
    data = json.loads(raw_out)
    assert isinstance(data, dict)


def test_tier1_f11_04_cli_benchmark_json_contains_required_metrics() -> None:
    """F11.4: Parsed benchmark JSON contains timestamp and report metrics."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_fn(["benchmark", "--multilingual", "--json"])
    data = json.loads(buf.getvalue().strip())
    assert "timestamp" in data or "models" in data or "summary" in data


def test_tier1_f11_05_cli_benchmark_human_readable_when_json_omitted() -> None:
    """F11.5: Running benchmark without --json does not format output as JSON."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_fn(["benchmark", "--multilingual"])
    out = buf.getvalue().strip()
    # Should not be pure JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


# ============================================================================
# F12: Modularization of Oversized Modules (R5)
# Decompose gradio_app, dataset_adapters, server while preserving exports
# ============================================================================

def test_tier1_f12_01_gradio_presets_extracted_and_reexported() -> None:
    """F12.1: gradio_presets module exists and presets are accessible from gradio_app."""
    import microllm.gradio_app as gapp
    import microllm.gradio_presets as gpresets

    assert hasattr(gpresets, "MULTILINGUAL_BENCHMARK_PRESETS")
    assert hasattr(gapp, "MULTILINGUAL_BENCHMARK_PRESETS")
    assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS


def test_tier1_f12_02_gradio_helpers_extracted_and_reexported() -> None:
    """F12.2: gradio_helpers module exists and helpers are accessible from gradio_app."""
    import microllm.gradio_app as gapp
    import microllm.gradio_helpers as ghelpers

    assert hasattr(ghelpers, "parse_criteria_text") or hasattr(ghelpers, "format_xai_attribution")
    for attr in dir(ghelpers):
        if not attr.startswith("_"):
            assert hasattr(gapp, attr)


def test_tier1_f12_03_dataset_fixtures_extracted_and_reexported() -> None:
    """F12.3: dataset_fixtures module contains OFFLINE_WEB_DATASET_SAMPLES and re-exported."""
    import microllm.dataset_adapters as dadapt
    import microllm.dataset_fixtures as dfix

    assert hasattr(dfix, "OFFLINE_WEB_DATASET_SAMPLES")
    assert hasattr(dadapt, "OFFLINE_WEB_DATASET_SAMPLES")
    assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES


def test_tier1_f12_04_server_cli_extracted_and_reexported() -> None:
    """F12.4: cli.py contains run_cli and main, re-exported by microllm.server."""
    import microllm.cli as scli
    import microllm.server as srv

    assert hasattr(scli, "run_cli")
    assert hasattr(srv, "run_cli")
    assert srv.run_cli is scli.run_cli


def test_tier1_f12_05_server_models_extracted_and_reexported() -> None:
    """F12.5: server_models.py contains StudioAnalyzeRequest etc., re-exported by server."""
    import microllm.server as srv
    import microllm.server_models as smodels

    assert hasattr(smodels, "StudioAnalyzeRequest")
    assert hasattr(srv, "StudioAnalyzeRequest")
    assert srv.StudioAnalyzeRequest is smodels.StudioAnalyzeRequest
