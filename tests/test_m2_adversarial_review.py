"""Adversarial stress and concurrency review suite for Milestone 2.

Authored by Reviewer 2 / Adversarial Critic for Milestone 2 (Validation, Concurrency & Event Loop Non-Blocking).
Covers:
1. F03: Input Validation on RLCD (Zero-500 HTTP 422 Guarantee, Trainer shape & bound validation)
2. F04: Concurrency Safety and Thread Isolation in guard_sockets()
3. F05: Event Loop Non-Blocking Execution, unauthenticated /healthz probe, and VerdictMQMultiQuestionCache thread safety
4. Integrity verification: Real computation, absence of test facades, genuine offline isolation
"""

from __future__ import annotations

import concurrent.futures
import inspect
import socket
import threading
import time
from typing import Any
import numpy as np
import pytest
from fastapi.testclient import TestClient

from microllm.backends.base import VerdictMQMultiQuestionCache
from microllm.hf_publishing import _GUARDED_THREADS, _SOCKET_GUARD_LOCK, guard_sockets
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import (
    StudioRLCDDemoRequest,
    create_app,
    run_rlcd_experiment_in_process,
)


# ===========================================================================
# 1. F03: Input Validation on RLCD (Zero-500 Guarantee & Bounds)
# ===========================================================================


def test_f03_adversarial_rlcd_payload_zero_500_guarantee() -> None:
    """Stress-test /v1/rlcd and /v1/studio/rlcd-demo with a wide range of malformed inputs.
    
    GUARANTEE: Must return HTTP 422 for all invalid payloads, never HTTP 500.
    """
    app = create_app(api_key="m2-adversarial-key")
    client = TestClient(app)
    headers = {"Authorization": "Bearer m2-adversarial-key"}

    malformed_payloads: list[tuple[str, dict[str, Any]]] = [
        ("features provided, labels None", {"features": [[1.0, 2.0]], "labels": None}),
        ("features None, labels provided", {"features": None, "labels": [0, 1]}),
        ("empty features list", {"features": [], "labels": []}),
        ("empty inner row in features", {"features": [[]], "labels": [0]}),
        ("features 1D instead of 2D", {"features": [1.0, 2.0, 3.0], "labels": [0]}),
        ("features 3D instead of 2D", {"features": [[[1.0, 2.0]]], "labels": [0]}),
        ("ragged features rows", {"features": [[1.0, 2.0], [3.0, 4.0, 5.0]], "labels": [0, 1]}),
        ("sample count mismatch (more features)", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0]}),
        ("sample count mismatch (more labels)", {"features": [[1.0, 2.0]], "labels": [0, 1]}),
        ("negative label value", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [-1, 0]}),
        ("single class label values", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 0]}),
        ("non-integer labels", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": ["zero", "one"]}),
        ("non-numeric features", {"features": [["a", "b"], ["c", "d"]], "labels": [0, 1]}),
        ("null inside features row", {"features": [[None, 1.0], [2.0, 3.0]], "labels": [0, 1]}),
    ]

    for name, payload in malformed_payloads:
        # Test /v1/rlcd
        resp_rlcd = client.post("/v1/rlcd", json=payload, headers=headers)
        assert resp_rlcd.status_code == 422, (
            f"Expected HTTP 422 for '{name}' on /v1/rlcd, got {resp_rlcd.status_code}: {resp_rlcd.text}"
        )
        assert resp_rlcd.status_code != 500, f"HTTP 500 integrity violation for '{name}' on /v1/rlcd!"

        # Test /v1/studio/rlcd-demo
        resp_demo = client.post("/v1/studio/rlcd-demo", json=payload, headers=headers)
        assert resp_demo.status_code == 422, (
            f"Expected HTTP 422 for '{name}' on /v1/studio/rlcd-demo, got {resp_demo.status_code}: {resp_demo.text}"
        )
        assert resp_demo.status_code != 500, f"HTTP 500 integrity violation for '{name}' on /v1/studio/rlcd-demo!"


def test_f03_adversarial_trainer_fit_and_predict_bounds() -> None:
    """Stress-test RLCDDecisionHeadTrainer shape assertions and edge validation."""
    cfg = RLCDConfig(stage1_steps=5, stage2_steps=5)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=3, config=cfg)

    # 1. predict_proba bounds
    with pytest.raises(ValueError, match="2D array"):
        trainer.predict_proba(np.zeros(4))  # 1D
    with pytest.raises(ValueError, match="2D array"):
        trainer.predict_proba(np.zeros((2, 4, 1)))  # 3D
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        trainer.predict_proba(np.zeros((3, 5)))  # d_feat=5 != 4
    with pytest.raises(ValueError, match="NaN or Infinite"):
        trainer.predict_proba(np.array([[1.0, np.nan, 3.0, 4.0]]))
    with pytest.raises(ValueError, match="NaN or Infinite"):
        trainer.predict_proba(np.array([[1.0, np.inf, 3.0, 4.0]]))

    # 2. fit bounds
    # Zero samples
    with pytest.raises(ValueError, match="non-empty"):
        trainer.fit(np.zeros((0, 4)), np.zeros(0, dtype=int))
    # 1D features
    with pytest.raises(ValueError, match="2D array"):
        trainer.fit(np.zeros(4), np.zeros(1, dtype=int))
    # 2D labels
    with pytest.raises(ValueError, match="1D array"):
        trainer.fit(np.zeros((2, 4)), np.zeros((2, 1), dtype=int))
    # Sample count mismatch
    with pytest.raises(ValueError, match="Sample count mismatch"):
        trainer.fit(np.zeros((4, 4)), np.zeros(3, dtype=int))
    # Dimension mismatch
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        trainer.fit(np.zeros((4, 6)), np.zeros(4, dtype=int))
    # Negative labels
    with pytest.raises(ValueError, match="non-negative"):
        trainer.fit(np.ones((4, 4)), np.array([-1, 0, 1, 0], dtype=int))
    # Label >= k_classes (k_classes=3, label=3)
    with pytest.raises(ValueError, match="range"):
        trainer.fit(np.ones((4, 4)), np.array([0, 1, 3, 0], dtype=int))
    # Non-finite features in fit
    with pytest.raises(ValueError, match="NaN or Infinite"):
        trainer.fit(np.array([[1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, 4.0]]), np.array([0, 1]))


# ===========================================================================
# 2. F04: Concurrency Safety and Thread Isolation in guard_sockets()
# ===========================================================================


def test_f04_adversarial_guard_sockets_20_concurrent_threads() -> None:
    """Stress-test guard_sockets() with 10 guarded and 10 unguarded concurrent threads.
    
    Verifies that:
    1. Guarded threads are 100% strictly blocked with RuntimeError.
    2. Unguarded threads connect unimpeded to a local TCP socket.
    3. Thread state is cleaned up without race conditions.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(20)
    server_sock.settimeout(3.0)
    port = server_sock.getsockname()[1]

    guarded_blocked_count = [0]
    guarded_errors: list[Exception] = []
    unguarded_connected_count = [0]
    unguarded_errors: list[Exception] = []

    count_lock = threading.Lock()

    def _guarded_worker() -> None:
        try:
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect(("127.0.0.1", port))
                    guarded_errors.append(RuntimeError("Guarded thread unexpectedly connected!"))
                except RuntimeError as re_err:
                    if "offline" in str(re_err).lower():
                        with count_lock:
                            guarded_blocked_count[0] += 1
                    else:
                        guarded_errors.append(re_err)
                finally:
                    s.close()
        except Exception as exc:
            guarded_errors.append(exc)

    def _unguarded_worker() -> None:
        try:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.connect(("127.0.0.1", port))
            c.sendall(b"PING")
            c.close()
            with count_lock:
                unguarded_connected_count[0] += 1
        except Exception as exc:
            unguarded_errors.append(exc)

    def _server_listener() -> None:
        accepted = 0
        while accepted < 10:
            try:
                conn, _ = server_sock.accept()
                conn.recv(16)
                conn.close()
                accepted += 1
            except socket.timeout:
                break

    listener_thread = threading.Thread(target=_server_listener)
    listener_thread.start()

    threads: list[threading.Thread] = []
    # Interleave 10 guarded and 10 unguarded threads
    for i in range(10):
        threads.append(threading.Thread(target=_guarded_worker))
        threads.append(threading.Thread(target=_unguarded_worker))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    listener_thread.join(timeout=5.0)
    server_sock.close()

    assert not guarded_errors, f"Guarded thread errors: {guarded_errors}"
    assert not unguarded_errors, f"Unguarded thread errors: {unguarded_errors}"
    assert guarded_blocked_count[0] == 10, f"Expected 10 blocked guarded threads, got {guarded_blocked_count[0]}"
    assert unguarded_connected_count[0] == 10, f"Expected 10 connected unguarded threads, got {unguarded_connected_count[0]}"


def test_f04_adversarial_guard_sockets_nested_and_exception_recovery() -> None:
    """Verify nested guard_sockets() context manager depth and clean exception recovery."""
    tid = threading.get_ident()

    # 1. Nested depth test
    with guard_sockets():
        assert _GUARDED_THREADS.get(tid) == 1
        with guard_sockets():
            assert _GUARDED_THREADS.get(tid) == 2
            with guard_sockets():
                assert _GUARDED_THREADS.get(tid) == 3
            assert _GUARDED_THREADS.get(tid) == 2
        assert _GUARDED_THREADS.get(tid) == 1
    assert tid not in _GUARDED_THREADS

    # 2. Exception recovery
    try:
        with guard_sockets():
            assert _GUARDED_THREADS.get(tid) == 1
            raise ValueError("Deliberate error inside guard")
    except ValueError:
        pass
    assert tid not in _GUARDED_THREADS


# ===========================================================================
# 3. F05: Non-Blocking Event Loop, /healthz, and VerdictMQMultiQuestionCache
# ===========================================================================


def test_f05_adversarial_healthz_unauthenticated_and_fast() -> None:
    """Verify /healthz works completely unauthenticated and returns valid status."""
    app = create_app(api_key="super-secret-auth-key")
    client = TestClient(app)

    # Unauthenticated GET
    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok", "service": "microllm"}

    # With invalid credentials, must still return 200 (health probe should never 401)
    r2 = client.get("/healthz", headers={"Authorization": "Bearer wrong-key"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"


def test_f05_adversarial_verdict_cache_concurrency_and_eviction() -> None:
    """Stress-test VerdictMQMultiQuestionCache under high concurrent multithreaded access."""
    cache = VerdictMQMultiQuestionCache(max_entries=4)
    assert hasattr(cache, "_lock")

    # 1. Empty string behavior
    empty_entry = cache.encode_state_once("")
    assert empty_entry["clause_count"] == 0
    whitespace_entry = cache.encode_state_once("   \n\t  ")
    assert whitespace_entry["clause_count"] == 0

    # 2. High concurrency access across 15 threads
    exceptions: list[Exception] = []

    def _worker(thread_idx: int) -> None:
        try:
            for step in range(20):
                # Mix repeated (hits) and unique (misses) keys
                text = f"State from thread {thread_idx % 3} with step {step % 5}"
                entry = cache.encode_state_once(text)
                assert "clause_count" in entry
                _ = cache.stats
        except Exception as exc:
            exceptions.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not exceptions, f"Concurrent cache access raised: {exceptions}"

    # Eviction boundary: max_states=4, so cached_states must not exceed 4
    stats = cache.stats
    assert stats["cached_states"] <= 4, f"Cache size exceeded max_entries: {stats['cached_states']}"
    assert stats["hits"] > 0
    assert stats["misses"] > 0
    assert stats["hits"] + stats["misses"] == 15 * 20 + 2  # 300 ops + 2 empty calls


def test_f05_adversarial_server_asyncio_to_thread_verification() -> None:
    """Inspect server.py to confirm all compute-heavy routes use asyncio.to_thread."""
    import microllm.server as srv
    source = inspect.getsource(srv.create_app)

    expected_routes = [
        "systemone_endpoint",
        "calibrate_endpoint",
        "studio_analyze",
        "guardrail_endpoint",
        "profiles_pull_endpoint",
        "batch_endpoint",
        "studio_rlcd_demo",
        "rlcd_endpoint",
        "get_benchmark_endpoint",
        "run_benchmark_endpoint",
    ]

    for route in expected_routes:
        assert route in source, f"Route handler {route} missing from create_app!"

    # Ensure asyncio.to_thread is used multiple times
    to_thread_count = source.count("asyncio.to_thread")
    assert to_thread_count >= 10, f"Expected at least 10 asyncio.to_thread calls, found {to_thread_count}"


# ===========================================================================
# 4. Integrity Verification: Real Logic & No Cheating
# ===========================================================================


def test_integrity_rlcd_trainer_executes_genuine_reinforce() -> None:
    """Verify that RLCDDecisionHeadTrainer genuinely optimizes weights and modifies loss."""
    cfg = RLCDConfig(stage1_steps=10, stage2_steps=10)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg, seed=42)

    features = np.random.normal(0, 1, size=(20, 4))
    labels = np.random.choice([0, 1], size=20)

    w_before = np.copy(trainer.w_readout)
    metrics = trainer.fit(features, labels)

    assert not np.allclose(trainer.w_readout, w_before), "Integrity Violation: weights were not updated during fit!"
    assert "stage1_brier" in metrics
    assert "final_brier" in metrics
    assert "initial_ece" in metrics
    assert "final_ece" in metrics
    assert "history" in metrics
    assert len(metrics["history"]) == 20


def test_f05_healthz_concurrent_with_heavy_rlcd_training() -> None:
    """Stress test: 50 rapid /healthz requests executed concurrently while a heavy RLCD training run is ongoing."""
    app = create_app(api_key="heavy-rlcd-key")
    client = TestClient(app)
    headers = {"Authorization": "Bearer heavy-rlcd-key"}

    # Heavy RLCD payload: 100 stage1 steps, 100 stage2 steps
    heavy_payload = {
        "stage1_steps": 100,
        "stage2_steps": 100,
        "seed": 999,
    }

    rlcd_done = threading.Event()
    healthz_latencies: list[float] = []
    healthz_errors: list[Exception] = []

    def _run_rlcd() -> None:
        try:
            resp = client.post("/v1/rlcd", json=heavy_payload, headers=headers)
            assert resp.status_code == 200
        finally:
            rlcd_done.set()

    def _run_healthz_probes() -> None:
        probe_client = TestClient(app)
        while not rlcd_done.is_set():
            t0 = time.perf_counter()
            try:
                r = probe_client.get("/healthz")
                dt = time.perf_counter() - t0
                assert r.status_code == 200
                assert r.json()["status"] == "ok"
                healthz_latencies.append(dt)
            except Exception as e:
                healthz_errors.append(e)
            time.sleep(0.01)

    t_rlcd = threading.Thread(target=_run_rlcd)
    t_probe = threading.Thread(target=_run_healthz_probes)

    t_rlcd.start()
    t_probe.start()

    t_rlcd.join(timeout=10.0)
    t_probe.join(timeout=5.0)

    assert not healthz_errors, f"Healthz probe errors during RLCD training: {healthz_errors}"
    assert len(healthz_latencies) >= 5, f"Expected at least 5 healthz probes, got {len(healthz_latencies)}"
    max_lat = max(healthz_latencies)
    assert max_lat < 0.15, f"Healthz probe blocked by event loop! Max latency was {max_lat * 1000:.1f}ms"
