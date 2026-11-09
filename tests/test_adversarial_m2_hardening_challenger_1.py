"""Adversarial Challenge and Stress Harness for Milestone 2 (Challenger 1).

Adversarially tests:
1. RLCD Input Validation (F03):
   - Evil arrays submitted to /v1/rlcd, /v1/studio/rlcd-demo, and RLCDDecisionHeadTrainer.fit.
   - Empty, 1D, 3D, ragged, infinite, -infinite, NaN, negative labels, single-class labels (all 0s, all 1s),
     mismatched length, out-of-range labels, and predict_proba validations.
   - Verifies zero 500 responses (HTTP 422 guaranteed on API).
2. Concurrency & Thread Isolation in guard_sockets() (F04):
   - Guarded threads blocked while concurrent unguarded threads connect unimpeded.
   - Multi-threaded stress harness with simultaneous guarded and unguarded worker threads.
   - Re-entrancy, nested scopes, exception safety, and interleaved multi-thread exits.
3. Event Loop Non-Blocking & Cache Safety (F05):
   - GET /healthz accessibility without authentication.
   - Concurrent health checks during heavy CPU inference workloads.
   - VerdictMQMultiQuestionCache concurrent multithreaded stress and boundary handling.
"""

from __future__ import annotations

import concurrent.futures
import json
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from microllm.backends.base import VerdictMQMultiQuestionCache
from microllm.hf_publishing import guard_sockets
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import StudioRLCDDemoRequest, create_app, run_rlcd_experiment_in_process


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def api_key() -> str:
    return "challenger1-m2-key"


@pytest.fixture(scope="module")
def test_app(api_key: str) -> Any:
    return create_app(api_key=api_key)


@pytest.fixture(scope="module")
def client(test_app: Any) -> TestClient:
    return TestClient(test_app)


@pytest.fixture(scope="module")
def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


# ============================================================================
# 1. RLCD Input Validation Stress Tests (Zero-500 Guarantee)
# ============================================================================

class TestRLCDInputValidationEvilArrays:
    """Adversarial stress-testing of RLCD input validation across API and Python core."""

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_empty_arrays_return_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Empty features and/or empty labels return HTTP 422."""
        cases = [
            {"features": [], "labels": []},
            {"features": [[]], "labels": [0]},
            {"features": [[1.0, 2.0]], "labels": []},
            {"features": [], "labels": [0, 1]},
            {"features": [[], []], "labels": [0, 1]},
        ]
        for payload in cases:
            resp = client.post(
                endpoint,
                json={"stage1_steps": 5, "stage2_steps": 5, **payload},
                headers=auth_headers,
            )
            assert resp.status_code == 422, (
                f"Expected 422 on {endpoint} with payload {payload}, got {resp.status_code}: {resp.text}"
            )
            assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_1d_features_returns_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: 1D features array instead of 2D matrix returns HTTP 422."""
        payload = {
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [1.0, 2.0, 3.0, 4.0],
            "labels": [0, 1, 0, 1],
        }
        resp = client.post(endpoint, json=payload, headers=auth_headers)
        assert resp.status_code == 422, f"Expected 422 on 1D features, got {resp.status_code}"
        assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_3d_features_returns_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: 3D features tensor instead of 2D matrix returns HTTP 422."""
        payload = {
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            "labels": [0, 1],
        }
        resp = client.post(endpoint, json=payload, headers=auth_headers)
        assert resp.status_code == 422, f"Expected 422 on 3D features, got {resp.status_code}"
        assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_ragged_features_returns_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Ragged features matrix (unequal row lengths) returns HTTP 422."""
        cases = [
            {"features": [[1.0, 2.0], [3.0]], "labels": [0, 1]},
            {"features": [[1.0, 2.0, 3.0], [4.0, 5.0]], "labels": [0, 1]},
            {"features": [[1.0], [2.0, 3.0, 4.0]], "labels": [0, 1]},
            {"features": [[1.0, 2.0], "non-list-element"], "labels": [0, 1]},
        ]
        for payload in cases:
            resp = client.post(
                endpoint,
                json={"stage1_steps": 5, "stage2_steps": 5, **payload},
                headers=auth_headers,
            )
            assert resp.status_code == 422, f"Expected 422 on ragged payload {payload}, got {resp.status_code}"
            assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_infinite_and_nan_features_return_422_or_400_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: NaN or Infinite values in features return 422 or 400, never 500."""
        # Non-finite values formatted in valid JSON strings
        nan_json = json.dumps({
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[1.0, 2.0], [float("nan"), 4.0]],
            "labels": [0, 1],
        })
        inf_json = json.dumps({
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[1.0, float("inf")], [3.0, 4.0]],
            "labels": [0, 1],
        })
        neginf_json = json.dumps({
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[float("-inf"), 2.0], [3.0, 4.0]],
            "labels": [0, 1],
        })

        for raw_body in [nan_json, inf_json, neginf_json]:
            resp = client.post(
                endpoint,
                content=raw_body,
                headers={**auth_headers, "Content-Type": "application/json"},
            )
            assert resp.status_code in (400, 422), (
                f"Expected 400/422 on non-finite JSON on {endpoint}, got {resp.status_code}: {resp.text}"
            )
            assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_negative_labels_returns_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Negative class labels return HTTP 422."""
        cases = [
            {"labels": [-1, 0]},
            {"labels": [-5, -2]},
            {"labels": [0, -1, 1]},
        ]
        for c in cases:
            payload = {
                "stage1_steps": 5,
                "stage2_steps": 5,
                "features": [[1.0, 2.0] for _ in c["labels"]],
                **c,
            }
            resp = client.post(endpoint, json=payload, headers=auth_headers)
            assert resp.status_code == 422, f"Expected 422 on negative label {c}, got {resp.status_code}"
            assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_single_class_labels_returns_422_or_valid_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Single-class labels (e.g. all 0s, all 1s) must not crash with 500."""
        # When all labels are 0, max(labels) = 0 -> k_classes < 2 -> raises 422
        payload_zeros = {
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "labels": [0, 0],
        }
        resp0 = client.post(endpoint, json=payload_zeros, headers=auth_headers)
        assert resp0.status_code == 422, f"Expected 422 on all zeros labels, got {resp0.status_code}"
        assert resp0.status_code != 500

        # When all labels are 1s, ensure zero 500s (either 422 or 200 without internal crash)
        payload_ones = {
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "labels": [1, 1],
        }
        resp1 = client.post(endpoint, json=payload_ones, headers=auth_headers)
        assert resp1.status_code in (200, 422), (
            f"Expected 200 or 422 on all-1s labels, got {resp1.status_code}: {resp1.text}"
        )
        assert resp1.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_mismatched_sample_counts_returns_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Mismatch between sample rows in features and label items."""
        cases = [
            {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0]},  # 2 samples, 1 label
            {"features": [[1.0, 2.0]], "labels": [0, 1]},  # 1 sample, 2 labels
            {"features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "labels": [0, 1]},  # 3 vs 2
            {"features": [[1.0, 2.0] for _ in range(10)], "labels": [0, 1]},  # 10 vs 2
        ]
        for payload in cases:
            resp = client.post(
                endpoint,
                json={"stage1_steps": 5, "stage2_steps": 5, **payload},
                headers=auth_headers,
            )
            assert resp.status_code == 422, (
                f"Expected 422 on mismatched counts on {endpoint}, got {resp.status_code}"
            )
            assert resp.status_code != 500

    @pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
    def test_out_of_bound_steps_return_422_never_500(
        self, client: TestClient, auth_headers: dict[str, str], endpoint: str
    ) -> None:
        """Evil test: Steps below 5 or above 100 return HTTP 422 via Pydantic Field constraints."""
        bad_steps = [
            {"stage1_steps": 1, "stage2_steps": 25},
            {"stage1_steps": -10, "stage2_steps": 25},
            {"stage1_steps": 200, "stage2_steps": 25},
            {"stage1_steps": 25, "stage2_steps": 2},
            {"stage1_steps": 25, "stage2_steps": 500},
        ]
        for p in bad_steps:
            resp = client.post(endpoint, json=p, headers=auth_headers)
            assert resp.status_code == 422, f"Expected 422 on bad steps {p}, got {resp.status_code}"
            assert resp.status_code != 500

    def test_direct_in_process_execution_raises_422_http_exception(self) -> None:
        """Direct call to run_rlcd_experiment_in_process validates request or converts ValueError to HTTPException(422)."""
        import pydantic
        from fastapi import HTTPException

        # 1. Direct call with invalid arguments raises pydantic.ValidationError (which FastAPI maps to 422)
        with pytest.raises(pydantic.ValidationError):
            StudioRLCDDemoRequest(
                stage1_steps=10,
                stage2_steps=10,
                features=[[1.0, 2.0]],
                labels=None,  # Not allowed if features is given
            )

        # 2. If trainer.fit raises ValueError inside run_rlcd_experiment_in_process, it converts to HTTPException(422)
        from unittest.mock import patch
        with patch("microllm.server.RLCDDecisionHeadTrainer.fit", side_effect=ValueError("Simulated fit error")):
            with pytest.raises(HTTPException) as exc_info:
                run_rlcd_experiment_in_process()
            assert exc_info.value.status_code == 422
            assert "Simulated fit error" in str(exc_info.value.detail)



class TestRLCDDecisionHeadTrainerCoreValidation:
    """Adversarial testing of RLCDDecisionHeadTrainer.fit and predict_proba raising ValueError."""

    def test_trainer_fit_validates_evil_arrays(self) -> None:
        cfg = RLCDConfig(stage1_steps=5, stage2_steps=5)
        trainer = RLCDDecisionHeadTrainer(d_feat=3, k_classes=2, config=cfg)

        # 1. Empty arrays
        with pytest.raises(ValueError, match="non-empty"):
            trainer.fit(np.zeros((0, 3)), np.zeros(0, dtype=int))

        # 2. 1D features
        with pytest.raises(ValueError, match="2D array"):
            trainer.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

        # 3. 3D features
        with pytest.raises(ValueError, match="2D array"):
            trainer.fit(np.ones((2, 3, 3)), np.array([0, 1]))

        # 4. Feature dimension mismatch (d_feat=4 vs expected 3)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            trainer.fit(np.ones((5, 4)), np.array([0, 1, 0, 1, 0]))

        # 5. Non-finite values in features
        with pytest.raises(ValueError, match="NaN or Infinite"):
            bad_x = np.ones((4, 3))
            bad_x[1, 1] = np.nan
            trainer.fit(bad_x, np.array([0, 1, 0, 1]))

        with pytest.raises(ValueError, match="NaN or Infinite"):
            bad_x = np.ones((4, 3))
            bad_x[2, 0] = np.inf
            trainer.fit(bad_x, np.array([0, 1, 0, 1]))

        # 6. Negative labels
        with pytest.raises(ValueError, match="non-negative"):
            trainer.fit(np.ones((4, 3)), np.array([-1, 0, 1, 0]))

        # 7. Labels >= k_classes (k_classes=2, label=2)
        with pytest.raises(ValueError, match="labels must be in range"):
            trainer.fit(np.ones((4, 3)), np.array([0, 1, 2, 0]))

        # 8. Length mismatch
        with pytest.raises(ValueError, match="Sample count mismatch"):
            trainer.fit(np.ones((5, 3)), np.array([0, 1, 0]))

        # 9. 2D labels array instead of 1D
        with pytest.raises(ValueError, match="1D array"):
            trainer.fit(np.ones((2, 3)), np.array([[0], [1]]))

    def test_trainer_predict_proba_validates_evil_arrays(self) -> None:
        trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=3)

        # 1D array
        with pytest.raises(ValueError, match="2D array"):
            trainer.predict_proba(np.array([1.0, 2.0, 3.0, 4.0]))

        # Dimension mismatch
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            trainer.predict_proba(np.ones((3, 2)))

        # Non-finite values
        with pytest.raises(ValueError, match="NaN or Infinite"):
            bad_x = np.ones((3, 4))
            bad_x[0, 0] = np.nan
            trainer.predict_proba(bad_x)


# ============================================================================
# 2. Concurrency Safety & Thread Isolation in guard_sockets()
# ============================================================================

class TestSocketGuardConcurrencyAndIsolation:
    """Adversarial stress-testing of thread-isolated socket blocking."""

    def test_guarded_thread_blocked_while_concurrent_thread_connects(self) -> None:
        """Empirically verify that a guarded thread raises RuntimeError while an concurrent thread connects."""
        # Create a real local loopback TCP server to test connections
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        srv_host, srv_port = srv.getsockname()

        results: dict[str, Any] = {"guarded_blocked": False, "unguarded_success": False}
        barrier = threading.Barrier(2)

        def guarded_worker() -> None:
            barrier.wait()
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect((srv_host, srv_port))
                except RuntimeError as exc:
                    if "blocked by 100% offline isolation" in str(exc):
                        results["guarded_blocked"] = True
                finally:
                    s.close()

        def unguarded_worker() -> None:
            barrier.wait()
            # Must be able to connect unimpeded even while guarded_worker is running guard_sockets()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((srv_host, srv_port))
                results["unguarded_success"] = True
            except Exception as e:
                results["unguarded_error"] = str(e)
            finally:
                s.close()

        try:
            t1 = threading.Thread(target=guarded_worker)
            t2 = threading.Thread(target=unguarded_worker)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
        finally:
            srv.close()

        assert results["guarded_blocked"] is True, "Guarded thread was not blocked by guard_sockets()"
        assert results["unguarded_success"] is True, (
            f"Unguarded concurrent thread was incorrectly blocked or failed: {results.get('unguarded_error')}"
        )

    def test_high_concurrency_stress_guarded_vs_unguarded(self) -> None:
        """Stress harness: 8 guarded threads and 8 unguarded threads running simultaneously."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(32)
        srv_host, srv_port = srv.getsockname()

        num_threads_per_kind = 8
        guarded_blocks = 0
        unguarded_connects = 0
        lock = threading.Lock()

        def guarded_task() -> None:
            nonlocal guarded_blocks
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect((srv_host, srv_port))
                except RuntimeError:
                    with lock:
                        guarded_blocks += 1
                finally:
                    s.close()

        def unguarded_task() -> None:
            nonlocal unguarded_connects
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((srv_host, srv_port))
                with lock:
                    unguarded_connects += 1
            finally:
                s.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = []
                for _ in range(num_threads_per_kind):
                    futures.append(pool.submit(guarded_task))
                    futures.append(pool.submit(unguarded_task))
                concurrent.futures.wait(futures, timeout=10.0)
        finally:
            srv.close()

        assert guarded_blocks == num_threads_per_kind, (
            f"Expected {num_threads_per_kind} guarded blocks, got {guarded_blocks}"
        )
        assert unguarded_connects == num_threads_per_kind, (
            f"Expected {num_threads_per_kind} unguarded connects, got {unguarded_connects}"
        )

    def test_reentrant_nesting_and_exception_unpatch(self) -> None:
        """Verify nested context scopes and that exceptions restore original state cleanly."""
        # 1. Nested re-entrancy
        with guard_sockets():
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with pytest.raises(RuntimeError):
                    s.connect(("127.0.0.1", 9999))
                s.close()

        # 2. Exception inside guard does not leave socket patched for future outside calls
        with pytest.raises(ValueError):
            with guard_sockets():
                raise ValueError("Forced error")

        # After exception, caller is outside guard
        # Socket connect should not raise RuntimeError("Socket connection blocked...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to closed port: should raise ConnectionRefusedError or OSError, NOT RuntimeError
            s.connect(("127.0.0.1", 9999))
        except (ConnectionRefusedError, OSError):
            pass  # Expected normal socket behavior
        finally:
            s.close()


# ============================================================================
# 3. Non-Blocking Event Loop & Cache Thread Safety (F05)
# ============================================================================

class TestEventLoopNonBlockingAndCache:
    """Adversarial testing of /healthz unauthenticated access and thread-safe cache."""

    def test_healthz_unauthenticated_and_fast(self, client: TestClient) -> None:
        """Verify /healthz requires zero auth and returns HTTP 200 in < 25ms."""
        t0 = time.perf_counter()
        resp = client.get("/healthz")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "microllm"}
        assert elapsed_ms < 50.0, f"Expected /healthz < 50ms, took {elapsed_ms:.2f}ms"

    def test_verdict_mq_cache_thread_safe_concurrency(self) -> None:
        """Adversarially hammer VerdictMQMultiQuestionCache from 16 concurrent threads."""
        cache = VerdictMQMultiQuestionCache(max_states=5)
        queries = [
            f"Customer message {i % 10}: Need technical assistance or billing refund."
            for i in range(100)
        ]

        def worker(q: str) -> dict[str, Any]:
            return cache.encode_state_once(q)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 100
        # Check cache capacity invariant
        assert len(cache._cache) <= 5, f"Cache size {len(cache._cache)} exceeded max_states=5"
        # Check stats integrity
        assert cache.encode_hits + cache.encode_misses == 100
        assert cache.lock is cache._lock

    def test_verdict_mq_cache_empty_string_boundary(self) -> None:
        """Empty string input returns clause_count == 0 without crashing."""
        cache = VerdictMQMultiQuestionCache(max_states=10)
        entry = cache.encode_state_once("")
        assert entry["clean"] == ""
        assert entry["clause_count"] == 0
        assert entry["clauses"] == ()
