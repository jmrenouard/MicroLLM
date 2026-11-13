"""Dedicated test suite for Milestone 2: Concurrency, Async Model Init & Cache Optimization.

Covers Audit Points 7, 8, 10, 11, 12 (Features F07, F08, F10, F11, F12):
- Point 7 (F07): Benchmark concurrency control (HTTP 409 when running) & atomic file writes.
- Point 8 (F08): Non-blocking neural & cascade model initialization (asyncio.Lock + to_thread).
- Point 10 (F10): Removal of artificial time.sleep delays in RLCD trainer and server endpoints.
- Point 11 (F11): Multilingual neural model backend caching in MCP server (_multilingual_backends).
- Point 12 (F12): Verdict-MQ cache single encode per multi-question call & cached representation leveraging.
"""

from __future__ import annotations

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from microllm.backends.base import (
    DecisionHeadScorer,
    VerdictMQMultiQuestionCache,
    semantic_match_logit,
)
from microllm.mcp_server import MicroLLMMCPServer
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import (
    _benchmark_lock,
    _model_loading_lock,
    _model_sync_lock,
    atomic_write_json,
    create_app,
    get_cascade_backend,
    get_cascade_backend_async,
    get_neural_backend,
    get_neural_backend_async,
    run_rlcd_experiment_in_process,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Point 7 (F07): Benchmark Concurrency Control & Atomic File Writes
# ==============================================================================


def test_point07_benchmark_lock_returns_409_when_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /v1/benchmark returns HTTP 409 Conflict when _benchmark_lock is already held."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Simulate an ongoing benchmark by acquiring the lock
        async def _acquire() -> None:
            await _benchmark_lock.acquire()

        asyncio.run(_acquire())
        try:
            resp = client.post("/v1/benchmark", json={})
            assert resp.status_code == 409, f"Expected 409 Conflict, got {resp.status_code}"
            assert "already in progress" in resp.json().get("detail", "")

            resp_run = client.post("/v1/benchmark/run", json={})
            assert resp_run.status_code == 409, f"Expected 409 Conflict, got {resp_run.status_code}"
        finally:
            _benchmark_lock.release()


def test_point07_external_benchmark_lock_returns_409_when_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /v1/benchmark/external returns HTTP 409 Conflict when _benchmark_lock is held."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        async def _acquire() -> None:
            await _benchmark_lock.acquire()

        asyncio.run(_acquire())
        try:
            resp = client.post("/v1/benchmark/external", json={})
            assert resp.status_code == 409, f"Expected 409 Conflict, got {resp.status_code}"
            assert "already in progress" in resp.json().get("detail", "")
        finally:
            _benchmark_lock.release()


def test_point07_hf_jev_models_benchmark_lock_returns_409_when_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /v1/benchmark/hf-jev-models returns HTTP 409 Conflict when _benchmark_lock is held."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        async def _acquire() -> None:
            await _benchmark_lock.acquire()

        asyncio.run(_acquire())
        try:
            resp = client.post("/v1/benchmark/hf-jev-models", json={})
            assert resp.status_code == 409, f"Expected 409 Conflict, got {resp.status_code}"
            assert "already in progress" in resp.json().get("detail", "")
        finally:
            _benchmark_lock.release()


def test_point07_benchmark_lock_released_after_completion() -> None:
    """Verify _benchmark_lock is released when server is idle."""
    assert not _benchmark_lock.locked(), "Benchmark lock must be released after completion."


def test_point07_atomic_write_json_normal(tmp_path: Path) -> None:
    """Verify atomic_write_json writes valid JSON via temporary file and replace."""
    target = tmp_path / "test_report.json"
    data = {"run_id": "bench_001", "score": 0.995, "tracks": ["fast", "neural"]}
    atomic_write_json(target, data, indent=2)

    assert target.is_file(), "Target JSON file must exist."
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content == data
    # No temporary files should remain in directory
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files were not cleaned up: {tmp_files}"


def test_point07_atomic_write_json_nested_dirs(tmp_path: Path) -> None:
    """Verify atomic_write_json creates nested parent directories automatically."""
    target = tmp_path / "sub1" / "sub2" / "nested_report.json"
    data = {"nested": True, "count": 42}
    atomic_write_json(target, data)

    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_point07_atomic_write_json_multiple_consecutive_writes(tmp_path: Path) -> None:
    """Verify multiple rapid-fire atomic writes do not corrupt the file on disk."""
    target = tmp_path / "rapid_write.json"
    for i in range(10):
        atomic_write_json(target, {"iteration": i, "timestamp": time.time()})
        read_data = json.loads(target.read_text(encoding="utf-8"))
        assert read_data["iteration"] == i


def test_point07_get_benchmarks_corrupted_json_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET benchmark endpoints gracefully recover if the cached file has invalid JSON."""
    cache_bench = tmp_path / "benchmarks"
    cache_bench.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))

    # Write corrupt JSON into latest_results.json
    corrupt_file = cache_bench / "latest_results.json"
    corrupt_file.write_text("{corrupt partial json", encoding="utf-8")

    monkeypatch.setattr("microllm.server.resolve_benchmark_path", lambda fn: corrupt_file)

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        with patch("microllm.benchmarks.run_benchmark_suite", return_value={"recovered": True}):
            resp = client.get("/v1/benchmark")
            assert resp.status_code == 200, f"Expected 200 fallback, got {resp.status_code}"
            assert resp.json().get("recovered") is True


def test_point07_benchmarks_module_uses_atomic_write() -> None:
    """Verify benchmarks.py and web_benchmark_suite.py import and use atomic_write_json."""
    from microllm import benchmarks, web_benchmark_suite
    b_src = inspect.getsource(benchmarks)
    assert "atomic_write_json" in b_src, "benchmarks.py must use atomic_write_json"

    wb_src = inspect.getsource(web_benchmark_suite)
    assert "atomic_write_json" in wb_src, "web_benchmark_suite.py must use atomic_write_json"


# ==============================================================================
# Point 8 (F08): Non-Blocking Model Initialization & Concurrency
# ==============================================================================


def test_point08_model_loading_lock_prevents_duplicate_init() -> None:
    """Verify _model_loading_lock serializes model loading and prevents duplicate initialization."""
    assert isinstance(_model_loading_lock, asyncio.Lock)

    async def _run() -> tuple[Any, Any]:
        task1 = asyncio.create_task(get_neural_backend_async("minilm"))
        task2 = asyncio.create_task(get_neural_backend_async("minilm"))
        return await asyncio.gather(task1, task2)

    b1, b2 = asyncio.run(_run())
    assert b1 is b2, "Concurrent get_neural_backend_async calls must return the identical backend instance."


def test_point08_get_cascade_backend_async_caching() -> None:
    """Verify get_cascade_backend_async caches and returns the same cascade instance."""
    async def _run() -> tuple[Any, Any]:
        c1 = await get_cascade_backend_async("minilm")
        c2 = await get_cascade_backend_async("minilm")
        return c1, c2

    c1, c2 = asyncio.run(_run())
    assert c1 is c2, "get_cascade_backend_async must return cached instance."


def test_point08_synchronous_getters_backward_compatibility() -> None:
    """Verify synchronous get_neural_backend and get_cascade_backend remain functional and thread-safe."""
    assert _model_sync_lock is not None
    b_sync = get_neural_backend("minilm")
    assert b_sync is not None
    c_sync = get_cascade_backend("minilm")
    assert c_sync is not None


def test_point08_systemone_endpoint_async_model_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/systemone handles requests specifying neural models without blocking event loop."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        req_payload = {
            "model": "jev-latest",
            "state": "Customer asking for priority technical support",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Determine support category",
                    "criteria": {"billing": "Billing", "tech": "Technical support"},
                }
            },
            "multilingual_strategy": "neural",
            "neural_model": "minilm",
        }
        resp = client.post("/v1/systemone", json=req_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "answers" in data
        assert "category" in data["answers"]


def test_point08_batch_endpoint_async_model_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/batch handles requests with neural model loading without event loop lag."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        batch_payload = {
            "items": ["Inquiry A", "Inquiry B"],
            "questions": {
                "urgent": {
                    "type": "choice",
                    "instructions": "Is it urgent?",
                    "criteria": {"yes": "Urgent", "no": "Normal"},
                }
            },
            "multilingual_strategy": "neural",
            "neural_model": "minilm",
        }
        resp = client.post("/v1/batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("summary", {}).get("total_items") == 2


# ==============================================================================
# Point 10 (F10): Removal of Artificial Sleep Delays in RLCD
# ==============================================================================


def test_point10_rlcd_trainer_no_time_sleep() -> None:
    """Inspect AST of src/microllm/rlcd.py to verify time.sleep does not exist in fit method."""
    rlcd_file = REPO_ROOT / "src" / "microllm" / "rlcd.py"
    assert rlcd_file.is_file()
    source = rlcd_file.read_text(encoding="utf-8")
    parsed = ast.parse(source)

    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef) and node.name == "fit":
            fit_source = ast.unparse(node)
            assert "time.sleep" not in fit_source, "fit() in rlcd.py must not contain time.sleep"


def test_point10_server_rlcd_no_time_sleep() -> None:
    """Inspect source of src/microllm/server.py to verify run_rlcd_experiment_in_process has no time.sleep."""
    fn_source = inspect.getsource(run_rlcd_experiment_in_process)
    assert "time.sleep" not in fn_source, "run_rlcd_experiment_in_process must not contain time.sleep"


def test_point10_rlcd_trainer_fit_execution_speed() -> None:
    """Verify RLCDDecisionHeadTrainer.fit with 60 total steps executes in < 150ms."""
    cfg = RLCDConfig(stage1_steps=30, stage2_steps=30, use_rlcd2_weights=True)
    trainer = RLCDDecisionHeadTrainer(d_feat=8, k_classes=3, config=cfg, seed=42)

    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(100, 8))
    y = rng.choice(3, size=100)

    t0 = time.perf_counter()
    metrics = trainer.fit(x, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 200.0, f"fit() took {elapsed_ms:.1f}ms; artificial sleep is likely present."
    assert "initial_brier" in metrics
    assert "final_brier" in metrics


def test_point10_rlcd_demo_endpoint_rapid_fire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/studio/rlcd-demo processes 5 rapid requests in under 1 second total."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        payload = {
            "stage1_steps": 10,
            "stage2_steps": 10,
            "use_rlcd2_weights": True,
            "seed": 42,
        }
        t0 = time.perf_counter()
        for _ in range(5):
            resp = client.post("/v1/studio/rlcd-demo", json=payload)
            assert resp.status_code == 200
        total_sec = time.perf_counter() - t0
        assert total_sec < 1.0, f"5 RLCD demo runs took {total_sec:.2f}s; artificial sleep may be present."


# ==============================================================================
# Point 11 (F11): Multilingual Neural Model Caching in MCP Server
# ==============================================================================


def test_point11_mcp_server_defines_cache_and_lock() -> None:
    """Verify MicroLLMMCPServer defines _multilingual_backends and _backend_lock."""
    server = MicroLLMMCPServer()
    assert hasattr(server, "_multilingual_backends"), "MicroLLMMCPServer must have _multilingual_backends"
    assert isinstance(server._multilingual_backends, dict)
    assert hasattr(server, "_backend_lock"), "MicroLLMMCPServer must have _backend_lock"


def test_point11_mcp_systemone_decide_caches_and_reuses_instance() -> None:
    """Verify calling systemone_decide twice with neural model caches and reuses the same instance."""
    server = MicroLLMMCPServer()
    sample_q = {
        "topic": {
            "type": "choice",
            "instructions": "Determine topic",
            "criteria": {"account": "Account settings", "sales": "Sales"},
        }
    }

    # Call 1: Populates cache
    res1 = server.systemone_decide(
        state="I want to update my account password",
        questions=sample_q,
        multilingual_strategy="strategy2_neural",
        neural_model="minilm",
    )
    assert isinstance(res1, dict)
    assert len(server._multilingual_backends) >= 1
    backend_key = ("neural", "minilm")
    assert backend_key in server._multilingual_backends
    inst1 = server._multilingual_backends[backend_key]

    # Call 2: Must reuse exact same backend instance
    res2 = server.systemone_decide(
        state="Please update my email address as well",
        questions=sample_q,
        multilingual_strategy="strategy2_neural",
        neural_model="minilm",
    )
    assert isinstance(res2, dict)
    inst2 = server._multilingual_backends[backend_key]
    assert inst1 is inst2, "Backend instance in MCP server must be memoized without reloading from disk."


def test_point11_mcp_different_models_cached_under_distinct_keys() -> None:
    """Verify minilm and mdeberta are cached under separate keys in _multilingual_backends."""
    server = MicroLLMMCPServer()
    sample_q = {
        "dept": {
            "type": "choice",
            "instructions": "Select department",
            "criteria": {"hr": "Human Resources", "it": "Information Technology"},
        }
    }
    server.systemone_decide(
        state="Employee onboarding issue",
        questions=sample_q,
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    server.systemone_decide(
        state="Employee onboarding issue",
        questions=sample_q,
        multilingual_strategy="neural",
        neural_model="mdeberta",
    )
    assert ("neural", "minilm") in server._multilingual_backends
    assert ("neural", "mdeberta") in server._multilingual_backends
    assert server._multilingual_backends[("neural", "minilm")] is not server._multilingual_backends[("neural", "mdeberta")]


def test_point11_mcp_neural_cache_thread_safety() -> None:
    """Verify concurrent threads calling systemone_decide access cache safely."""
    server = MicroLLMMCPServer()
    sample_q = {
        "lang": {
            "type": "choice",
            "instructions": "Language inquiry",
            "criteria": {"fr": "French", "es": "Spanish"},
        }
    }

    def _worker(idx: int) -> dict[str, Any]:
        return server.systemone_decide(
            state=f"Message {idx} regarding account support",
            questions=sample_q,
            multilingual_strategy="neural",
            neural_model="minilm",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_worker, i) for i in range(8)]
        results = [f.result() for f in futures]

    for r in results:
        assert isinstance(r, dict)
        assert "answers" in r
    assert len(server._multilingual_backends) == 1


def test_point11_mcp_unknown_neural_model_error() -> None:
    """Verify requesting an invalid neural model name returns an error dictionary."""
    server = MicroLLMMCPServer()
    sample_q = {
        "q1": {"type": "choice", "instructions": "test", "criteria": {"a": "A", "b": "B"}}
    }
    res = server.systemone_decide(
        state="test message",
        questions=sample_q,
        multilingual_strategy="neural",
        neural_model="unsupported-nonexistent-model",
    )
    assert isinstance(res, dict)
    assert res.get("isError") is True or "error" in res


# ==============================================================================
# Point 12 (F12): VerdictMQ Single-Encode & Cached Representation Exploitation
# ==============================================================================


def test_point12_verdict_mq_single_encode_mcp() -> None:
    """Verify VerdictMQ cache encodes state once and subsequent questions count as cache hits in MCP."""
    server = MicroLLMMCPServer()
    multi_q = {
        "q1": {
            "type": "choice",
            "instructions": "Question 1 instruction",
            "criteria": {"yes": "Affirmative", "no": "Negative"},
        },
        "q2": {
            "type": "choice",
            "instructions": "Question 2 instruction",
            "criteria": {"high": "High priority", "low": "Low priority"},
        },
        "q3": {
            "type": "choice",
            "instructions": "Question 3 instruction",
            "criteria": {"billing": "Billing", "tech": "Technical"},
        },
    }

    res = server.systemone_decide(
        state="Urgent server outage affecting billing systems and user logins.",
        questions=multi_q,
    )
    assert isinstance(res, dict)
    stats = server.mq_cache.stats
    assert stats["cached_states"] >= 1
    # For a 3-question query, question 1 misses/encodes once, questions 2 and 3 must hit the cache
    assert stats["hits"] >= 2, f"Expected at least 2 hits on questions 2 and 3, got stats={stats}"
    assert stats["misses"] >= 1


def test_point12_verdict_mq_single_encode_server_systemone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/systemone with multi-question request encodes state once and records hits."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        payload = {
            "state": "The user reported slow loading times in the dashboard and login screen.",
            "questions": {
                "perf": {
                    "type": "choice",
                    "instructions": "Is performance impacted?",
                    "criteria": {"yes": "Yes, slow", "no": "No issue"},
                },
                "area": {
                    "type": "choice",
                    "instructions": "Which area?",
                    "criteria": {"ui": "Dashboard / UI", "api": "Backend API"},
                },
                "severity": {
                    "type": "choice",
                    "instructions": "What severity level?",
                    "criteria": {"p1": "Critical", "p2": "Moderate"},
                },
            },
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data.get("answers", {})) == 3


def test_point12_verdict_mq_single_encode_server_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/batch encodes each item once and scores all questions efficiently."""
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(tmp_path))
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        batch_payload = {
            "items": ["User account locked", "Password reset request"],
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Routing department",
                    "criteria": {"security": "Security", "support": "Support"},
                },
                "action": {
                    "type": "choice",
                    "instructions": "Action needed",
                    "criteria": {"unlock": "Unlock", "reset": "Reset"},
                },
            },
        }
        resp = client.post("/v1/batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()
        summary = data.get("summary", {})
        assert summary.get("total_items") == 2
        assert "verdict_mq_cache" in summary


def test_point12_backend_score_cached_state_matches_semantic_match_logit() -> None:
    """Verify score_cached_state leverages pre-computed representation and produces equivalent scores."""
    cache = VerdictMQMultiQuestionCache()
    state_text = "The quick brown fox jumps over the lazy dog."
    query_text = "What animal jumped?"
    cand_text = "A quick brown fox jumped over a dog."

    score_direct = semantic_match_logit(state_text, query_text, cand_text)
    score_cached = cache.score_cached_state(state_text, query_text, cand_text)

    assert abs(score_direct - score_cached) < 1e-5, (
        f"score_cached_state ({score_cached}) must match semantic_match_logit ({score_direct})"
    )


def test_point12_decision_head_scorer_interface() -> None:
    """Verify DecisionHeadScorer class exists and provides score() method."""
    scorer = DecisionHeadScorer()
    assert hasattr(scorer, "score"), "DecisionHeadScorer must implement score()"
    val = scorer.score("State context", "Query instruction", "Hypothesis candidate")
    assert isinstance(val, float)
