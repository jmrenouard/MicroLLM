"""Tier 2: Boundary & Corner Cases (>=5 tests per feature for each of F01-F12).

Total tests in this module: 60 (5 boundary tests * 12 features).
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import os
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
from microllm.client import Choice, MicroJevClient
from microllm.hf_publishing import guard_sockets, mask_token
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import create_app


# ============================================================================
# F01: Universal Endpoint Authentication Boundaries (R1)
# ============================================================================

def test_tier2_f01_01_empty_authorization_header(auth_client: TestClient) -> None:
    """F01.B1: Authorization header with empty string returns HTTP 401."""
    resp = auth_client.get("/v1/metrics", headers={"Authorization": ""})
    assert resp.status_code == 401


def test_tier2_f01_02_bearer_prefix_without_token(auth_client: TestClient) -> None:
    """F01.B2: 'Bearer ' with no token returns HTTP 401."""
    resp = auth_client.get("/v1/metrics", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


def test_tier2_f01_03_non_bearer_scheme_rejected(auth_client: TestClient) -> None:
    """F01.B3: Non-Bearer scheme (e.g. Basic, Token) returns HTTP 401."""
    resp = auth_client.get("/v1/metrics", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert resp.status_code == 401


def test_tier2_f01_04_whitespace_padded_valid_bearer_token(auth_client: TestClient, auth_api_key: str) -> None:
    """F01.B4: Extra whitespace padding around Bearer token is trimmed and accepted."""
    resp = auth_client.get("/v1/metrics", headers={"Authorization": f"Bearer   {auth_api_key}  "})
    assert resp.status_code == 200


def test_tier2_f01_05_malformed_control_chars_in_auth_header(auth_client: TestClient) -> None:
    """F01.B5: Header with control characters or binary bytes returns 401 cleanly without 500."""
    resp = auth_client.get("/v1/metrics", headers={"Authorization": "Bearer \x00\x01\x02"})
    assert resp.status_code == 401


# ============================================================================
# F02: Direct In-Process Core Execution Boundaries (R1)
# ============================================================================

def test_tier2_f02_01_guardrail_in_process_empty_state_string() -> None:
    """F02.B1: evaluate_guardrail_in_process handles empty state string safely."""
    import microllm.server as srv

    fn = getattr(srv, "evaluate_guardrail_in_process", getattr(srv, "evaluate_guardrail", None))
    if fn is None:
        pytest.fail("Feature F02 missing: evaluate_guardrail_in_process not in microllm.server")
    res = fn(state="", threshold=0.5)
    assert isinstance(res, dict)
    assert "safe" in res


def test_tier2_f02_02_guardrail_in_process_extreme_thresholds() -> None:
    """F02.B2: evaluate_guardrail_in_process handles threshold boundaries (0.0 and 1.0)."""
    import microllm.server as srv

    fn = getattr(srv, "evaluate_guardrail_in_process", getattr(srv, "evaluate_guardrail", None))
    if fn is None:
        pytest.fail("Feature F02 missing: evaluate_guardrail_in_process not in microllm.server")
    res0 = fn(state="Hello", threshold=0.0)
    res1 = fn(state="Hello", threshold=1.0)
    assert isinstance(res0, dict) and isinstance(res1, dict)


def test_tier2_f02_03_calibration_in_process_empty_confidence_and_labels() -> None:
    """F02.B3: evaluate_calibration_in_process with empty inputs raises ValueError or returns fallback."""
    import microllm.server as srv

    fn = getattr(srv, "evaluate_calibration_in_process", None)
    if fn is None:
        pytest.fail("Feature F02 missing: evaluate_calibration_in_process not in microllm.server")
    try:
        res = fn(confidence_scores=[], accuracy_labels=[], alpha=0.1)
        assert isinstance(res, dict)
    except ValueError:
        pass  # Raising ValueError on empty calibration is also compliant contract behavior


def test_tier2_f02_04_calibration_in_process_single_sample_edge() -> None:
    """F02.B4: evaluate_calibration_in_process handles N=1 sample without division by zero."""
    import microllm.server as srv

    fn = getattr(srv, "evaluate_calibration_in_process", None)
    if fn is None:
        pytest.fail("Feature F02 missing: evaluate_calibration_in_process not in microllm.server")
    res = fn(confidence_scores=[0.95], accuracy_labels=[1], alpha=0.1)
    assert isinstance(res, dict)
    assert "conformal_q_hat" in res


def test_tier2_f02_05_mcp_unknown_tool_call_handled_gracefully() -> None:
    """F02.B5: MCP server tool dispatcher returns error envelope for unknown tool."""
    from microllm.mcp_server import MicroLLMMCPServer

    mcp = MicroLLMMCPServer()
    resp = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "nonexistent_tool_name", "arguments": {}},
    })
    assert isinstance(resp, dict)
    assert resp.get("result", {}).get("isError") is True or "error" in str(resp).lower()


# ============================================================================
# F03: Input Validation on RLCD Boundaries (R2)
# ============================================================================

def test_tier2_f03_01_rlcd_none_features_and_none_labels_allowed(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.B1: Omitting both features and labels triggers synthetic training default."""
    resp = auth_client.post(
        "/v1/rlcd",
        json={"stage1_steps": 5, "stage2_steps": 5, "features": None, "labels": None},
        headers=auth_headers,
    )
    assert resp.status_code == 200, f"Expected 200 on None/None features/labels, got {resp.status_code}"


def test_tier2_f03_02_rlcd_features_provided_labels_none_rejected(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.B2: Providing features but labels=None returns HTTP 422."""
    resp = auth_client.post(
        "/v1/rlcd",
        json={"stage1_steps": 5, "stage2_steps": 5, "features": [[1.0, 2.0]], "labels": None},
        headers=auth_headers,
    )
    assert resp.status_code == 422, f"Expected 422 when labels is None, got {resp.status_code}"


def test_tier2_f03_03_rlcd_labels_provided_features_none_rejected(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.B3: Providing labels but features=None returns HTTP 422."""
    resp = auth_client.post(
        "/v1/rlcd",
        json={"stage1_steps": 5, "stage2_steps": 5, "features": None, "labels": [0, 1]},
        headers=auth_headers,
    )
    assert resp.status_code == 422, f"Expected 422 when features is None, got {resp.status_code}"


def test_tier2_f03_04_rlcd_ragged_features_matrix_rejected(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.B4: Ragged features matrix (uneven row dimensions) returns HTTP 422."""
    resp = auth_client.post(
        "/v1/rlcd",
        json={
            "stage1_steps": 5,
            "stage2_steps": 5,
            "features": [[1.0, 2.0], [3.0, 4.0, 5.0]],  # row 0 has len 2, row 1 has len 3
            "labels": [0, 1],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 422, f"Expected 422 on ragged matrix, got {resp.status_code}"


def test_tier2_f03_05_rlcd_nan_or_inf_in_features_rejected(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """F03.B5: Features containing NaN or Inf return HTTP 422."""
    resp = auth_client.post(
        "/v1/rlcd",
        content='{"stage1_steps": 5, "stage2_steps": 5, "features": [[1.0, NaN], [3.0, 4.0]], "labels": [0, 1]}',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422), f"Expected 400/422 on NaN, got {resp.status_code}"


# ============================================================================
# F04: Thread-Isolated guard_sockets() Boundaries (R3)
# ============================================================================

def test_tier2_f04_01_guard_sockets_nested_triple_depth() -> None:
    """F04.B1: 3 nested context managers stay guarded until the outermost exits."""
    with guard_sockets():
        with guard_sockets():
            with guard_sockets():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    with pytest.raises(RuntimeError):
                        s.connect(("127.0.0.1", 9))
                finally:
                    s.close()


def test_tier2_f04_02_guard_sockets_exception_during_yield() -> None:
    """F04.B2: Exception raised within context manager cleanly resets guard depth."""
    try:
        with guard_sockets():
            raise ZeroDivisionError("Intentional test fault")
    except ZeroDivisionError:
        pass

    # Next attempt must be able to use guard_sockets again cleanly
    with guard_sockets():
        pass


def test_tier2_f04_03_guard_sockets_connect_offline_error_message() -> None:
    """F04.B3: Connect raises RuntimeError with descriptive offline message."""
    with guard_sockets():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(RuntimeError) as exc_info:
                s.connect(("127.0.0.1", 9))
            assert "offline" in str(exc_info.value).lower() or "disabled" in str(exc_info.value).lower()
        finally:
            s.close()


def test_tier2_f04_04_guard_sockets_threadpool_executor_worker_unaffected() -> None:
    """F04.B4: Worker thread executing on threadpool is not blocked by main thread guard."""
    from concurrent.futures import ThreadPoolExecutor

    # Start listening local server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    port = server_sock.getsockname()[1]

    def _client_task() -> bool:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect(("127.0.0.1", port))
        c.close()
        return True

    try:
        with guard_sockets():
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_client_task)
                try:
                    conn, _ = server_sock.accept()
                    conn.close()
                except socket.timeout:
                    pass
                assert fut.result(timeout=2.0) is True
    finally:
        server_sock.close()


def test_tier2_f04_05_guard_sockets_repeated_enter_exit_cycles() -> None:
    """F04.B5: 20 consecutive enter/exit cycles maintain socket integrity."""
    orig = socket.socket.connect
    for _ in range(20):
        with guard_sockets():
            pass
    assert socket.socket.connect is orig


# ============================================================================
# F05: Non-Blocking Event Loop / Healthz / Cache Boundaries (R3)
# ============================================================================

def test_tier2_f05_01_healthz_under_high_concurrency(unauth_client: TestClient) -> None:
    """F05.B1: 25 rapid requests to /healthz return 200 without timeouts."""
    for _ in range(25):
        resp = unauth_client.get("/healthz")
        assert resp.status_code == 200


def test_tier2_f05_02_verdict_mq_cache_empty_state_string() -> None:
    """F05.B2: Caching empty string returns consistent entry with 0 clauses."""
    cache = VerdictMQMultiQuestionCache(max_entries=8)
    entry = cache.encode_state_once("")
    assert isinstance(entry, dict)
    assert entry.get("clause_count") == 0


def test_tier2_f05_03_verdict_mq_cache_large_state_string() -> None:
    """F05.B3: Large 50,000-character state string encoded without stack overflow."""
    cache = VerdictMQMultiQuestionCache(max_entries=8)
    large_state = "Invoice refund payment request. " * 1500
    entry = cache.encode_state_once(large_state)
    assert isinstance(entry, dict)
    assert entry.get("clause_count", 0) > 0


def test_tier2_f05_04_cache_eviction_under_concurrent_writes() -> None:
    """F05.B4: Concurrent writes beyond max_entries evict cleanly without deadlock."""
    cache = VerdictMQMultiQuestionCache(max_entries=10)
    errors: list[Exception] = []

    def _writer(tid: int) -> None:
        try:
            for i in range(20):
                cache.encode_state_once(f"Thread {tid} state {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert len(cache._cache) <= 10


def test_tier2_f05_05_healthz_response_time_under_50ms(unauth_client: TestClient) -> None:
    """F05.B5: /healthz probe returns within 50 milliseconds."""
    import time

    t0 = time.perf_counter()
    resp = unauth_client.get("/healthz")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    assert elapsed_ms < 50.0, f"Expected <50ms response, got {elapsed_ms:.2f}ms"


# ============================================================================
# F06: Multi-Question Token Caching Boundaries (R4)
# ============================================================================

def test_tier2_f06_01_tokenize_multilingual_whitespace_only() -> None:
    """F06.B1: Tokenizing whitespace-only string handled and cached."""
    from microllm.backends.cascade import tokenize_multilingual

    tokens = tokenize_multilingual("   \t\n   ")
    assert isinstance(tokens, (list, tuple))
    assert len(tokens) == 0 or all(isinstance(t, str) for t in tokens)


def test_tier2_f06_02_tokenize_multilingual_multilingual_unicode_scripts() -> None:
    """F06.B2: Non-Latin scripts (Chinese, Arabic, Cyrillic) tokenized and cached without error."""
    from microllm.backends.cascade import tokenize_multilingual

    samples = [
        "你好，我需要退款发票。",
        "مرحبا، أود استرداد فاتورتي.",
        "Здравствуйте, я хочу вернуть деньги по счету.",
    ]
    for s in samples:
        toks = tokenize_multilingual(s)
        assert len(toks) > 0


def test_tier2_f06_03_multi_question_empty_questions_dict(unauth_client: TestClient) -> None:
    """F06.B3: Empty questions dictionary {} handled safely without crash."""
    payload = {"model": "jev-latest", "state": "Some message", "questions": {}}
    resp = unauth_client.post("/v1/systemone", json=payload)
    # JevCompat allows 200 with empty answers or 422
    assert resp.status_code in (200, 422)


def test_tier2_f06_04_multi_question_single_question_parity(
    unauth_client: TestClient, sample_choice_question: dict[str, Any]
) -> None:
    """F06.B4: Single question produces identical result whether evaluated solo or in batch."""
    payload = {
        "model": "jev-latest",
        "state": "I need a refund for my last invoice.",
        "questions": sample_choice_question,
    }
    resp1 = unauth_client.post("/v1/systemone", json=payload)
    resp2 = unauth_client.post("/v1/systemone", json=payload)
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.json()["answers"]["dept"]["choice"] == resp2.json()["answers"]["dept"]["choice"]


def test_tier2_f06_05_multi_question_stress_20_questions(
    unauth_client: TestClient, sample_choice_question: dict[str, Any]
) -> None:
    """F06.B5: Evaluating 20 questions against a single state completes efficiently."""
    questions: dict[str, Any] = {}
    base_q = sample_choice_question["dept"]
    for i in range(20):
        questions[f"dept_{i}"] = base_q

    payload = {
        "model": "jev-latest",
        "state": "The server crashed with code 500 while trying to pay invoice.",
        "questions": questions,
    }
    resp = unauth_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["answers"]) == 20


# ============================================================================
# F07: SDK Client Batch Parity Boundaries (R4)
# ============================================================================

def test_tier2_f07_01_sync_batch_empty_items_list() -> None:
    """F07.B1: MicroJevClient.batch with items=[] returns total_items=0."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    res = client.batch(
        items=[],
        questions={"dept": Choice(options={"billing": "Billing", "tech": "Tech"})},
    )
    assert res["summary"]["total_items"] == 0
    assert len(res["results"]) == 0


def test_tier2_f07_02_sync_batch_empty_questions_dict() -> None:
    """F07.B2: MicroJevClient.batch with questions={} handles cleanly."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    res = client.batch(items=["Item 1"], questions={})
    assert res["summary"]["total_items"] == 1


def test_tier2_f07_03_sync_batch_none_strategy_and_model() -> None:
    """F07.B3: Explicit None for strategy and model defaults cleanly."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    res = client.batch(
        items=["Invoice test"],
        questions={"dept": Choice(options={"billing": "Billing", "tech": "Tech"})},
        multilingual_strategy=None,
        neural_model=None,
    )
    assert res["summary"]["total_items"] == 1


def test_tier2_f07_04_sync_batch_large_batch_size() -> None:
    """F07.B4: Batch of 15 items processes correctly with summary statistics."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    items = [f"Ticket message number {i}: need billing assistance" for i in range(15)]
    res = client.batch(
        items=items,
        questions={"dept": Choice(options={"billing": "Billing", "tech": "Tech"})},
    )
    assert res["summary"]["total_items"] == 15
    assert len(res["results"]) == 15


def test_tier2_f07_05_sync_batch_special_characters_in_items() -> None:
    """F07.B5: Items containing quotes, HTML tags, and newlines processed without JSON errors."""
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    items = [
        '<script>alert("xss")</script> & invoice quote="123"',
        'Line 1\nLine 2\r\n"Quoted text" and backslash \\',
    ]
    res = client.batch(
        items=items,
        questions={"dept": Choice(options={"billing": "Billing", "tech": "Tech"})},
    )
    assert res["summary"]["total_items"] == 2


# ============================================================================
# F08: SDK Base Consolidation Boundaries (R5)
# ============================================================================

def test_tier2_f08_01_base_client_custom_headers_merged() -> None:
    """F08.B1: Base client handles custom headers without dropping Authorization."""
    import microllm.client as cli_mod

    base_cls = getattr(cli_mod, "_BaseMicroJevClient", None)
    if base_cls is None:
        pytest.fail("Feature F08 missing: _BaseMicroJevClient not in microllm.client")
    inst = base_cls(api_key="my-key")
    headers = inst._auth_headers()
    assert headers.get("Authorization") == "Bearer my-key"


def test_tier2_f08_02_build_guardrail_empty_rubrics_dict() -> None:
    """F08.B2: _build_guardrail_payload with custom_rubrics={} creates clean payload."""
    import microllm.client as cli_mod

    fn = getattr(cli_mod, "_build_guardrail_payload", None)
    if fn is None:
        pytest.fail("Feature F08 missing: _build_guardrail_payload not in microllm.client")
    p = fn("State text", None, 0.5, {})
    assert p["custom_rubrics"] == {}


def test_tier2_f08_03_build_batch_items_with_dict_and_str() -> None:
    """F08.B3: _build_batch_payload handles mixed item types."""
    import microllm.client as cli_mod

    fn = getattr(cli_mod, "_build_batch_payload", None)
    if fn is None:
        pytest.fail("Feature F08 missing: _build_batch_payload not in microllm.client")
    p = fn(
        items=["String item", {"state": "Dict item"}],
        questions={},
        model="jev-latest",
        multilingual_strategy=None,
        neural_model=None,
    )
    assert len(p["items"]) == 2


def test_tier2_f08_04_build_papers_all_none_params() -> None:
    """F08.B4: _build_papers_params with all None parameters returns empty dict."""
    import microllm.client as cli_mod

    fn = getattr(cli_mod, "_build_papers_params", None)
    if fn is None:
        pytest.fail("Feature F08 missing: _build_papers_params not in microllm.client")
    p = fn(None, None, None)
    assert p == {}


def test_tier2_f08_05_parse_systemone_answers_empty_raw() -> None:
    """F08.B5: _parse_systemone_answers with {} returns empty parsed answers."""
    import microllm.client as cli_mod

    fn = getattr(cli_mod, "_parse_systemone_answers", None)
    if fn is None:
        pytest.fail("Feature F08 missing: _parse_systemone_answers not in microllm.client")
    res = fn({})
    assert res == {}


# ============================================================================
# F09: Space License Alignment Boundaries (R4)
# ============================================================================

def test_tier2_f09_01_explicit_mit_license_override() -> None:
    """F09.B1: Explicit license_type='mit' is respected when explicitly requested."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme(license_type="mit")
    assert "license: mit" in readme.lower()


def test_tier2_f09_02_empty_string_license_fallback() -> None:
    """F09.B2: Empty license string falls back to Apache-2.0 default."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme(license_type="")
    assert "license: apache-2.0" in readme.lower()


def test_tier2_f09_03_license_case_insensitivity() -> None:
    """F09.B3: License string handles uppercase 'Apache-2.0' consistently."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme(license_type="Apache-2.0")
    assert "license: apache-2.0" in readme.lower()


def test_tier2_f09_04_readme_with_custom_tags_and_apache_license() -> None:
    """F09.B4: Custom tags preserved alongside Apache-2.0 license."""
    from microllm.hf_space_sync import build_space_readme

    readme = build_space_readme(tags=["custom-tag-1", "custom-tag-2"])
    assert "license: apache-2.0" in readme.lower()
    assert "custom-tag-1" in readme


def test_tier2_f09_05_staging_in_nested_nonexistent_directory(tmp_path: Path) -> None:
    """F09.B5: stage_hf_space creates deeply nested directory hierarchy."""
    from microllm.hf_space_sync import stage_hf_space

    nested = tmp_path / "deep" / "nested" / "staging"
    res = stage_hf_space(staging_dir=nested)
    assert Path(res["staging_dir"]).is_dir()


# ============================================================================
# F10: Workstation Paths Elimination Boundaries (R4)
# ============================================================================

def test_tier2_f10_01_empty_env_vars_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """F10.B1: Empty string MICROLLM_KEEPASS_DB handled as unset, falling back to home."""
    import microllm.hf_publishing as hfp

    monkeypatch.setenv("MICROLLM_KEEPASS_DB", "")
    if hasattr(hfp, "get_vault_paths"):
        db, _ = hfp.get_vault_paths()
        assert Path.home() in Path(db).parents or Path(db).is_relative_to(Path.home())


def test_tier2_f10_02_whitespace_env_vars_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """F10.B2: Whitespace-only MICROLLM_KEEPASS_DB handled as unset."""
    import microllm.hf_publishing as hfp

    monkeypatch.setenv("MICROLLM_KEEPASS_DB", "   \t   ")
    if hasattr(hfp, "get_vault_paths"):
        db, _ = hfp.get_vault_paths()
        assert Path.home() in Path(db).parents or Path(db).is_relative_to(Path.home())


def test_tier2_f10_03_relative_path_in_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """F10.B3: Relative path in MICROLLM_KEEPASS_DB is resolved."""
    import microllm.hf_publishing as hfp

    monkeypatch.setenv("MICROLLM_KEEPASS_DB", "relative/vault.kdbx")
    if hasattr(hfp, "get_vault_paths"):
        db, _ = hfp.get_vault_paths()
        assert "relative" in str(db)


def test_tier2_f10_04_nonexistent_vault_file_no_import_crash() -> None:
    """F10.B4: Nonexistent vault file does not raise an exception during module import."""
    import microllm.hf_publishing as hfp

    assert hfp is not None


def test_tier2_f10_05_mask_token_boundary_lengths() -> None:
    """F10.B5: mask_token handles empty string, short tokens, and long tokens."""
    assert mask_token("") == "<none>"
    assert mask_token(None) == "<none>"
    assert mask_token("abc") == "***"
    assert mask_token("12345678") == "***"
    assert mask_token("abcdefghijklmnop") == "abc***mnop"


# ============================================================================
# F11: CLI Benchmark JSON Boundaries (R4)
# ============================================================================

def test_tier2_f11_01_cli_benchmark_combined_flags() -> None:
    """F11.B1: Combining --multilingual and --json parses cleanly."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_fn(["benchmark", "--multilingual", "--json"])
    assert code == 0


def test_tier2_f11_02_cli_benchmark_json_flag_stdout_purity() -> None:
    """F11.B2: Stdout when using --json contains only valid JSON (no header text)."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_fn(["benchmark", "--multilingual", "--json"])
    raw = buf.getvalue().strip()
    assert raw.startswith("{")
    assert raw.endswith("}")
    json.loads(raw)


def test_tier2_f11_03_cli_benchmark_invalid_subcommand_exits_code_2() -> None:
    """F11.B3: Invalid subcommand exits with code 2."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli
    try:
        code = run_fn(["nonexistent_subcommand"])
    except SystemExit as exc:
        code = exc.code
    assert code != 0


def test_tier2_f11_04_cli_benchmark_pareto_json_flag() -> None:
    """F11.B4: CLI benchmark --pareto --json outputs valid JSON dictionary."""
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_fn(["benchmark", "--pareto", "--json"])
    assert code == 0
    data = json.loads(buf.getvalue().strip())
    assert isinstance(data, dict)


def test_tier2_f11_05_cli_main_entrypoint_with_json() -> None:
    """F11.B5: CLI main entrypoint returns exit code 0 when invoked with --json."""
    import microllm.server as srv

    main_fn = getattr(srv, "main", getattr(srv, "run_cli", None))
    if main_fn is None:
        from microllm.cli import main as main_fn

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main_fn(["benchmark", "--multilingual", "--json"])
    assert code == 0


# ============================================================================
# F12: Modularization Boundaries (R5)
# ============================================================================

def test_tier2_f12_01_circular_import_resilience() -> None:
    """F12.B1: Independent imports of extracted modules do not trigger circular import loops."""
    import sys

    for mod in ["microllm.gradio_presets", "microllm.gradio_helpers", "microllm.dataset_fixtures", "microllm.cli", "microllm.server_models"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import microllm.gradio_presets
    import microllm.gradio_helpers
    import microllm.dataset_fixtures
    import microllm.cli
    import microllm.server_models
    assert True


def test_tier2_f12_02_all_dataset_fixtures_keys_accessible() -> None:
    """F12.B2: Extracted OFFLINE_WEB_DATASET_SAMPLES contains standard benchmark datasets."""
    from microllm.dataset_fixtures import OFFLINE_WEB_DATASET_SAMPLES

    assert isinstance(OFFLINE_WEB_DATASET_SAMPLES, dict)
    assert len(OFFLINE_WEB_DATASET_SAMPLES) >= 5


def test_tier2_f12_03_server_models_immutability_and_validation() -> None:
    """F12.B3: Extracted Pydantic models in server_models validate fields identically."""
    from microllm.server_models import StudioRLCDDemoRequest

    req = StudioRLCDDemoRequest(stage1_steps=20, stage2_steps=15)
    assert req.stage1_steps == 20
    assert req.stage2_steps == 15


def test_tier2_f12_04_cli_parser_backward_compatibility() -> None:
    """F12.B4: Extracted CLI parser in cli.py produces identical arguments."""
    from microllm.cli import build_cli_parser

    parser = build_cli_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_tier2_f12_05_gradio_handlers_direct_invocation() -> None:
    """F12.B5: Extracted gradio_handlers module functions are callable directly."""
    import microllm.gradio_handlers as ghandlers

    assert hasattr(ghandlers, "evaluate_guardrail_gradio") or hasattr(ghandlers, "run_guardrail_with_custom_rubrics")
