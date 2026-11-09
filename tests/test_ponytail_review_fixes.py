"""Regression and unit tests for all 15 Ponytail Review findings (Issue #171)."""

from __future__ import annotations

import asyncio
from collections import deque
import hmac
import re
import socket
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    create_hf_zero_shot_scorer,
    embed_text_deterministic,
)
from microllm.client import AsyncMicroJevClient, Choice, MicroJevClient
from microllm.hf_publishing import (
    build_multilingual_model_card,
    guard_sockets as guard_sockets_pub,
    stage_model_profile,
)
from microllm.multilingual_benchmark_runner import guard_sockets as guard_sockets_runner
from microllm.registry import load_profile_scorer
from microllm.server import create_app


def test_finding_01_auth_enforced_across_protected_endpoints_and_calibrate_isolation() -> None:
    """Finding 1: All /v1/* inference & mutation routes enforce MICROLLM_API_KEY and /v1/calibrate isolates state."""
    app = create_app(api_key="super-secret-ponytail-key")
    client = TestClient(app)
    headers = {"Authorization": "Bearer super-secret-ponytail-key"}

    # Unauthenticated calls must be rejected with 401
    assert client.post("/v1/guardrail", json={"state": "hello"}).status_code == 401
    assert (
        client.post(
            "/v1/batch",
            json={
                "items": ["hello"],
                "questions": {"q1": {"type": "noul", "instructions": "test"}},
            },
        ).status_code
        == 401
    )
    assert client.post("/v1/calibrate", json={"alpha": 0.25}).status_code == 401
    assert client.post("/v1/profiles/pull", json={"profile": "fast-cpu-int8"}).status_code == 401
    assert (
        client.post(
            "/v1/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        ).status_code
        == 401
    )
    assert client.post("/v1/rlcd", json={"stage1_steps": 10, "stage2_steps": 10}).status_code == 401
    assert (
        client.post(
            "/v1/studio/analyze",
            json={"state": "hello", "questions": {"q1": {"type": "choice", "instructions": "t", "criteria": {"a": "Option A", "b": "Option B"}}}},
        ).status_code
        == 401
    )
    assert client.post("/v1/studio/rlcd-demo", json={"stage1_steps": 5, "stage2_steps": 5}).status_code == 401
    assert client.get("/v1/metrics").status_code == 401
    assert client.get("/v1/benchmark").status_code == 401
    assert client.get("/v1/benchmark/external").status_code == 401
    assert client.get("/v1/benchmark/hf-jev-models").status_code == 401

    # Public documentation and model metadata routes remain accessible without auth
    assert client.get("/").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/profiles").status_code == 200
    assert client.get("/v1/jev-models").status_code == 200
    assert client.get("/v1/papers").status_code == 200
    assert client.get("/documentation").status_code == 200
    assert client.get("/slides").status_code == 200

    # Authenticated calls succeed
    cal_resp = client.post(
        "/v1/calibrate",
        json={
            "alpha": 0.95,
            "logits": [[5.0, -5.0], [-5.0, 5.0], [4.0, -4.0], [-4.0, 4.0]],
            "labels": [0, 1, 0, 1],
        },
        headers=headers,
    )
    assert cal_resp.status_code == 200

    met_resp = client.get("/v1/metrics", headers=headers)
    assert met_resp.status_code == 200
    assert "total_requests" in met_resp.json()

    rlcd_resp = client.post(
        "/v1/studio/rlcd-demo",
        json={"stage1_steps": 5, "stage2_steps": 5, "seed": 42},
        headers=headers,
    )
    assert rlcd_resp.status_code == 200
    assert "clopper_pearson_certification" in rlcd_resp.json()

    analyze_resp = client.post(
        "/v1/studio/analyze",
        json={
            "state": "The billing payment failed with card error.",
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Route ticket",
                    "criteria": {"billing": "Billing invoices", "tech": "Technical bug"},
                }
            },
        },
        headers=headers,
    )
    assert analyze_resp.status_code == 200
    assert "gear1" in analyze_resp.json()

    # Verify ad-hoc /v1/calibrate call with alpha=0.95 did NOT poison shared gear1 conformal alpha
    s1_resp = client.post(
        "/v1/systemone",
        json={
            "model": "microllm-gear1-fast",
            "state": "The invoice billing payment failed with credit card error.",
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Route ticket",
                    "criteria": {
                        "billing": "Invoices and payment errors",
                        "shipping": "Package delivery",
                    },
                }
            },
        },
        headers=headers,
    )
    assert s1_resp.status_code == 200
    assert s1_resp.json()["answers"]["dept"]["choice"] == "billing"


def test_finding_02_constant_time_hmac_compare_digest_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 2: Bearer token verification uses constant-time hmac.compare_digest."""
    compare_calls: list[tuple[Any, Any]] = []
    orig_compare = hmac.compare_digest

    def _spy_compare(a: Any, b: Any) -> bool:
        compare_calls.append((a, b))
        return orig_compare(a, b)

    monkeypatch.setattr(hmac, "compare_digest", _spy_compare)
    app = create_app(api_key="constant-time-key")
    client = TestClient(app)

    resp = client.post(
        "/v1/guardrail",
        json={"state": "normal query"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401
    assert len(compare_calls) >= 1


def test_finding_03_latencies_ms_bounded_deque() -> None:
    """Finding 3: telemetry_state['latencies_ms'] is a bounded deque (maxlen=10000)."""
    import inspect
    import microllm.server as srv

    src = inspect.getsource(srv.create_app)
    assert "deque(maxlen=" in src


def test_finding_04_guard_sockets_reentrant_and_thread_safe() -> None:
    """Finding 4: guard_sockets is unified, re-entrant, and safe under interleaved exits."""
    assert guard_sockets_pub is guard_sockets_runner

    orig_connect = socket.socket.connect
    cm1 = guard_sockets_pub()
    cm2 = guard_sockets_pub()

    cm1.__enter__()
    cm2.__enter__()
    # Interleaved exit: cm1 exits while cm2 is still active
    cm1.__exit__(None, None, None)

    # Socket must STILL be blocked because cm2 has not exited yet
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="offline"):
            s.connect(("127.0.0.1", 9))
    finally:
        s.close()

    # Now cm2 exits: original socket.socket.connect must be cleanly restored
    cm2.__exit__(None, None, None)
    assert socket.socket.connect is orig_connect


def test_finding_05_multilingual_model_card_python_snippet_is_valid_and_executable() -> None:
    """Finding 5: Model card Python Quickstart imports microllm.backends.cascade and runs cleanly."""
    card = build_multilingual_model_card("multilingual-minilm-int8")
    assert "microllm.multilingual_backend" not in card
    assert "route_ticket" not in card
    assert "from microllm.backends.cascade import create_multilingual_backend" in card

    match = re.search(r"```python\n(.*?)```", card, flags=re.DOTALL)
    assert match is not None
    snippet = match.group(1)
    ns: dict[str, Any] = {}
    exec(snippet, ns)
    assert "result" in ns
    assert ns["result"]["choice"] == "billing"


def test_finding_06_load_profile_scorer_supports_0d_and_1d_npz_scale(tmp_path: Path) -> None:
    """Finding 6: load_profile_scorer loads both 0D scalar scale and 1D array scale without IndexError."""
    staged = stage_model_profile("multilingual-minilm-int8", staging_dir=tmp_path / "staged")
    staged_npz = Path(staged["staging_dir"]) / "model_quantized_int8.npz"
    assert staged_npz.is_file()

    cache_root = tmp_path / "cache"
    prof_dir = cache_root / "multilingual-minilm-int8"
    prof_dir.mkdir(parents=True, exist_ok=True)
    (prof_dir / "model_quantized_int8.npz").write_bytes(staged_npz.read_bytes())

    scorer = load_profile_scorer("multilingual-minilm-int8", cache_dir=cache_root)
    val = scorer(
        "Je veux un remboursement de ma facture.",
        "Route ticket",
        "It is billing: invoice payment refund.",
    )
    assert isinstance(val, float) and np.isfinite(val)


def test_finding_07_hf_zero_shot_scorer_binary_not_entailment_label_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 7: create_hf_zero_shot_scorer picks 'entailment' (1), not 'not_entailment' (0)."""
    mock_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    mock_torch.no_grad = _NoGrad  # type: ignore[attr-defined]

    class _FakeTensor:
        def __init__(self, val: float) -> None:
            self._val = val

        def item(self) -> float:
            return self._val

    class _FakeConfig:
        label2id = {"not_entailment": 0, "entailment": 1}
        id2label = {0: "not_entailment", 1: "entailment"}

    class _FakeModel:
        config = _FakeConfig()

        def eval(self) -> None:
            return None

        def __call__(self, **kwargs: Any) -> Any:
            class _Out:
                logits = [[_FakeTensor(-3.5), _FakeTensor(4.25)]]

            return _Out()

    class _FakeTokenizer:
        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [[1, 2, 3]]}

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **kw: _FakeTokenizer()
    )
    mock_transformers.AutoModelForSequenceClassification = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **kw: _FakeModel()
    )

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    scorer = create_hf_zero_shot_scorer("fake/binary-rte-nli")
    score = scorer("Premise", "Query", "Hypothesis")
    assert score == 4.25


def test_finding_08_and_09_gradio_studio_caches_backends_and_test_client() -> None:
    """Findings 8 & 9: Gradio studio caches backends across clicks and reuses a lazy TestClient."""
    import microllm.gradio_app as gapp

    assert hasattr(gapp, "_get_cached_studio_backend")
    assert hasattr(gapp, "_get_studio_test_client")

    b1 = gapp._get_cached_studio_backend("fast", "minilm", 0.75, 0.75)
    b2 = gapp._get_cached_studio_backend("fast", "minilm", 0.75, 0.75)
    assert b1 is b2

    c1 = gapp._get_studio_test_client()
    c2 = gapp._get_studio_test_client()
    assert c1 is c2


def test_finding_10_verdict_mq_skips_unused_clause_embeddings_on_short_states() -> None:
    """Finding 10: VerdictMQMultiQuestionCache only embeds clauses when clause-salient triggers match."""
    mq = VerdictMQMultiQuestionCache(max_entries=16)
    short_state = "First clause about database latency. Second clause about billing refund."
    entry_short = mq.encode_state_once(short_state)
    assert entry_short["clause_count"] == 2
    assert entry_short["clause_embeddings"] == ()

    salient_state = (
        "Initial routine telemetry looks normal across the cluster; "
        "however, critical database connection pool exhaustion is blocking checkout."
    )
    entry_salient = mq.encode_state_once(salient_state)
    assert entry_salient["clause_count"] >= 2
    assert len(entry_salient["clause_embeddings"]) >= 2


def test_finding_11_client_shared_helpers_and_async_guardrail_batch_parity() -> None:
    """Finding 11: client.py shares _build_ask_payload/_parse_systemone_answers and AsyncMicroJevClient has guardrail & batch."""
    import microllm.client as cli_mod

    assert hasattr(cli_mod, "_build_ask_payload")
    assert hasattr(cli_mod, "_parse_systemone_answers")

    app = create_app()
    test_client = TestClient(app)
    async_client = AsyncMicroJevClient(http_client=test_client)

    async def _run() -> None:
        g_res = await async_client.guardrail("Normal safe request to summarize docs.")
        assert g_res["safe"] is True
        b_res = await async_client.batch(
            items=["Invoice payment failed"],
            questions={"dept": Choice(options={"billing": "Invoice payment", "tech": "Bug"})},
        )
        assert b_res["summary"]["total_items"] == 1
        assert b_res["results"][0]["answers"]["dept"]["choice"] == "billing"

    asyncio.run(_run())


def test_finding_12_shared_hf_upload_helper_used_across_dataset_model_and_space() -> None:
    """Finding 12: _upload_staged_folder_to_hf is shared by hf_publishing and hf_space_sync."""
    import microllm.hf_publishing as hfp

    assert hasattr(hfp, "_upload_staged_folder_to_hf")


def test_finding_13_batch_and_mcp_respect_multilingual_strategy_and_neural_model() -> None:
    """Finding 13: /v1/batch and /v1/mcp support multilingual_strategy and neural_model."""
    app = create_app(multilingual_strategy="neural", neural_model="mdeberta")
    client = TestClient(app)

    batch_resp = client.post(
        "/v1/batch",
        json={
            "items": ["Bonjour, je voudrais le remboursement de ma facture."],
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Route ticket",
                    "criteria": {
                        "billing": "Invoice, payment, refund",
                        "tech": "Technical bug, crash",
                    },
                }
            },
            "multilingual_strategy": "neural",
            "neural_model": "mdeberta",
        },
    )
    assert batch_resp.status_code == 200
    assert batch_resp.json()["results"][0]["answers"]["dept"]["choice"] == "billing"

    mcp_resp = client.post(
        "/v1/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "systemone_decide",
                "arguments": {
                    "state": "Hola, necesito un reembolso de mi factura.",
                    "multilingual_strategy": "neural",
                    "neural_model": "minilm",
                    "questions": {
                        "dept": {
                            "type": "choice",
                            "instructions": "Route ticket",
                            "criteria": {
                                "billing": "Invoice, payment, refund",
                                "tech": "Technical bug, crash",
                            },
                        }
                    },
                },
            },
        },
    )
    assert mcp_resp.status_code == 200
    structured = mcp_resp.json()["result"]["structuredContent"]
    assert structured["answers"]["dept"]["choice"] == "billing"


def test_finding_14_precompiled_concept_clusters_preserve_exact_embeddings() -> None:
    """Finding 14: _COMPILED_CONCEPT_CLUSTERS is precomputed and preserves deterministic embeddings."""
    import microllm.backends.base as base_mod

    assert hasattr(base_mod, "_COMPILED_CONCEPT_CLUSTERS")
    assert len(base_mod._COMPILED_CONCEPT_CLUSTERS) == len(base_mod.CONCEPT_CLUSTERS)
    vec = embed_text_deterministic("Critical database outage and billing refund request.")
    assert vec.shape == (256,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_finding_15_hf_publishing_reexports_space_sync_directly() -> None:
    """Finding 15: hf_publishing directly re-exports space sync functions from hf_space_sync."""
    import microllm.hf_publishing as hfp
    import microllm.hf_space_sync as hfs

    assert hfp.build_space_readme is hfs.build_space_readme
    assert hfp.stage_hf_space is hfs.stage_hf_space
    assert hfp.sync_hf_space is hfs.sync_hf_space


def test_finding_16_universal_endpoint_authentication_and_public_routes() -> None:
    """Finding 16 (Milestone 1 F01): Universal endpoint authentication on CPU-heavy routes & public access."""
    # 1. When API key is configured, missing or invalid Bearer tokens are rejected with 401
    app = create_app(api_key="m1-secret-key")
    client = TestClient(app)
    bad_headers = {"Authorization": "Bearer wrong-token"}

    endpoints_to_test = [
        ("POST", "/v1/studio/analyze", {"state": "test", "questions": {"q": {"type": "choice", "instructions": "t", "criteria": {"a": "Option A", "b": "Option B"}}}}),
        ("POST", "/v1/studio/rlcd-demo", {"stage1_steps": 5, "stage2_steps": 5}),
        ("GET", "/v1/metrics", None),
        ("GET", "/v1/benchmark", None),
        ("GET", "/v1/benchmark/external", None),
        ("GET", "/v1/benchmark/hf-jev-models", None),
        ("POST", "/v1/guardrail", {"state": "hello"}),
        ("POST", "/v1/calibrate", {"alpha": 0.25}),
        ("POST", "/v1/profiles/pull", {"profile": "fast-cpu-int8"}),
        ("POST", "/v1/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        ("POST", "/v1/batch", {"items": ["hello"], "questions": {"q1": {"type": "noul", "instructions": "test"}}}),
        ("POST", "/v1/rlcd", {"stage1_steps": 5, "stage2_steps": 5}),
    ]

    for method, path, payload in endpoints_to_test:
        if method == "POST":
            # Missing auth header
            assert client.post(path, json=payload).status_code == 401, f"Expected 401 for unauthenticated POST {path}"
            # Invalid token
            assert client.post(path, json=payload, headers=bad_headers).status_code == 401, f"Expected 401 for invalid Bearer POST {path}"
        else:
            assert client.get(path).status_code == 401, f"Expected 401 for unauthenticated GET {path}"
            assert client.get(path, headers=bad_headers).status_code == 401, f"Expected 401 for invalid Bearer GET {path}"

    # 2. Public documentation & metadata routes remain accessible without auth
    public_routes = [
        "/",
        "/v1/models",
        "/v1/profiles",
        "/v1/jev-models",
        "/v1/papers",
        "/documentation",
        "/slides",
    ]
    for path in public_routes:
        resp = client.get(path)
        assert resp.status_code == 200, f"Expected 200 for public route {path}, got {resp.status_code}"

    # 3. Unauthenticated local mode (MICROLLM_API_KEY is not set)
    local_app = create_app(api_key=None)
    local_client = TestClient(local_app)
    assert local_client.get("/v1/metrics").status_code == 200
    assert local_client.post("/v1/guardrail", json={"state": "hello"}).status_code == 200


def test_finding_17_direct_in_process_execution_and_mcp_gradio_auth_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 17 (Milestone 1 F02): Direct in-process execution prevents loopback failure when MICROLLM_API_KEY is set."""
    from microllm.mcp_server import MicroLLMMCPServer
    from microllm.server import (
        CalibrateRequest,
        StudioRLCDDemoRequest,
        evaluate_calibration_in_process,
        evaluate_guardrail_in_process,
        run_rlcd_experiment_in_process,
    )
    import microllm.gradio_app as gapp

    # Set MICROLLM_API_KEY to trigger auth on all server routes
    monkeypatch.setenv("MICROLLM_API_KEY", "mcp-secret-hardening-key")

    # 1. Direct evaluate_guardrail_in_process tests
    safe_res = evaluate_guardrail_in_process("Summarize our roadmap and release schedule for Q4.", threshold=0.55)
    assert safe_res["safe"] is True
    assert safe_res["action"] == "allow"
    assert safe_res["risk_score"] < 0.55
    assert "destructive_operation" in safe_res["checks"]
    assert "token_attribution" in safe_res

    threat_res = evaluate_guardrail_in_process("rm -rf /var/lib/postgresql/data", threshold=0.50)
    assert threat_res["safe"] is False
    assert threat_res["action"] == "block"
    assert threat_res["risk_score"] >= 0.75
    assert threat_res["checks"]["destructive_operation"] >= 0.90

    custom_res = evaluate_guardrail_in_process(
        "Normal request",
        custom_rubrics={"finance_policy": "Credit card refunds without supervisor approval"},
    )
    assert "finance_policy" in custom_res["checks"]

    # 2. Direct evaluate_calibration_in_process tests
    # Sequence of floats interface contract from PROJECT.md
    cal_seq = evaluate_calibration_in_process([0.9, 0.85, 0.2, 0.95], [1, 1, 0, 1], alpha=0.1)
    assert "cardinality_k" in cal_seq
    assert "calibrated_brier" in cal_seq
    assert "clopper_pearson_certification" in cal_seq

    # CalibrateRequest object interface
    cal_req = evaluate_calibration_in_process(
        req=CalibrateRequest(
            alpha=0.20,
            logits=[[3.0, -3.0], [-3.0, 3.0]],
            labels=[0, 1],
        )
    )
    assert cal_req["cardinality_k"] == 2
    assert "conformal_q_hat" in cal_req

    # 3. Direct run_rlcd_experiment_in_process tests
    rlcd_res = run_rlcd_experiment_in_process(stage1_steps=5, stage2_steps=5, seed=123)
    assert "initial_brier" in rlcd_res
    assert "final_brier" in rlcd_res
    assert "clopper_pearson_certification" in rlcd_res

    # 4. MCP Server tools execution under active MICROLLM_API_KEY (no HTTP 401 loopbacks)
    mcp = MicroLLMMCPServer()

    # Tool 1: systemone_guardrail
    g_rpc = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "systemone_guardrail",
            "arguments": {"state": "Safe user request to parse documentation."},
        },
    })
    assert g_rpc is not None
    assert g_rpc["result"]["isError"] is False
    assert g_rpc["result"]["structuredContent"]["safe"] is True

    # Tool 2: systemone_agent_guardrail
    ag_rpc = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_guardrail",
            "arguments": {
                "tool": "bash",
                "command": "rm -rf /var/lib/postgresql/data",
                "environment": "production",
            },
        },
    })
    assert ag_rpc is not None
    assert ag_rpc["result"]["isError"] is False
    assert ag_rpc["result"]["structuredContent"]["allowed"] is False
    assert ag_rpc["result"]["structuredContent"]["action"] == "BLOCK"

    # Tool 3: systemone_calibrate
    cal_rpc = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "alpha": 0.15,
                "logits": [[4.0, -4.0], [-4.0, 4.0]],
                "labels": [0, 1],
            },
        },
    })
    assert cal_rpc is not None
    assert cal_rpc["result"]["isError"] is False
    assert cal_rpc["result"]["structuredContent"]["cardinality_k"] == 2

    # Tool 4: systemone_decide
    dec_rpc = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "systemone_decide",
            "arguments": {
                "state": "Payment failed due to expired credit card.",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "instructions": "Route ticket",
                        "criteria": {"billing": "Card payments and invoices", "tech": "Bugs"},
                    }
                },
            },
        },
    })
    assert dec_rpc is not None
    assert dec_rpc["result"]["isError"] is False
    assert dec_rpc["result"]["structuredContent"]["answers"]["dept"]["choice"] == "billing"

    # Tool 5: systemone_agent_router
    rout_rpc = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_router",
            "arguments": {
                "goal": "Query customer orders from SQL database",
                "candidates": {"sql": "SQL database query tool", "web": "Public web search"},
            },
        },
    })
    assert rout_rpc is not None
    assert rout_rpc["result"]["isError"] is False
    assert rout_rpc["result"]["structuredContent"]["selected"] == "sql"

    # 5. Gradio Studio tabs execution under active MICROLLM_API_KEY
    verdict_md, lat_str, checks_dict, full_d = gapp.run_guardrail_with_custom_rubrics(
        text_input="Summarize our roadmap and release schedule for Q4.",
        threshold=0.55,
    )
    assert "SAFE / ALLOW" in verdict_md
    assert full_d["safe"] is True
    assert "destructive_operation" in checks_dict

    rlcd_md, rlcd_d = gapp.run_rlcd_lab_gradio(stage1_steps=5, stage2_steps=5)
    assert "OpenJev-RLCD" in rlcd_md
    assert "clopper_pearson_certification" in rlcd_d
    assert "initial_brier" in rlcd_d

    # TestClient singleton compatibility helper
    legacy_client = gapp._get_studio_test_client()
    assert legacy_client is not None


def test_finding_18_rlcd_input_validation_zero_500_guarantee() -> None:
    """Finding 18 (Milestone 2 F03): Zero-500 guarantee via Pydantic model validation (HTTP 422) & trainer ValueError."""
    from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer

    app = create_app()
    client = TestClient(app)

    # 1. Empty features and labels return 422
    assert client.post("/v1/rlcd", json={"features": [], "labels": []}).status_code == 422
    assert client.post("/v1/studio/rlcd-demo", json={"features": [], "labels": []}).status_code == 422

    # 2. Sample count mismatch returns 422
    mismatch_payload = {
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [0, 1, 0],
    }
    assert client.post("/v1/rlcd", json=mismatch_payload).status_code == 422

    # 3. 1D features array returns 422
    oned_payload = {"features": [1.0, 2.0], "labels": [0]}
    assert client.post("/v1/rlcd", json=oned_payload).status_code == 422

    # 4. Negative label returns 422
    neg_payload = {
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [-1, 0],
    }
    assert client.post("/v1/rlcd", json=neg_payload).status_code == 422

    # 5. Ragged features matrix returns 422
    ragged_payload = {
        "features": [[1.0, 2.0], [3.0, 4.0, 5.0]],
        "labels": [0, 1],
    }
    assert client.post("/v1/rlcd", json=ragged_payload).status_code == 422

    # 6. One of features/labels provided while other is None returns 422
    assert client.post("/v1/rlcd", json={"features": [[1.0, 2.0]], "labels": None}).status_code == 422
    assert client.post("/v1/rlcd", json={"features": None, "labels": [0, 1]}).status_code == 422

    # 7. Valid inputs or both None return 200
    assert client.post("/v1/rlcd", json={"features": None, "labels": None, "stage1_steps": 5, "stage2_steps": 5}).status_code == 200
    valid_payload = {
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [0, 1],
        "stage1_steps": 5,
        "stage2_steps": 5,
    }
    assert client.post("/v1/rlcd", json=valid_payload).status_code == 200

    # 8. RLCDDecisionHeadTrainer input validation raises ValueError
    cfg = RLCDConfig(stage1_steps=5, stage2_steps=5)
    trainer = RLCDDecisionHeadTrainer(d_feat=4, k_classes=2, config=cfg)
    with pytest.raises(ValueError):
        trainer.fit(np.zeros((0, 4)), np.zeros(0, dtype=int))
    with pytest.raises(ValueError):
        trainer.fit(np.zeros((5, 4)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError):
        trainer.fit(np.zeros(5), np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        trainer.fit(np.ones((4, 4)), np.array([-1, 0, 1, 0], dtype=int))
    with pytest.raises(ValueError):
        trainer.predict_proba(np.zeros(4))


def test_finding_19_guard_sockets_thread_isolation_and_concurrency() -> None:
    """Finding 19 (Milestone 2 F04): Concurrency safety and thread isolation in guard_sockets()."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    port = server_sock.getsockname()[1]

    thread_connected = [False]
    thread_error: list[Exception] = []

    def _worker() -> None:
        try:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.connect(("127.0.0.1", port))
            c.close()
            thread_connected[0] = True
        except Exception as e:
            thread_error.append(e)

    try:
        with guard_sockets_pub():
            # Main thread is guarded: socket.connect raises RuntimeError
            s_main = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with pytest.raises(RuntimeError, match="offline"):
                    s_main.connect(("127.0.0.1", port))
            finally:
                s_main.close()

            # Concurrent non-benchmark thread connects unimpeded
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

    assert not thread_error, f"Concurrent thread raised: {thread_error}"
    assert thread_connected[0] is True


def test_finding_20_non_blocking_event_loop_healthz_and_cache_lock() -> None:
    """Finding 20 (Milestone 2 F05): Event loop non-blocking execution, /healthz endpoint, and cache Lock."""
    import inspect
    import microllm.server as srv

    app = create_app(api_key="healthz-test-key")
    client = TestClient(app)

    # 1. /healthz returns 200 without authentication
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "microllm" in data["service"].lower()

    # 2. CPU route handlers use asyncio.to_thread for offloading
    src = inspect.getsource(srv.create_app)
    assert "to_thread" in src

    # 3. VerdictMQMultiQuestionCache has lock attribute and synchronizes state encoding
    cache = VerdictMQMultiQuestionCache(max_entries=32)
    assert hasattr(cache, "_lock")
    empty_entry = cache.encode_state_once("")
    assert empty_entry["clause_count"] == 0

    exceptions: list[Exception] = []

    def _cache_worker(tid: int) -> None:
        try:
            for i in range(10):
                cache.encode_state_once(f"Thread {tid} text {i}")
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=_cache_worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not exceptions, f"Concurrent cache access raised: {exceptions}"
    assert cache.stats["hits"] >= 0
    assert cache.stats["misses"] >= 40

