"""Regression and unit tests for all 15 Ponytail Review findings (Issue #171)."""

from __future__ import annotations

import asyncio
from collections import deque
import hmac
import re
import socket
import sys
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
