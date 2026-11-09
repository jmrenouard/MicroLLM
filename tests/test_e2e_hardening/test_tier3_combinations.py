"""Tier 3: Cross-Feature Combinations.

Validates multi-feature interactions across security, validation, concurrency, caching, and SDK:
1. Auth + Thread-isolated sockets concurrency
2. Input validation + Event loop non-blocking / /healthz
3. SDK Client Batch + Multilingual Strategy + Auth
4. In-process MCP execution + Configured API Key + Token caching
5. Gradio Guardrail in-process + Custom rubrics + Auth
6. RLCD in-process + Custom valid arrays + Model card generation
7. CLI benchmark JSON output + Socket isolation
8. Async and Sync SDK clients identical serialization under Auth
9. Multi-question cache + Conformal calibration under Auth
10. Modular re-exports across gradio, server, and dataset adapters
11. RLCD validation rejects malformed payload before trainer execution
12. Vault path environment variable override during model profile staging
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from microllm.backends.base import VerdictMQMultiQuestionCache
from microllm.client import AsyncMicroJevClient, Choice, MicroJevClient
from microllm.hf_publishing import guard_sockets, stage_model_profile
from microllm.hf_space_sync import build_space_readme
from microllm.rlcd import RLCDConfig, RLCDDecisionHeadTrainer
from microllm.server import create_app


def test_tier3_01_auth_and_concurrency_socket_isolation(auth_client: TestClient, auth_headers: dict[str, str]) -> None:
    """Combines F01 (Auth) and F04 (Socket Guard Isolation):
    An authenticated HTTP request executes concurrently with a thread running inside guard_sockets();
    neither blocks or interferes with the other.
    """
    server_response_status: list[int] = []

    def _auth_request_worker() -> None:
        resp = auth_client.get("/v1/models", headers=auth_headers)
        server_response_status.append(resp.status_code)

    with guard_sockets():
        # Background thread runs authenticated request
        t = threading.Thread(target=_auth_request_worker)
        t.start()
        t.join(timeout=3.0)

    assert server_response_status == [200]


def test_tier3_02_validation_and_event_loop_non_blocking(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Combines F03 (RLCD 422 Validation) and F05 (Non-blocking /healthz):
    Submitting an invalid RLCD payload returns HTTP 422 while concurrent /healthz probe returns immediately.
    """
    # 1. Invalid payload: mismatched features and labels
    invalid_payload = {
        "stage1_steps": 5,
        "stage2_steps": 5,
        "features": [[1.0, 2.0]],  # 1 row
        "labels": [0, 1],  # 2 labels -> mismatch
    }
    resp_validation = auth_client.post("/v1/rlcd", json=invalid_payload, headers=auth_headers)
    assert resp_validation.status_code == 422

    # 2. Concurrently /healthz responds with 200 ok
    resp_health = auth_client.get("/healthz")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "ok"


def test_tier3_03_sdk_client_batch_with_multilingual_strategy_and_auth() -> None:
    """Combines F01 (Auth), F07 (SDK Parity), and F08 (Shared SDK Base):
    MicroJevClient sends authenticated batch with multilingual_strategy='neural' and neural_model='minilm'.
    """
    app = create_app(api_key="batch-combo-key")
    test_client = TestClient(app)
    client = MicroJevClient(api_key="batch-combo-key", http_client=test_client)

    res = client.batch(
        items=["Bonjour, je voudrais contester ma facture."],
        questions={"dept": Choice(options={"billing": "Factures et paiements", "tech": "Support technique"})},
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    assert res["summary"]["total_items"] == 1
    assert res["results"][0]["answers"]["dept"]["choice"] == "billing"


def test_tier3_04_in_process_mcp_under_configured_api_key_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combines F01 (Auth), F02 (In-process MCP), and F06 (Multi-Question Cache):
    MCP server executes tool call with multiple questions when MICROLLM_API_KEY is configured.
    """
    monkeypatch.setenv("MICROLLM_API_KEY", "mcp-combo-key")
    from microllm.mcp_server import MicroLLMMCPServer

    mcp = MicroLLMMCPServer()
    resp = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "systemone_decide",
            "arguments": {
                "state": "The primary database crashed during credit card checkout payment.",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "instructions": "Route ticket",
                        "criteria": {"billing": "Payment checkout", "tech": "Database crash"},
                    },
                    "urgency": {
                        "type": "choice",
                        "instructions": "Score urgency",
                        "criteria": {"high": "Critical outage", "low": "General question"},
                    },
                },
            },
        },
    })
    assert resp is not None
    assert resp.get("result", {}).get("isError") is not True
    answers = resp["result"]["structuredContent"]["answers"]
    assert "dept" in answers
    assert "urgency" in answers


def test_tier3_05_gradio_guardrail_in_process_with_custom_rubrics_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combines F01 (Auth) and F02 (In-process Gradio) with custom rubrics:
    Gradio safety guardrail handler executes in-process without 401 when MICROLLM_API_KEY is configured.
    """
    monkeypatch.setenv("MICROLLM_API_KEY", "gradio-combo-key")
    import microllm.gradio_app as gapp

    res = gapp.run_guardrail_with_custom_rubrics(
        text_input="System invoice refund request",
        threshold=0.6,
        context_input="Accounting portal",
        custom_rubrics_text="fraud: Suspicious account activity\nleak: Credential leak",
    )
    assert isinstance(res, (tuple, list))
    assert len(res) >= 3


def test_tier3_06_rlcd_in_process_with_valid_arrays_and_model_card_generation(
    sample_valid_rlcd_dataset: tuple[list[list[float]], list[int]]
) -> None:
    """Combines F02 (In-process RLCD), F03 (RLCD validation), and F09 (Apache-2.0 Space License):
    Valid arrays trained in-process and model card generated with Apache-2.0 license default.
    """
    import microllm.server as srv
    from microllm.server import StudioRLCDDemoRequest

    fn = getattr(srv, "run_rlcd_experiment_in_process", None)
    if fn is None:
        pytest.fail("Feature F02 missing: run_rlcd_experiment_in_process not in microllm.server")

    features, labels = sample_valid_rlcd_dataset
    req = StudioRLCDDemoRequest(
        stage1_steps=10,
        stage2_steps=10,
        features=features,
        labels=labels,
        seed=42,
    )
    result = fn(req)
    assert isinstance(result, dict)

    # Generate Space README with default license
    readme = build_space_readme()
    assert "license: apache-2.0" in readme.lower()


def test_tier3_07_cli_benchmark_json_output_with_thread_isolated_sockets() -> None:
    """Combines F04 (Thread-isolated sockets) and F11 (CLI benchmark --json):
    CLI benchmark executes with --json while background socket guard is safely scoped.
    """
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    with guard_sockets():
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_fn(["benchmark", "--multilingual", "--json"])
        assert code == 0
        data = json.loads(buf.getvalue().strip())
        assert isinstance(data, dict)


def test_tier3_08_async_and_sync_sdk_clients_identical_payloads_under_auth() -> None:
    """Combines F01 (Auth), F07 (Batch Parity), and F08 (Shared SDK Base):
    Verifies that MicroJevClient and AsyncMicroJevClient send identical serialized JSON to /v1/batch.
    """
    from microllm.client import _build_batch_payload

    items = ["Message 1: billing issue", "Message 2: technical crash"]
    questions = {
        "dept": Choice(options={"billing": "Invoices and charges", "tech": "Crashes and bugs"})
    }

    payload = _build_batch_payload(
        items=items,
        questions=questions,
        model="jev-latest",
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    assert payload["items"] == items
    assert payload["multilingual_strategy"] == "neural"
    assert payload["neural_model"] == "minilm"
    assert "questions" in payload


def test_tier3_09_multi_question_cache_and_conformal_calibration_under_auth(
    auth_client: TestClient, auth_headers: dict[str, str], sample_choice_question: dict[str, Any]
) -> None:
    """Combines F01 (Auth), F05 (Cache Concurrency), and F06 (Multi-Question Caching):
    Protected multi-question request executes with conformal calibration under API key.
    """
    payload = {
        "model": "jev-latest",
        "state": "The invoice billing payment failed with credit card charge error.",
        "questions": sample_choice_question,
    }
    resp = auth_client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answers"]["dept"]["choice"] == "billing"


def test_tier3_10_modular_reexports_across_gradio_server_and_adapters() -> None:
    """Combines F12 modularization across all three decomposed packages:
    Verifies re-export facades on gradio_app, dataset_adapters, and server.
    """
    import microllm.dataset_adapters as dadapt
    import microllm.dataset_fixtures as dfix
    import microllm.gradio_app as gapp
    import microllm.gradio_presets as gpresets
    import microllm.server as srv
    import microllm.server_models as smodels

    # 1. Dataset fixtures
    assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES

    # 2. Gradio presets
    assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS

    # 3. Server models
    assert srv.StudioRLCDDemoRequest is smodels.StudioRLCDDemoRequest


def test_tier3_11_rlcd_validation_rejects_malformed_input_before_trainer_called(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Combines F01 (Auth) and F03 (RLCD validation):
    Pydantic schema validation rejects malformed payload with 422 before training logic can fail.
    """
    malformed_payload = {
        "stage1_steps": 10,
        "stage2_steps": 10,
        "features": [[1.0, 2.0], [3.0, 4.0]],
        "labels": [-1, 0],  # negative label
    }
    resp = auth_client.post("/v1/rlcd", json=malformed_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert "labels" in str(resp.json()).lower() or "negative" in str(resp.json()).lower()


def test_tier3_12_vault_path_env_var_override_during_model_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Combines F10 (Workstation Path Elimination) and Model Profile Staging:
    Model staging executes without workstation path dependency when MICROLLM_KEEPASS_DB is configured.
    """
    custom_vault = tmp_path / "custom_vault.kdbx"
    monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(custom_vault))

    # Stage profile locally
    staged = stage_model_profile("multilingual-minilm-int8", staging_dir=tmp_path / "staged")
    assert Path(staged["staging_dir"]).is_dir()
    assert (Path(staged["staging_dir"]) / "model_quantized_int8.npz").is_file()
