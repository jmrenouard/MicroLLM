"""Tier 4: Real-World Application Scenarios.

Validates end-to-end production workloads, client queries under auth, CLI runs,
and agentic workflows:
1. Enterprise client session full lifecycle (catalog, guardrail, decision, telemetry)
2. Multilingual batch support ticket classification pipeline (EN, FR, ES)
3. MCP agentic tool workflow execution under API key
4. Decision Studio diagnostic pipeline with custom rubrics and XAI
5. RLCD active learning cycle (feature extraction, head training, conformal bounds)
6. Concurrent mixed workload stress test under high request density
7. CLI benchmark telemetry automated ingestion pipeline
8. Model publishing packaging pipeline with Apache-2.0 licensing
9. Unauthorized attacker breach scenario (complete 401 perimeter defense)
10. Malformed payload adversarial injection defense (zero-500 guarantee)
11. High-volume multi-question throughput with state token cache reuse
12. Graceful recovery and resource bounded leakage verification
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
from microllm.client import Choice, MicroJevClient
from microllm.hf_publishing import guard_sockets, stage_model_profile
from microllm.hf_space_sync import stage_hf_space
from microllm.server import create_app


def test_tier4_01_production_client_session_full_lifecycle(
    auth_client: TestClient, auth_headers: dict[str, str], sample_choice_question: dict[str, Any]
) -> None:
    """Scenario 1: Enterprise Client Session Full Lifecycle.
    1. Query models catalog (/v1/models) [Public]
    2. Run safety guardrail (/v1/guardrail) [Protected]
    3. Execute systemone decision (/v1/systemone) [Protected]
    4. Fetch telemetry metrics (/v1/metrics) [Protected]
    """
    # 1. Models catalog
    resp_models = auth_client.get("/v1/models")
    assert resp_models.status_code == 200
    models_list = resp_models.json().get("models", [])
    assert len(models_list) >= 1

    # 2. Guardrail
    guardrail_payload = {
        "state": "Customer asking for invoice refund after billing charge error.",
        "threshold": 0.5,
    }
    resp_guardrail = auth_client.post("/v1/guardrail", json=guardrail_payload, headers=auth_headers)
    assert resp_guardrail.status_code == 200
    guard_data = resp_guardrail.json()
    assert guard_data.get("safe") is True

    # 3. SystemOne decision
    decision_payload = {
        "model": "jev-latest",
        "state": "Customer asking for invoice refund after billing charge error.",
        "questions": sample_choice_question,
    }
    resp_decision = auth_client.post("/v1/systemone", json=decision_payload, headers=auth_headers)
    assert resp_decision.status_code == 200
    assert resp_decision.json()["answers"]["dept"]["choice"] == "billing"

    # 4. Metrics
    resp_metrics = auth_client.get("/v1/metrics", headers=auth_headers)
    assert resp_metrics.status_code == 200
    metrics_data = resp_metrics.json()
    assert metrics_data["total_requests"] >= 1


def test_tier4_02_batch_support_ticket_classification_pipeline() -> None:
    """Scenario 2: Multilingual Support Ticket Classification Pipeline.
    Evaluates customer tickets in English, French, and Spanish with strategy='neural'.
    """
    app = create_app()
    test_client = TestClient(app)
    client = MicroJevClient(http_client=test_client)

    tickets = [
        "EN: The payment transaction was declined by the credit card gateway.",
        "FR: Bonjour, je souhaite obtenir un remboursement suite à une double facturation.",
        "ES: Hola, tengo un cargo no reconocido en mi tarjeta de crédito.",
    ]
    questions = {
        "department": Choice(
            options={
                "billing": "Invoice payment, refund, credit card charges",
                "technical": "System crash, bug, outage",
                "shipping": "Order tracking, delivery package",
            }
        )
    }

    results = client.batch(
        items=tickets,
        questions=questions,
        multilingual_strategy="neural",
        neural_model="minilm",
    )
    assert results["summary"]["total_items"] == 3
    for item in results["results"]:
        assert item["answers"]["department"]["choice"] == "billing"


def test_tier4_03_mcp_agentic_workflow_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 3: MCP Agentic Tool Workflow Execution under API key.
    An AI agent invokes tools sequentially: guardrail check -> calibration -> decision.
    """
    monkeypatch.setenv("MICROLLM_API_KEY", "mcp-agentic-key")
    from microllm.mcp_server import MicroLLMMCPServer

    mcp = MicroLLMMCPServer()

    # Step 1: Guardrail check
    guard_res = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "systemone_guardrail",
            "arguments": {"state": "Please summarize the invoice terms."},
        },
    })
    assert guard_res is not None
    assert guard_res.get("result", {}).get("isError") is not True
    assert "safe" in guard_res["result"]["structuredContent"]

    # Step 2: Calibration
    cal_res = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "alpha": 0.1,
                "logits": [[4.0, -4.0], [-3.0, 3.0], [5.0, -5.0], [-2.0, 2.0]],
                "labels": [0, 1, 0, 1],
            },
        },
    })
    assert cal_res is not None
    assert cal_res.get("result", {}).get("isError") is not True
    assert "conformal_q_hat" in cal_res["result"]["structuredContent"]

    # Step 3: Decision
    dec_res = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "systemone_decide",
            "arguments": {
                "state": "Server crashed with database connection timeout.",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "instructions": "Route ticket",
                        "criteria": {"billing": "Billing", "tech": "Database crashes"},
                    }
                },
            },
        },
    })
    assert dec_res is not None
    assert dec_res.get("result", {}).get("isError") is not True
    assert dec_res["result"]["structuredContent"]["answers"]["dept"]["choice"] == "tech"


def test_tier4_04_decision_studio_workflow_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4: Gradio Decision Studio User Journey.
    Executes in-process evaluation with custom rubrics under configured API key.
    """
    monkeypatch.setenv("MICROLLM_API_KEY", "studio-e2e-key")
    import microllm.gradio_app as gapp

    res = gapp.run_guardrail_with_custom_rubrics(
        text_input="Customer wants refund for duplicate monthly charge.",
        threshold=0.5,
        context_input="Finance portal",
        custom_rubrics_text="duplicate_charge: Multiple charges for same billing period",
    )
    assert isinstance(res, (tuple, list))
    badge, risk_txt, groundedness_txt, details_txt = res[:4]
    assert "safe" in badge.lower() or "block" in badge.lower() or "flagged" in badge.lower()


def test_tier4_05_rlcd_active_learning_cycle(
    auth_client: TestClient, auth_headers: dict[str, str], sample_valid_rlcd_dataset: tuple[list[list[float]], list[int]]
) -> None:
    """Scenario 5: End-to-End RLCD Active Learning Cycle.
    Submits collected dataset features and trains Stage 1 and Stage 2 decision head.
    """
    features, labels = sample_valid_rlcd_dataset
    payload = {
        "stage1_steps": 15,
        "stage2_steps": 10,
        "features": features,
        "labels": labels,
        "seed": 42,
    }
    resp = auth_client.post("/v1/rlcd", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    metrics = resp.json()
    assert "stage1_brier" in metrics or "final_brier" in metrics or "stage1_train_acc" in metrics or "stage2_accuracy" in metrics
    assert "conformal_q_hat" in metrics or "calibrated_brier" in metrics or "initial_brier" in metrics


def test_tier4_06_concurrent_mixed_workload_stress(
    auth_client: TestClient, auth_headers: dict[str, str], sample_choice_question: dict[str, Any]
) -> None:
    """Scenario 6: Concurrent Mixed Workload Stress Test.
    Mix of 10 concurrent requests (/healthz, /v1/systemone, /v1/guardrail) completes with zero 5xx errors.
    """
    errors: list[Exception] = []
    statuses: list[int] = []

    def _make_req(kind: str) -> None:
        try:
            if kind == "healthz":
                r = auth_client.get("/healthz")
            elif kind == "guardrail":
                r = auth_client.post(
                    "/v1/guardrail",
                    json={"state": "Normal test message"},
                    headers=auth_headers,
                )
            else:
                r = auth_client.post(
                    "/v1/systemone",
                    json={
                        "model": "jev-latest",
                        "state": "Urgent invoice billing question",
                        "questions": sample_choice_question,
                    },
                    headers=auth_headers,
                )
            statuses.append(r.status_code)
        except Exception as e:
            errors.append(e)

    tasks = ["healthz", "guardrail", "systemone"] * 4
    threads = [threading.Thread(target=_make_req, args=(kind,)) for kind in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Errors encountered during stress run: {errors}"
    assert all(s == 200 for s in statuses), f"Non-200 status codes: {statuses}"


def test_tier4_07_cli_benchmark_telemetry_pipeline() -> None:
    """Scenario 7: CLI Automated Benchmark Telemetry Ingestion Pipeline.
    Runs microllm benchmark --multilingual --json and parses JSON output into telemetry report.
    """
    import microllm.server as srv

    run_fn = getattr(srv, "run_cli", None)
    if run_fn is None:
        from microllm.cli import run_cli
        run_fn = run_cli

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = run_fn(["benchmark", "--multilingual", "--json"])
    assert exit_code == 0
    report = json.loads(buf.getvalue().strip())
    assert isinstance(report, dict)
    assert len(report.keys()) >= 1


def test_tier4_08_model_publishing_packaging_pipeline(tmp_path: Path) -> None:
    """Scenario 8: Model Publishing & Space Packaging Pipeline with Apache-2.0.
    Builds Space staging folder, generates README, and verifies Apache-2.0 license.
    """
    space_staging = tmp_path / "space_package"
    staged = stage_hf_space(staging_dir=space_staging)
    assert Path(staged["staging_dir"]).is_dir()

    readme_path = Path(staged["staging_dir"]) / "README.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")
    assert "license: apache-2.0" in text.lower()


def test_tier4_09_unauthorized_attacker_scenario(
    auth_client: TestClient, sample_choice_question: dict[str, Any]
) -> None:
    """Scenario 9: Unauthorized Attacker Scenario (Complete 401 Perimeter Defense).
    Simulated attacker probes protected endpoints with invalid or missing credentials;
    verified complete 401 lockdown.
    """
    attack_headers = {"Authorization": "Bearer hacker-brute-force-key"}
    endpoints = [
        ("POST", "/v1/studio/analyze", {"state": "test", "questions": sample_choice_question}),
        ("POST", "/v1/studio/rlcd-demo", {"stage1_steps": 10}),
        ("GET", "/v1/metrics", None),
        ("GET", "/v1/benchmark", None),
        ("POST", "/v1/rlcd", {"stage1_steps": 10}),
        ("POST", "/v1/guardrail", {"state": "test"}),
    ]

    for method, path, payload in endpoints:
        if method == "POST":
            r1 = auth_client.post(path, json=payload)
            r2 = auth_client.post(path, json=payload, headers=attack_headers)
        else:
            r1 = auth_client.get(path)
            r2 = auth_client.get(path, headers=attack_headers)

        assert r1.status_code == 401, f"Breach on {path} without auth: {r1.status_code}"
        assert r2.status_code == 401, f"Breach on {path} with bad auth: {r2.status_code}"


def test_tier4_10_malformed_payload_injection_defense(
    auth_client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Scenario 10: Malformed Payload Adversarial Injection Defense (Zero-500 Guarantee).
    Client injects malformed arrays, NaNs, negative labels, and non-2D matrices into RLCD.
    Verified zero 500 errors (100% 422).
    """
    adversarial_payloads = [
        {"features": [], "labels": []},
        {"features": [1.0, 2.0], "labels": [0]},
        {"features": [[1.0, 2.0]], "labels": [0, 1]},
        {"features": [[1.0, 2.0], [3.0]], "labels": [0, 1]},
        {"features": [[1.0, 2.0]], "labels": [-1]},
    ]

    for p in adversarial_payloads:
        p["stage1_steps"] = 10
        p["stage2_steps"] = 10
        resp = auth_client.post("/v1/rlcd", json=p, headers=auth_headers)
        assert resp.status_code == 422, f"Failed on payload {p}: got {resp.status_code} instead of 422"


def test_tier4_11_high_volume_multi_question_throughput(
    unauth_client: TestClient, sample_choice_question: dict[str, Any]
) -> None:
    """Scenario 11: High-Volume Multi-Question Throughput with State Token Cache Reuse.
    Evaluates 15 distinct questions on single state; verifies successful execution without degradation.
    """
    questions: dict[str, Any] = {}
    base_q = sample_choice_question["dept"]
    for i in range(15):
        questions[f"dept_routed_{i}"] = base_q

    payload = {
        "model": "jev-latest",
        "state": "The database connection failed during payment checkout invoice generation.",
        "questions": questions,
    }
    resp = unauth_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["answers"]) == 15


def test_tier4_12_graceful_recovery_and_zero_leakage() -> None:
    """Scenario 12: Graceful Recovery and Resource Bounded Leakage.
    Consecutive errors and cache operations maintain memory capacity boundaries.
    """
    cache = VerdictMQMultiQuestionCache(max_entries=10)
    for i in range(50):
        cache.encode_state_once(f"Arbitrary dynamic telemetry query {i}")

    # Maximum entries in internal cache never exceeds 10
    assert len(cache._cache) <= 10
