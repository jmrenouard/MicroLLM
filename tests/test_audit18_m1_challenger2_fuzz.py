"""Adversarial Fuzzing and Stress-Testing Suite for Milestone 1.

Authored by Empirical Challenger 2.
Focus: Points 4 & 5 (OOM Prevention, Boolean Rejection, Array Dimension Boundaries, MCP Fuzzing).
"""

from __future__ import annotations

import math
import os
import psutil
import pytest
from fastapi.testclient import TestClient

from microllm.mcp_server import MicroLLMMCPServer
from microllm.server import create_app
from microllm.server_models import CalibrateRequest, StudioRLCDDemoRequest


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    """Fixture providing initialized FastAPI test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="module")
def mcp_server() -> MicroLLMMCPServer:
    """Fixture providing initialized MicroLLM MCP server instance."""
    return MicroLLMMCPServer()


# =============================================================================
# 1. Booleans in Labels, Features, and Logits
# =============================================================================


@pytest.mark.parametrize("endpoint", ["/v1/rlcd", "/v1/studio/rlcd-demo"])
def test_fuzz_booleans_in_rlcd(api_client: TestClient, endpoint: str) -> None:
    """Verify boolean values in labels and features are rejected with HTTP 422."""
    cases = [
        {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [True, False]},
        {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, True]},
        {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [False, 1]},
        {"features": [[True, 2.0], [3.0, 4.0]], "labels": [0, 1]},
        {"features": [[1.0, False], [3.0, 4.0]], "labels": [0, 1]},
        {"features": [[1.0, 2.0], [True, False]], "labels": [0, 1]},
    ]
    for payload in cases:
        resp = api_client.post(endpoint, json=payload)
        assert resp.status_code == 422, f"Expected 422 for {payload}, got {resp.status_code}"
        assert "cannot be a boolean" in resp.text


def test_fuzz_booleans_in_calibrate(api_client: TestClient) -> None:
    """Verify boolean values in labels, logits, and hyperparameters are rejected with HTTP 422."""
    cases = [
        {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [True, False]},
        {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, True]},
        {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [False, 1]},
        {"logits": [[True, 2.0], [3.0, 4.0]], "labels": [0, 1]},
        {"logits": [[1.0, False], [3.0, 4.0]], "labels": [0, 1]},
        {"logits": [[1.0, 2.0], [True, False]], "labels": [0, 1]},
        {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1], "alpha": True},
        {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1], "target_risk": False},
    ]
    for payload in cases:
        resp = api_client.post("/v1/calibrate", json=payload)
        assert resp.status_code == 422, f"Expected 422 for {payload}, got {resp.status_code}"


# =============================================================================
# 2. Extreme Class Values & OOM Prevention
# =============================================================================


def test_fuzz_extreme_class_values_and_oom_prevention(api_client: TestClient) -> None:
    """Verify extreme class values (e.g. 1.6 Billion) do not cause OOM and return HTTP 422 immediately."""
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)

    # 1. 1.6 Billion classes on /v1/rlcd and /v1/studio/rlcd-demo
    for ep in ["/v1/rlcd", "/v1/studio/rlcd-demo"]:
        resp = api_client.post(
            ep,
            json={"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1_600_000_000]},
        )
        assert resp.status_code == 422
        assert "exceeds maximum allowed class index 255" in resp.text

    # 2. Class label 256 (K=257 > 256 limit)
    resp_256 = api_client.post(
        "/v1/rlcd",
        json={"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 256]},
    )
    assert resp_256.status_code == 422
    assert "exceeds maximum allowed class index 255" in resp_256.text

    # 3. Negative labels
    for bad_lbl in [-1, -999]:
        resp_neg = api_client.post(
            "/v1/rlcd",
            json={"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [bad_lbl, 0]},
        )
        assert resp_neg.status_code == 422
        assert "must be a non-negative integer" in resp_neg.text

    # 4. Out-of-range labels on /v1/calibrate
    resp_cal_oor = api_client.post(
        "/v1/calibrate",
        json={"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 2]},  # K=2, valid is 0..1
    )
    assert resp_cal_oor.status_code == 422
    assert "Each label must be in 0..1" in resp_cal_oor.text

    mem_after = proc.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before
    assert mem_delta < 50.0, f"Memory spiked by {mem_delta:.2f} MB during OOM fuzzing!"


# =============================================================================
# 3. Dimension Boundaries (N=10,001, d_feat=1,025, K=257)
# =============================================================================


def test_fuzz_dimension_boundaries_rejections(api_client: TestClient) -> None:
    """Verify upper dimension limits are strictly enforced with HTTP 422."""
    # N = 10,001
    resp_n = api_client.post(
        "/v1/rlcd",
        json={"features": [[0.1, 0.2]] * 10001, "labels": [0, 1] * 5000 + [0]},
    )
    assert resp_n.status_code == 422
    assert "Sample count N=10001 exceeds maximum limit of 10,000" in resp_n.text

    # d_feat = 1,025
    resp_d = api_client.post(
        "/v1/rlcd",
        json={"features": [[0.1] * 1025, [0.2] * 1025], "labels": [0, 1]},
    )
    assert resp_d.status_code == 422
    assert "Feature dimension d_feat=1025 exceeds maximum limit of 1,024" in resp_d.text

    # K = 257 on /v1/calibrate
    resp_k = api_client.post(
        "/v1/calibrate",
        json={"logits": [[0.1] * 257, [0.2] * 257], "labels": [0, 1]},
    )
    assert resp_k.status_code == 422
    assert "Number of classes K=257 exceeds maximum limit of 256" in resp_k.text


def test_dimension_exact_upper_bounds_acceptance() -> None:
    """Verify exact valid boundaries (N=10,000, d_feat=1,024, K=256, max label=255) pass validation."""
    # 1. N = 10,000
    req_n = StudioRLCDDemoRequest(
        features=[[0.1, 0.2]] * 10000,
        labels=[0, 1] * 5000,
    )
    assert len(req_n.features) == 10000

    # 2. d_feat = 1,024
    req_d = StudioRLCDDemoRequest(
        features=[[0.1] * 1024, [0.2] * 1024],
        labels=[0, 1],
    )
    assert len(req_d.features[0]) == 1024

    # 3. K = 256
    req_k = CalibrateRequest(
        logits=[[0.1] * 256, [0.2] * 256],
        labels=[0, 255],
    )
    assert len(req_k.logits[0]) == 256

    # 4. max label = 255 (K=256)
    req_lbl = StudioRLCDDemoRequest(
        features=[[0.1, 0.2], [0.3, 0.4]],
        labels=[0, 255],
    )
    assert max(req_lbl.labels) == 255


# =============================================================================
# 4. Malformed Structures (1D, 3D, Ragged, Strings, None, Degenerate)
# =============================================================================


@pytest.mark.parametrize(
    ("name", "endpoint", "payload", "expected_fragment"),
    [
        ("1D_features", "/v1/rlcd", {"features": [1.0, 2.0], "labels": [0, 1]}, "valid list"),
        ("3D_features", "/v1/rlcd", {"features": [[[1.0]], [[2.0]]], "labels": [0, 1]}, "Input should be a valid number"),
        ("ragged_features", "/v1/rlcd", {"features": [[1.0, 2.0], [1.0, 2.0, 3.0]], "labels": [0, 1]}, "expected 2"),
        ("string_in_features", "/v1/rlcd", {"features": [["invalid", 2.0], [3.0, 4.0]], "labels": [0, 1]}, "valid number"),
        ("string_in_labels", "/v1/rlcd", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": ["zero", 1]}, "valid integer"),
        ("none_in_features", "/v1/rlcd", {"features": [[None, 2.0], [3.0, 4.0]], "labels": [0, 1]}, "valid number"),
        ("none_in_labels", "/v1/rlcd", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [None, 1]}, "valid integer"),
        ("empty_features", "/v1/rlcd", {"features": [], "labels": []}, "non-empty lists"),
        ("count_mismatch", "/v1/rlcd", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0]}, "Sample count mismatch"),
        ("single_class", "/v1/rlcd", {"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 0]}, "at least 2 distinct classes"),
        ("1D_logits", "/v1/calibrate", {"logits": [1.0, 2.0], "labels": [0, 1]}, "Input should be a valid list"),
        ("3D_logits", "/v1/calibrate", {"logits": [[[1.0, 2.0]], [[3.0, 4.0]]], "labels": [0, 1]}, "valid number"),
        ("ragged_logits", "/v1/calibrate", {"logits": [[1.0, 2.0], [1.0, 2.0, 3.0]], "labels": [0, 1]}, "identical cardinality K=2"),
        ("string_in_logits", "/v1/calibrate", {"logits": [["invalid", 2.0], [3.0, 4.0]], "labels": [0, 1]}, "valid number"),
        ("none_in_logits", "/v1/calibrate", {"logits": [[None, 2.0], [3.0, 4.0]], "labels": [0, 1]}, "valid number"),
        ("single_class_logits", "/v1/calibrate", {"logits": [[1.0], [2.0]], "labels": [0, 0]}, "at least 2 classes"),
        ("empty_logits", "/v1/calibrate", {"logits": [], "labels": []}, "non-empty lists"),
    ],
)
def test_fuzz_malformed_structures(
    api_client: TestClient,
    name: str,
    endpoint: str,
    payload: dict,
    expected_fragment: str,
) -> None:
    """Verify malformed array structures return HTTP 422 with descriptive validation errors."""
    resp = api_client.post(endpoint, json=payload)
    assert resp.status_code == 422, f"Failed on {name}: expected 422, got {resp.status_code}"
    assert expected_fragment.lower() in resp.text.lower(), f"Failed on {name}: expected '{expected_fragment}' in '{resp.text}'"


# =============================================================================
# 5. Non-Finite Floats (NaN, Infinity, -Infinity)
# =============================================================================


@pytest.mark.parametrize(
    ("name", "endpoint", "raw_body"),
    [
        ("nan_features", "/v1/rlcd", '{"features": [[NaN, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
        ("inf_features", "/v1/rlcd", '{"features": [[Infinity, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
        ("neginf_features", "/v1/rlcd", '{"features": [[-Infinity, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
        ("nan_logits", "/v1/calibrate", '{"logits": [[NaN, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
        ("inf_logits", "/v1/calibrate", '{"logits": [[Infinity, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
        ("neginf_logits", "/v1/calibrate", '{"logits": [[-Infinity, 2.0], [3.0, 4.0]], "labels": [0, 1]}'),
    ],
)
def test_fuzz_non_finite_floats_rejected(
    api_client: TestClient,
    name: str,
    endpoint: str,
    raw_body: str,
) -> None:
    """Verify NaN, Infinity, and -Infinity are rejected with HTTP 422."""
    resp = api_client.post(endpoint, content=raw_body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422, f"Failed on {name}: expected 422, got {resp.status_code}"
    assert "Non-finite numbers (NaN or Infinity)" in resp.text


# =============================================================================
# 6. MCP systemone_calibrate Fuzzing Matrix
# =============================================================================


@pytest.mark.parametrize(
    ("case_name", "args"),
    [
        ("bool_in_labels", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [True, False]}),
        ("bool_in_logits", {"logits": [[True, 2.0], [3.0, 4.0]], "labels": [0, 1]}),
        ("extreme_label_1.6B", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1600000000]}),
        ("extreme_label_256", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 256]}),
        ("negative_label", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [-1, 0]}),
        ("label_out_of_range", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 2]}),
        ("N_10001", {"logits": [[0.1, 0.2]] * 10001, "labels": [0, 1] * 5000 + [0]}),
        ("K_257", {"logits": [[0.1] * 257, [0.2] * 257], "labels": [0, 1]}),
        ("1D_logits", {"logits": [1.0, 2.0], "labels": [0, 1]}),
        ("3D_logits", {"logits": [[[1.0, 2.0]], [[3.0, 4.0]]], "labels": [0, 1]}),
        ("ragged_logits", {"logits": [[1.0, 2.0], [1.0, 2.0, 3.0]], "labels": [0, 1]}),
        ("string_in_logits", {"logits": [["a", 2.0], [3.0, 4.0]], "labels": [0, 1]}),
        ("string_in_labels", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": ["zero", 1]}),
        ("none_in_logits", {"logits": [[None, 2.0], [3.0, 4.0]], "labels": [0, 1]}),
        ("none_in_labels", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [None, 1]}),
        ("nan_in_logits", {"logits": [[float("nan"), 2.0], [3.0, 4.0]], "labels": [0, 1]}),
        ("inf_in_logits", {"logits": [[float("inf"), 2.0], [3.0, 4.0]], "labels": [0, 1]}),
        ("empty_logits", {"logits": [], "labels": []}),
        ("mismatch_count", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0]}),
        ("single_class", {"logits": [[1.0], [2.0]], "labels": [0, 0]}),
        ("bool_alpha", {"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1], "alpha": True}),
    ],
)
def test_fuzz_mcp_calibrate_handle_rpc(mcp_server: MicroLLMMCPServer, case_name: str, args: dict) -> None:
    """Verify MCP systemone_calibrate tool returns isError: True on invalid payloads, never raising unhandled exceptions."""
    rpc_req = {
        "jsonrpc": "2.0",
        "id": 4200,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": args,
        },
    }
    resp = mcp_server.handle_rpc(rpc_req)
    assert resp is not None
    assert "result" in resp
    res = resp["result"]
    assert res.get("isError") is True, f"Failed on {case_name}: expected isError=True, got {res}"
    assert len(res.get("content", [])) > 0
    err_text = res["content"][0].get("text", "")
    assert "validation error" in err_text.lower() or "error" in err_text.lower()


def test_fuzz_mcp_calibrate_direct_call_tool(mcp_server: MicroLLMMCPServer) -> None:
    """Verify direct _call_tool invocations safely return error dictionaries for invalid or malformed arguments."""
    # 1. Invalid payload via dict
    out = mcp_server._call_tool("systemone_calibrate", {"logits": [[1.0, 2.0]], "labels": [999]})
    assert out.get("isError") is True
    assert "Validation Error" in out["content"][0]["text"]

    # 2. String arguments
    out_str = mcp_server._call_tool("systemone_calibrate", "not_a_dict")
    assert out_str.get("isError") is True
    assert "Validation Error" in out_str["content"][0]["text"]

    # 3. Via handle_rpc with non-dict arguments (e.g. integer or list)
    for bad_args in [12345, [1, 2, 3]]:
        resp = mcp_server.handle_rpc({
            "jsonrpc": "2.0",
            "id": 4201,
            "method": "tools/call",
            "params": {"name": "systemone_calibrate", "arguments": bad_args},
        })
        assert resp is not None
        assert resp["result"]["isError"] is True
