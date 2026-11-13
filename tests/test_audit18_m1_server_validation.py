"""Regression test suite for Milestone 1: Server, API Security & Input Validation.

Covers Audit Points 1, 2, 3, 4, 5 (Features F01, F02, F03, F04, F05):
- Point 1 (F01): Dockerfile cache directory creation, ownership, and MICROLLM_CACHE_DIR environment variable.
- Point 2 (F02): Server path resolution fallbacks outside git repository checkouts (docs, slides, assets, papers, benchmarks).
- Point 3 (F03): Secure profile pull path (removal of client-controlled cache_dir).
- Point 4 (F04): Memory OOM bounds and boolean rejection in /v1/rlcd and /v1/calibrate.
- Point 5 (F05): Input schema and array validation in MCP systemone_calibrate tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from microllm.mcp_server import MicroLLMMCPServer
from microllm.server import (
    create_app,
    evaluate_calibration_in_process,
    resolve_assets_dir,
    resolve_benchmark_path,
    resolve_docs_file,
    resolve_papers_dir,
    resolve_repo_root,
    run_rlcd_experiment_in_process,
)
from microllm.server_models import (
    CalibrateRequest,
    ProfilePullRequest,
    StudioRLCDDemoRequest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Point 1 (F01): Docker cache directory, environment variable, and ownership
# =============================================================================


def test_dockerfile_cache_permissions_and_env() -> None:
    """Verify Dockerfile sets MICROLLM_CACHE_DIR=/app/.cache and grants appuser ownership before USER switch."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.is_file(), "Dockerfile must exist at repository root."

    content = dockerfile_path.read_text(encoding="utf-8")

    assert "ENV PYTHONUNBUFFERED=1" in content
    assert "MICROLLM_CACHE_DIR=/app/.cache" in content, "MICROLLM_CACHE_DIR must be defined in ENV."
    assert "mkdir -p /app/.cache" in content, "/app/.cache directory must be created during build."
    assert "chown -R appuser:appgroup /app" in content, "/app must be owned by appuser:appgroup."

    chown_pos = content.find("chown -R appuser:appgroup /app")
    user_pos = content.find("USER appuser")
    assert chown_pos != -1, "chown command not found."
    assert user_pos != -1, "USER appuser not found."
    assert chown_pos < user_pos, "chown must execute before switching to USER appuser."


# =============================================================================
# Point 2 (F02): Path resolution fallbacks outside git repository
# =============================================================================


def test_path_resolution_in_git_checkout() -> None:
    """Verify path resolution helpers locate repository directories during development."""
    repo = resolve_repo_root()
    assert repo is not None, "Expected resolve_repo_root() to detect repo root in checkout."
    assert (repo / "pyproject.toml").is_file()

    bench_p = resolve_benchmark_path("test_bench.json")
    assert bench_p.parent == repo / "benchmarks"

    doc_p = resolve_docs_file("index.html")
    assert doc_p is not None and doc_p.is_file()

    assets_dir = resolve_assets_dir()
    assert assets_dir is not None and assets_dir.is_dir()

    papers_dir = resolve_papers_dir()
    assert papers_dir is not None and papers_dir.is_dir()


def test_path_resolution_fallback_outside_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify path resolution gracefully falls back to cache dir and returns fallbacks outside git repo."""
    fake_cache = tmp_path / "mock_cache"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(fake_cache))

    with patch("microllm.server.resolve_repo_root", return_value=None):
        # 1. Benchmarks fallback to get_cache_dir() / "benchmarks"
        bench_p = resolve_benchmark_path("custom_bench.json")
        assert bench_p.parent == fake_cache / "benchmarks"
        assert bench_p.parent.is_dir()

        # 2. Docs, assets, papers fallbacks when files do not exist
        missing_doc = resolve_docs_file("nonexistent_doc.html")
        assert missing_doc is None

        # 3. HTTP server responses outside git repo
        app = create_app()
        client = TestClient(app)

        with patch("microllm.server.resolve_docs_file", return_value=None):
            resp_doc = client.get("/documentation")
            assert resp_doc.status_code == 200
            assert "<h1>MicroLLM Documentation</h1>" in resp_doc.text

            resp_slides = client.get("/slides")
            assert resp_slides.status_code == 200
            assert "<h1>MicroLLM LightPath Presentation Slides</h1>" in resp_slides.text

        with patch("microllm.server.resolve_assets_dir", return_value=None):
            resp_asset = client.get("/assets/missing.svg")
            assert resp_asset.status_code == 404
            assert resp_asset.json()["detail"] == "SVG asset not found."

        with patch("microllm.server.resolve_papers_dir", return_value=None):
            resp_paper = client.get("/papers/missing.pdf")
            assert resp_paper.status_code == 404
            assert resp_paper.json()["detail"] == "PDF paper not found."


# =============================================================================
# Point 3 (F03): Secure profile pull path (removal of client cache_dir)
# =============================================================================


def test_profile_pull_request_schema_has_no_cache_dir() -> None:
    """Verify ProfilePullRequest model does not contain client-controlled cache_dir."""
    fields = ProfilePullRequest.model_fields
    assert "cache_dir" not in fields, "cache_dir must be removed from ProfilePullRequest schema."
    assert "profile" in fields
    assert "download_weights" in fields
    assert "from_hub" in fields


def test_post_profiles_pull_ignores_client_path_and_uses_cache_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify POST /v1/profiles/pull writes to server cache and cannot be redirected via request body."""
    server_cache = tmp_path / "server_cache"
    unauthorized_dir = tmp_path / "unauthorized_arbitrary_dir"
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(server_cache))

    app = create_app()
    client = TestClient(app)

    # Sending cache_dir in JSON should be ignored (or discarded by schema) and NOT create unauthorized_dir
    resp = client.post(
        "/v1/profiles/pull",
        json={
            "profile": "fast-cpu-int8",
            "cache_dir": str(unauthorized_dir),
            "download_weights": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["profile"] == "fast-cpu-int8"
    assert data["installed"] is True

    # Confirm installation occurred under server_cache and NOT unauthorized_dir
    assert not unauthorized_dir.exists(), "Arbitrary client path must not be created."
    assert (server_cache / "fast-cpu-int8" / "profile_manifest.json").is_file()


# =============================================================================
# Point 4 (F04): Memory OOM bounds and boolean rejection
# =============================================================================


def test_rlcd_and_calibrate_reject_boolean_types() -> None:
    """Verify booleans in labels, features, or logits raise ValueError/ValidationError."""
    # 1. StudioRLCDDemoRequest rejects bools in labels and features
    with pytest.raises(ValidationError) as exc1:
        StudioRLCDDemoRequest(
            features=[[1.0, 2.0], [3.0, 4.0]],
            labels=[True, 0],  # type: ignore[list-item]
        )
    assert "cannot be a boolean" in str(exc1.value)

    with pytest.raises(ValidationError) as exc2:
        StudioRLCDDemoRequest(
            features=[[True, 2.0], [3.0, 4.0]],  # type: ignore[list-item]
            labels=[1, 0],
        )
    assert "cannot be a boolean" in str(exc2.value)

    # 2. CalibrateRequest rejects bools in labels and logits
    with pytest.raises(ValidationError) as exc3:
        CalibrateRequest(
            logits=[[1.0, 2.0], [3.0, 4.0]],
            labels=[True, 0],  # type: ignore[list-item]
        )
    assert "cannot be a boolean" in str(exc3.value)

    with pytest.raises(ValidationError) as exc4:
        CalibrateRequest(
            logits=[[True, 2.0], [3.0, 4.0]],  # type: ignore[list-item]
            labels=[1, 0],
        )
    assert "cannot be a boolean" in str(exc4.value)

    # 3. HTTP Endpoints return HTTP 422 Unprocessable Entity
    app = create_app()
    client = TestClient(app)

    resp_rlcd = client.post(
        "/v1/rlcd",
        json={"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [True, 0]},
    )
    assert resp_rlcd.status_code == 422
    assert "cannot be a boolean" in resp_rlcd.text

    resp_cal = client.post(
        "/v1/calibrate",
        json={"logits": [[1.0, 2.0], [3.0, 4.0]], "labels": [True, 0]},
    )
    assert resp_cal.status_code == 422
    assert "cannot be a boolean" in resp_cal.text


def test_rlcd_and_calibrate_payload_bounds() -> None:
    """Verify sample count N <= 10,000, feature dim <= 1,024, K <= 256, and 0 <= y < K."""
    app = create_app()
    client = TestClient(app)

    # 1. Sample count N > 10,000 rejected
    huge_n_labels = [0, 1] * 5001  # 10,002 samples
    huge_n_features = [[0.1, 0.2]] * 10002
    with pytest.raises(ValidationError) as exc_n:
        StudioRLCDDemoRequest(features=huge_n_features, labels=huge_n_labels)
    assert "Sample count N=10002 exceeds maximum limit of 10,000" in str(exc_n.value)

    # 2. Feature dimension d_feat > 1,024 rejected
    huge_d_features = [[0.1] * 1025, [0.2] * 1025]
    with pytest.raises(ValidationError) as exc_d:
        StudioRLCDDemoRequest(features=huge_d_features, labels=[0, 1])
    assert "Feature dimension d_feat=1025 exceeds maximum limit of 1,024" in str(exc_d.value)

    # 3. Number of classes K > 256 rejected
    huge_k_logits = [[0.0] * 257, [1.0] * 257]
    with pytest.raises(ValidationError) as exc_k:
        CalibrateRequest(logits=huge_k_logits, labels=[0, 1])
    assert "Number of classes K=257 exceeds maximum limit of 256" in str(exc_k.value)

    # 4. Extreme label value (e.g. 1,600,000,000 preventing 12.8 GB OOM)
    with pytest.raises(ValidationError) as exc_oom:
        StudioRLCDDemoRequest(
            features=[[1.0, 2.0], [3.0, 4.0]],
            labels=[0, 1_600_000_000],
        )
    assert "exceeds maximum allowed class index 255" in str(exc_oom.value)

    # 5. Label out of range y >= K in CalibrateRequest
    with pytest.raises(ValidationError) as exc_range:
        CalibrateRequest(
            logits=[[1.0, 2.0], [3.0, 4.0]],  # K=2
            labels=[0, 2],  # y=2 is invalid for K=2 (must be 0..1)
        )
    assert "Each label must be in 0..1" in str(exc_range.value)

    # 6. Verify HTTP 422 response on OOM attack payload
    resp = client.post(
        "/v1/rlcd",
        json={"features": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1_600_000_000]},
    )
    assert resp.status_code == 422
    assert "exceeds maximum allowed class index 255" in resp.text


# =============================================================================
# Point 5 (F05): MCP systemone_calibrate input validation
# =============================================================================


def test_mcp_calibrate_valid_execution() -> None:
    """Verify MCP systemone_calibrate tool succeeds on valid inputs."""
    mcp = MicroLLMMCPServer()
    resp = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "alpha": 0.10,
                "logits": [[4.0, -4.0], [-3.0, 3.0], [5.0, -5.0], [-2.0, 2.0]],
                "labels": [0, 1, 0, 1],
            },
        },
    })
    assert resp is not None
    assert "result" in resp
    assert resp["result"]["isError"] is False
    assert resp["result"]["structuredContent"]["cardinality_k"] == 2
    assert "conformal_q_hat" in resp["result"]["structuredContent"]


def test_mcp_calibrate_catches_validation_errors() -> None:
    """Verify MCP systemone_calibrate tool returns isError: True on invalid payloads instead of crashing."""
    mcp = MicroLLMMCPServer()

    # Case A: Booleans in labels
    resp_bool = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "logits": [[1.0, -1.0], [-1.0, 1.0]],
                "labels": [True, 0],
            },
        },
    })
    assert resp_bool is not None
    res_b = resp_bool["result"]
    assert res_b["isError"] is True
    assert "Validation Error" in res_b["content"][0]["text"]
    assert "cannot be a boolean" in res_b["content"][0]["text"]

    # Case B: Length mismatch between logits and labels
    resp_mismatch = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "logits": [[1.0, -1.0], [-1.0, 1.0]],
                "labels": [0],
            },
        },
    })
    assert resp_mismatch is not None
    res_m = resp_mismatch["result"]
    assert res_m["isError"] is True
    assert "Validation Error" in res_m["content"][0]["text"]
    assert "equal length" in res_m["content"][0]["text"]

    # Case C: Label out of range y >= K
    resp_range = mcp.handle_rpc({
        "jsonrpc": "2.0",
        "id": 104,
        "method": "tools/call",
        "params": {
            "name": "systemone_calibrate",
            "arguments": {
                "logits": [[1.0, -1.0], [-1.0, 1.0]],
                "labels": [0, 99],
            },
        },
    })
    assert resp_range is not None
    res_r = resp_range["result"]
    assert res_r["isError"] is True
    assert "Validation Error" in res_r["content"][0]["text"]
    assert "Each label must be in 0..1" in res_r["content"][0]["text"]

    # Case D: Direct _call_tool returns isError dict
    direct_res = mcp._call_tool(
        "systemone_calibrate",
        {"logits": [[1.0, 2.0]], "labels": [5]},
    )
    assert direct_res.get("isError") is True
    assert "Validation Error" in direct_res["content"][0]["text"]
