"""TDD tests for `POST /v1/guard/evaluate`, `GET /v1/guard/contract`, SDK methods, and `microllm-guard` CLI (Issue #279)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from microllm import AsyncMicroJevClient, MicroJevClient, create_app
from microllm.cli import run_cli
from microllm.guard import GATEKEEPER_SCHEMA_VERSION
from microllm.guard_cli import (
    EXIT_ALLOW,
    EXIT_BLOCK,
    EXIT_ESCALATE,
    EXIT_INVALID_INPUT,
    run_guard_cli,
)


def test_http_guard_contract_and_evaluate_endpoints() -> None:
    """Verify GET /v1/guard/contract and POST /v1/guard/evaluate enforce microllm.guard.v1 contract and metrics."""
    app = create_app()
    client = TestClient(app)

    contract_resp = client.get("/v1/guard/contract")
    assert contract_resp.status_code == 200
    contract = contract_resp.json()
    assert contract["properties"]["schema_version"]["const"] == GATEKEEPER_SCHEMA_VERSION
    assert set(contract["required"]) == {"schema_version", "decision", "confidence", "abstain", "reason"}

    # Allow
    r_allow = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "git status"},
    )
    assert r_allow.status_code == 200
    d_allow = r_allow.json()
    assert d_allow["schema_version"] == "microllm.guard.v1"
    assert d_allow["decision"] == "allow"
    assert d_allow["abstain"] is False
    assert d_allow["latency_ms"] < 50.0

    # Block
    r_block = client.post(
        "/v1/guard/evaluate",
        json={"tool": "sql", "command": "DROP TABLE users;"},
    )
    assert r_block.status_code == 200
    d_block = r_block.json()
    assert d_block["decision"] == "block"
    assert d_block["abstain"] is False

    # Escalate (conformal abstention)
    r_esc = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "kubectl apply -f deploy/prod.yaml"},
    )
    assert r_esc.status_code == 200
    d_esc = r_esc.json()
    assert d_esc["decision"] == "escalate"
    assert d_esc["abstain"] is True

    # Metrics updated accurately
    r_metrics = client.get("/v1/metrics")
    assert r_metrics.status_code == 200
    metrics = r_metrics.json()
    assert metrics["guard_requests"] == 3
    assert metrics["guard_blocked"] == 1
    assert metrics["guard_escalated"] == 1
    assert metrics["abstained_decisions"] >= 1


def test_http_guard_evaluate_auth_and_422_validation() -> None:
    """Verify HTTP 401 auth enforcement and HTTP 422 loc[0] == 'body' validation (zero 5xx)."""
    app = create_app(api_key="super-secret-guard-key")
    client = TestClient(app)

    # Missing Bearer header -> 401
    r_unauth = client.post("/v1/guard/evaluate", json={"tool_name": "bash", "command": "ls"})
    assert r_unauth.status_code == 401

    # Wrong Bearer token -> 401
    r_bad = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "ls"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r_bad.status_code == 401

    headers = {"Authorization": "Bearer super-secret-guard-key"}

    # Authorized valid request -> 200
    r_ok = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "git status"},
        headers=headers,
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["decision"] == "allow"

    # Malformed policy_boundaries -> 422 with loc[0] == "body"
    r_422 = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "ls", "policy_boundaries": {"allowed": "not-a-list"}},
        headers=headers,
    )
    assert r_422.status_code == 422
    detail = r_422.json()["detail"]
    assert isinstance(detail, list) and detail
    assert detail[0]["loc"][0] == "body"

    # Non-finite JSON (NaN) -> 422 (never 500)
    r_nan = client.post(
        "/v1/guard/evaluate",
        content='{"tool_name": "bash", "command": "ls", "threshold": NaN}',
        headers={**headers, "Content-Type": "application/json"},
    )
    assert r_nan.status_code == 422
    assert r_nan.json()["detail"][0]["loc"][0] == "body"

    # Unknown extra field -> 422
    r_extra = client.post(
        "/v1/guard/evaluate",
        json={"tool_name": "bash", "command": "ls", "unexpected_key": 123},
        headers=headers,
    )
    assert r_extra.status_code == 422


def test_sync_and_async_sdk_guard_evaluate_and_contract() -> None:
    """Verify MicroJevClient and AsyncMicroJevClient guard_evaluate and guard_contract methods."""
    app = create_app()
    tc = TestClient(app)
    sync_client = MicroJevClient(http_client=tc)

    schema = sync_client.guard_contract()
    assert schema["properties"]["schema_version"]["const"] == "microllm.guard.v1"

    res_allow = sync_client.guard_evaluate(
        tool_name="bash",
        command="rm -rf ./dist",
        policy_boundaries={"allowed": ["rm -rf ./dist"]},
    )
    assert res_allow["schema_version"] == "microllm.guard.v1"
    assert res_allow["decision"] == "allow"
    assert res_allow["abstain"] is False

    async def _run_async() -> tuple[dict[str, object], dict[str, object]]:
        ac = AsyncMicroJevClient(http_client=tc)
        c = await ac.guard_contract()
        d = await ac.guard_evaluate(
            tool_name="bash",
            command="kubectl apply -f deploy/prod.yaml",
            abstain_policy="block",
        )
        return c, d

    async_schema, async_dec = asyncio.run(_run_async())
    assert async_schema["properties"]["schema_version"]["const"] == "microllm.guard.v1"
    assert async_dec["decision"] == "block"
    assert async_dec["abstain"] is True

    with pytest.raises(ValueError):
        sync_client.guard_evaluate("bash", "ls", policy_boundaries={"allowed": "ls"})  # type: ignore[dict-item]


def test_guard_cli_flags_stdin_policy_file_and_exit_codes(tmp_path: Path) -> None:
    """Verify microllm-guard CLI exit codes (0/2/3/1), strict --json output, stdin piping, and policy files."""
    # 1. Allow via CLI flags -> exit 0, single clean JSON object on stdout
    out_allow = io.StringIO()
    err_allow = io.StringIO()
    code_allow = run_guard_cli(
        ["--tool", "bash", "--command", "git status", "--json"],
        stdin_stream=io.StringIO(""),
        stdout_stream=out_allow,
        stderr_stream=err_allow,
    )
    assert code_allow == EXIT_ALLOW == 0
    assert err_allow.getvalue() == ""
    lines = [ln for ln in out_allow.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed_allow = json.loads(lines[0])
    assert parsed_allow["schema_version"] == "microllm.guard.v1"
    assert parsed_allow["decision"] == "allow"
    assert parsed_allow["abstain"] is False

    # 2. Block via stdin JSON -> exit 2
    out_block = io.StringIO()
    code_block = run_guard_cli(
        ["--json"],
        stdin_stream=io.StringIO(json.dumps({"tool_name": "bash", "command": "rm -rf /"})),
        stdout_stream=out_block,
        stderr_stream=io.StringIO(),
    )
    assert code_block == EXIT_BLOCK == 2
    parsed_block = json.loads(out_block.getvalue())
    assert parsed_block["decision"] == "block"

    # 3. Escalate via --args-json -> exit 3
    out_esc = io.StringIO()
    code_esc = run_guard_cli(
        ["--tool", "bash", "--args-json", '{"cmd": "kubectl apply -f deploy/prod.yaml"}', "--json"],
        stdin_stream=io.StringIO(""),
        stdout_stream=out_esc,
        stderr_stream=io.StringIO(),
    )
    assert code_esc == EXIT_ESCALATE == 3
    parsed_esc = json.loads(out_esc.getvalue())
    assert parsed_esc["decision"] == "escalate"
    assert parsed_esc["abstain"] is True

    # 4. --fail-closed maps abstention to block -> exit 2
    out_fc = io.StringIO()
    code_fc = run_guard_cli(
        ["--tool", "bash", "--command", "kubectl apply -f deploy/prod.yaml", "--fail-closed", "--json"],
        stdin_stream=io.StringIO(""),
        stdout_stream=out_fc,
        stderr_stream=io.StringIO(),
    )
    assert code_fc == EXIT_BLOCK == 2
    parsed_fc = json.loads(out_fc.getvalue())
    assert parsed_fc["decision"] == "block"
    assert parsed_fc["abstain"] is True

    # 5. --policy-file allows scoped cleanup -> exit 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"allowed": ["rm -rf ./dist"], "forbidden": ["rm -rf /var/lib/postgresql"]}),
        encoding="utf-8",
    )
    out_pol = io.StringIO()
    code_pol = run_guard_cli(
        ["--tool", "bash", "--command", "rm -rf ./dist", "--policy-file", str(policy_path), "--json"],
        stdin_stream=io.StringIO(""),
        stdout_stream=out_pol,
        stderr_stream=io.StringIO(),
    )
    assert code_pol == EXIT_ALLOW == 0
    assert json.loads(out_pol.getvalue())["decision"] == "allow"

    # 6. --contract prints schema and exits 0
    out_contract = io.StringIO()
    code_contract = run_guard_cli(
        ["--contract"],
        stdin_stream=io.StringIO(""),
        stdout_stream=out_contract,
        stderr_stream=io.StringIO(),
    )
    assert code_contract == EXIT_ALLOW == 0
    assert json.loads(out_contract.getvalue())["properties"]["schema_version"]["const"] == "microllm.guard.v1"

    # 7. Invalid JSON, NaN, missing policy file, empty input, and CLI flag typos -> exit 1 (never 2!)
    for bad_argv, bad_stdin in [
        (["--json"], "{invalid json"),
        (["--json"], '{"tool_name": "bash", "threshold": NaN}'),
        (["--json"], ""),
        (["--tool", "bash", "--command", "ls", "--policy-file", str(tmp_path / "missing.json"), "--json"], ""),
        (["--unknown-flag", "--json"], ""),
    ]:
        out_err = io.StringIO()
        err_buf = io.StringIO()
        ec = run_guard_cli(
            bad_argv,
            stdin_stream=io.StringIO(bad_stdin),
            stdout_stream=out_err,
            stderr_stream=err_buf,
        )
        assert ec == EXIT_INVALID_INPUT == 1, f"Expected exit 1 for {bad_argv}/{bad_stdin}, got {ec}"
        parsed_err = json.loads(out_err.getvalue())
        assert "error" in parsed_err


def test_microllm_guard_subcommand_and_subprocess_execution() -> None:
    """Verify `microllm guard` subcommand and `python -m microllm.guard_cli` subprocess invocation."""
    assert run_cli(["guard", "--contract"]) == 0

    proc = subprocess.run(
        [sys.executable, "-m", "microllm.guard_cli", "--json"],
        input=json.dumps({"tool_name": "bash", "command": "rm -rf /"}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == EXIT_BLOCK == 2
    data = json.loads(proc.stdout.strip())
    assert data["schema_version"] == "microllm.guard.v1"
    assert data["decision"] == "block"
