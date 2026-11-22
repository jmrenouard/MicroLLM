"""Drop-in MCP JSON-RPC 2.0 pre-execution guard middleware and stdio proxy (`microllm.guard.v1`).

Screens every `tools/call` frame offline on local CPU in under 5 ms before forwarding
to any upstream MCP server, with zero third-party MCP SDK dependencies and zero `shell=True` calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence, TextIO

from pydantic import ValidationError

from microllm.guard import (
    GATEKEEPER_SCHEMA_VERSION,
    EnvironmentType,
    GatekeeperDecision,
    GatekeeperRequest,
    GuardConfig,
    MicroLLMGuard,
)

_COMMAND_ARGUMENT_KEYS: tuple[str, ...] = (
    "command",
    "cmd",
    "query",
    "sql",
    "script",
    "code",
)


def _is_notification(request: Mapping[str, Any]) -> bool:
    has_id = "id" in request
    method = str(request.get("method", "")).strip()
    return (not has_id) or (method.startswith("notifications/") and request.get("id") is None)


def extract_mcp_tool_request(
    params: Mapping[str, Any],
    *,
    environment: EnvironmentType | None = None,
) -> GatekeeperRequest:
    """Convert MCP `tools/call` params (`{name, arguments}`) into a `GatekeeperRequest`."""
    raw_name = params.get("name")
    tool_name = str(raw_name).strip() if raw_name is not None and str(raw_name).strip() else "mcp_tool"
    raw_args = params.get("arguments")

    command: str | None = None
    arguments: dict[str, Any] | list[Any] | str | None = None
    if isinstance(raw_args, Mapping):
        args_dict = dict(raw_args)
        for key in _COMMAND_ARGUMENT_KEYS:
            val = args_dict.get(key)
            if isinstance(val, str) and val.strip():
                command = val
                remaining = {k: v for k, v in args_dict.items() if k != key}
                arguments = remaining if remaining else None
                break
        else:
            arguments = args_dict
    elif isinstance(raw_args, (list, str)):
        arguments = raw_args
    elif raw_args is not None:
        arguments = str(raw_args)

    return GatekeeperRequest.model_validate(
        {
            "tool_name": tool_name,
            "command": command,
            "arguments": arguments,
            "environment": environment,
        }
    )


def _format_mcp_blocked_response(
    *,
    jsonrpc: str,
    req_id: Any,
    decision: GatekeeperDecision,
) -> dict[str, Any]:
    payload = decision.to_dict()
    return {
        "jsonrpc": jsonrpc,
        "id": req_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                }
            ],
            "structuredContent": payload,
            "isError": True,
        },
    }


def _format_mcp_validation_error_response(
    *,
    jsonrpc: str,
    req_id: Any,
    message: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": GATEKEEPER_SCHEMA_VERSION,
        "decision": "block",
        "confidence": 1.0,
        "abstain": False,
        "reason": f"Blocked invalid MCP tool call payload: {message}",
    }
    return {
        "jsonrpc": jsonrpc,
        "id": req_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                }
            ],
            "structuredContent": payload,
            "isError": True,
        },
    }


class MCPGuardMiddleware:
    """In-process JSON-RPC 2.0 middleware screening MCP `tools/call` frames with `MicroLLMGuard`."""

    def __init__(
        self,
        upstream: Any,
        guard: MicroLLMGuard | None = None,
        *,
        config: GuardConfig | None = None,
        block_on_escalate: bool = True,
        exempt_tools: Sequence[str] | None = None,
    ) -> None:
        if not (
            callable(upstream)
            or hasattr(upstream, "handle_rpc")
            or hasattr(upstream, "handle_jsonrpc")
        ):
            raise TypeError(
                "upstream must be callable or expose 'handle_rpc' / 'handle_jsonrpc'."
            )
        self.upstream = upstream
        self.guard = guard or MicroLLMGuard(config=config)
        self.block_on_escalate = bool(block_on_escalate)
        self.exempt_tools: frozenset[str] = frozenset(
            str(name).strip() for name in (exempt_tools or ()) if str(name).strip()
        )
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "total_tool_calls": 0,
            "allowed": 0,
            "blocked": 0,
            "escalated": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Return a snapshot of intercepted MCP tool call counters."""
        with self._stats_lock:
            return dict(self._stats)

    def _forward_single(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        if hasattr(self.upstream, "handle_rpc"):
            return self.upstream.handle_rpc(request)
        if hasattr(self.upstream, "handle_jsonrpc"):
            resp = self.upstream.handle_jsonrpc(request)
            return resp if isinstance(resp, dict) or resp is None else None
        handler: Callable[[Mapping[str, Any]], dict[str, Any] | None] = self.upstream
        return handler(request)

    def handle_rpc(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Screen a single JSON-RPC 2.0 MCP message and forward safe calls upstream."""
        if not isinstance(request, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object."},
            }

        jsonrpc = str(request.get("jsonrpc", "2.0"))
        req_id = request.get("id")
        method = str(request.get("method", "")).strip()
        is_notif = _is_notification(request)

        if method != "tools/call":
            return self._forward_single(request)

        params = request.get("params")
        if not isinstance(params, Mapping):
            if is_notif:
                return None
            return {
                "jsonrpc": jsonrpc,
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: expected object."},
            }

        tool_name = str(params.get("name", "")).strip()
        if tool_name and tool_name in self.exempt_tools:
            return self._forward_single(request)

        try:
            guard_req = extract_mcp_tool_request(params)
            decision = self.guard.evaluate(guard_req)
        except (ValidationError, ValueError, TypeError) as exc:
            with self._stats_lock:
                self._stats["total_tool_calls"] += 1
                self._stats["blocked"] += 1
            if is_notif:
                return None
            return _format_mcp_validation_error_response(
                jsonrpc=jsonrpc,
                req_id=req_id,
                message=str(exc),
            )

        with self._stats_lock:
            self._stats["total_tool_calls"] += 1
            if decision.decision == "allow":
                self._stats["allowed"] += 1
            elif decision.decision == "block":
                self._stats["blocked"] += 1
            else:
                self._stats["escalated"] += 1

        should_short_circuit = decision.decision == "block" or (
            decision.decision == "escalate" and self.block_on_escalate
        )
        if should_short_circuit:
            if is_notif:
                return None
            return _format_mcp_blocked_response(
                jsonrpc=jsonrpc,
                req_id=req_id,
                decision=decision,
            )

        return self._forward_single(request)

    def handle_jsonrpc(
        self,
        payload: Mapping[str, Any] | list[Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Process a single JSON-RPC 2.0 object or batch array with pre-execution screening."""
        if isinstance(payload, list):
            if not payload:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: empty batch array."},
                }
            responses: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, Mapping):
                    responses.append(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request in batch."},
                        }
                    )
                else:
                    res = self.handle_rpc(item)
                    if res is not None:
                        responses.append(res)
            return responses if responses else None

        if not isinstance(payload, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected object or array."},
            }
        return self.handle_rpc(payload)


class _StdioSubprocessMCPUpstream:
    """Line-delimited JSON-RPC 2.0 bridge forwarding approved frames to a child MCP stdio process."""

    def __init__(self, proc: subprocess.Popen[str], client_stdout: TextIO | None = None) -> None:
        self._proc = proc
        self._client_stdout = client_stdout

    def handle_rpc(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        if self._proc.stdin is None or self._proc.stdout is None:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": "Upstream MCP stdio pipes are unavailable."},
            }

        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            if _is_notification(request):
                return None
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": f"Failed writing to upstream MCP server: {exc}"},
            }

        if _is_notification(request):
            return None

        while True:
            raw_line = self._proc.stdout.readline()
            if not raw_line:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32603, "message": "Upstream MCP server closed stdout unexpectedly."},
                }
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32603, "message": f"Malformed JSON from upstream MCP server: {exc}"},
                }
            if isinstance(parsed, dict):
                # Pass through upstream server-initiated notifications without desynchronizing request IDs
                if "id" not in parsed and "method" in parsed:
                    if self._client_stdout is not None:
                        self._client_stdout.write(
                            json.dumps(parsed, ensure_ascii=False) + "\n"
                        )
                        self._client_stdout.flush()
                    continue
                return parsed
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": "Unexpected non-object response from upstream MCP server."},
            }


def _reject_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant '{token}' is not permitted.")


def run_mcp_guard_proxy(
    upstream_command: Sequence[str],
    *,
    guard: MicroLLMGuard | None = None,
    config: GuardConfig | None = None,
    block_on_escalate: bool = True,
    stdin_stream: TextIO | None = None,
    stdout_stream: TextIO | None = None,
    stderr_stream: TextIO | None = None,
) -> int:
    """Run a transparent JSON-RPC 2.0 MCP stdio guard proxy wrapping an upstream MCP command.

    Never uses `shell=True`; passes `upstream_command` strictly as an argv list to `subprocess.Popen`.
    """
    cmd_list = [str(arg) for arg in upstream_command if str(arg) != ""]
    if not cmd_list:
        raise ValueError("upstream_command must be a non-empty list of command arguments.")

    in_stream = sys.stdin if stdin_stream is None else stdin_stream
    out_stream = sys.stdout if stdout_stream is None else stdout_stream
    err_stream = sys.stderr if stderr_stream is None else stderr_stream

    # Forward upstream stderr only if err_stream has a real OS file descriptor; otherwise redirect to DEVNULL/PIPE
    sub_stderr: Any = subprocess.PIPE
    try:
        err_stream.fileno()
        sub_stderr = err_stream
    except Exception:
        sub_stderr = subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sub_stderr,
        text=True,
        encoding="utf-8",
        bufsize=1,
        shell=False,
    )

    upstream = _StdioSubprocessMCPUpstream(proc, client_stdout=out_stream)
    middleware = MCPGuardMiddleware(
        upstream=upstream,
        guard=guard,
        config=config,
        block_on_escalate=block_on_escalate,
    )

    try:
        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line, parse_constant=_reject_non_finite_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                out_stream.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
                out_stream.flush()
                continue

            resp = middleware.handle_jsonrpc(req)
            if resp is not None:
                out_stream.write(json.dumps(resp, ensure_ascii=False, allow_nan=False) + "\n")
                out_stream.flush()
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

    return 0


__all__ = [
    "MCPGuardMiddleware",
    "extract_mcp_tool_request",
    "run_mcp_guard_proxy",
]
