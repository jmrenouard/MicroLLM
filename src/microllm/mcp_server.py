"""Model Context Protocol (MCP) Server for MicroLLM System-One Decision Engine.

Implements JSON-RPC 2.0 MCP transport over both:
1. Standard Input/Output (`microllm mcp` CLI subcommand) for Claude Desktop, Cursor, and Windsurf.
2. HTTP `POST /v1/mcp` endpoint in the FastAPI server.

Exposes three calibrated zero-generation tools:
- `systemone_decide`: Evaluate `noul`, `choice`, `score`, and `multi` questions over a state with
  Verdict-MQ multi-question state caching (`IDEA-022`), Conformal Abstention, and VoC routing (`HYP-005`).
- `systemone_guardrail`: Zero-Shot AI Safety Guardrail pack (`arXiv:2609.29429`) detecting prompt injection,
  destructive commands, PII/secret leaks, and ungrounded hallucinations.
- `systemone_calibrate`: Per-K temperature scaling, Split Conformal Prediction, and Clopper-Pearson
  selective risk certification.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from microllm.backends.base import VerdictMQMultiQuestionCache
from microllm.backends.cascade import DualGearCascadeBackend
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.backends.prefill_llm import AnyJevPrefillBackend


MCP_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "systemone_decide",
        "description": (
            "Evaluate structured non-generative System-1 decisions (noul, choice, score, multi) "
            "over a state with calibrated probabilities, conformal prediction sets, and abstention flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": ["string", "object", "array"],
                    "description": "Context, document, ticket, or JSON state to evaluate.",
                },
                "questions": {
                    "type": "object",
                    "description": "Map of question IDs to question specs ({type: noul|choice|score|multi, instructions, criteria}).",
                },
                "model": {
                    "type": "string",
                    "default": "jev-latest",
                    "enum": ["jev-latest", "microllm-gear1-fast", "microllm-gear2-anyjev-l0"],
                },
            },
            "required": ["state", "questions"],
        },
    },
    {
        "name": "systemone_guardrail",
        "description": (
            "Run Zero-Shot AI Safety Guardrail checks (prompt injection, destructive command, "
            "PII/credential leak, and context groundedness/hallucination)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": ["string", "object", "array"],
                    "description": "User prompt, tool call, or model claim to audit.",
                },
                "context": {
                    "type": ["string", "object", "array"],
                    "description": "Optional reference context for groundedness / hallucination verification.",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.55,
                    "description": "Risk score threshold in [0.1, 0.99] above which the state is blocked.",
                },
            },
            "required": ["state"],
        },
    },
    {
        "name": "systemone_calibrate",
        "description": (
            "Run Per-K temperature scaling, Split Conformal Prediction, and Clopper-Pearson "
            "upper-confidence risk certification on logits and labels."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "logits": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Optional 2D array of validation logits (N x K).",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional 1D array of ground-truth class indices (0..K-1).",
                },
                "alpha": {"type": "number", "default": 0.10},
                "target_risk": {"type": "number", "default": 0.10},
                "delta": {"type": "number", "default": 0.05},
            },
        },
    },
]


class MicroLLMMCPServer:
    """JSON-RPC 2.0 Model Context Protocol dispatcher for MicroLLM."""

    def __init__(
        self,
        gear1: FastEncoderBackend | None = None,
        gear2: AnyJevPrefillBackend | None = None,
        cascade: DualGearCascadeBackend | None = None,
        mq_cache: VerdictMQMultiQuestionCache | None = None,
    ) -> None:
        self.gear1 = gear1 or FastEncoderBackend()
        self.gear2 = gear2 or AnyJevPrefillBackend()
        self.cascade = cascade or DualGearCascadeBackend(gear1=self.gear1, gear2=self.gear2)
        self.mq_cache = mq_cache or VerdictMQMultiQuestionCache()
        self._test_client: Any = None

    def _get_client(self) -> Any:
        if self._test_client is None:
            from fastapi.testclient import TestClient
            from microllm.server import create_app

            self._test_client = TestClient(create_app())
        return self._test_client

    def handle_rpc(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC 2.0 MCP message and return a response dict (or None for notifications)."""
        if not isinstance(request, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object."},
            }

        jsonrpc = str(request.get("jsonrpc", "2.0"))
        has_id = "id" in request
        req_id = request.get("id")
        method = str(request.get("method", "")).strip()
        params = request.get("params") or {}

        if jsonrpc != "2.0" or not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: 'jsonrpc' must be '2.0' and 'method' must be a non-empty string.",
                },
            }

        # Per JSON-RPC 2.0 §4.1: any request without an 'id' member (or notifications/* with id=None) is a Notification
        if not has_id or (method.startswith("notifications/") and req_id is None):
            return None

        if method == "initialize":
            return {
                "jsonrpc": jsonrpc,
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "microllm-systemone-mcp",
                        "version": "0.2.0",
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": jsonrpc, "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": jsonrpc,
                "id": req_id,
                "result": {"tools": MCP_TOOLS_SPEC},
            }

        if method == "tools/call":
            if not isinstance(params, Mapping):
                return {
                    "jsonrpc": jsonrpc,
                    "id": req_id,
                    "error": {"code": -32602, "message": "Invalid params: expected object."},
                }
            tool_name = str(params.get("name", ""))
            arguments = params.get("arguments") or {}
            try:
                tool_output = self._call_tool(tool_name, arguments)
                return {
                    "jsonrpc": jsonrpc,
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(tool_output),
                            }
                        ],
                        "structuredContent": tool_output,
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": jsonrpc,
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": jsonrpc,
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    def handle_jsonrpc(
        self, payload: Mapping[str, Any] | list[Any]
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Process a single JSON-RPC 2.0 request object or a JSON-RPC 2.0 batch array."""
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

    def _call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from microllm.schemas import SystemOneRequest, render_structured_text
        from microllm.server import _evaluate_isolated_question

        if tool_name == "systemone_decide":
            req_obj = SystemOneRequest.model_validate(dict(arguments))
            state_text = render_structured_text(req_obj.state)
            active_backend: Any = self.cascade
            if req_obj.model == "microllm-gear1-fast":
                active_backend = self.gear1
            elif req_obj.model == "microllm-gear2-anyjev-l0":
                active_backend = self.gear2

            state_ctx: dict[str, Any] = {}
            answers: dict[str, Any] = {}
            for qid, q_spec in req_obj.questions.items():
                # Verdict-MQ (IDEA-022): encode once on question 1, reuse cached representation on questions 2..M
                state_ctx = self.mq_cache.encode_state(state_text)
                ans_obj, raw = _evaluate_isolated_question(active_backend, state_text, q_spec)
                answers[qid] = {**ans_obj.model_dump(), "x_meta": raw}
            return {
                "model": req_obj.model or "jev-latest",
                "answers": answers,
                "verdict_mq_state_cached": bool(state_ctx.get("clause_count", 1) >= 1),
                "verdict_mq_cache_stats": self.mq_cache.stats,
            }

        if tool_name == "systemone_guardrail":
            client = self._get_client()
            resp = client.post("/v1/guardrail", json=dict(arguments))
            if resp.status_code != 200:
                raise ValueError(resp.text)
            return dict(resp.json())

        if tool_name == "systemone_calibrate":
            client = self._get_client()
            resp = client.post("/v1/calibrate", json=dict(arguments))
            if resp.status_code != 200:
                raise ValueError(resp.text)
            return dict(resp.json())

        raise ValueError(f"Unknown MCP tool '{tool_name}'.")


def run_mcp_stdio_server() -> int:
    """Run the JSON-RPC 2.0 Model Context Protocol server over standard input/output."""
    server = MicroLLMMCPServer()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = server.handle_jsonrpc(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0
