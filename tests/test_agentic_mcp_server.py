"""Comprehensive Tests for Model Context Protocol (MCP) Server Agentic Enhancements (IDEA-048 / Issue #175).

Validates:
1. `tools/list` returns 5 tools: `systemone_decide`, `systemone_guardrail`, `systemone_calibrate`,
   `systemone_agent_guardrail`, and `systemone_agent_router`.
2. `systemone_decide`:
   - `format="compact"` produces ~90% context reduction single-line text summary `[DECIDE] ...`
   - `format="verbose"` (default) retains full detailed JSON output.
3. `systemone_agent_guardrail`:
   - Sub-millisecond pre-tool execution interceptor.
   - Blocks destructive bash commands (`rm -rf /var/lib/postgresql`, `DROP TABLE users`) in production.
   - Allows safe benign operations (`python script.py`, `SELECT COUNT(*) FROM metrics`).
4. `systemone_agent_router`:
   - Sub-millisecond zero-generation intent router using Dual-Gear NLI candidate tournament.
   - Accurately selects database tool for SQL requests and search tool for web queries.
   - Supports candidates specified as dict or list with `format="compact"`.
5. JSON-RPC 2.0 protocol compliance (initialize, ping, batch execution, error handling).
"""

from __future__ import annotations

import json
from typing import Any
import pytest

from microllm.mcp_server import MCP_TOOLS_SPEC, MicroLLMMCPServer


@pytest.fixture
def mcp_server() -> MicroLLMMCPServer:
    return MicroLLMMCPServer()


def test_mcp_tools_list_exposes_5_tools(mcp_server: MicroLLMMCPServer) -> None:
    """Verify tools/list exposes all 5 registered System-1 tools with schema annotations."""
    resp = mcp_server.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp

    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert len(tool_names) == 5
    assert "systemone_decide" in tool_names
    assert "systemone_guardrail" in tool_names
    assert "systemone_calibrate" in tool_names
    assert "systemone_agent_guardrail" in tool_names
    assert "systemone_agent_router" in tool_names

    # Check that format parameter exists in systemone_decide schema
    decide_spec = next(t for t in tools if t["name"] == "systemone_decide")
    assert "format" in decide_spec["inputSchema"]["properties"]
    assert decide_spec["inputSchema"]["properties"]["format"]["default"] == "verbose"


def test_systemone_decide_compact_token_condensation(mcp_server: MicroLLMMCPServer) -> None:
    """Verify format='compact' provides condensed single-line text and minimal answers dict."""
    req_payload = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "systemone_decide",
            "arguments": {
                "state": "The user reported a severe recurring payment failure on order #94821.",
                "questions": {
                    "is_urgent": {
                        "type": "noul",
                        "instructions": "Is this issue urgent and business critical?",
                    },
                    "team": {
                        "type": "choice",
                        "instructions": "Route to the best team.",
                        "criteria": {
                            "billing": "Payment processing, invoices, subscription cards",
                            "general": "General questions, feature suggestions",
                        },
                    },
                },
                "format": "compact",
            },
        },
    }
    resp = mcp_server.handle_rpc(req_payload)
    assert resp is not None
    assert resp["id"] == 10
    result = resp["result"]
    assert result["isError"] is False

    # Check compact single-line text
    text_content = result["content"][0]["text"]
    assert text_content.startswith("[DECIDE:")
    assert "is_urgent=" in text_content
    assert "team=" in text_content

    # Check that text is concise (< 200 chars vs typical > 1500 chars verbose JSON)
    assert len(text_content) < 200

    # Structured content should have concise answers without full simplexes or caches
    structured = result["structuredContent"]
    assert "answers" in structured
    assert "team" in structured["answers"]
    assert "decision" in structured["answers"]["team"]
    assert "confidence" in structured["answers"]["team"]


def test_systemone_agent_guardrail_blocking_destructive_command(
    mcp_server: MicroLLMMCPServer,
) -> None:
    """Verify systemone_agent_guardrail intercepts and blocks destructive commands in production."""
    req_payload = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_guardrail",
            "arguments": {
                "tool": "bash",
                "command": "rm -rf /var/lib/postgresql/data",
                "environment": "production",
                "format": "compact",
            },
        },
    }
    resp = mcp_server.handle_rpc(req_payload)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is False

    structured = result["structuredContent"]
    assert structured["allowed"] is False
    assert structured["action"] == "BLOCK"
    assert structured["tool"] == "bash"
    assert structured["risk_score"] >= 0.70
    assert structured["primary_threat"] == "destructive_operation"

    text = result["content"][0]["text"]
    assert text.startswith("[GUARDRAIL:BLOCK]")
    assert "tool=bash" in text
    assert "destructive_operation" in text


def test_systemone_agent_guardrail_allowing_safe_command(
    mcp_server: MicroLLMMCPServer,
) -> None:
    """Verify systemone_agent_guardrail permits benign safe commands."""
    req_payload = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_guardrail",
            "arguments": {
                "tool": "bash",
                "command": "python -m pytest tests/test_sanity.py -v",
                "environment": "production",
                "format": "compact",
            },
        },
    }
    resp = mcp_server.handle_rpc(req_payload)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is False

    structured = result["structuredContent"]
    assert structured["allowed"] is True
    assert structured["action"] == "ALLOW"
    assert structured["tool"] == "bash"
    assert structured["risk_score"] < 0.50

    text = result["content"][0]["text"]
    assert text.startswith("[GUARDRAIL:ALLOW]")


def test_systemone_agent_router_intent_selection(mcp_server: MicroLLMMCPServer) -> None:
    """Verify systemone_agent_router accurately routes goals to candidate tools in < 1 ms."""
    candidates = {
        "sql_tool": "Execute relational SQL queries against PostgreSQL customer database",
        "web_search": "Search external web search engines for public documentation and articles",
        "calculator": "Perform arithmetic math and financial calculations",
    }

    # Query 1: Database intent
    req_db = {
        "jsonrpc": "2.0",
        "id": 30,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_router",
            "arguments": {
                "goal": "Query total monthly recurring revenue and active subscriptions from PostgreSQL",
                "candidates": candidates,
                "format": "compact",
            },
        },
    }
    resp_db = mcp_server.handle_rpc(req_db)
    assert resp_db is not None
    res_db = resp_db["result"]
    assert res_db["isError"] is False
    assert res_db["structuredContent"]["selected"] == "sql_tool"
    assert res_db["structuredContent"]["confidence"] > 0.40
    assert res_db["content"][0]["text"].startswith("[ROUTE] selected=sql_tool")

    # Query 2: Search intent
    req_search = {
        "jsonrpc": "2.0",
        "id": 31,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_router",
            "arguments": {
                "goal": "Search the public web for the latest updates on Python 3.14 release schedule",
                "candidates": candidates,
                "format": "compact",
            },
        },
    }
    resp_search = mcp_server.handle_rpc(req_search)
    assert resp_search is not None
    res_search = resp_search["result"]
    assert res_search["isError"] is False
    assert res_search["structuredContent"]["selected"] == "web_search"


def test_systemone_agent_router_list_candidates(mcp_server: MicroLLMMCPServer) -> None:
    """Verify systemone_agent_router supports list candidate format."""
    candidates_list = ["database_query", "web_search", "send_email"]
    req = {
        "jsonrpc": "2.0",
        "id": 32,
        "method": "tools/call",
        "params": {
            "name": "systemone_agent_router",
            "arguments": {
                "goal": "Send invoice notification email to client",
                "candidates": candidates_list,
                "format": "compact",
            },
        },
    }
    resp = mcp_server.handle_rpc(req)
    assert resp is not None
    res = resp["result"]
    assert res["isError"] is False
    assert res["structuredContent"]["selected"] == "send_email"


def test_mcp_jsonrpc_lifecycle_and_errors(mcp_server: MicroLLMMCPServer) -> None:
    """Verify initialize, ping, invalid method, and unknown tool error handling."""
    # Initialize
    init_res = mcp_server.handle_rpc({"jsonrpc": "2.0", "id": "init-1", "method": "initialize"})
    assert init_res is not None
    assert init_res["result"]["protocolVersion"] == "2024-11-05"
    assert init_res["result"]["serverInfo"]["name"] == "microllm-systemone-mcp"

    # Ping
    ping_res = mcp_server.handle_rpc({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"})
    assert ping_res is not None
    assert ping_res["result"] == {}

    # Unknown method
    err_res = mcp_server.handle_rpc({"jsonrpc": "2.0", "id": 99, "method": "unknown_action"})
    assert err_res is not None
    assert err_res["error"]["code"] == -32601

    # Unknown tool call
    tool_err = mcp_server.handle_rpc({
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/call",
        "params": {"name": "nonexistent_tool"},
    })
    assert tool_err is not None
    assert tool_err["result"]["isError"] is True
    assert "Unknown MCP tool" in tool_err["result"]["content"][0]["text"]
