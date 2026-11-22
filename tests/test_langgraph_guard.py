"""TDD tests for zero-dependency LangGraph pre-tool guard node and tool wrapper (Issue #285)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import pytest

from microllm import GuardConfig
from microllm.middleware import (
    GuardAbstainedError,
    GuardBlockedError,
    LangGraphGuardNode,
    guard_router_edge,
    guard_tool_callable,
    microllm_guard_tool,
)


@dataclass
class _FakeAIMessage:
    """Minimal duck-typed LangChain AIMessage stand-in (zero langchain install required)."""

    content: str
    tool_calls: list[dict[str, Any]]


@dataclass
class _FakeState:
    """Minimal duck-typed LangGraph state object."""

    messages: list[Any]
    guard_route: str | None = None
    guard_decisions: list[dict[str, Any]] | None = None


def test_langgraph_guard_node_supports_dict_and_attribute_messages_and_batches() -> None:
    """Verify LangGraphGuardNode handles plain dicts, attribute objects, and multi-call routing priority."""
    node = LangGraphGuardNode()

    # 1. Plain dict state with safe tool call -> execute_tools
    safe_state = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "name": "bash", "args": {"command": "git status"}},
                ],
            }
        ]
    }
    safe_update = node(safe_state)
    assert safe_update["guard_route"] == "execute_tools"
    assert guard_router_edge(safe_update) == "execute_tools"
    assert len(safe_update["guard_decisions"]) == 1
    assert safe_update["guard_decisions"][0]["schema_version"] == "microllm.guard.v1"
    assert safe_update["guard_decisions"][0]["decision"] == "allow"
    assert "messages" not in safe_update

    # 2. Attribute-based AIMessage with mixed [allow, block] batch -> block_tool + synthetic feedback for BOTH IDs
    mixed_state = _FakeState(
        messages=[
            _FakeAIMessage(
                content="",
                tool_calls=[
                    {"id": "call_ok", "name": "bash", "args": {"command": "git status"}},
                    {"id": "call_bad", "name": "sql", "args": {"query": "DROP TABLE users;"}},
                ],
            )
        ]
    )
    mixed_update = node(mixed_state)
    assert mixed_update["guard_route"] == "block_tool"
    assert guard_router_edge(mixed_update) == "block_tool"
    assert len(mixed_update["guard_decisions"]) == 2
    assert len(mixed_update["messages"]) == 2
    msg_ids = {m["tool_call_id"] for m in mixed_update["messages"]}
    assert msg_ids == {"call_ok", "call_bad"}
    blocked_msg = next(m for m in mixed_update["messages"] if m["tool_call_id"] == "call_bad")
    parsed_content = json.loads(blocked_msg["content"])
    assert parsed_content["decision"] == "block"

    # 3. Multi-call batch with [allow, ambiguous] (no hard block) -> escalate_human
    esc_state = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "git status"}'}},
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "kubectl apply -f deploy/prod.yaml"}'},
                    },
                ],
            }
        ]
    }
    esc_update = asyncio.run(node.ainvoke(esc_state))
    assert esc_update["guard_route"] == "escalate_human"
    assert guard_router_edge(esc_update) == "escalate_human"
    assert esc_update["guard_decisions"][1]["abstain"] is True

    # 4. Malformed JSON string arguments fail closed to block_tool
    bad_json_state = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c_mal", "type": "function", "function": {"name": "bash", "arguments": "{broken json"}},
                ],
            }
        ]
    }
    bad_update = node(bad_json_state)
    assert bad_update["guard_route"] == "block_tool"
    assert bad_update["guard_decisions"][0]["decision"] == "block"

    # 5. No pending tool calls -> clean no-op execute_tools
    empty_update = node({"messages": [{"role": "assistant", "content": "All done."}]})
    assert empty_update == {"guard_decisions": [], "guard_route": "execute_tools"}


def test_sync_and_async_microllm_guard_tool_decorator_raise_and_return_modes() -> None:
    """Verify @microllm_guard_tool enforces pre-execution checks on sync/async functions in both modes."""
    executed: list[str] = []

    @microllm_guard_tool
    def run_bash(command: str) -> str:
        executed.append(command)
        return f"ran:{command}"

    @microllm_guard_tool(on_block="return", on_abstain="return")
    async def run_async_sql(query: str) -> dict[str, Any]:
        executed.append(query)
        return {"ok": True, "query": query}

    # Sync allow
    assert run_bash("git status") == "ran:git status"
    assert executed == ["git status"]

    # Sync block raises GuardBlockedError before execution
    with pytest.raises(GuardBlockedError) as blk_exc:
        run_bash("rm -rf /")
    assert blk_exc.value.decision.decision == "block"
    assert executed == ["git status"]

    # Sync ambiguous raises GuardAbstainedError before execution
    with pytest.raises(GuardAbstainedError) as esc_exc:
        run_bash("kubectl apply -f deploy/prod.yaml")
    assert esc_exc.value.decision.abstain is True
    assert executed == ["git status"]

    # Async allow
    res_ok = asyncio.run(run_async_sql("SELECT count(*) FROM orders"))
    assert res_ok["ok"] is True

    # Async block in return mode returns structured v1 dict without executing
    res_blk = asyncio.run(run_async_sql("DROP TABLE orders;"))
    assert res_blk["schema_version"] == "microllm.guard.v1"
    assert res_blk["decision"] == "block"
    assert "DROP TABLE orders;" not in executed

    # Async ambiguous in return mode returns structured escalate dict without executing
    res_esc = asyncio.run(run_async_sql("ALTER TABLE orders ADD COLUMN flag BOOLEAN"))
    assert res_esc["decision"] == "escalate"
    assert res_esc["abstain"] is True
    assert "ALTER TABLE orders ADD COLUMN flag BOOLEAN" not in executed


def test_guard_tool_callable_handles_path_objects_policy_boundaries_and_sub_5ms_sla() -> None:
    """Verify non-JSON Python arg objects (Path), scoped policy_boundaries, and < 5 ms CPU latency."""
    wrapped_cleanup = guard_tool_callable(
        lambda command: f"ok:{command}",
        tool_name="bash",
        config=GuardConfig(policy_boundaries={"allowed": ["rm -rf ./dist"]}),
    )
    assert wrapped_cleanup("rm -rf ./dist") == "ok:rm -rf ./dist"

    wrapped_reader = guard_tool_callable(
        lambda path: f"read:{path}",
        tool_name="read_file",
    )
    assert wrapped_reader(Path("README.md")) == "read:README.md"

    # Verify zero heavy LangGraph/LangChain runtime dependencies added to pyproject.toml
    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    deps_block = pyproject_text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "langgraph" not in deps_block.lower()
    assert "langchain" not in deps_block.lower()

    # Sub-5 ms steady-state CPU p95 SLA
    node = LangGraphGuardNode()
    st = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "1", "name": "bash", "args": {"command": "git status"}}]}
        ]
    }
    node(st)
    lats: list[float] = []
    for _ in range(30):
        t0 = time.perf_counter()
        node(st)
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats.sort()
    p95 = lats[int(0.95 * (len(lats) - 1))]
    assert p95 < 5.0, f"Expected LangGraph guard node CPU p95 < 5.0 ms, got {p95:.3f} ms"
