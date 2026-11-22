"""Zero-dependency, duck-typed LangGraph pre-tool guard node and tool wrapper (`microllm.guard.v1`).

Works with LangGraph `StateGraph`, LangChain `AIMessage`/`ToolMessage` attribute objects,
and plain dict states without requiring `langgraph` or `langchain` at import or runtime.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from typing import Any, Callable, Literal, Mapping, Sequence, overload

from pydantic import ValidationError

from microllm.guard import (
    GATEKEEPER_SCHEMA_VERSION,
    GatekeeperDecision,
    GuardConfig,
    MicroLLMGuard,
    _get_default_guard,
)
from microllm.middleware.mcp_guard import extract_mcp_tool_request

GuardRoute = Literal["execute_tools", "block_tool", "escalate_human"]
ErrorMode = Literal["raise", "return"]


class GuardBlockedError(PermissionError):
    """Raised when `MicroLLMGuard` blocks a tool call prior to execution."""

    def __init__(self, decision: GatekeeperDecision) -> None:
        self.decision = decision
        super().__init__(f"Tool '{decision.tool_name}' blocked: {decision.reason}")


class GuardAbstainedError(RuntimeError):
    """Raised when `MicroLLMGuard` abstains/escalates a tool call for human or System-2 review."""

    def __init__(self, decision: GatekeeperDecision) -> None:
        self.decision = decision
        super().__init__(f"Tool '{decision.tool_name}' escalated (abstain=True): {decision.reason}")


def _reject_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant '{token}' is not permitted.")


def _parse_tool_call_entry(entry: Any) -> tuple[str, str, Any]:
    """Normalize LangChain/OpenAI/dict tool call objects into `(call_id, tool_name, raw_args)`."""
    if isinstance(entry, Mapping):
        call_id = str(entry.get("id") or "")
        if "function" in entry and isinstance(entry["function"], Mapping):
            fn_obj = entry["function"]
            tool_name = str(fn_obj.get("name") or "").strip() or "tool"
            raw_args = fn_obj.get("arguments")
        else:
            tool_name = str(entry.get("name") or entry.get("tool_name") or "").strip() or "tool"
            raw_args = entry.get("args") if "args" in entry else entry.get("arguments")
    else:
        call_id = str(getattr(entry, "id", "") or "")
        fn_attr = getattr(entry, "function", None)
        if fn_attr is not None:
            tool_name = str(getattr(fn_attr, "name", "") or "").strip() or "tool"
            raw_args = getattr(fn_attr, "arguments", None)
        else:
            tool_name = str(getattr(entry, "name", "") or "").strip() or "tool"
            raw_args = getattr(entry, "args", getattr(entry, "arguments", None))

    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if stripped.startswith(("{", "[")):
            try:
                raw_args = json.loads(stripped, parse_constant=_reject_non_finite_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Malformed JSON arguments for tool '{tool_name}': {exc}") from exc

    return call_id, tool_name, raw_args


def _extract_pending_tool_calls(state: Any) -> list[Any]:
    """Extract pending `tool_calls` from the latest message in a LangGraph state."""
    if isinstance(state, Mapping):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)

    if not messages or not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []

    last_msg = messages[-1]
    if isinstance(last_msg, Mapping):
        raw_calls = last_msg.get("tool_calls")
    else:
        raw_calls = getattr(last_msg, "tool_calls", None)

    if not raw_calls or not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return []
    return list(raw_calls)


def _route_from_decisions(decisions: Sequence[Mapping[str, Any]]) -> GuardRoute:
    has_escalate = False
    for dec in decisions:
        d = str(dec.get("decision", "")).lower()
        if d == "block":
            return "block_tool"
        if d == "escalate" or bool(dec.get("abstain", False)):
            has_escalate = True
    return "escalate_human" if has_escalate else "execute_tools"


def _make_blocked_fallback_decision(tool_name: str, reason: str, env: str) -> dict[str, Any]:
    return {
        "schema_version": GATEKEEPER_SCHEMA_VERSION,
        "decision": "block",
        "confidence": 1.0,
        "abstain": False,
        "reason": reason,
        "risk_score": 1.0,
        "primary_threat": "invalid_tool_arguments",
        "conformal_set": ["block"],
        "checks": {},
        "action_probabilities": {"allow": 0.0, "escalate": 0.0, "block": 1.0},
        "latency_ms": 0.0,
        "tool_name": tool_name,
        "environment": env,
        "abstain_policy": "block",
        "hijack_suspected": False,
    }


def _default_blocked_message_factory(
    call_id: str,
    tool_name: str,
    decision_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": json.dumps(decision_dict, ensure_ascii=False, allow_nan=False),
    }


class LangGraphGuardNode:
    """Drop-in LangGraph pre-tool guard node inspecting pending `tool_calls` in `< 5 ms` CPU."""

    def __init__(
        self,
        guard: MicroLLMGuard | None = None,
        *,
        config: GuardConfig | None = None,
        message_factory: Callable[[str, str, dict[str, Any]], Any] | None = None,
    ) -> None:
        if guard is not None:
            self.guard = guard
        elif config is not None:
            self.guard = MicroLLMGuard(config=config)
        else:
            self.guard = _get_default_guard()
        self.message_factory = message_factory or _default_blocked_message_factory

    def __call__(self, state: Any, config: Any | None = None) -> dict[str, Any]:
        """Synchronously screen all pending `tool_calls` on the latest assistant message."""
        del config
        pending = _extract_pending_tool_calls(state)
        if not pending:
            return {
                "guard_decisions": [],
                "guard_route": "execute_tools",
            }

        decisions: list[dict[str, Any]] = []
        parsed_entries: list[tuple[str, str, dict[str, Any]]] = []
        env_label = self.guard.config.environment

        for entry in pending:
            call_id = ""
            tool_name = "tool"
            try:
                call_id, tool_name, raw_args = _parse_tool_call_entry(entry)
                guard_req = extract_mcp_tool_request({"name": tool_name, "arguments": raw_args})
                decision = self.guard.evaluate(guard_req)
                dec_dict = decision.to_dict()
            except (ValidationError, ValueError, TypeError) as exc:
                dec_dict = _make_blocked_fallback_decision(
                    tool_name=tool_name,
                    reason=f"Blocked malformed tool call payload: {exc}",
                    env=env_label,
                )

            decisions.append(dec_dict)
            parsed_entries.append((call_id, tool_name, dec_dict))

        route = _route_from_decisions(decisions)
        update: dict[str, Any] = {
            "guard_decisions": decisions,
            "guard_route": route,
        }
        if route == "block_tool":
            # Emit tool feedback for every call in the aborted batch so no tool_call_id is orphaned
            blocked_messages: list[Any] = []
            for call_id, tool_name, dec_dict in parsed_entries:
                if dec_dict["decision"] == "block":
                    blocked_messages.append(self.message_factory(call_id, tool_name, dec_dict))
                else:
                    aborted = dict(dec_dict)
                    aborted["reason"] = (
                        f"Batch aborted because a sibling tool call was blocked ({dec_dict['reason']})."
                    )
                    blocked_messages.append(self.message_factory(call_id, tool_name, aborted))
            update["messages"] = blocked_messages
        return update

    async def ainvoke(self, state: Any, config: Any | None = None) -> dict[str, Any]:
        """Asynchronously screen pending `tool_calls` without blocking the event loop."""
        return await asyncio.to_thread(self.__call__, state, config)


def guard_router_edge(state: Any) -> GuardRoute:
    """Conditional edge function for LangGraph `add_conditional_edges` after `LangGraphGuardNode`."""
    raw_route = state.get("guard_route") if isinstance(state, Mapping) else getattr(state, "guard_route", None)
    if raw_route in ("execute_tools", "block_tool", "escalate_human"):
        return raw_route

    raw_decisions = (
        state.get("guard_decisions")
        if isinstance(state, Mapping)
        else getattr(state, "guard_decisions", None)
    )
    if isinstance(raw_decisions, Sequence) and not isinstance(raw_decisions, (str, bytes)):
        mapping_decisions = [d for d in raw_decisions if isinstance(d, Mapping)]
        if mapping_decisions:
            return _route_from_decisions(mapping_decisions)

    return "execute_tools"


def _to_json_safe(value: Any) -> Any:
    """Convert non-JSON Python objects (e.g. Path, UUID) to strings while preserving non-finite floats."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return str(value)


def _evaluate_bound_call(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    tool_name: str | None,
    active_guard: MicroLLMGuard,
) -> GatekeeperDecision:
    eff_name = tool_name or getattr(func, "__name__", "tool")
    bound_args: dict[str, Any] = {}
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound_args = dict(bound.arguments)
    except (TypeError, ValueError):
        bound_args = {f"arg_{i}": v for i, v in enumerate(args)}
        bound_args.update(kwargs)

    safe_args = _to_json_safe(bound_args)
    guard_req = extract_mcp_tool_request({"name": eff_name, "arguments": safe_args})
    return active_guard.evaluate(guard_req)


def _handle_decision(
    decision: GatekeeperDecision,
    *,
    on_block: ErrorMode,
    on_abstain: ErrorMode,
) -> tuple[bool, dict[str, Any] | None]:
    """Return `(should_execute, structured_return_or_none)` or raise typed guard exception."""
    if decision.decision == "block":
        if on_block == "return":
            return False, decision.to_dict()
        raise GuardBlockedError(decision)
    if decision.decision == "escalate" or decision.abstain:
        if on_abstain == "return":
            return False, decision.to_dict()
        raise GuardAbstainedError(decision)
    return True, None


def guard_tool_callable(
    func: Callable[..., Any],
    *,
    tool_name: str | None = None,
    guard: MicroLLMGuard | None = None,
    config: GuardConfig | None = None,
    on_block: ErrorMode = "raise",
    on_abstain: ErrorMode = "raise",
) -> Callable[..., Any]:
    """Wrap a sync or async Python tool callable with sub-5 ms `MicroLLMGuard` pre-execution checks."""
    if on_block not in ("raise", "return"):
        raise ValueError("on_block must be 'raise' or 'return'.")
    if on_abstain not in ("raise", "return"):
        raise ValueError("on_abstain must be 'raise' or 'return'.")

    active_guard = (
        guard
        if guard is not None
        else (MicroLLMGuard(config=config) if config is not None else _get_default_guard())
    )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            decision = await asyncio.to_thread(
                _evaluate_bound_call,
                func,
                args,
                kwargs,
                tool_name=tool_name,
                active_guard=active_guard,
            )
            should_run, ret_payload = _handle_decision(
                decision,
                on_block=on_block,
                on_abstain=on_abstain,
            )
            if not should_run:
                return ret_payload
            return await func(*args, **kwargs)

        return _async_wrapper

    @functools.wraps(func)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        decision = _evaluate_bound_call(
            func,
            args,
            kwargs,
            tool_name=tool_name,
            active_guard=active_guard,
        )
        should_run, ret_payload = _handle_decision(
            decision,
            on_block=on_block,
            on_abstain=on_abstain,
        )
        if not should_run:
            return ret_payload
        return func(*args, **kwargs)

    return _sync_wrapper


@overload
def microllm_guard_tool(func: Callable[..., Any]) -> Callable[..., Any]: ...


@overload
def microllm_guard_tool(
    func: None = None,
    *,
    tool_name: str | None = None,
    guard: MicroLLMGuard | None = None,
    config: GuardConfig | None = None,
    on_block: ErrorMode = "raise",
    on_abstain: ErrorMode = "raise",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def microllm_guard_tool(
    func: Callable[..., Any] | None = None,
    *,
    tool_name: str | None = None,
    guard: MicroLLMGuard | None = None,
    config: GuardConfig | None = None,
    on_block: ErrorMode = "raise",
    on_abstain: ErrorMode = "raise",
) -> Any:
    """Decorator wrapping sync or async tool functions with `MicroLLMGuard` pre-execution gating."""
    if func is not None:
        return guard_tool_callable(
            func,
            tool_name=tool_name,
            guard=guard,
            config=config,
            on_block=on_block,
            on_abstain=on_abstain,
        )

    def _decorator(target: Callable[..., Any]) -> Callable[..., Any]:
        return guard_tool_callable(
            target,
            tool_name=tool_name,
            guard=guard,
            config=config,
            on_block=on_block,
            on_abstain=on_abstain,
        )

    return _decorator


__all__ = [
    "ErrorMode",
    "GuardAbstainedError",
    "GuardBlockedError",
    "GuardRoute",
    "LangGraphGuardNode",
    "guard_router_edge",
    "guard_tool_callable",
    "microllm_guard_tool",
]
