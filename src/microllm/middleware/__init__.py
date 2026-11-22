"""Drop-in zero-dependency agent safety middleware for MicroLLM (`microllm.guard.v1`)."""

from microllm.middleware.claude_code_hook import (
    DEFAULT_CLAUDE_HOOK_MATCHER,
    PROJECT_POLICY_FILENAME,
    build_claude_hooks_config,
    evaluate_claude_pre_tool_use,
    format_claude_hook_output,
    load_claude_project_config,
    run_claude_hook_stdin,
    write_claude_hooks_config,
)
from microllm.middleware.langgraph_guard import (
    ErrorMode,
    GuardAbstainedError,
    GuardBlockedError,
    GuardRoute,
    LangGraphGuardNode,
    guard_router_edge,
    guard_tool_callable,
    microllm_guard_tool,
)
from microllm.middleware.mcp_guard import (
    MCPGuardMiddleware,
    extract_mcp_tool_request,
    run_mcp_guard_proxy,
)

__all__ = [
    "DEFAULT_CLAUDE_HOOK_MATCHER",
    "ErrorMode",
    "GuardAbstainedError",
    "GuardBlockedError",
    "GuardRoute",
    "LangGraphGuardNode",
    "MCPGuardMiddleware",
    "PROJECT_POLICY_FILENAME",
    "build_claude_hooks_config",
    "evaluate_claude_pre_tool_use",
    "extract_mcp_tool_request",
    "format_claude_hook_output",
    "guard_router_edge",
    "guard_tool_callable",
    "load_claude_project_config",
    "microllm_guard_tool",
    "run_claude_hook_stdin",
    "run_mcp_guard_proxy",
    "write_claude_hooks_config",
]
