# MicroLLM Guard Middleware — Drop-In Pre-Execution Gatekeeper Guide (`microllm.guard.v1`)

> **Sub-5 ms CPU System-1 pre-execution gatekeeper for AI coding and tool-calling agents**: zero GPU required, `0.00 USD` API cost, Split Conformal abstention (`x_abstain`), certainty-claim hijack defense, and non-overridable `policy_boundaries`.

- **Formal Contract Schema**: [`src/microllm/data/gatekeeper_contract_v1.json`](../src/microllm/data/gatekeeper_contract_v1.json) (`schema_version = "microllm.guard.v1"`)
- **Verified CPU Benchmark Report**: [`docs/GUARD_VS_LLM_JUDGE_BENCHMARK.md`](GUARD_VS_LLM_JUDGE_BENCHMARK.md) (`benchmarks/guard_vs_llm_judge_results.json`)
- **Core Engine & Adapters**: [`src/microllm/guard.py`](../src/microllm/guard.py) · [`src/microllm/guard_cli.py`](../src/microllm/guard_cli.py) · [`src/microllm/middleware/`](../src/microllm/middleware/__init__.py)

---

## Verified Performance Snapshot (220 Curated Agent Tool Calls, CPU Only)

| Gatekeeper Mode | CPU `p50` | CPU `p95` | API Cost / 1M Calls | Binary Safety F1 | False Positive Rate | System-2 Avoidance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`microllm-guard` (Gear 1 CPU)** | **`0.367 ms`** | **`0.554 ms`** | **`0.00 USD`** | **`100.00%`** | **`0.00%`** | `84.09%` local |
| **`microllm-guard` (Dual-Gear Cascade)** | **`0.280 ms`** | **`0.353 ms`** | **`0.00 USD`** | **`100.00%`** | **`0.00%`** | `84.09%` local |
| **Hybrid Cascade (`microllm-guard` + Frontier on `x_abstain`)** | **`0.367 ms`** | `137.2 ms` (mean) | **`260.13 USD`** | **`99.05%` (acc)** | **`0.00%`** | **`84.1%` cost saved** |
| System-2 Frontier Cloud LLM-Judge (Reference) | `780.0 ms` | `1420.0 ms` | `1635.00 USD` | `96.20%` | `4.20%` | `0.0%` |

---

## 1. `jmrenouard/api_executor` & Local HTTP / CLI Contract (`microllm.guard.v1`)

MicroLLM exposes the versioned `microllm.guard.v1` contract identically over **HTTP**, **Python SDK**, and **CLI** so external orchestrators such as `jmrenouard/api_executor` can gate every tool invocation locally in under 1 ms.

### 1.1 Local HTTP Endpoints

Start the local FastAPI server:

```bash
microllm serve --port 8000
```

- **`GET /v1/guard/contract`**: Returns the packaged Draft 2020-12 JSON Schema (`gatekeeper_contract_v1.json`).
- **`POST /v1/guard/evaluate`**: Evaluates a proposed tool call and returns a `GatekeeperDecision` JSON document. Honors `Authorization: Bearer <MICROLLM_API_KEY>` when configured and rejects `NaN`/`Infinity` with HTTP `422` (`loc[0] == "body"`).
- **`GET /v1/metrics`**: Tracks cumulative `guard_requests`, `guard_blocked`, and `guard_escalated` counters.

#### Request Payload (`POST /v1/guard/evaluate`)

```json
{
  "tool_name": "bash",
  "command": "kubectl apply -f deploy/prod-api.yaml",
  "arguments": {
    "namespace": "production"
  },
  "environment": "production",
  "policy_boundaries": {
    "allowed": ["git status", "pytest -q"],
    "forbidden": ["rm -rf /", "DROP TABLE"]
  },
  "abstain_policy": "escalate"
}
```

#### Response Payload (`microllm.guard.v1`)

```json
{
  "schema_version": "microllm.guard.v1",
  "decision": "escalate",
  "confidence": 0.62,
  "abstain": true,
  "reason": "Escalated for human or System-2 review in production environment (ambiguous privileged operation, risk=0.6200 >= 0.5000).",
  "risk_score": 0.62,
  "checks": {
    "prompt_injection": 0.08,
    "destructive_operation": 0.62,
    "pii_or_secret_leak": 0.09,
    "hallucination_risk": 0.08
  },
  "action_probabilities": {
    "allow": 0.14,
    "escalate": 0.62,
    "block": 0.24
  },
  "conformal_set": ["escalate", "block"],
  "primary_threat": "ambiguous_privileged_operation",
  "environment": "production",
  "latency_ms": 0.41
}
```

### 1.2 Standalone CLI (`microllm-guard` & `microllm guard`)

The installed console script `microllm-guard` (and `microllm guard`) supports both flag-based invocation and stdin JSON piping with deterministic exit codes:

| Exit Code | Decision | Recommended Orchestrator Action |
| :---: | :--- | :--- |
| **`0`** | `allow` (`abstain: false`) | Proceed with tool execution immediately |
| **`2`** | `block` | Reject tool execution and return `reason` to the agent |
| **`3`** | `escalate` (`abstain: true`) | Pause for human confirmation or escalate to a System-2 LLM judge |
| **`1`** | Invalid input / parse error | Fail closed and log malformed invocation |

```bash
# Inspect the versioned JSON Schema contract
microllm-guard --contract

# Flag-based evaluation with machine-readable JSON output
microllm-guard \
  --tool bash \
  --command "rm -rf /var/lib/postgresql/data" \
  --env production \
  --json

# Pipe a JSON payload over stdin (e.g. from api_executor)
echo '{"tool_name":"bash","command":"git status","environment":"production"}' \
  | microllm-guard --json

# Fail-closed mode (converts conformal abstentions into exit code 2 / block)
microllm-guard \
  --tool kubectl \
  --command "kubectl apply -f deploy/prod.yaml" \
  --env production \
  --fail-closed \
  --json
```

### 1.3 Typed Python SDK (Sync & Async)

```python
from microllm import AsyncMicroJevClient, MicroJevClient

client = MicroJevClient(base_url="http://127.0.0.1:8000")
verdict = client.guard_evaluate(
    tool_name="bash",
    command="pytest -q",
    environment="production",
    policy_boundaries={"allowed": ["pytest -q"], "forbidden": ["rm -rf /"]},
)
assert verdict["decision"] == "allow"
assert verdict["abstain"] is False

# Or strictly in-process without HTTP overhead (< 0.4 ms p50 CPU):
from microllm import MicroLLMGuard

guard = MicroLLMGuard()
decision = guard.evaluate(
    {"tool_name": "bash", "command": "git status", "environment": "production"}
)
print(decision.decision, decision.confidence, decision.abstain)
```

---

## 2. Drop-In MCP Pre-Execution Guard Middleware & Stdio Proxy

MicroLLM provides two zero-dependency Model Context Protocol (MCP) integration layers in [`src/microllm/middleware/mcp_guard.py`](../src/microllm/middleware/mcp_guard.py):

### 2.1 Zero-Code Stdio Proxy (`microllm-guard mcp-proxy`)

Wrap any existing MCP stdio server without modifying its source code. All `tools/call` frames (including batch arrays and notification-spoofed frames) are screened locally before forwarding to the upstream child process (executed with strict `shell=False` argv lists):

```json
{
  "mcpServers": {
    "guarded-ops": {
      "command": "microllm-guard",
      "args": [
        "mcp-proxy",
        "--env", "production",
        "--policy-file", ".microllm-guard.json",
        "--",
        "python", "-m", "my_mcp_server"
      ]
    }
  }
}
```

- Allowed calls pass transparently to the upstream MCP server; upstream server-initiated notifications are forwarded without ID desynchronization.
- Blocked or escalated calls return a structured MCP tool error response (`result.isError = true`) containing the full `microllm.guard.v1` JSON verdict in both `result.structuredContent` and `result.content[0].text` (`decision: "block"` or `"escalate"`, `abstain: true|false`, `reason`).

### 2.2 In-Process `MCPGuardMiddleware`

```python
from microllm import GuardConfig
from microllm.middleware import MCPGuardMiddleware

middleware = MCPGuardMiddleware(
    upstream=my_mcp_server,
    config=GuardConfig(environment="production"),
)

# Wrap any JSON-RPC 2.0 MCP server exposing handle_jsonrpc(payload) / handle_rpc(req):
guarded_response = middleware.handle_jsonrpc(
    {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "bash",
            "arguments": {"command": "rm -rf /etc/passwd"},
        },
    }
)
assert guarded_response["result"]["isError"] is True
assert guarded_response["result"]["structuredContent"]["decision"] == "block"
```

MicroLLM's built-in MCP tool `systemone_agent_guardrail` ([`src/microllm/mcp_server.py`](../src/microllm/mcp_server.py)) also includes the additive `microllm.guard.v1` fields (`schema_version`, `decision`, `confidence`, `abstain`) while keeping 100% backward compatibility with existing keys.

---

## 3. Claude Code `PreToolUse` Hook Adapter

Protect Claude Code sessions (`Bash`, `Write`, `Edit`, `MultiEdit`, `WebFetch`, and `mcp__*` tools) in under 1 ms per tool call using [`src/microllm/middleware/claude_code_hook.py`](../src/microllm/middleware/claude_code_hook.py).

### 3.1 Generate `.claude/settings.json`

```bash
# Preview the hook configuration safely
microllm-guard init-claude-hooks --print

# Or merge idempotently into .claude/settings.json
microllm-guard init-claude-hooks --write .claude/settings.json
```

Generated `.claude/settings.json` snippet:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|WebFetch|mcp__.*",
        "hooks": [
          {
            "type": "command",
            "command": "microllm-guard claude-hook"
          }
        ]
      }
    ]
  }
}
```

### 3.2 Optional Project Policy (`.microllm-guard.json`)

Place a `.microllm-guard.json` file at your project root to customize environment thresholds, allowed cleanup commands, and forbidden boundaries:

```json
{
  "environment": "production",
  "abstain_policy": "escalate",
  "policy_boundaries": {
    "allowed": [
      "pytest -q",
      "rm -rf dist/ build/"
    ],
    "forbidden": [
      "rm -rf /",
      "/etc/shadow",
      ".ssh/authorized_keys"
    ]
  }
}
```

- `allow` -> `permissionDecision: "allow"` (exit `0`).
- `block` -> `permissionDecision: "deny"` (exit `0` with structured `hookSpecificOutput`).
- `escalate` (`abstain: true`) -> `permissionDecision: "ask"` (prompts the developer for interactive confirmation).
- Malformed or unreadable stdin JSON fails closed (`deny`).

---

## 4. Zero-Dependency LangGraph Pre-Tool Guard Node & Decorator

[`src/microllm/middleware/langgraph_guard.py`](../src/microllm/middleware/langgraph_guard.py) integrates into any LangGraph `StateGraph` via pure duck typing without requiring `langgraph` or `langchain-core` as package dependencies.

### 4.1 `StateGraph` Pre-Tool Routing Node

```python
from microllm import GuardConfig
from microllm.middleware import LangGraphGuardNode, guard_router_edge

guard_node = LangGraphGuardNode(config=GuardConfig(environment="production"))

# Wire inside a LangGraph StateGraph:
# graph.add_node("microllm_guard", guard_node)
# graph.add_conditional_edges(
#     "microllm_guard",
#     guard_router_edge,
#     {
#         "execute_tools": "tools",
#         "block_tool": "agent",
#         "escalate_human": "human_review",
#     },
# )
```

- Supports both LangChain-style message objects (`.tool_calls`) and plain `dict` states (`{"messages": [...]}`).
- Enforces strict multi-call batch priority (`block_tool` > `escalate_human` > `execute_tools`) and emits feedback messages covering every `tool_call_id` in an aborted batch to prevent orphaned tool call errors.

### 4.2 Sync & Async `@microllm_guard_tool` Decorator

```python
from microllm import GuardConfig
from microllm.middleware import GuardAbstainedError, GuardBlockedError, microllm_guard_tool

@microllm_guard_tool(
    tool_name="run_shell",
    config=GuardConfig(environment="production"),
)
def run_shell(command: str) -> str:
    return f"executed: {command}"

# Or return structured feedback instead of raising:
@microllm_guard_tool(
    tool_name="deploy_manifest",
    config=GuardConfig(environment="production"),
    on_block="return",
    on_abstain="return",
)
async def deploy_manifest(command: str, namespace: str) -> str:
    return f"deployed {namespace}"
```

---

## 5. Reproducing the System-1 vs System-2 LLM-Judge Benchmark

Verify all latency, cost, F1, FPR, calibration, and Hybrid Cascade metrics strictly offline on CPU:

```bash
# Via the unified CLI
microllm benchmark --guard-judge

# Via the standalone benchmark script (writes JSON + Markdown reports)
python scripts/run_guard_vs_judge_benchmark.py

# Optional: compare live against any local or remote OpenAI-compatible judge
microllm benchmark --guard-judge \
  --live-judge-url http://127.0.0.1:11434 \
  --live-judge-model llama3.1:8b
```

See [`docs/GUARD_VS_LLM_JUDGE_BENCHMARK.md`](GUARD_VS_LLM_JUDGE_BENCHMARK.md) for the complete 6-track breakdown and automated quality gates.
