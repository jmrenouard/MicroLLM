# Integrate MicroLLM Guard Middleware

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that `microllm-guard` is on your system `PATH`.
2. Make sure that you have write access to the target agent workspace.
3. Select the target environment (`production`, `staging`, or `development`).

## Safety Instructions

**CAUTION:** Test `.microllm-guard.json` locally before you set `--fail-closed` in production.
Incorrect `forbidden` rules or low thresholds can stop valid agent tool calls.

## Procedure

1. Examine the `microllm.guard.v1` JSON schema contract:

   ```bash
   microllm-guard --contract
   ```

2. Write a `.microllm-guard.json` policy file in your project root:

   ```json
   {
     "environment": "production",
     "abstain_policy": "escalate",
     "policy_boundaries": {
       "allowed": [
         "git status",
         "pytest -q"
       ],
       "forbidden": [
         "rm -rf /",
         "DROP TABLE",
         "/etc/shadow"
       ]
     }
   }
   ```

3. Evaluate one sample command with CLI flags and `--policy-file`:

   ```bash
   microllm-guard --tool bash --command "git status" --policy-file .microllm-guard.json --json
   ```

4. If you protect an MCP stdio server, start `microllm-guard mcp-proxy` with your upstream command:

   ```bash
   microllm-guard mcp-proxy --env production --policy-file .microllm-guard.json -- python -m my_mcp_server
   ```

5. If you protect a Claude Code workspace, write `.claude/settings.json` with `init-claude-hooks`:

   ```bash
   microllm-guard init-claude-hooks --write .claude/settings.json
   ```

6. If you call the HTTP gatekeeper from an external orchestrator, send `POST` requests to `/v1/guard/evaluate`:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/guard/evaluate \
     -H "Content-Type: application/json" \
     -d '{"tool_name":"bash","command":"pytest -q","environment":"production"}'
   ```

7. If you use a Python LangGraph workflow, add `LangGraphGuardNode` and `guard_router_edge` to your graph:

   ```python
   from microllm import GuardConfig
   from microllm.middleware import LangGraphGuardNode, guard_router_edge

   guard_node = LangGraphGuardNode(config=GuardConfig(environment="production"))
   ```

**NOTE:** Exit code `0` shows `allow`, exit code `2` shows `block`, and exit code `3` shows `escalate`.

## Verification Check

1. Do a check on an allowed command and make sure that exit code is `0`:

   ```bash
   microllm-guard --tool bash --command "git status" --policy-file .microllm-guard.json --json
   ```

2. Do a check on a dangerous command and make sure that exit code is `2`:

   ```bash
   microllm-guard --tool bash --command "rm -rf /etc/passwd" --policy-file .microllm-guard.json --json
   ```

3. Do a check on a privileged mutation command and make sure that exit code is `3`:

   ```bash
   microllm-guard --tool bash --command "kubectl apply -f deploy/prod.yaml" --env production --json
   ```

4. If `microllm serve` is active on port `8000`, examine `guard_requests` on `/v1/metrics`:

   ```bash
   curl -fsS http://127.0.0.1:8000/v1/metrics
   ```

## Rollback Procedure

If `microllm-guard` blocks valid developer commands after integration, do these steps:

1. Remove the `PreToolUse` entry for `microllm-guard claude-hook` from `.claude/settings.json`.
2. If you use `mcp-proxy`, replace the proxy command with the direct upstream MCP server command.
3. Replace `.microllm-guard.json` with the backup copy or remove the unverified `.microllm-guard.json` file.
4. Do a check with `git status` in the agent tool chain to make sure that standard execution starts again.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
