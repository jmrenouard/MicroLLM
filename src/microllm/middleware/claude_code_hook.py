"""Drop-in Claude Code `PreToolUse` hook adapter and settings generator (`microllm.guard.v1`).

Evaluates Claude Code `Bash`, `Write`, `Edit`, `MultiEdit`, `WebFetch`, and `mcp__*`
tool calls locally on CPU in under 5 ms with zero third-party dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
import posixpath
import re
import sys
import unicodedata
from typing import Any, Mapping, TextIO

from pydantic import ValidationError

from microllm.guard import (
    GATEKEEPER_SCHEMA_VERSION,
    AbstainPolicyType,
    EnvironmentType,
    GatekeeperDecision,
    GatekeeperRequest,
    GuardConfig,
    MicroLLMGuard,
    _get_default_guard,
    _normalize_environment,
    _normalize_policy_boundaries,
)
from microllm.middleware.mcp_guard import extract_mcp_tool_request

DEFAULT_CLAUDE_HOOK_MATCHER = "Bash|Write|Edit|MultiEdit|WebFetch|mcp__.*"
PROJECT_POLICY_FILENAME = ".microllm-guard.json"

_SENSITIVE_WRITE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[\\/])etc[\\/](?:g?shadow|sudoers|passwd|master\.passwd)$", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])\.ssh[\\/](?:authorized_keys|id_rsa|id_ed25519)$", re.IGNORECASE),
)


def _reject_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant '{token}' is not permitted.")


def _loads_strict_json(raw: str, label: str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_non_finite_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {label}: {exc}") from exc


def load_claude_project_config(
    *,
    cwd: str | Path | None = None,
    policy_file: str | Path | None = None,
) -> GuardConfig | None:
    """Load optional `.microllm-guard.json` project policy config from explicit path or workspace root."""
    target: Path | None = None
    if policy_file is not None:
        target = Path(policy_file)
        if not target.is_file():
            raise ValueError(f"Policy file not found: {target}")
    elif cwd is not None and str(cwd).strip():
        candidate = Path(str(cwd)) / PROJECT_POLICY_FILENAME
        if candidate.is_file():
            target = candidate

    if target is None:
        return None

    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read policy config '{target}': {exc}") from exc

    parsed = _loads_strict_json(raw, f"policy config '{target}'")
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Policy config '{target}' must be a JSON object.")

    # Support flat {allowed, forbidden} or structured {environment, threshold, abstain_policy, policy_boundaries}
    if "policy_boundaries" in parsed or any(
        k in parsed for k in ("environment", "threshold", "abstain_policy", "custom_rubrics")
    ):
        raw_bounds = parsed.get("policy_boundaries")
        boundaries = _normalize_policy_boundaries(raw_bounds) if raw_bounds is not None else None
        raw_env = parsed.get("environment", "production")
        env: EnvironmentType = _normalize_environment(raw_env)
        raw_thresh = parsed.get("threshold")
        thresh = float(raw_thresh) if raw_thresh is not None and not isinstance(raw_thresh, bool) else None
        if isinstance(raw_thresh, bool):
            raise ValueError("threshold in policy config cannot be a boolean.")
        raw_abstain = parsed.get("abstain_policy", "escalate")
        if raw_abstain not in ("escalate", "block"):
            raise ValueError("abstain_policy in policy config must be 'escalate' or 'block'.")
        abstain_pol: AbstainPolicyType = raw_abstain
        raw_rubrics = parsed.get("custom_rubrics")
        if raw_rubrics is not None and not isinstance(raw_rubrics, Mapping):
            raise ValueError("custom_rubrics in policy config must be an object.")
        custom_rubrics = (
            {str(k): str(v) for k, v in raw_rubrics.items()} if raw_rubrics is not None else None
        )
        return GuardConfig(
            environment=env,
            threshold=thresh,
            abstain_policy=abstain_pol,
            policy_boundaries=boundaries,
            custom_rubrics=custom_rubrics,
        )

    boundaries = _normalize_policy_boundaries(parsed)
    return GuardConfig(policy_boundaries=boundaries)


def _extract_claude_tool_request(
    payload: Mapping[str, Any],
    *,
    environment: EnvironmentType | None = None,
) -> tuple[GatekeeperRequest, str | None]:
    """Translate a Claude Code `PreToolUse` JSON payload into a `GatekeeperRequest`.

    Returns `(request, sensitive_path_reason)` where `sensitive_path_reason` is non-None
    if a Write/Edit/MultiEdit targets a critical system credential file.
    """
    raw_tool = payload.get("tool_name")
    if not isinstance(raw_tool, str) or not raw_tool.strip():
        raise ValueError("PreToolUse payload requires a non-empty string 'tool_name'.")
    tool_name = raw_tool.strip()
    tool_input = payload.get("tool_input")
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, Mapping):
        raise ValueError("PreToolUse payload 'tool_input' must be a JSON object.")

    sensitive_reason: str | None = None
    lower_tool = tool_name.lower()

    if lower_tool == "bash":
        cmd = tool_input.get("command")
        command = str(cmd) if cmd is not None else None
        extra = {k: v for k, v in tool_input.items() if k != "command"}
        req = GatekeeperRequest.model_validate(
            {
                "tool_name": tool_name,
                "command": command,
                "arguments": extra if extra else None,
                "environment": environment,
            }
        )
        return req, None

    if lower_tool in ("write", "edit", "multiedit"):
        file_path = str(tool_input.get("file_path") or tool_input.get("path") or "").strip()
        if file_path:
            nfkc_path = unicodedata.normalize("NFKC", file_path).replace("\\", "/")
            norm_path = posixpath.normpath(nfkc_path)
            if any(
                pat.search(nfkc_path) or pat.search(norm_path)
                for pat in _SENSITIVE_WRITE_PATH_PATTERNS
            ):
                sensitive_reason = f"Write/edit targets protected system credential path '{file_path}'."

        text_parts: list[str] = []
        if file_path:
            text_parts.append(f"file_path={file_path}")
        if lower_tool == "write" and tool_input.get("content") is not None:
            text_parts.append(str(tool_input["content"]))
        elif lower_tool == "edit" and tool_input.get("new_string") is not None:
            text_parts.append(str(tool_input["new_string"]))
        elif lower_tool == "multiedit":
            edits = tool_input.get("edits")
            if isinstance(edits, list):
                for item in edits:
                    if isinstance(item, Mapping) and item.get("new_string") is not None:
                        text_parts.append(str(item["new_string"]))

        req = GatekeeperRequest.model_validate(
            {
                "tool_name": tool_name,
                "command": "\n".join(text_parts) if text_parts else None,
                "arguments": None if text_parts else dict(tool_input),
                "environment": environment,
            }
        )
        return req, sensitive_reason

    if lower_tool == "webfetch":
        url = str(tool_input.get("url") or "").strip()
        prompt = str(tool_input.get("prompt") or "").strip()
        combined = "\n".join(p for p in (url, prompt) if p)
        req = GatekeeperRequest.model_validate(
            {
                "tool_name": tool_name,
                "command": combined if combined else None,
                "arguments": None if combined else dict(tool_input),
                "environment": environment,
            }
        )
        return req, None

    # Default: MCP tools (`mcp__*`) and custom Claude Code tools
    return extract_mcp_tool_request({"name": tool_name, "arguments": tool_input}, environment=environment), None


def format_claude_hook_output(decision: GatekeeperDecision) -> dict[str, Any]:
    """Map a `GatekeeperDecision` to Claude Code's `PreToolUse` `hookSpecificOutput` JSON."""
    if decision.decision == "allow":
        perm = "allow"
    elif decision.decision == "block":
        perm = "deny"
    else:
        perm = "ask"

    dec_dict = decision.to_dict()
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": perm,
            "permissionDecisionReason": decision.reason,
            "additionalContext": json.dumps(
                {
                    "schema_version": decision.schema_version,
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "abstain": decision.abstain,
                    "primary_threat": decision.primary_threat,
                    "latency_ms": decision.latency_ms,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
        },
        "guard": dec_dict,
    }


def _format_fail_closed_hook_error(message: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"microllm-guard fail-closed error: {message}",
        },
        "error": message,
        "schema_version": GATEKEEPER_SCHEMA_VERSION,
    }


def evaluate_claude_pre_tool_use(
    payload: Mapping[str, Any],
    *,
    guard: MicroLLMGuard | None = None,
    policy_file: str | Path | None = None,
    environment: EnvironmentType | None = None,
    fail_closed: bool = False,
) -> tuple[dict[str, Any], GatekeeperDecision]:
    """Evaluate a parsed Claude Code `PreToolUse` payload and return `(hook_json, decision)`."""
    if not isinstance(payload, Mapping):
        raise ValueError("PreToolUse payload must be a JSON object.")

    cwd = payload.get("cwd")
    project_cfg = load_claude_project_config(
        cwd=str(cwd) if isinstance(cwd, str) else None,
        policy_file=policy_file,
    )

    active_guard = guard
    if active_guard is None:
        if project_cfg is not None:
            eff_abstain: AbstainPolicyType = "block" if fail_closed else project_cfg.abstain_policy
            eff_env: EnvironmentType = environment if environment is not None else project_cfg.environment
            active_guard = MicroLLMGuard(
                GuardConfig(
                    environment=eff_env,
                    threshold=project_cfg.threshold,
                    abstain_policy=eff_abstain,
                    policy_boundaries=project_cfg.policy_boundaries,
                    custom_rubrics=project_cfg.custom_rubrics,
                )
            )
        elif environment is None and not fail_closed:
            active_guard = _get_default_guard()
        else:
            active_guard = MicroLLMGuard(
                GuardConfig(
                    environment=environment or "production",
                    abstain_policy="block" if fail_closed else "escalate",
                )
            )

    req, sensitive_path_reason = _extract_claude_tool_request(payload, environment=environment)
    decision = active_guard.evaluate(req)

    if sensitive_path_reason is not None and decision.decision != "block":
        decision = decision.model_copy(
            update={
                "decision": "block",
                "confidence": 0.96,
                "abstain": False,
                "risk_score": max(decision.risk_score, 0.96),
                "primary_threat": "policy_boundary_violation",
                "reason": sensitive_path_reason,
                "conformal_set": ["block"],
            }
        )

    return format_claude_hook_output(decision), decision


def run_claude_hook_stdin(
    *,
    policy_file: str | Path | None = None,
    environment: EnvironmentType | None = None,
    fail_closed: bool = False,
    exit_on_block: bool = False,
    stdin_stream: TextIO | None = None,
    stdout_stream: TextIO | None = None,
    stderr_stream: TextIO | None = None,
) -> int:
    """Read Claude Code `PreToolUse` JSON from stdin, evaluate locally, and emit hook JSON."""
    in_stream = sys.stdin if stdin_stream is None else stdin_stream
    out_stream = sys.stdout if stdout_stream is None else stdout_stream
    err_stream = sys.stderr if stderr_stream is None else stderr_stream

    raw = in_stream.read().strip()
    if not raw:
        err_obj = _format_fail_closed_hook_error("Empty stdin payload received by PreToolUse hook.")
        out_stream.write(json.dumps(err_obj, ensure_ascii=False, allow_nan=False) + "\n")
        out_stream.flush()
        err_stream.write("microllm-guard claude-hook error: Empty stdin payload.\n")
        err_stream.flush()
        return 2

    try:
        parsed = _loads_strict_json(raw, "PreToolUse stdin")
        hook_out, decision = evaluate_claude_pre_tool_use(
            parsed,
            policy_file=policy_file,
            environment=environment,
            fail_closed=fail_closed,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        err_obj = _format_fail_closed_hook_error(str(exc))
        out_stream.write(json.dumps(err_obj, ensure_ascii=False, allow_nan=False) + "\n")
        out_stream.flush()
        err_stream.write(f"microllm-guard claude-hook error: {exc}\n")
        err_stream.flush()
        return 2

    out_stream.write(json.dumps(hook_out, ensure_ascii=False, allow_nan=False) + "\n")
    out_stream.flush()
    if exit_on_block and decision.decision == "block":
        return 2
    return 0


def build_claude_hooks_config(
    *,
    command: str = "microllm-guard claude-hook",
    matcher: str = DEFAULT_CLAUDE_HOOK_MATCHER,
) -> dict[str, Any]:
    """Generate a valid Claude Code `.claude/settings.json` `PreToolUse` hook configuration block."""
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": matcher,
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                        }
                    ],
                }
            ]
        }
    }


def write_claude_hooks_config(
    target_path: str | Path,
    *,
    command: str = "microllm-guard claude-hook",
    matcher: str = DEFAULT_CLAUDE_HOOK_MATCHER,
) -> dict[str, Any]:
    """Merge the `microllm-guard` `PreToolUse` hook cleanly into a `.claude/settings.json` file."""
    path = Path(target_path)
    new_cfg = build_claude_hooks_config(command=command, matcher=matcher)
    new_entry = new_cfg["hooks"]["PreToolUse"][0]

    merged: dict[str, Any] = {}
    if path.is_file():
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Unable to read Claude settings file '{path}': {exc}") from exc
        if raw:
            existing = _loads_strict_json(raw, f"Claude settings file '{path}'")
            if not isinstance(existing, dict):
                raise ValueError(f"Existing Claude settings file '{path}' must be a JSON object.")
            merged = existing

    hooks_section = merged.get("hooks")
    if hooks_section is None:
        hooks_section = {}
        merged["hooks"] = hooks_section
    if not isinstance(hooks_section, dict):
        raise ValueError(f"'hooks' in '{path}' must be a JSON object.")

    pre_list = hooks_section.get("PreToolUse")
    if pre_list is None:
        pre_list = []
        hooks_section["PreToolUse"] = pre_list
    if not isinstance(pre_list, list):
        raise ValueError(f"'hooks.PreToolUse' in '{path}' must be a JSON array.")

    already_present = any(
        isinstance(entry, Mapping)
        and any(
            isinstance(h, Mapping) and h.get("command") == command
            for h in (entry.get("hooks") or ())
        )
        for entry in pre_list
    )
    if not already_present:
        pre_list.append(new_entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


__all__ = [
    "DEFAULT_CLAUDE_HOOK_MATCHER",
    "PROJECT_POLICY_FILENAME",
    "build_claude_hooks_config",
    "evaluate_claude_pre_tool_use",
    "format_claude_hook_output",
    "load_claude_project_config",
    "run_claude_hook_stdin",
    "write_claude_hooks_config",
]
