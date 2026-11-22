"""Standalone CLI entry point for `microllm-guard` (`microllm.guard.v1` gatekeeper).

Deterministic process exit codes:
- `0`: decision == "allow"
- `1`: invalid input, malformed JSON, missing policy file, or usage error
- `2`: decision == "block"
- `3`: decision == "escalate" (conformal abstention / human review)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence, TextIO

from pydantic import ValidationError

from microllm.guard import (
    GATEKEEPER_SCHEMA_VERSION,
    GatekeeperRequest,
    GuardConfig,
    MicroLLMGuard,
    _get_default_guard,
    _normalize_policy_boundaries,
    load_gatekeeper_contract,
)

EXIT_ALLOW = 0
EXIT_INVALID_INPUT = 1
EXIT_BLOCK = 2
EXIT_ESCALATE = 3


class _GuardArgumentParser(argparse.ArgumentParser):
    """ArgumentParser raising ValueError on usage errors so exit code 1 never collides with EXIT_BLOCK (2)."""

    def error(self, message: str) -> Any:
        raise ValueError(message)


def _reject_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant '{token}' is not permitted.")


def _loads_strict_json(raw: str, source_label: str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_non_finite_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {source_label}: {exc}") from exc


def load_policy_file(path: str | Path) -> dict[str, list[str]]:
    """Load `policy_boundaries` (`allowed`/`forbidden`) from a JSON policy file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Policy file not found: {file_path}")
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to read policy file '{file_path}': {exc}") from exc
    parsed = _loads_strict_json(raw, f"policy file '{file_path}'")
    if not isinstance(parsed, Mapping):
        raise ValueError("Policy file must contain a JSON object.")
    boundaries_obj = parsed.get("policy_boundaries", parsed)
    normalized = _normalize_policy_boundaries(boundaries_obj)
    return normalized if normalized is not None else {}


def build_guard_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `microllm-guard` evaluation mode."""
    parser = _GuardArgumentParser(
        prog="microllm-guard",
        description=(
            "MicroLLM System-1 Pre-Execution Gatekeeper CLI (`microllm.guard.v1`). "
            "Exit codes: 0=allow, 2=block, 3=escalate, 1=invalid input."
        ),
    )
    parser.add_argument("--tool", dest="tool_name", default=None, help="Tool name (e.g. bash, sql, file_write)")
    parser.add_argument("--command", default=None, help="Command, SQL query, or payload text to evaluate")
    parser.add_argument("--args-json", dest="args_json", default=None, help="JSON-encoded tool arguments")
    parser.add_argument("--context", default=None, help="Optional reference context for groundedness verification")
    parser.add_argument(
        "--env",
        "--environment",
        dest="environment",
        default=None,
        help="Execution environment ('production', 'staging', or 'development')",
    )
    parser.add_argument(
        "--policy-file",
        dest="policy_file",
        default=None,
        help="Path to JSON file defining policy_boundaries ({allowed: [...], forbidden: [...]})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Risk score threshold in [0.1, 0.99]",
    )
    parser.add_argument(
        "--fail-closed",
        dest="fail_closed",
        action="store_true",
        help="Map conformal abstention to 'block' (exit 2) instead of 'escalate' (exit 3)",
    )
    parser.add_argument(
        "--abstain-policy",
        dest="abstain_policy",
        choices=["escalate", "block"],
        default=None,
        help="Action taken on abstain=True ('escalate' or 'block')",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit strict single-object microllm.guard.v1 JSON to stdout",
    )
    parser.add_argument(
        "--stdin",
        dest="from_stdin",
        action="store_true",
        help="Explicitly read base JSON request from stdin even when --command or --args-json is supplied",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="Print the packaged microllm.guard.v1 JSON Schema and exit 0",
    )
    return parser


def decision_to_exit_code(decision: str) -> int:
    """Map a `microllm.guard.v1` decision string to its deterministic process exit code."""
    if decision == "allow":
        return EXIT_ALLOW
    if decision == "block":
        return EXIT_BLOCK
    if decision == "escalate":
        return EXIT_ESCALATE
    return EXIT_INVALID_INPUT


def _emit_error(
    message: str,
    *,
    as_json: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if as_json:
        err_payload = {
            "schema_version": GATEKEEPER_SCHEMA_VERSION,
            "error": message,
        }
        stdout.write(json.dumps(err_payload, ensure_ascii=False, allow_nan=False) + "\n")
        stdout.flush()
    else:
        stderr.write(f"microllm-guard error: {message}\n")
        stderr.flush()
    return EXIT_INVALID_INPUT


def run_guard_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin_stream: TextIO | None = None,
    stdout_stream: TextIO | None = None,
    stderr_stream: TextIO | None = None,
) -> int:
    """Execute `microllm-guard` CLI and return deterministic exit code (0/2/3/1)."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    in_stream = sys.stdin if stdin_stream is None else stdin_stream
    out_stream = sys.stdout if stdout_stream is None else stdout_stream
    err_stream = sys.stderr if stderr_stream is None else stderr_stream
    wants_json = "--json" in args_list

    parser = build_guard_cli_parser()
    try:
        parsed = parser.parse_args(args_list)
    except SystemExit as exc:
        return 0 if exc.code == 0 else EXIT_INVALID_INPUT
    except ValueError as exc:
        return _emit_error(str(exc), as_json=wants_json, stdout=out_stream, stderr=err_stream)

    if parsed.contract:
        out_stream.write(json.dumps(load_gatekeeper_contract(), indent=2, ensure_ascii=False) + "\n")
        out_stream.flush()
        return EXIT_ALLOW

    payload: dict[str, Any] = {}
    has_cli_payload = any(val is not None for val in (parsed.command, parsed.args_json))
    has_cli_input = has_cli_payload or parsed.tool_name is not None

    stdin_raw = ""
    if parsed.from_stdin or not has_cli_payload:
        try:
            is_tty = bool(in_stream.isatty())
        except Exception:
            is_tty = False
        if not is_tty:
            stdin_raw = in_stream.read().strip()

    if stdin_raw:
        try:
            stdin_obj = _loads_strict_json(stdin_raw, "stdin")
        except ValueError as exc:
            return _emit_error(str(exc), as_json=parsed.as_json, stdout=out_stream, stderr=err_stream)
        if not isinstance(stdin_obj, Mapping):
            return _emit_error(
                "stdin payload must be a JSON object.",
                as_json=parsed.as_json,
                stdout=out_stream,
                stderr=err_stream,
            )
        payload.update(dict(stdin_obj))
    elif not has_cli_input:
        return _emit_error(
            "No tool call provided. Supply --command/--args-json/--tool or pipe JSON via stdin.",
            as_json=parsed.as_json,
            stdout=out_stream,
            stderr=err_stream,
        )

    if parsed.tool_name is not None:
        payload.pop("tool", None)
        payload["tool_name"] = parsed.tool_name
    if parsed.command is not None:
        payload["command"] = parsed.command
    if parsed.args_json is not None:
        try:
            payload["arguments"] = _loads_strict_json(parsed.args_json, "--args-json")
        except ValueError as exc:
            return _emit_error(str(exc), as_json=parsed.as_json, stdout=out_stream, stderr=err_stream)
    if parsed.context is not None:
        payload["context"] = parsed.context
    if parsed.environment is not None:
        payload["environment"] = parsed.environment
    if parsed.threshold is not None:
        payload["threshold"] = parsed.threshold

    eff_abstain_policy = parsed.abstain_policy
    if parsed.fail_closed:
        eff_abstain_policy = "block"
    if eff_abstain_policy is not None:
        payload["abstain_policy"] = eff_abstain_policy

    file_boundaries: dict[str, list[str]] | None = None
    if parsed.policy_file is not None:
        try:
            file_boundaries = load_policy_file(parsed.policy_file)
        except ValueError as exc:
            return _emit_error(str(exc), as_json=parsed.as_json, stdout=out_stream, stderr=err_stream)

    try:
        req = GatekeeperRequest.model_validate(payload)
        guard = (
            MicroLLMGuard(GuardConfig(policy_boundaries=file_boundaries))
            if file_boundaries is not None
            else _get_default_guard()
        )
        decision = guard.evaluate(req)
    except (ValidationError, ValueError, TypeError) as exc:
        return _emit_error(str(exc), as_json=parsed.as_json, stdout=out_stream, stderr=err_stream)

    if parsed.as_json:
        out_stream.write(json.dumps(decision.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
    else:
        out_stream.write(
            f"[GUARD:{decision.decision.upper()}] tool={decision.tool_name} "
            f"conf={decision.confidence:.2f} abstain={decision.abstain} "
            f"risk={decision.risk_score:.2f} ({decision.primary_threat}) "
            f"[{decision.latency_ms:.2f}ms] {decision.reason}\n"
        )
    out_stream.flush()
    return decision_to_exit_code(decision.decision)


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entry point for `microllm-guard`."""
    code = run_guard_cli(argv)
    if argv is None:
        raise SystemExit(code)
    return code


__all__ = [
    "EXIT_ALLOW",
    "EXIT_BLOCK",
    "EXIT_ESCALATE",
    "EXIT_INVALID_INPUT",
    "build_guard_cli_parser",
    "decision_to_exit_code",
    "load_policy_file",
    "main",
    "run_guard_cli",
]


if __name__ == "__main__":
    main()
