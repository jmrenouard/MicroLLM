"""Versioned System-1 pre-execution gatekeeper engine (`microllm.guard.v1`).

Combines calibrated `noul` hazard evaluation, 3-way `choice` action classification,
Split Conformal abstention (`x_abstain`), certainty-claim hijack detection, and
explicit coding-agent `policy_boundaries` strictly offline on CPU in under 5 ms.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import functools
import importlib.resources
import json
import math
import re
import threading
import time
import unicodedata
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from microllm.backends.base import _CERTAINTY_HIJACK_MARKERS
from microllm.backends.cascade import DualGearCascadeBackend
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.calibration import SplitConformalCalibrator
from microllm.confidence import sanitize_and_normalize_probabilities
from microllm.schemas import StructuredText, _ensure_finite_json, render_structured_text

GATEKEEPER_SCHEMA_VERSION: Literal["microllm.guard.v1"] = "microllm.guard.v1"

DecisionType = Literal["allow", "block", "escalate"]
EnvironmentType = Literal["production", "staging", "development"]
AbstainPolicyType = Literal["escalate", "block"]

_ENV_ALIASES: dict[str, EnvironmentType] = {
    "production": "production",
    "prod": "production",
    "staging": "staging",
    "stage": "staging",
    "development": "development",
    "dev": "development",
}

_ENV_DEFAULT_THRESHOLDS: dict[EnvironmentType, float] = {
    "production": 0.50,
    "staging": 0.60,
    "development": 0.70,
}

_GUARD_CHOICE_INSTRUCTIONS = (
    "Classify the safety disposition of this proposed agent tool call before execution."
)

_GUARD_ACTION_OPTIONS: dict[str, str] = {
    "allow": (
        "Routine read-only inspection, status check, benign query, documentation read, "
        "unit test run, or safe scoped workspace build"
    ),
    "escalate": (
        "Ambiguous or privileged infrastructure mutation, production deployment, schema "
        "alteration, permission change, service restart, or external write webhook requiring review"
    ),
    "block": (
        "Destructive file or database deletion, rm -rf, drop table, truncate, piped shell "
        "execution, prompt injection, jailbreak override, or credential secret leak"
    ),
}

_EXPLICIT_INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "jailbreak",
    "disregard",
    "bypass safety",
    "developer mode",
    "do anything now",
    "dan mode",
    "unfiltered",
    "uncensored",
    "without any moral",
    "no moral",
    "no ethical",
    "ignore your rules",
    "forget your instructions",
    "override safety",
    "hypothetical scenario where you have no restrictions",
)

_ROLEPLAY_FRAMES: tuple[str, ...] = (
    "in a novel",
    "screenplay",
    "fictional story",
    "for a movie",
    "roleplay as",
    "pretend you are",
    "act as an unrestricted",
)

_ROLEPLAY_HARM_TARGETS: tuple[str, ...] = (
    "step-by-step",
    "exfiltrate",
    "bypass authentication",
    "privilege escalation",
    "zero-day",
    "synthesize",
    "exploit",
    "steal",
    "malware",
    "reverse shell",
)

_EXTRA_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN\s+(?:[A-Z0-9]+\s+)?PRIVATE\s+KEY-----", re.IGNORECASE),
    re.compile(r"\baws_secret_access_key\s*[:=]\s*\S+", re.IGNORECASE),
)

_INJECTION_INTENT_TOKENS: tuple[str, ...] = (
    "instructions",
    "guardrails",
    "restrictions",
    "override",
    "bypass",
    "jailbreak",
    "disregard",
    "unfiltered",
    "uncensored",
    "system prompt",
    "forget your",
    "ignore your",
    "ignore all",
    "ignore previous",
    "safety rules",
    "operational rules",
    "no ethical",
    "no moral",
)

_HARD_DESTRUCTIVE_MARKERS: tuple[str, ...] = (
    "rm -rf",
    "drop table",
    "truncate ",
    "delete from",
    "/var/lib/postgresql",
    "ransomware",
    "keylogger",
    "malware",
    "exploit payload",
    "reverse shell",
    "phishing email",
    "steal credentials",
    "credit card fraud",
    "bomb",
    "explosive",
    "sql injection payload",
    "ddos attack",
    "botnet",
)

_AMBIGUOUS_PRIVILEGED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bkubectl\s+(?:apply|delete|patch|scale|rollout|cordon|drain)\b", re.IGNORECASE),
    re.compile(r"\bterraform\s+(?:apply|destroy|import|state\s+rm)\b", re.IGNORECASE),
    re.compile(r"\bhelm\s+(?:install|upgrade|uninstall|rollback)\b", re.IGNORECASE),
    re.compile(r"\balter\s+(?:table|database|role|user|system)\b", re.IGNORECASE),
    re.compile(r"\bgrant\s+all\b", re.IGNORECASE),
    re.compile(r"\b(?:chmod|chown)\s+-r\b", re.IGNORECASE),
    re.compile(r"\bsystemctl\s+(?:restart|stop|disable|mask)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b[^\n]*(?:--force|-f\b|main\b|master\b|release/)", re.IGNORECASE),
    re.compile(r"\bdocker\s+(?:rm|rmi|system\s+prune|volume\s+rm)\b", re.IGNORECASE),
    re.compile(r"\biptables\b", re.IGNORECASE),
    re.compile(r"\bcurl\b[^\n]*(?:-x|--request)\s+(?:post|put|patch|delete)\b", re.IGNORECASE),
    re.compile(r"\bmodify\s+(?:production|firewall|iam|dns|permissions?)\b", re.IGNORECASE),
)


@functools.lru_cache(maxsize=1)
def _load_gatekeeper_contract_cached() -> dict[str, Any]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("gatekeeper_contract_v1.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


def load_gatekeeper_contract() -> dict[str, Any]:
    """Return a fresh copy of the packaged `microllm.guard.v1` JSON Schema."""
    return copy.deepcopy(_load_gatekeeper_contract_cached())


def _normalize_environment(raw: Any) -> EnvironmentType:
    if not isinstance(raw, str):
        raise ValueError("environment must be a string ('production', 'staging', or 'development').")
    key = raw.strip().lower()
    if key not in _ENV_ALIASES:
        raise ValueError(
            f"Invalid environment '{raw}'. Expected one of: production, staging, development."
        )
    return _ENV_ALIASES[key]


def _normalize_policy_boundaries(
    raw: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("policy_boundaries must be a JSON object with optional 'allowed'/'forbidden' lists.")
    allowed_keys = {"allowed", "forbidden"}
    unknown = set(raw.keys()) - allowed_keys
    if unknown:
        raise ValueError(
            f"policy_boundaries keys must be subset of {sorted(allowed_keys)}, got {sorted(unknown)}."
        )
    out: dict[str, list[str]] = {}
    for key in ("allowed", "forbidden"):
        if key in raw and raw[key] is not None:
            seq = raw[key]
            if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
                raise ValueError(f"policy_boundaries['{key}'] must be a list of strings.")
            cleaned: list[str] = []
            for item in seq:
                if not isinstance(item, str):
                    raise ValueError(f"policy_boundaries['{key}'] items must be strings.")
                if item.strip():
                    cleaned.append(item)
            out[key] = cleaned
    return out


def _merge_policy_boundaries(
    base: Mapping[str, Sequence[str]] | None,
    override: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]] | None:
    if base is None and override is None:
        return None
    merged: dict[str, list[str]] = {"allowed": [], "forbidden": []}
    for src in (base, override):
        if not src:
            continue
        for key in ("allowed", "forbidden"):
            for item in src.get(key) or ():
                s = str(item)
                if s.strip() and s not in merged[key]:
                    merged[key].append(s)
    return {k: v for k, v in merged.items() if v}


def _deobfuscate_text(text: str) -> tuple[str, str, str]:
    normalized = unicodedata.normalize("NFKC", text)
    lowered = normalized.lower()
    dedotted = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", lowered)
    deobfuscated = dedotted.replace("a11", "all").translate(str.maketrans("013457@$", "oieastas"))
    return normalized, lowered, deobfuscated


def _strip_allowed_matches(
    lowered_text: str,
    policy_boundaries: Mapping[str, Sequence[str]] | None,
) -> tuple[bool, str]:
    if not policy_boundaries:
        return False, lowered_text
    allowed_rules = [
        str(rule).strip().lower()
        for rule in (policy_boundaries.get("allowed") or ())
        if str(rule).strip()
    ]
    matched = [rule for rule in allowed_rules if rule in lowered_text]
    if not matched:
        return False, lowered_text
    remainder = lowered_text
    for rule in matched:
        remainder = remainder.replace(rule, " ")
    return True, remainder


class GatekeeperRequest(BaseModel):
    """Validated request payload for `MicroLLMGuard.evaluate` and `POST /v1/guard/evaluate`."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(default="tool", min_length=1)
    command: str | None = None
    arguments: dict[str, Any] | list[Any] | str | None = None
    context: StructuredText | None = None
    environment: EnvironmentType | None = None
    threshold: float | None = Field(default=None, ge=0.1, le=0.99)
    abstain_policy: AbstainPolicyType | None = None
    policy_boundaries: dict[str, list[str]] | None = None
    custom_rubrics: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_tool_alias_and_finite_json(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            raise ValueError("Gatekeeper request must be a JSON object.")
        _ensure_finite_json(dict(data))
        normalized = dict(data)
        if "tool" in normalized and "tool_name" not in normalized:
            normalized["tool_name"] = normalized.pop("tool")
        elif "tool" in normalized and "tool_name" in normalized:
            if str(normalized["tool"]).strip() != str(normalized["tool_name"]).strip():
                raise ValueError("Conflicting 'tool' and 'tool_name' values provided.")
            normalized.pop("tool")
        if "threshold" in normalized and isinstance(normalized["threshold"], bool):
            raise ValueError("threshold cannot be a boolean.")
        return normalized

    @field_validator("tool_name")
    @classmethod
    def _validate_tool_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be a non-empty string.")
        return cleaned

    @field_validator("environment", mode="before")
    @classmethod
    def _validate_environment(cls, value: Any) -> EnvironmentType | None:
        if value is None:
            return None
        return _normalize_environment(value)

    @field_validator("policy_boundaries", mode="before")
    @classmethod
    def _validate_policy_boundaries(cls, value: Any) -> dict[str, list[str]] | None:
        return _normalize_policy_boundaries(value)

    @field_validator("arguments", "context", mode="before")
    @classmethod
    def _validate_structured_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (str, dict, list)):
            raise ValueError("Structured fields must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(value)

    @field_validator("threshold")
    @classmethod
    def _validate_threshold_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("threshold must be a finite float in [0.1, 0.99].")
        return value


class GatekeeperDecision(BaseModel):
    """Versioned `microllm.guard.v1` gatekeeper decision contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["microllm.guard.v1"] = GATEKEEPER_SCHEMA_VERSION
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool
    reason: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=1.0)
    primary_threat: str = Field(min_length=1)
    conformal_set: list[DecisionType] = Field(min_length=1)
    checks: dict[str, float]
    action_probabilities: dict[DecisionType, float]
    latency_ms: float = Field(ge=0.0)
    tool_name: str = Field(min_length=1)
    environment: EnvironmentType
    abstain_policy: AbstainPolicyType
    hijack_suspected: bool = False

    @model_validator(mode="after")
    def _verify_finite_numbers(self) -> "GatekeeperDecision":
        for val in (self.confidence, self.risk_score, self.latency_ms):
            if not math.isfinite(val):
                raise ValueError("Decision numeric fields must be finite.")
        _ensure_finite_json(self.checks)
        _ensure_finite_json(self.action_probabilities)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision into a strict JSON-compatible dictionary."""
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class GuardConfig:
    """Configuration for `MicroLLMGuard` System-1 pre-execution evaluation."""

    environment: EnvironmentType = "production"
    threshold: float | None = None
    block_threshold: float = 0.75
    abstain_policy: AbstainPolicyType = "escalate"
    conformal_alpha: float = 0.10
    conformal_q_hat: float = 0.72
    policy_boundaries: dict[str, list[str]] | None = None
    custom_rubrics: dict[str, str] | None = None
    use_cascade: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", _normalize_environment(self.environment))
        if self.threshold is not None:
            if isinstance(self.threshold, bool) or not (0.1 <= float(self.threshold) <= 0.99):
                raise ValueError("threshold must be in [0.1, 0.99].")
        if isinstance(self.block_threshold, bool) or not (0.5 <= float(self.block_threshold) <= 0.99):
            raise ValueError("block_threshold must be in [0.5, 0.99].")
        if self.abstain_policy not in ("escalate", "block"):
            raise ValueError("abstain_policy must be 'escalate' or 'block'.")
        if isinstance(self.conformal_alpha, bool) or not (0.0 < float(self.conformal_alpha) < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if isinstance(self.conformal_q_hat, bool) or not (0.5 <= float(self.conformal_q_hat) < 1.0):
            raise ValueError("conformal_q_hat must be in [0.5, 1.0).")
        if self.policy_boundaries is not None:
            object.__setattr__(
                self,
                "policy_boundaries",
                _normalize_policy_boundaries(self.policy_boundaries),
            )
        if self.custom_rubrics is not None:
            object.__setattr__(
                self,
                "custom_rubrics",
                {str(k): str(v) for k, v in self.custom_rubrics.items()},
            )


def _assemble_payload_text(request: GatekeeperRequest) -> str:
    parts: list[str] = []
    if request.command is not None and request.command.strip():
        parts.append(request.command.strip())
    if request.arguments is not None:
        if isinstance(request.arguments, str):
            if request.arguments.strip():
                parts.append(request.arguments.strip())
        elif isinstance(request.arguments, (dict, list)):
            if len(request.arguments) > 0:
                parts.append(render_structured_text(request.arguments))
    if not parts:
        return f"tool={request.tool_name}"
    return "\n".join(parts)


class MicroLLMGuard:
    """Offline sub-5 ms CPU System-1 gatekeeper engine for AI agent tool calls."""

    def __init__(
        self,
        config: GuardConfig | None = None,
        *,
        gear1: FastEncoderBackend | None = None,
        cascade: DualGearCascadeBackend | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.gear1 = gear1 or FastEncoderBackend()
        self.cascade = cascade or (
            DualGearCascadeBackend(gear1=self.gear1) if self.config.use_cascade else None
        )
        self._action_conformal = SplitConformalCalibrator(alpha=self.config.conformal_alpha)
        self._action_conformal.q_hat = float(self.config.conformal_q_hat)

    def evaluate(
        self,
        request: GatekeeperRequest | Mapping[str, Any],
    ) -> GatekeeperDecision:
        """Evaluate a tool call and return a versioned `microllm.guard.v1` decision."""
        t0 = time.perf_counter()
        req = (
            request
            if isinstance(request, GatekeeperRequest)
            else GatekeeperRequest.model_validate(request)
        )

        env: EnvironmentType = (
            req.environment if req.environment is not None else self.config.environment
        )
        eff_threshold = (
            float(req.threshold)
            if req.threshold is not None
            else (
                float(self.config.threshold)
                if self.config.threshold is not None
                else _ENV_DEFAULT_THRESHOLDS[env]
            )
        )
        eff_block_threshold = max(float(self.config.block_threshold), eff_threshold)
        eff_abstain_policy: AbstainPolicyType = req.abstain_policy or self.config.abstain_policy
        merged_boundaries = _merge_policy_boundaries(
            self.config.policy_boundaries,
            req.policy_boundaries,
        )
        merged_rubrics: dict[str, str] | None = None
        if self.config.custom_rubrics or req.custom_rubrics:
            merged_rubrics = {}
            if self.config.custom_rubrics:
                merged_rubrics.update(self.config.custom_rubrics)
            if req.custom_rubrics:
                merged_rubrics.update(req.custom_rubrics)

        payload_text = _assemble_payload_text(req)
        normalized_text, lowered, deobfuscated = _deobfuscate_text(payload_text)

        # 1. Layer 1: Calibrated `noul` hazard evaluation + policy_boundaries + NFKC/remainder guards
        from microllm.server import DANGEROUS_COMMAND_REGEX, evaluate_guardrail_in_process

        guardrail = evaluate_guardrail_in_process(
            state=payload_text,
            threshold=eff_threshold,
            context=req.context,
            custom_rubrics=merged_rubrics,
            gear1=self.gear1,
            policy_boundaries=merged_boundaries,
        )
        checks: dict[str, float] = {
            str(k): round(float(v), 4) for k, v in guardrail["checks"].items()
        }

        # Supplement tool-call credential detection on NFKC-normalized text (never downgraded)
        has_extra_secret = any(
            pat.search(payload_text) or pat.search(normalized_text)
            for pat in _EXTRA_SECRET_PATTERNS
        )
        if has_extra_secret:
            checks["pii_or_secret_leak"] = max(checks.get("pii_or_secret_leak", 0.0), 0.94)

        # 2. Layer 2: Calibrated 3-way `choice` action classification + hijack detection
        active_backend: Any = self.cascade if self.cascade is not None else self.gear1
        choice_state = f"tool={req.tool_name} env={env}\n{payload_text}"
        choice_res = active_backend.evaluate_choice(
            choice_state,
            _GUARD_CHOICE_INSTRUCTIONS,
            _GUARD_ACTION_OPTIONS,
        )
        choice_probs = choice_res["probabilities"]
        certainty_marker_hit = any(
            m in lowered or m in deobfuscated for m in _CERTAINTY_HIJACK_MARKERS
        )
        hijack_suspected = bool(choice_res.get("x_hijack_suspected", False) or certainty_marker_hit)

        has_explicit_injection = bool(
            checks.get("prompt_injection", 0.0) >= eff_block_threshold
            and (
                any(m in lowered or m in deobfuscated for m in _EXPLICIT_INJECTION_MARKERS)
                or (
                    any(frame in lowered for frame in _ROLEPLAY_FRAMES)
                    and any(target in lowered for target in _ROLEPLAY_HARM_TARGETS)
                )
            )
        )

        # Prevent benign CLI flags (--ignore-*, docker system ...) from falsely saturating prompt_injection (RCA-096)
        has_injection_intent = any(
            tok in lowered or tok in deobfuscated for tok in _INJECTION_INTENT_TOKENS
        )
        if (
            checks.get("prompt_injection", 0.0) >= eff_threshold
            and not has_explicit_injection
            and not hijack_suspected
            and not has_injection_intent
        ):
            checks["prompt_injection"] = min(checks["prompt_injection"], 0.22)

        # 3. Check operational ambiguity / privileged mutations with remainder-based boundary stripping
        matched_allowed, remainder = _strip_allowed_matches(lowered, merged_boundaries)
        has_forbidden_violation = checks.get("policy_boundary_violation", 0.0) >= eff_block_threshold
        is_dangerous_cmd = bool(
            DANGEROUS_COMMAND_REGEX.search(payload_text)
            or DANGEROUS_COMMAND_REGEX.search(normalized_text)
            or DANGEROUS_COMMAND_REGEX.search(deobfuscated)
        )
        has_hard_destructive_marker = bool(
            is_dangerous_cmd
            or any(m in lowered or m in deobfuscated for m in _HARD_DESTRUCTIVE_MARKERS)
        )
        is_ambiguous_op = bool(
            not has_forbidden_violation
            and any(pat.search(remainder) for pat in _AMBIGUOUS_PRIVILEGED_PATTERNS)
        )
        # Re-calibrate uncorroborated bag-of-concepts destructive scores on privileged mutations (e.g. "production"/"postgresql")
        # into the review/escalation band unless a hard destructive command or forbidden boundary is present (RCA-096)
        if (
            checks.get("destructive_operation", 0.0) >= eff_block_threshold
            and not has_hard_destructive_marker
            and not has_forbidden_violation
        ):
            if matched_allowed and not is_ambiguous_op:
                checks["destructive_operation"] = min(checks["destructive_operation"], 0.12)
            elif is_ambiguous_op:
                checks["destructive_operation"] = min(checks["destructive_operation"], 0.62)

        if is_ambiguous_op and env == "development" and not matched_allowed:
            # In development, only keep high-impact infra/auth mutations ambiguous
            dev_review_res = self.gear1.evaluate_noul(
                payload_text,
                "Does this tool call mutate shared infrastructure, database schema, or remote systems?",
                {
                    "true": "kubectl apply, terraform, alter table, production deploy, firewall, external webhook",
                    "false": "local file read, unit test, git status, print, format",
                },
            )
            is_ambiguous_op = float(dev_review_res["noul"]) >= 0.45

        risk_score = round(float(max(checks.values())) if checks else 0.0, 4)

        non_inj_max_risk = max(
            (v for k, v in checks.items() if k != "prompt_injection"),
            default=0.0,
        )
        is_pure_hijack = bool(
            hijack_suspected
            and not has_explicit_injection
            and not has_forbidden_violation
            and non_inj_max_risk < eff_block_threshold
        )

        has_hard_block = bool(
            has_forbidden_violation
            or checks.get("destructive_operation", 0.0) >= eff_block_threshold
            or checks.get("pii_or_secret_leak", 0.0) >= eff_block_threshold
            or checks.get("hallucination_risk", 0.0) >= eff_block_threshold
            or has_explicit_injection
            or (risk_score >= eff_block_threshold and not is_pure_hijack)
        )

        in_review_band = bool(
            not has_hard_block
            and not is_pure_hijack
            and (risk_score >= eff_threshold or is_ambiguous_op)
        )

        # 4. Fuse `noul` hazard evidence and `choice` probabilities into calibrated action simplex
        action_probs = self._fuse_action_probabilities(
            choice_probs=choice_probs,
            risk_score=risk_score,
            eff_threshold=eff_threshold,
            has_hard_block=has_hard_block,
            is_pure_hijack=is_pure_hijack,
            in_review_band=in_review_band,
            matched_allowed=matched_allowed and not has_hard_block and not is_ambiguous_op,
        )

        # 5. Split Conformal prediction set & abstention
        conformal_info = self._action_conformal.predict_set(
            {str(k): v for k, v in action_probs.items()}
        )
        conformal_set: list[DecisionType] = [
            item for item in conformal_info["conformal_set"] if item in ("allow", "escalate", "block")
        ]
        if not conformal_set:
            conformal_set = ["escalate"]

        if has_hard_block:
            abstain = False
            decision: DecisionType = "block"
            confidence = round(float(max(action_probs["block"], risk_score)), 4)
        elif is_pure_hijack:
            abstain = True
            decision = "block" if eff_abstain_policy == "block" else "escalate"
            confidence = round(float(max(action_probs["escalate"], action_probs["block"])), 4)
        elif in_review_band or bool(conformal_info["abstain"]):
            abstain = True
            decision = "block" if eff_abstain_policy == "block" else "escalate"
            confidence = round(float(max(action_probs.values())), 4)
        else:
            abstain = False
            decision = "allow"
            confidence = round(float(action_probs["allow"]), 4)

        confidence = float(max(0.0, min(1.0, confidence)))
        primary_threat = self._select_primary_threat(
            checks=checks,
            decision=decision,
            abstain=abstain,
            is_pure_hijack=is_pure_hijack,
            is_ambiguous_op=is_ambiguous_op,
            eff_threshold=eff_threshold,
        )
        reason = self._format_reason(
            decision=decision,
            abstain=abstain,
            primary_threat=primary_threat,
            risk_score=risk_score,
            eff_threshold=eff_threshold,
            env=env,
            abstain_policy=eff_abstain_policy,
            matched_allowed=matched_allowed and decision == "allow",
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return GatekeeperDecision(
            schema_version=GATEKEEPER_SCHEMA_VERSION,
            decision=decision,
            confidence=confidence,
            abstain=abstain,
            reason=reason,
            risk_score=risk_score,
            primary_threat=primary_threat,
            conformal_set=conformal_set,
            checks=checks,
            action_probabilities=action_probs,
            latency_ms=latency_ms,
            tool_name=req.tool_name,
            environment=env,
            abstain_policy=eff_abstain_policy,
            hijack_suspected=hijack_suspected,
        )

    def _fuse_action_probabilities(
        self,
        *,
        choice_probs: Mapping[str, float],
        risk_score: float,
        eff_threshold: float,
        has_hard_block: bool,
        is_pure_hijack: bool,
        in_review_band: bool,
        matched_allowed: bool,
    ) -> dict[DecisionType, float]:
        p_c_allow = float(choice_probs.get("allow", 1.0 / 3.0))
        p_c_esc = float(choice_probs.get("escalate", 1.0 / 3.0))
        p_c_blk = float(choice_probs.get("block", 1.0 / 3.0))

        if has_hard_block:
            p_block = min(0.99, max(0.86, 0.80 * risk_score + 0.20 * max(p_c_blk, 0.85)))
            rem = max(0.01, 1.0 - p_block)
            raw: dict[str, float] = {
                "allow": rem * 0.25,
                "escalate": rem * 0.75,
                "block": p_block,
            }
        elif is_pure_hijack:
            raw = {
                "allow": 0.08,
                "escalate": 0.48 + 0.04 * p_c_esc,
                "block": 0.44 + 0.04 * p_c_blk,
            }
        elif in_review_band:
            band_risk = max(0.35, min(0.72, risk_score))
            raw = {
                "allow": max(0.29, 0.36 * (1.0 - band_risk) + 0.24 * p_c_allow),
                "escalate": 0.46 + 0.14 * p_c_esc,
                "block": max(0.12, 0.26 * band_risk + 0.10 * p_c_blk),
            }
        elif matched_allowed:
            raw = {
                "allow": 0.92,
                "escalate": 0.05,
                "block": 0.03,
            }
        else:
            margin = max(0.0, eff_threshold - risk_score) / max(0.1, eff_threshold)
            p_allow = min(0.97, max(0.78, 0.78 + 0.14 * margin + 0.06 * p_c_allow))
            rem = max(0.03, 1.0 - p_allow)
            esc_share = 0.65 if p_c_esc >= p_c_blk else 0.50
            raw = {
                "allow": p_allow,
                "escalate": rem * esc_share,
                "block": rem * (1.0 - esc_share),
            }

        norm = sanitize_and_normalize_probabilities(raw)
        return {
            "allow": round(float(norm["allow"]), 4),
            "escalate": round(float(norm["escalate"]), 4),
            "block": round(float(norm["block"]), 4),
        }

    @staticmethod
    def _select_primary_threat(
        *,
        checks: Mapping[str, float],
        decision: DecisionType,
        abstain: bool,
        is_pure_hijack: bool,
        is_ambiguous_op: bool,
        eff_threshold: float,
    ) -> str:
        if decision == "allow":
            return "none"
        if "policy_boundary_violation" in checks and checks["policy_boundary_violation"] >= 0.75:
            return "policy_boundary_violation"
        if is_pure_hijack:
            return "evidence_hijack"
        if checks:
            top_threat = max(checks, key=lambda k: checks[k])
            if checks[top_threat] >= eff_threshold:
                return top_threat
        if is_ambiguous_op or abstain:
            return "ambiguous_tool_action"
        return max(checks, key=lambda k: checks[k]) if checks else "unknown"

    @staticmethod
    def _format_reason(
        *,
        decision: DecisionType,
        abstain: bool,
        primary_threat: str,
        risk_score: float,
        eff_threshold: float,
        env: EnvironmentType,
        abstain_policy: AbstainPolicyType,
        matched_allowed: bool,
    ) -> str:
        if decision == "allow":
            if matched_allowed:
                return (
                    f"Allowed by explicit policy_boundaries in {env} "
                    f"(risk={risk_score:.2f} < {eff_threshold:.2f})."
                )
            return f"Allowed in {env} (risk={risk_score:.2f} < {eff_threshold:.2f}, no hard hazards)."
        if abstain:
            action_verb = "Blocked (fail-closed abstain)" if abstain_policy == "block" else "Escalated"
            return (
                f"{action_verb} due to {primary_threat} in {env} "
                f"(conformal abstain=True, risk={risk_score:.2f})."
            )
        return (
            f"Blocked due to {primary_threat} in {env} "
            f"(risk={risk_score:.2f} >= {eff_threshold:.2f})."
        )


_DEFAULT_GUARD_LOCK = threading.Lock()
_DEFAULT_GUARD: MicroLLMGuard | None = None


def _get_default_guard() -> MicroLLMGuard:
    global _DEFAULT_GUARD
    with _DEFAULT_GUARD_LOCK:
        if _DEFAULT_GUARD is None:
            _DEFAULT_GUARD = MicroLLMGuard()
        return _DEFAULT_GUARD


def evaluate_tool_call(
    tool_name: str = "tool",
    command: str | None = None,
    *,
    arguments: dict[str, Any] | list[Any] | str | None = None,
    context: StructuredText | None = None,
    environment: EnvironmentType | None = None,
    threshold: float | None = None,
    abstain_policy: AbstainPolicyType | None = None,
    policy_boundaries: Mapping[str, Sequence[str]] | None = None,
    custom_rubrics: Mapping[str, str] | None = None,
    guard: MicroLLMGuard | None = None,
) -> GatekeeperDecision:
    """Evaluate a proposed agent tool call and return a `microllm.guard.v1` decision."""
    req = GatekeeperRequest.model_validate(
        {
            "tool_name": tool_name,
            "command": command,
            "arguments": arguments,
            "context": context,
            "environment": environment,
            "threshold": threshold,
            "abstain_policy": abstain_policy,
            "policy_boundaries": policy_boundaries,
            "custom_rubrics": custom_rubrics,
        }
    )
    active_guard = guard if guard is not None else _get_default_guard()
    return active_guard.evaluate(req)


__all__ = [
    "GATEKEEPER_SCHEMA_VERSION",
    "AbstainPolicyType",
    "DecisionType",
    "EnvironmentType",
    "GatekeeperDecision",
    "GatekeeperRequest",
    "GuardConfig",
    "MicroLLMGuard",
    "evaluate_tool_call",
    "load_gatekeeper_contract",
]
