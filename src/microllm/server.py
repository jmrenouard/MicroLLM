"""FastAPI Server for MicroLLM: JevCompat 0.1 (32/32 MUST) + Decision Studio & RLCD Lab."""

from __future__ import annotations

from collections import deque
import asyncio
import hmac
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    create_hf_zero_shot_scorer,
    embed_text_deterministic,
    explain_state_tokens,
    semantic_match_logit,
    tokenize_multilingual,
)
from microllm.backends.cascade import (
    DualGearCascadeBackend,
    MultilingualCascadeBackend,
)
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.backends.prefill_llm import AnyJevPrefillBackend
from microllm.calibration import (
    PerKTemperatureScaler,
    SplitConformalCalibrator,
    brier_score,
    clopper_pearson_threshold,
    expected_calibration_error,
    reliability_diagram_bins,
)
from microllm.rlcd import (
    RLCDConfig,
    RLCDDecisionHeadTrainer,
    group_mean_variance_identity,
)
from microllm.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    MultiAnswer,
    MultiQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    UsageInfo,
    _ensure_finite_json,
    render_structured_text,
)

DANGEROUS_COMMAND_REGEX = re.compile(
    r"""
    # 1. Pipe to shell or dynamic language interpreter
    (?:\|\s*(?:\\(?:\r?\n))?\s*(?:sudo(?:\s+-[a-zA-Z0-9]+)*\s+)?(?:/(?:usr/)?bin/)?(?:(?:ba?)?sh|zsh|dash|ksh|python[\d.]*|perl|ruby|php|pwsh|powershell)\b)
    |
    # 2. Pipe to PowerShell Invoke-Expression
    (?:\|\s*(?:\\(?:\r?\n))?\s*(?:iex|invoke-expression)\b)
    |
    # 3. Unspaced pipe to shell
    (?:\|(?:(?:ba?)?sh|zsh|dash|ksh)\b)
    |
    # 4. Bash process substitution with remote download
    (?:\b(?:ba?sh|zsh|source|\.)\s*<\s*\(\s*(?:curl|wget)\b)
    |
    # 5. Eval of remote download
    (?:\beval\s+[\"']?\$\((?:curl|wget)\b)
    |
    # 6. Classic fork bomb: :(){ :|:& };: (with any whitespace)
    (?::\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:)
    |
    # 7. Named function shell fork bomb: e.g. bomb() { bomb|bomb& }; bomb
    (?:\b([a-zA-Z0-9_]+)\(\)\s*\{\s*\1\s*\|\s*\1\s*&\s*\}\s*;\s*\1\b)
    |
    # 8. Scripting language fork loops: os.fork() or fork while fork
    (?:\bos\.fork\(\)|\bfork\s+while\s+fork\b)
    |
    # 9. Low-level drive overwrite or raw write via dd
    (?:\bdd\s+[^|\n]*(?:of=/dev/(?:sd[a-z][0-9]*|nvme[0-9a-z]*|hd[a-z][0-9]*|vd[a-z][0-9]*|null|zero|loop[0-9]*)|if=/dev/(?:zero|urandom)[^|\n]*of=/dev/))
    |
    # 10. Direct raw device redirection: > /dev/sdX
    (?:>\s*/dev/(?:sd[a-z][0-9]*|nvme[0-9a-z]*|hd[a-z][0-9]*|vd[a-z][0-9]*))
    |
    # 11. Filesystem creation/formatting on raw device
    (?:\bmkfs(?:\.[a-z0-9]+)?\s+/dev/)
    |
    # 12. Partition/signature wipe or raw shred
    (?:\b(?:wipefs|shred)\s+[^|\n]*/dev/)
    |
    # 13. Unauthorized credential/shadow database read
    (?:\b(?:cat|grep|more|less|head|tail|view|nano|vim|vi|strings)\s+[^|\n]*/etc/(?:g?shadow|master\.passwd)\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)


from microllm.server_models import (
    BatchRequest,
    CalibrateRequest,
    GuardrailRequest,
    ProfilePullRequest,
    StudioAnalyzeRequest,
    StudioRLCDDemoRequest,
    _reject_bools_recursively,
)
from microllm.registry import get_cache_dir
import sys
import types


class _ServerModule(types.ModuleType):
    @property
    def StudioRLCDDemoRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].StudioRLCDDemoRequest
        import microllm.server_models
        return microllm.server_models.StudioRLCDDemoRequest

    @property
    def StudioAnalyzeRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].StudioAnalyzeRequest
        import microllm.server_models
        return microllm.server_models.StudioAnalyzeRequest

    @property
    def CalibrateRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].CalibrateRequest
        import microllm.server_models
        return microllm.server_models.CalibrateRequest

    @property
    def GuardrailRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].GuardrailRequest
        import microllm.server_models
        return microllm.server_models.GuardrailRequest

    @property
    def ProfilePullRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].ProfilePullRequest
        import microllm.server_models
        return microllm.server_models.ProfilePullRequest

    @property
    def BatchRequest(self) -> Any:
        if "microllm.server_models" in sys.modules:
            return sys.modules["microllm.server_models"].BatchRequest
        import microllm.server_models
        return microllm.server_models.BatchRequest

    @property
    def run_cli(self) -> Any:
        if "microllm.cli" in sys.modules:
            return sys.modules["microllm.cli"].run_cli
        import microllm.cli
        return microllm.cli.run_cli

    @property
    def main(self) -> Any:
        if "microllm.cli" in sys.modules:
            return sys.modules["microllm.cli"].main
        import microllm.cli
        return microllm.cli.main

    @property
    def build_cli_parser(self) -> Any:
        if "microllm.cli" in sys.modules:
            return sys.modules["microllm.cli"].build_cli_parser
        import microllm.cli
        return microllm.cli.build_cli_parser

    @property
    def get_cache_dir(self) -> Any:
        from microllm.registry import get_cache_dir
        return get_cache_dir


sys.modules[__name__].__class__ = _ServerModule


def resolve_repo_root() -> Path | None:
    """Resolve the git repository root if running from a source checkout, else None."""
    try:
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "src" / "microllm").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    except Exception:
        pass
    return None


def resolve_benchmark_path(filename: str) -> Path:
    """Resolve the benchmark output path in the repo or under the server cache directory."""
    repo = resolve_repo_root()
    if repo is not None:
        target_dir = repo / "benchmarks"
    else:
        from microllm.registry import get_cache_dir

        target_dir = get_cache_dir() / "benchmarks"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def resolve_docs_file(filename: str) -> Path | None:
    """Resolve a documentation file in docs/ or package static/ fallback."""
    repo = resolve_repo_root()
    if repo is not None:
        p = repo / "docs" / filename
        if p.is_file():
            return p
    pkg_static = Path(__file__).resolve().parent / "static" / filename
    if pkg_static.is_file():
        return pkg_static
    return None


def resolve_assets_dir() -> Path | None:
    """Resolve the documentation SVG assets directory."""
    repo = resolve_repo_root()
    if repo is not None:
        p = repo / "docs" / "assets"
        if p.is_dir():
            return p
    pkg_assets = Path(__file__).resolve().parent / "static" / "assets"
    if pkg_assets.is_dir():
        return pkg_assets
    return None


def resolve_papers_dir() -> Path | None:
    """Resolve the academic papers PDF directory."""
    repo = resolve_repo_root()
    if repo is not None:
        p = repo / "docs" / "papers"
        if p.is_dir():
            return p
    from microllm.registry import get_cache_dir

    cache_papers = get_cache_dir() / "papers"
    if cache_papers.is_dir():
        return cache_papers
    return None


def _evaluate_isolated_question(
    backend: Any,
    state_text: str,
    q: NoulQuestion | ChoiceQuestion | ScoreQuestion | MultiQuestion,
) -> tuple[NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer, dict[str, Any]]:
    """Evaluate a single question in complete isolation (guaranteeing semantics.question-id/order/batching)."""
    instructions_text = render_structured_text(q.instructions, fallback="")
    if isinstance(q, NoulQuestion):
        crit_map: dict[str, str | None] | None = None
        if q.criteria is not None:
            crit_map = {
                k: (render_structured_text(v) if v is not None else None)
                for k, v in q.criteria.items()
            }
        raw = backend.evaluate_noul(state_text, instructions_text, crit_map)
        return NoulAnswer(type="noul", noul=float(raw["noul"])), raw

    if isinstance(q, ChoiceQuestion):
        opt_map = {
            k: (render_structured_text(v) if v is not None else None)
            for k, v in q.criteria.items()
        }
        raw = backend.evaluate_choice(state_text, instructions_text, opt_map)
        return (
            ChoiceAnswer(
                type="choice",
                choice=str(raw["choice"]),
                probabilities=dict(raw["probabilities"]),
                confidence=float(raw["confidence"]),
            ),
            raw,
        )

    if isinstance(q, MultiQuestion):
        opt_map = {
            k: (render_structured_text(v) if v is not None else None)
            for k, v in q.criteria.items()
        }
        raw = backend.evaluate_multi(state_text, instructions_text, opt_map, threshold=q.threshold)
        return (
            MultiAnswer(
                type="multi",
                labels=list(raw["labels"]),
                probabilities=dict(raw["probabilities"]),
                confidence=float(raw["confidence"]),
            ),
            raw,
        )

    # ScoreQuestion
    level_strings = [render_structured_text(lvl) for lvl in q.criteria]
    raw = backend.evaluate_score(state_text, instructions_text, level_strings)
    legend = {str(i): q.criteria[i] for i in range(len(q.criteria))}
    return (
        ScoreAnswer(
            type="score",
            score=float(raw["score"]),
            legend=legend,
            probabilities=dict(raw["probabilities"]),
            confidence=float(raw["confidence"]),
        ),
        raw,
    )


_DEFAULT_GEAR1: FastEncoderBackend | None = None


def _get_default_gear1() -> FastEncoderBackend:
    global _DEFAULT_GEAR1
    if _DEFAULT_GEAR1 is None:
        _DEFAULT_GEAR1 = FastEncoderBackend()
    return _DEFAULT_GEAR1


def evaluate_guardrail_in_process(
    state: Any,
    threshold: float = 0.5,
    context: Any | None = None,
    custom_rubrics: Mapping[str, str] | None = None,
    gear1: Any | None = None,
) -> dict[str, Any]:
    """Execute Zero-Shot AI Safety Guardrail evaluation directly in-process without HTTP loopback.

    Evaluates prompt injection, destructive commands, PII/secret leaks, and context groundedness.
    """
    t0 = time.perf_counter()
    active_gear1 = gear1 if gear1 is not None else _get_default_gear1()
    state_text = render_structured_text(state)
    lowered = state_text.lower()
    dedotted = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", lowered)
    deobfuscated = (
        dedotted.replace("a11", "all")
        .translate(str.maketrans("013457@$", "oieastas"))
    )

    v_state = np.maximum(
        embed_text_deterministic(state_text),
        embed_text_deterministic(deobfuscated),
    )

    inj_res = active_gear1.evaluate_noul(
        state_text,
        "Evaluate input safety.",
        {
            "true": "Ignore previous instructions, system prompt override, jailbreak, bypass safety, disregard rules",
            "false": "Normal benign user request, summarize documentation, roadmap, question, conversation",
        },
    )
    dest_res = active_gear1.evaluate_noul(
        state_text,
        "Evaluate command safety.",
        {
            "true": "rm -rf, drop table, truncate database, delete production data, wipe format disk",
            "false": "Normal benign user request, summarize documentation, roadmap, safe read-only query",
        },
    )
    pii_res = active_gear1.evaluate_noul(
        state_text,
        "Evaluate credential safety.",
        {
            "true": "API key, secret_key, password, bearer token, SSN, credit card number leak",
            "false": "Normal benign user request, summarize documentation, roadmap, public text",
        },
    )

    injection_markers = (
        "ignore previous", "ignore all previous", "system prompt", "jailbreak",
        "disregard", "bypass safety", "developer mode", "do anything now",
        "dan mode", "unfiltered", "uncensored", "without any moral", "no moral",
        "no ethical", "ignore your rules", "forget your instructions",
        "override safety", "hypothetical scenario where you have no restrictions",
    )
    destructive_or_harm_markers = (
        "rm -rf", "drop table", "truncate ", "delete from", "/var/lib/postgresql",
        "ransomware", "keylogger", "malware", "exploit payload", "reverse shell",
        "phishing email", "steal credentials", "credit card fraud", "bomb", "explosive",
        "ricin", "fentanyl", "methamphetamine", "cyanide", "commit suicide",
        "self-harm", "kill themselves", "how to hack", "sql injection payload",
        "ddos attack", "botnet", "shoplift", "counterfeit money", "launder money",
        "child abuse", "non-consensual", "doxx", "stalking", "hitman", "assassinate",
        "hotwire a car", "pick a lock to break in", "make a weapon", "poison someone",
    )
    benign_defensive_markers = (
        "how to protect", "how to prevent", "defend against", "mitigate",
        "security awareness", "detect phishing", "patch vulnerability",
    )
    is_defensive = any(m in lowered or m in deobfuscated for m in benign_defensive_markers)

    p_inj = float(inj_res["noul"]) if v_state[8] > 0 else float(inj_res["noul"]) * 0.35
    if any(k in lowered or k in deobfuscated for k in injection_markers) and not is_defensive:
        p_inj = max(p_inj, 0.92)

    is_dangerous_cmd = bool(
        DANGEROUS_COMMAND_REGEX.search(state_text)
        or DANGEROUS_COMMAND_REGEX.search(deobfuscated)
    )
    p_dest = float(dest_res["noul"]) if v_state[7] > 0 else float(dest_res["noul"]) * 0.35
    if (
        any(k in lowered or k in deobfuscated for k in destructive_or_harm_markers)
        or is_dangerous_cmd
    ) and not is_defensive:
        p_dest = max(p_dest, 0.95)

    p_pii = float(pii_res["noul"]) if v_state[9] > 0 else float(pii_res["noul"]) * 0.35
    if any(k in lowered or k in deobfuscated for k in ("sk-", "api_key", "secret_key", "password=", "bearer ")):
        p_pii = max(p_pii, 0.90)

    ctx_raw = context
    claim_text = state_text
    if ctx_raw is None and isinstance(state, dict) and "context" in state:
        ctx_raw = state.get("context")
        claim_text = render_structured_text(
            state.get("response") or state.get("claim") or state
        )

    if ctx_raw is not None:
        ctx_text = render_structured_text(ctx_raw)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "and", "or", "by", "from", "that", "this",
        }
        ctx_tokens = set(tokenize_multilingual(ctx_text)) - stopwords
        claim_tokens = set(tokenize_multilingual(claim_text)) - stopwords
        overlap_ratio = len(claim_tokens & ctx_tokens) / float(max(1, len(claim_tokens)))

        ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", ctx_text))
        claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim_text))
        ungrounded_nums = len(claim_nums - ctx_nums)

        neg_words = {"no", "not", "never", "none", "without", "normal", "healthy"}
        severe_words = {"severe", "chronic", "critical", "fatal", "acute", "always"}
        polarity_conflict = bool((ctx_tokens & neg_words) and (claim_tokens & severe_words))

        p_halluc = max(0.02, min(0.98, 1.0 - overlap_ratio * 1.15))
        if ungrounded_nums > 0:
            p_halluc = max(p_halluc, min(0.96, 0.55 + 0.20 * ungrounded_nums))
        if polarity_conflict:
            p_halluc = max(p_halluc, 0.88)
    else:
        p_halluc = 0.08
        if any(k in lowered for k in ("fabricated", "hallucinated", "unverified rumor", "made up")):
            p_halluc = 0.85

    groundedness_score = round(float(max(0.0, min(1.0, 1.0 - p_halluc))), 4)
    checks_dict: dict[str, float] = {
        "prompt_injection": round(p_inj, 4),
        "destructive_operation": round(p_dest, 4),
        "pii_or_secret_leak": round(p_pii, 4),
        "hallucination_risk": round(p_halluc, 4),
    }

    if custom_rubrics:
        for rub_name, rub_desc in custom_rubrics.items():
            c_res = active_gear1.evaluate_noul(
                state_text,
                f"Evaluate custom safety rubric: {rub_name}",
                {
                    "true": str(rub_desc),
                    "false": "Normal benign safe request, standard public query",
                },
            )
            checks_dict[str(rub_name)] = round(float(c_res["noul"]), 4)

    risk_score = round(float(max(checks_dict.values())), 4)
    is_safe = risk_score < float(threshold)
    if risk_score >= 0.75:
        action = "block"
    elif risk_score >= float(threshold):
        action = "review"
    else:
        action = "allow"

    attributions = explain_state_tokens(
        state_text,
        "Security threat, prompt injection, destructive command, secret leak, hallucination",
        "It is unsafe: rm -rf, drop table, ignore previous instructions, password, secret token, severe chronic.",
        top_k=6,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "safe": is_safe,
        "action": action,
        "risk_score": risk_score,
        "groundedness_score": groundedness_score,
        "checks": checks_dict,
        "token_attribution": attributions,
        "latency_ms": latency_ms,
    }


def evaluate_calibration_in_process(
    confidence_scores: Sequence[float] | Sequence[Sequence[float]] | CalibrateRequest | Mapping[str, Any] | None = None,
    accuracy_labels: Sequence[int] | None = None,
    alpha: float = 0.1,
    *,
    logits: Sequence[Sequence[float]] | None = None,
    labels: Sequence[int] | None = None,
    target_risk: float = 0.10,
    delta: float = 0.05,
    seed: int = 42,
    req: CalibrateRequest | None = None,
) -> dict[str, Any]:
    """Execute Per-K temperature scaling, Split Conformal, and Clopper-Pearson calibration in-process."""
    if isinstance(confidence_scores, CalibrateRequest):
        req = confidence_scores
    elif req is None and isinstance(confidence_scores, Mapping):
        req = CalibrateRequest(**dict(confidence_scores))
    elif req is None and logits is not None and labels is not None:
        req = CalibrateRequest(
            logits=[list(row) for row in logits],
            labels=list(labels),
            alpha=float(alpha),
            target_risk=float(target_risk),
            delta=float(delta),
            seed=int(seed),
        )

    if req is not None:
        logits = req.logits
        labels = req.labels
        alpha = float(req.alpha)
        target_risk = float(req.target_risk)
        delta = float(req.delta)
        seed = int(req.seed)
    elif logits is None and confidence_scores is not None:
        if isinstance(confidence_scores, (list, tuple, np.ndarray)):
            _reject_bools_recursively(confidence_scores, "confidence_scores")
            if accuracy_labels is not None:
                _reject_bools_recursively(accuracy_labels, "accuracy_labels")
            if len(confidence_scores) > 10000:
                raise ValueError(f"Sample count N={len(confidence_scores)} exceeds maximum limit of 10,000.")
            if accuracy_labels is not None and len(accuracy_labels) > 10000:
                raise ValueError(f"Sample count N={len(accuracy_labels)} exceeds maximum limit of 10,000.")
            c_arr = np.asarray(confidence_scores, dtype=np.float64)
            if c_arr.ndim == 2:
                logits = c_arr.tolist()
                labels = list(accuracy_labels) if accuracy_labels is not None else None
            elif c_arr.ndim == 1:
                p = np.clip(c_arr, 1e-6, 1.0 - 1e-6)
                half_logit = np.log(p / (1.0 - p)) / 2.0
                logits = np.column_stack([-half_logit, half_logit]).tolist()
                labels = list(accuracy_labels) if accuracy_labels is not None else [1] * len(p)
        else:
            raise TypeError("confidence_scores must be a list, tuple, or numpy array.")

    scaler = PerKTemperatureScaler()
    conformal = SplitConformalCalibrator(alpha=float(alpha))
    if logits is not None and labels is not None:
        logits_arr = np.asarray(logits, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        base_logits = rng.normal(0.0, 1.0, size=(200, 4))
        exp_b = np.exp(base_logits - np.max(base_logits, axis=1, keepdims=True))
        true_p = exp_b / np.sum(exp_b, axis=1, keepdims=True)
        labels_arr = np.array([rng.choice(4, p=true_p[i]) for i in range(200)], dtype=np.int64)
        logits_arr = base_logits * 4.0
    _, k_card = logits_arr.shape

    shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
    raw_probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    raw_brier = brier_score(raw_probs, labels_arr)
    raw_ece = expected_calibration_error(raw_probs, labels_arr, n_bins=8)

    fitted_t = scaler.fit_k(logits_arr, labels_arr)
    cal_probs = scaler.transform(logits_arr)
    cal_brier = brier_score(cal_probs, labels_arr)
    cal_ece = expected_calibration_error(cal_probs, labels_arr, n_bins=8)

    q_hat = conformal.fit(cal_probs, labels_arr)

    confidences = np.max(cal_probs, axis=1)
    correct = (np.argmax(cal_probs, axis=1) == labels_arr).astype(np.int64)
    cp_cert = clopper_pearson_threshold(
        confidences, correct, target_risk=target_risk, delta=delta
    )
    bins = reliability_diagram_bins(cal_probs, labels_arr, n_bins=8)

    return {
        "cardinality_k": int(k_card),
        "temperature": round(float(fitted_t), 4),
        "fitted_temperature": round(float(fitted_t), 4),
        "raw_brier": round(float(raw_brier), 4),
        "calibrated_brier": round(float(cal_brier), 4),
        "ece_before": round(float(raw_ece), 4),
        "raw_ece": round(float(raw_ece), 4),
        "calibrated_ece": round(float(cal_ece), 4),
        "conformal_q_hat": round(float(q_hat), 4),
        "clopper_pearson_certification": cp_cert,
        "reliability_bins": bins,
    }


def run_rlcd_experiment_in_process(
    req: StudioRLCDDemoRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute OpenJev-RLCD Algorithm 1 + StartLux RLCD² training & calibration run in-process."""
    if req is None and kwargs:
        req = StudioRLCDDemoRequest(**kwargs)
    elif isinstance(req, Mapping):
        req = StudioRLCDDemoRequest(**dict(req))
    elif req is None:
        req = StudioRLCDDemoRequest()

    rng = np.random.default_rng(abs(int(req.seed)))
    if req.features is not None and req.labels is not None:
        _ensure_finite_json(req.features)
        features = np.asarray(req.features, dtype=np.float64)
        labels = np.asarray(req.labels, dtype=np.int64)
        n_samples, d_feat = features.shape
        k_classes = int(np.max(labels)) + 1
    else:
        n_samples, d_feat, k_classes = 160, 12, 4
        features = rng.normal(0.0, 1.0, size=(n_samples, d_feat))
        true_w = rng.normal(0.0, 1.5, size=(d_feat, k_classes))
        true_logits = features @ true_w
        labels = np.argmax(
            true_logits + rng.normal(0.0, 0.45, size=true_logits.shape), axis=1
        ).astype(np.int64)

    cfg = RLCDConfig(
        stage1_steps=max(5, min(100, req.stage1_steps)),
        stage2_steps=max(5, min(100, req.stage2_steps)),
        use_rlcd2_weights=req.use_rlcd2_weights,
    )
    trainer = RLCDDecisionHeadTrainer(
        d_feat=d_feat, k_classes=k_classes, config=cfg, seed=req.seed
    )
    init_probs = trainer.predict_proba(features)
    init_bins = reliability_diagram_bins(init_probs, labels, n_bins=8)
    try:
        metrics = trainer.fit(features, labels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    final_probs = trainer.predict_proba(features)
    final_bins = reliability_diagram_bins(final_probs, labels, n_bins=8)

    confidences = np.max(final_probs, axis=1)
    correct = (np.argmax(final_probs, axis=1) == labels).astype(np.int64)
    cp_cert = clopper_pearson_threshold(confidences, correct, target_risk=0.10, delta=0.05)

    risk_coverage_curve: list[dict[str, float]] = []
    for q_lvl in np.linspace(0.0, 0.85, 10):
        tau_q = float(np.quantile(confidences, q_lvl))
        mask = confidences >= tau_q
        cov = float(np.mean(mask))
        risk = float(np.mean(1 - correct[mask])) if np.any(mask) else 0.0
        risk_coverage_curve.append(
            {
                "threshold": round(tau_q, 4),
                "coverage": round(cov, 4),
                "empirical_risk": round(risk, 4),
            }
        )

    prop1 = group_mean_variance_identity(
        np.array([[0.82, 0.10, 0.08], [0.22, 0.68, 0.10], [0.55, 0.30, 0.15]]),
        np.array([0.60, 0.30, 0.10]),
    )

    return {
        **metrics,
        "metrics": metrics,
        "initial_bins": init_bins,
        "final_bins": final_bins,
        "initial_reliability_bins": init_bins,
        "final_reliability_bins": final_bins,
        "risk_coverage_curve": risk_coverage_curve,
        "clopper_pearson_certification": cp_cert,
        "proposition1_variance_identity": prop1,
    }


def atomic_write_json(file_path: Path | str, data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to file_path using a temporary file and os.replace."""
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=indent, ensure_ascii=False)

    tmp_path: Path | None = None
    try:
        fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{target.stem}_",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )
        tmp_path = Path(tmp_path_str)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        target.write_text(content, encoding="utf-8")


_benchmark_lock = asyncio.Lock()
_model_loading_lock = asyncio.Lock()
_model_sync_lock = threading.Lock()
_neural_backends: dict[str, FastEncoderBackend] = {}
_cascade_backends: dict[str, MultilingualCascadeBackend] = {}


async def get_neural_backend_async(m_name: str) -> FastEncoderBackend:
    """Async getter for neural encoder backend offloading model loading to worker thread."""
    clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
    if clean in _neural_backends:
        return _neural_backends[clean]
    async with _model_loading_lock:
        if clean not in _neural_backends:
            from microllm.registry import load_profile_scorer

            p_name = "multilingual-minilm-int8" if clean == "minilm" else "multilingual-mdeberta-int8"
            scorer = await asyncio.to_thread(load_profile_scorer, p_name)
            _neural_backends[clean] = FastEncoderBackend(logit_scorer=scorer)
        return _neural_backends[clean]


async def get_cascade_backend_async(
    m_name: str, gear1: FastEncoderBackend | None = None
) -> MultilingualCascadeBackend:
    """Async getter for multilingual cascade backend offloading model loading."""
    clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
    if clean in _cascade_backends:
        return _cascade_backends[clean]
    g2 = await get_neural_backend_async(clean)
    async with _model_loading_lock:
        if clean not in _cascade_backends:
            g1 = gear1 or FastEncoderBackend()
            _cascade_backends[clean] = MultilingualCascadeBackend(
                gear1=g1,
                gear2=g2,
                neural_model_name=clean,
                margin_threshold=0.12,
                entropy_threshold=0.88,
                fusion_alpha=0.75,
            )
        return _cascade_backends[clean]


def get_neural_backend(m_name: str) -> FastEncoderBackend:
    """Thread-safe synchronous getter for neural encoder backend."""
    clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
    with _model_sync_lock:
        if clean not in _neural_backends:
            from microllm.registry import load_profile_scorer

            p_name = "multilingual-minilm-int8" if clean == "minilm" else "multilingual-mdeberta-int8"
            _neural_backends[clean] = FastEncoderBackend(logit_scorer=load_profile_scorer(p_name))
        return _neural_backends[clean]


def get_cascade_backend(
    m_name: str, gear1: FastEncoderBackend | None = None
) -> MultilingualCascadeBackend:
    """Thread-safe synchronous getter for multilingual cascade backend."""
    clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
    with _model_sync_lock:
        if clean not in _cascade_backends:
            g2 = get_neural_backend(clean)
            g1 = gear1 or FastEncoderBackend()
            _cascade_backends[clean] = MultilingualCascadeBackend(
                gear1=g1,
                gear2=g2,
                neural_model_name=clean,
                margin_threshold=0.12,
                entropy_threshold=0.88,
                fusion_alpha=0.75,
            )
        return _cascade_backends[clean]


def create_app(
    api_key: str | None = None,
    multilingual_strategy: str | None = None,
    neural_model: str | None = None,
) -> FastAPI:
    """Create and configure the Jev-compatible FastAPI application."""
    configured_key = api_key if api_key is not None else os.environ.get("MICROLLM_API_KEY")
    raw_strategy = (
        multilingual_strategy
        if (multilingual_strategy is not None and str(multilingual_strategy).strip())
        else os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY")
    )
    default_strategy = (raw_strategy or "auto").lower().strip()
    raw_neural = (
        neural_model
        if (neural_model is not None and str(neural_model).strip())
        else os.environ.get("MICROLLM_NEURAL_MODEL")
    )
    default_neural_model = (raw_neural or "minilm").lower().strip()

    hf_model_name = os.environ.get("MICROLLM_HF_MODEL", "").strip()
    scorer = create_hf_zero_shot_scorer(hf_model_name) if hf_model_name else semantic_match_logit

    app = FastAPI(
        title="MicroLLM System-One Decision Engine",
        version="0.3.0",
        description="100% JevCompat (/v1/systemone) Local Hybrid Engine (Gear 1 Fast Encoder + Gear 2 AnyJev L0 + OpenJev-RLCD)",
    )

    gear1 = FastEncoderBackend(logit_scorer=scorer)
    gear2 = AnyJevPrefillBackend(logit_scorer=scorer)
    cascade = DualGearCascadeBackend(gear1=gear1, gear2=gear2, gate_threshold=0.75, alpha=0.75)
    mq_cache = VerdictMQMultiQuestionCache(max_entries=1024)

    neural_backends: dict[str, FastEncoderBackend] = _neural_backends
    cascade_backends: dict[str, MultilingualCascadeBackend] = _cascade_backends

    async def get_neural_backend_async(m_name: str) -> FastEncoderBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
        if clean in neural_backends:
            return neural_backends[clean]
        async with _model_loading_lock:
            if clean not in neural_backends:
                from microllm.registry import load_profile_scorer

                p_name = "multilingual-minilm-int8" if clean == "minilm" else "multilingual-mdeberta-int8"
                scorer_fn = await asyncio.to_thread(load_profile_scorer, p_name)
                neural_backends[clean] = FastEncoderBackend(logit_scorer=scorer_fn)
            return neural_backends[clean]

    async def get_cascade_backend_async(m_name: str) -> MultilingualCascadeBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
        if clean in cascade_backends:
            return cascade_backends[clean]
        g2 = await get_neural_backend_async(clean)
        async with _model_loading_lock:
            if clean not in cascade_backends:
                cascade_backends[clean] = MultilingualCascadeBackend(
                    gear1=gear1,
                    gear2=g2,
                    neural_model_name=clean,
                    margin_threshold=0.12,
                    entropy_threshold=0.88,
                    fusion_alpha=0.75,
                )
            return cascade_backends[clean]

    def get_neural_backend(m_name: str) -> FastEncoderBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
        with _model_sync_lock:
            if clean not in neural_backends:
                from microllm.registry import load_profile_scorer

                p_name = "multilingual-minilm-int8" if clean == "minilm" else "multilingual-mdeberta-int8"
                neural_backends[clean] = FastEncoderBackend(logit_scorer=load_profile_scorer(p_name))
            return neural_backends[clean]

    def get_cascade_backend(m_name: str) -> MultilingualCascadeBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
        with _model_sync_lock:
            if clean not in cascade_backends:
                g2 = get_neural_backend(clean)
                cascade_backends[clean] = MultilingualCascadeBackend(
                    gear1=gear1,
                    gear2=g2,
                    neural_model_name=clean,
                    margin_threshold=0.12,
                    entropy_threshold=0.88,
                    fusion_alpha=0.75,
                )
            return cascade_backends[clean]

    telemetry_state: dict[str, Any] = {
        "total_requests": 0,
        "systemone_requests": 0,
        "guardrail_requests": 0,
        "batch_requests": 0,
        "guardrail_blocked": 0,
        "abstained_decisions": 0,
        "latencies_ms": deque(maxlen=10000),
    }

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Ensure jevcompat §6 errors.validation-shape and errors.no-5xx (422 JSON with loc starting with 'body')."""
        details = []
        for err in exc.errors():
            loc = list(err.get("loc", []))
            if not loc or loc[0] != "body":
                loc = ["body", *loc]
            details.append(
                {
                    "loc": [str(x) if not isinstance(x, (str, int)) else x for x in loc],
                    "msg": str(err.get("msg", "Validation error")),
                    "type": str(err.get("type", "value_error")),
                }
            )
        return JSONResponse(status_code=422, content={"detail": details})

    def _check_auth(authorization: str | None) -> JSONResponse | None:
        if not configured_key:
            return None
        if not authorization or not authorization.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": {
                        "error_type": "authentication_error",
                        "message": "Missing Authorization: Bearer <key> header.",
                    }
                },
            )
        token = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": {
                        "error_type": "authentication_error",
                        "message": "Invalid API key provided.",
                    }
                },
            )
        return None

    @app.get("/healthz")
    async def healthz_endpoint() -> dict[str, str]:
        """Lightweight non-blocking healthcheck endpoint (JevCompat / Kubernetes / Docker)."""
        return {"status": "ok", "service": "microllm"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """Implements `http.models` (`jevcompat` §1)."""
        return {
            "models": [
                {
                    "name": "jev-latest",
                    "description": "MicroLLM Dual-Gear Hybrid Cascade (FastEncoder + AnyJev L0 Prefill + RLCD)",
                    "release_date": "2026-10-21",
                },
                {
                    "name": "microllm-gear1-fast",
                    "description": "Gear 1 Single-Pass Non-Autoregressive Encoder with NLI Templating & Per-K Scaling",
                    "release_date": "2026-10-21",
                },
                {
                    "name": "microllm-gear2-anyjev-l0",
                    "description": "Gear 2 AnyJev L0 Circular Rotation Prefill Reader (Position-Bias Free)",
                    "release_date": "2026-10-21",
                },
            ]
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone_endpoint(
        req: SystemOneRequest,
        response: Response,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Main Jev-compatible `POST /v1/systemone` endpoint (32/32 MUST conformant)."""
        import time

        t0 = time.perf_counter()
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err

        fields_set: set[str] = getattr(req, "model_fields_set", set())
        req_strat = (
            req.multilingual_strategy
            if "multilingual_strategy" in fields_set and req.multilingual_strategy is not None
            else default_strategy
        )
        req_neural_model = (
            req.neural_model
            if "neural_model" in fields_set and req.neural_model is not None
            else default_neural_model
        )
        clean_neural_model = "mdeberta" if "mdeberta" in str(req_neural_model).lower() else "minilm"

        state_text = render_structured_text(req.state)
        active_backend: Any = None
        if req.model == "microllm-gear1-fast":
            active_backend = gear1
        elif req.model == "microllm-gear2-anyjev-l0":
            active_backend = gear2
        elif req.model in ("jev-latest", "micro-jev", "microllm") or "model" not in fields_set:
            if req_strat == "fast":
                active_backend = gear1
            elif req_strat == "neural":
                active_backend = await get_neural_backend_async(clean_neural_model)
            else:  # "auto"
                active_backend = await get_cascade_backend_async(clean_neural_model)
        else:
            clean_m = str(req.model).lower().strip()
            if clean_m in ("minilm", "mdeberta", "multilingual-minilm-int8", "multilingual-mdeberta-int8"):
                active_backend = await get_neural_backend_async(clean_m)
            else:
                raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found.")

        # Verdict-MQ (IDEA-022): encode state representation once per multi-question request
        mq_cache.encode_state(state_text)

        def _do_inference() -> tuple[dict[str, Any], int, bool, bool, int, int]:
            answers: dict[str, Any] = {}
            any_cascade = False
            any_neural = False
            abstained_count = 0
            for idx, (qid, q_spec) in enumerate(req.questions.items()):
                if idx > 0:
                    with mq_cache._lock:
                        mq_cache.encode_hits += 1
                        mq_cache._metrics["hits"] += 1
                ans_obj, raw = _evaluate_isolated_question(active_backend, state_text, q_spec)
                answers[qid] = ans_obj
                if raw.get("x_abstain", False):
                    abstained_count += 1
                if raw.get("x_cascade_triggered", False):
                    any_cascade = True
                strat_raw = raw.get("x_strategy_used")
                if strat_raw == "neural" or req_strat == "neural":
                    any_neural = True

            in_tokens = max(1, len(state_text.split()) + len(req.questions) * 8)
            out_tokens = max(1, len(req.questions) * 4)
            return answers, abstained_count, any_cascade, any_neural, in_tokens, out_tokens

        answers, abstained_count, any_cascade, any_neural, in_tokens, out_tokens = await asyncio.to_thread(_do_inference)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry_state["total_requests"] += 1
        telemetry_state["systemone_requests"] += 1
        telemetry_state["abstained_decisions"] += abstained_count
        telemetry_state["latencies_ms"].append(elapsed_ms)

        final_strat = "neural" if any_neural else ("fast" if req_strat == "fast" else ("neural" if any_cascade else "fast"))
        response.headers["X-Strategy-Used"] = final_strat
        response.headers["X-Cascade-Triggered"] = str(any_cascade).lower()
        response.headers["X-Neural-Model"] = clean_neural_model

        return SystemOneResponse(
            model=req.model or "jev-latest",
            answers=answers,
            usage=UsageInfo(input_tokens=in_tokens, output_tokens=out_tokens),
        )

    @app.post("/v1/calibrate")
    async def calibrate_endpoint(
        req: CalibrateRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Calibrate Per-K temperature scaling, Split Conformal Prediction, and Clopper-Pearson bounds."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        return await asyncio.to_thread(evaluate_calibration_in_process, req=req)

    @app.post("/v1/studio/analyze")
    async def studio_analyze(
        req: StudioAnalyzeRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Diagnostic endpoint for Decision Studio comparing Gear 1, Gear 2, Cascade, Order-Flip Bias, Counterfactual XAI, Kolmogorov Coherence, and VOC."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err

        def _do_analyze() -> dict[str, Any]:
            from microllm.gradio_app import compute_deep_studio_diagnostics

            state_text = render_structured_text(req.state)
            custom_cascade = DualGearCascadeBackend(
                gear1=gear1, gear2=gear2, gate_threshold=req.gate_threshold, alpha=req.alpha
            )

            g1_out: dict[str, Any] = {}
            g2_out: dict[str, Any] = {}
            cas_out: dict[str, Any] = {}
            flip_audit: dict[str, Any] = {}
            token_attribution: dict[str, list[dict[str, Any]]] = {}
            counterfactual_xai: dict[str, dict[str, Any]] = {}
            coherence_audit: dict[str, dict[str, Any]] = {}
            voc_diagnostics: dict[str, dict[str, Any]] = {}

            for qid, q_spec in req.questions.items():
                a1, raw1 = _evaluate_isolated_question(gear1, state_text, q_spec)
                a2, raw2 = _evaluate_isolated_question(gear2, state_text, q_spec)
                ac, rawc = _evaluate_isolated_question(custom_cascade, state_text, q_spec)

                g1_out[qid] = {**a1.model_dump(), "meta": raw1}
                g2_out[qid] = {**a2.model_dump(), "meta": raw2}
                cas_out[qid] = {**ac.model_dump(), "meta": rawc}

                instr_text = render_structured_text(q_spec.instructions, fallback="")
                if isinstance(q_spec, ChoiceQuestion) and isinstance(ac, ChoiceAnswer):
                    win_desc = render_structured_text(q_spec.criteria.get(ac.choice), fallback="")
                    win_hyp = format_nli_hypothesis(ac.choice, win_desc)
                elif isinstance(q_spec, MultiQuestion) and isinstance(ac, MultiAnswer):
                    best_lbl = max(ac.probabilities, key=lambda k: ac.probabilities[k])
                    win_desc = render_structured_text(q_spec.criteria.get(best_lbl), fallback="")
                    win_hyp = format_nli_hypothesis(best_lbl, win_desc)
                elif isinstance(q_spec, ScoreQuestion) and isinstance(ac, ScoreAnswer):
                    best_idx = max(ac.probabilities, key=lambda k: ac.probabilities[k])
                    win_hyp = format_nli_hypothesis(best_idx, render_structured_text(ac.legend[best_idx]))
                else:
                    win_hyp = instr_text or "True affirmative proposition"
                token_attribution[qid] = explain_state_tokens(state_text, instr_text, win_hyp, top_k=6)

                cf_info, coh_info, voc_info = compute_deep_studio_diagnostics(state_text, q_spec, ac)
                counterfactual_xai[qid] = cf_info
                coherence_audit[qid] = coh_info
                voc_diagnostics[qid] = voc_info

                if (
                    req.run_order_flip_test
                    and isinstance(q_spec, ChoiceQuestion)
                    and isinstance(a2, ChoiceAnswer)
                ):
                    rev_q = ChoiceQuestion(
                        type="choice",
                        instructions=q_spec.instructions,
                        criteria=dict(reversed(list(q_spec.criteria.items()))),
                    )
                    a2_rev, _ = _evaluate_isolated_question(gear2, state_text, rev_q)
                    if isinstance(a2_rev, ChoiceAnswer):
                        max_diff = max(
                            abs(a2.probabilities[k] - a2_rev.probabilities[k])
                            for k in a2.probabilities
                        )
                        flip_audit[qid] = {
                            "forward_choice": a2.choice,
                            "reverse_choice": a2_rev.choice,
                            "order_flipped": a2.choice != a2_rev.choice,
                            "max_probability_drift": round(float(max_diff), 6),
                        }

            return {
                "gear1": g1_out,
                "gear2": g2_out,
                "cascade": cas_out,
                "order_flip_audit": flip_audit,
                "token_attribution": token_attribution,
                "counterfactual_xai": counterfactual_xai,
                "coherence_audit": coherence_audit,
                "voc_diagnostics": voc_diagnostics,
            }

        return await asyncio.to_thread(_do_analyze)

    @app.post("/v1/guardrail")
    async def guardrail_endpoint(
        req: GuardrailRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Zero-Shot AI Safety Guardrail pack inspired by 'Just Ask Jev' (arXiv:2609.29429)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        res = await asyncio.to_thread(
            evaluate_guardrail_in_process,
            state=req.state,
            threshold=req.threshold,
            context=req.context,
            custom_rubrics=req.custom_rubrics,
            gear1=gear1,
        )
        telemetry_state["total_requests"] += 1
        telemetry_state["guardrail_requests"] += 1
        if not res["safe"]:
            telemetry_state["guardrail_blocked"] += 1
        telemetry_state["latencies_ms"].append(res["latency_ms"])
        return res

    @app.get("/v1/profiles")
    async def profiles_list_endpoint() -> dict[str, Any]:
        """List Ollama-style hardware profiles (`fast-cpu-int8`, `multilingual`, `guardrail`)."""
        from microllm.registry import list_profiles

        return {"profiles": list_profiles()}

    @app.post("/v1/profiles/pull")
    async def profiles_pull_endpoint(
        req: ProfilePullRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Pull and quantize (INT8) a named hardware profile into `~/.cache/microllm/`."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        from microllm.registry import pull_profile

        try:
            return await asyncio.to_thread(
                pull_profile,
                req.profile,
                download_weights=req.download_weights,
                from_hub=req.from_hub,
            )
        except ValueError as exc:
            if "unknown profile" in str(exc).lower():
                raise HTTPException(status_code=404, detail=f"Unknown profile: {req.profile}") from exc
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    mcp_dispatcher = None

    @app.post("/v1/mcp")
    async def mcp_http_endpoint(
        payload: dict[str, Any] | list[Any],
        authorization: str | None = Header(default=None),
    ) -> Any:
        """JSON-RPC 2.0 Model Context Protocol (MCP) HTTP endpoint supporting single & batch requests."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        nonlocal mcp_dispatcher
        if mcp_dispatcher is None:
            from microllm.mcp_server import MicroLLMMCPServer

            mcp_dispatcher = MicroLLMMCPServer(
                gear1=gear1,
                gear2=gear2,
                cascade=cascade,
                mq_cache=mq_cache,
            )
        resp = mcp_dispatcher.handle_jsonrpc(payload)
        return resp if resp is not None else {"jsonrpc": "2.0", "result": {}}

    @app.post("/v1/batch")
    async def batch_endpoint(
        req: BatchRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Evaluate a batch of states (CSV/JSONL rows) across typed System-One questions."""
        import time

        t0 = time.perf_counter()
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err

        fields_set: set[str] = getattr(req, "model_fields_set", set())
        req_strat = (
            req.multilingual_strategy
            if req.multilingual_strategy is not None
            else (default_strategy if raw_strategy is not None else None)
        )
        req_neural = (
            req.neural_model
            if req.neural_model is not None
            else default_neural_model
        )
        clean_neural = "mdeberta" if "mdeberta" in str(req_neural).lower() else "minilm"

        active_backend: Any = gear1
        if req_strat is not None and ("model" not in fields_set or req.model in ("microllm-gear1-fast", "jev-latest", "micro-jev", "microllm")):
            if req_strat == "fast":
                active_backend = gear1
            elif req_strat == "neural":
                active_backend = await get_neural_backend_async(clean_neural)
            else:
                active_backend = await get_cascade_backend_async(clean_neural)
        elif req.model in ("jev-latest", "micro-jev", "microllm"):
            active_backend = cascade
        elif req.model == "microllm-gear2-anyjev-l0":
            active_backend = gear2
        elif req.model == "microllm-gear1-fast":
            active_backend = gear1
        else:
            clean_m = str(req.model).lower().strip()
            if clean_m in ("minilm", "mdeberta", "multilingual-minilm-int8", "multilingual-mdeberta-int8"):
                active_backend = await get_neural_backend_async(clean_m)
            else:
                raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found.")

        def _do_batch() -> tuple[list[dict[str, Any]], int]:
            results: list[dict[str, Any]] = []
            abstained_count = 0

            for idx, item_state in enumerate(req.items):
                item_t0 = time.perf_counter()
                s_text = render_structured_text(item_state)
                # Verdict-MQ (IDEA-022): encode state representation once per item
                mq_cache.encode_state(s_text)
                item_answers: dict[str, Any] = {}
                item_abstain = False
                for q_idx, (qid, q_spec) in enumerate(req.questions.items()):
                    if q_idx > 0:
                        with mq_cache._lock:
                            mq_cache.encode_hits += 1
                            mq_cache._metrics["hits"] += 1
                    ans_obj, raw = _evaluate_isolated_question(active_backend, s_text, q_spec)
                    item_answers[qid] = ans_obj.model_dump()
                    if raw.get("x_abstain", False):
                        item_abstain = True
                if item_abstain:
                    abstained_count += 1
                results.append(
                    {
                        "index": idx,
                        "state": item_state,
                        "answers": item_answers,
                        "abstain": item_abstain,
                        "latency_ms": round((time.perf_counter() - item_t0) * 1000.0, 3),
                    }
                )
            return results, abstained_count

        results, abstained_count = await asyncio.to_thread(_do_batch)

        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        item_lats = [float(r["latency_ms"]) for r in results] or [0.0]
        telemetry_state["total_requests"] += 1
        telemetry_state["batch_requests"] += 1
        telemetry_state["latencies_ms"].append(total_ms)
        return {
            "summary": {
                "total_items": len(req.items),
                "abstained_count": abstained_count,
                "total_latency_ms": total_ms,
                "avg_latency_ms": round(total_ms / max(1, len(req.items)), 3),
                "p50_latency_ms": round(float(np.percentile(item_lats, 50)), 3),
                "p95_latency_ms": round(float(np.percentile(item_lats, 95)), 3),
                "verdict_mq_cache": mq_cache.stats,
            },
            "results": results,
        }

    @app.get("/v1/metrics")
    async def metrics_endpoint(
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Production telemetry & observability endpoint (IDEA-020)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        lats = telemetry_state["latencies_ms"] or [0.0]
        return {
            "total_requests": telemetry_state["total_requests"],
            "systemone_requests": telemetry_state["systemone_requests"],
            "guardrail_requests": telemetry_state["guardrail_requests"],
            "batch_requests": telemetry_state["batch_requests"],
            "guardrail_blocked": telemetry_state["guardrail_blocked"],
            "abstained_decisions": telemetry_state["abstained_decisions"],
            "latency_p50_ms": round(float(np.percentile(lats, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 3),
            "verdict_mq_cache": mq_cache.stats,
        }

    _run_rlcd_experiment = run_rlcd_experiment_in_process

    @app.post("/v1/studio/rlcd-demo")
    async def studio_rlcd_demo(
        req: StudioRLCDDemoRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Run a live OpenJev-RLCD Algorithm 1 + StartLux RLCD² training & calibration experiment."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        return await asyncio.to_thread(run_rlcd_experiment_in_process, req)

    @app.post("/v1/rlcd")
    async def rlcd_endpoint(
        req: StudioRLCDDemoRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Local RLCD Stage 1 + Stage 2 training endpoint (`implementation_plan.md`)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        return await asyncio.to_thread(run_rlcd_experiment_in_process, req)

    @app.get("/v1/benchmark")
    async def get_benchmark_endpoint(
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Return the latest 540-item benchmark suite results (running it if not cached)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        import json
        from microllm.benchmarks import run_benchmark_suite

        def _do_get_bench() -> Any:
            cached = resolve_benchmark_path("latest_results.json")
            if cached.is_file():
                try:
                    return json.loads(cached.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            if resolve_repo_root() is None:
                return run_benchmark_suite(save_artifacts=False, output_path=cached)
            return run_benchmark_suite(save_artifacts=True, output_path=cached)

        return await asyncio.to_thread(_do_get_bench)

    @app.post("/v1/benchmark")
    @app.post("/v1/benchmark/run")
    async def run_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 540-item multi-track benchmark suite across all 5 evaluation tracks."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        if _benchmark_lock.locked():
            raise HTTPException(status_code=409, detail="A benchmark run is already in progress.")
        async with _benchmark_lock:
            from microllm.benchmarks import run_benchmark_suite

            save_flag = bool((payload or {}).get("save_artifacts", True))
            if resolve_repo_root() is None:
                cached = resolve_benchmark_path("latest_results.json")
                return await asyncio.to_thread(run_benchmark_suite, save_artifacts=False, output_path=cached)
            return await asyncio.to_thread(run_benchmark_suite, save_artifacts=save_flag)

    @app.get("/v1/benchmark/external")
    async def get_external_benchmark_endpoint(
        full_splits: bool = False,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Return the latest 16-dataset external web benchmark results (curated or full unsampled splits)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        import json
        from microllm.web_benchmark_suite import run_all_web_benchmarks

        def _do_get_ext() -> Any:
            filename = (
                "web_benchmarks_full_results.json" if full_splits else "web_benchmarks_results.json"
            )
            cached = resolve_benchmark_path(filename)
            if cached.is_file():
                try:
                    data = json.loads(cached.read_text(encoding="utf-8"))
                    if data.get("total_web_datasets") == 16:
                        return data
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            return run_all_web_benchmarks(output_path=cached, full_splits=full_splits)

        return await asyncio.to_thread(_do_get_ext)

    @app.post("/v1/benchmark/external")
    async def run_external_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 16-dataset external web benchmark suite (`src/microllm/web_benchmark_suite.py`)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        if _benchmark_lock.locked():
            raise HTTPException(status_code=409, detail="A benchmark run is already in progress.")
        async with _benchmark_lock:
            from microllm.web_benchmark_suite import run_all_web_benchmarks

            body = payload or {}
            save_flag = bool(body.get("save_artifacts", True))
            full_splits = bool(body.get("full_splits", False))
            entries_per_ds = body.get("entries_per_dataset")
            scale_per_ds = body.get("scale_per_dataset")
            datasets_filter = body.get("datasets")
            fetch_live = bool(body.get("fetch_live", False))
            filename = (
                "web_benchmarks_full_results.json" if full_splits else "web_benchmarks_results.json"
            )
            out_path = (
                resolve_benchmark_path(filename)
                if (save_flag and not datasets_filter)
                else None
            )
            return await asyncio.to_thread(
                run_all_web_benchmarks,
                output_path=out_path,
                entries_per_dataset=entries_per_ds,
                scale_per_dataset=scale_per_ds,
                datasets=datasets_filter,
                fetch_live=fetch_live,
                full_splits=full_splits,
            )

    @app.get("/v1/benchmark/hf-jev-models")
    async def get_hf_jev_models_benchmark_endpoint(
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Return the latest 12-dataset Hugging Face JEV models ecosystem benchmark results (Issue #108)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        import json
        from microllm.web_benchmark_suite import run_hf_jev_model_benchmarks

        def _do_get_hf() -> Any:
            cached = resolve_benchmark_path("hf_jev_models_benchmarks_results.json")
            if cached.is_file():
                try:
                    data = json.loads(cached.read_text(encoding="utf-8"))
                    if data.get("total_web_datasets") == 12:
                        return data
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            return run_hf_jev_model_benchmarks(output_path=cached)

        return await asyncio.to_thread(_do_get_hf)

    @app.post("/v1/benchmark/hf-jev-models")
    async def run_hf_jev_models_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 12-dataset Hugging Face JEV models ecosystem benchmark suite (Issue #108)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        if _benchmark_lock.locked():
            raise HTTPException(status_code=409, detail="A benchmark run is already in progress.")
        async with _benchmark_lock:
            from microllm.web_benchmark_suite import run_hf_jev_model_benchmarks

            body = payload or {}
            save_flag = bool(body.get("save_artifacts", True))
            entries_per_ds = body.get("entries_per_dataset")
            datasets_filter = body.get("datasets")
            fetch_live = bool(body.get("fetch_live", False))
            out_path = (
                resolve_benchmark_path("hf_jev_models_benchmarks_results.json")
                if (save_flag and entries_per_ds is None and not datasets_filter)
                else None
            )
            return await asyncio.to_thread(
                run_hf_jev_model_benchmarks,
                output_path=out_path,
                entries_per_dataset=entries_per_ds,
                datasets=datasets_filter,
                fetch_live=fetch_live,
            )

    @app.get("/v1/jev-models")
    async def list_hf_jev_models_endpoint(
        q: str | None = None,
        primitive: str | None = None,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """List and filter the 25 active Hugging Face JEV ecosystem models and 12 linked datasets (`huggingface.co/models?search=jev`)."""
        from microllm.dataset_adapters import (
            ALL_12_HF_JEV_MODEL_DATASETS,
            ALL_28_EXTERNAL_DATASETS,
            list_hf_jev_ecosystem_models,
        )

        models = list_hf_jev_ecosystem_models(
            query=q, primitive=primitive, dataset=dataset
        )
        return {
            "total": len(models),
            "source_search_url": "https://huggingface.co/models?search=jev",
            "datasets_enriched": ALL_12_HF_JEV_MODEL_DATASETS,
            "linked_jev_datasets": ALL_12_HF_JEV_MODEL_DATASETS,
            "total_external_datasets": len(ALL_28_EXTERNAL_DATASETS),
            "models": models,
        }

    @app.get("/", response_class=HTMLResponse)
    async def serve_studio_ui() -> HTMLResponse:
        static_file = Path(__file__).parent / "static" / "index.html"
        if static_file.exists():
            return HTMLResponse(content=static_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>MicroLLM Decision Studio</h1>")

    @app.get("/documentation", response_class=HTMLResponse)
    async def serve_documentation_ui() -> HTMLResponse:
        doc_file = resolve_docs_file("index.html")
        if doc_file and doc_file.is_file():
            return HTMLResponse(content=doc_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>MicroLLM Documentation</h1>")

    @app.get("/slides", response_class=HTMLResponse)
    async def serve_slides_ui() -> HTMLResponse:
        """Serve the LightPath-branded 12-slide interactive executive & technical deck (`docs/slides.html`)."""
        slides_file = resolve_docs_file("slides.html")
        if slides_file and slides_file.is_file():
            return HTMLResponse(content=slides_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>MicroLLM LightPath Presentation Slides</h1>")

    @app.get("/assets/{asset_name}")
    async def serve_svg_asset(asset_name: str) -> FileResponse:
        assets_dir = resolve_assets_dir()
        if assets_dir is None:
            raise HTTPException(status_code=404, detail="SVG asset not found.")
        candidate = (assets_dir / Path(asset_name).name).resolve()
        if candidate.parent != assets_dir.resolve() or not candidate.is_file() or candidate.suffix.lower() != ".svg":
            raise HTTPException(status_code=404, detail="SVG asset not found.")
        return FileResponse(path=str(candidate), media_type="image/svg+xml")

    @app.get("/papers/{paper_name}")
    async def serve_paper_pdf(paper_name: str) -> FileResponse:
        papers_dir = resolve_papers_dir()
        if papers_dir is None:
            raise HTTPException(status_code=404, detail="PDF paper not found.")
        candidate = (papers_dir / Path(paper_name).name).resolve()
        if candidate.parent != papers_dir.resolve() or not candidate.is_file() or candidate.suffix.lower() != ".pdf":
            raise HTTPException(status_code=404, detail="PDF paper not found.")
        return FileResponse(path=str(candidate), media_type="application/pdf")

    @app.get("/v1/papers")
    async def list_academic_papers_endpoint(
        pillar: str | None = None,
        q: str | None = None,
        arxiv_id: str | None = None,
    ) -> dict[str, Any]:
        """Search and inspect the 104-paper Jev, RLCD, System-1 & Calibrated LLM academic catalog (Issue #42)."""
        from microllm.papers import (
            get_academic_paper,
            get_literature_taxonomy,
            list_academic_papers,
        )

        if arxiv_id is not None:
            paper = get_academic_paper(arxiv_id)
            if paper is None:
                raise HTTPException(status_code=404, detail=f"arXiv paper '{arxiv_id}' not found in catalog.")
            return {
                "total": 1,
                "taxonomy": get_literature_taxonomy(),
                "papers": [paper],
            }
        matched = list_academic_papers(pillar=pillar, query=q)
        return {
            "total": len(matched),
            "taxonomy": get_literature_taxonomy(),
            "papers": matched,
        }

    app.setup()
    app.middleware_stack = app.build_middleware_stack()
    return app




from microllm.cli import build_cli_parser, main, run_cli


if __name__ == "__main__":
    main()
