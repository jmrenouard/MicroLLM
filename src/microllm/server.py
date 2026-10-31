"""FastAPI Server for MicroLLM: JevCompat 0.1 (32/32 MUST) + Decision Studio & RLCD Lab."""

from __future__ import annotations

from collections import deque
import hmac
import os
from pathlib import Path
from typing import Any
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    create_hf_zero_shot_scorer,
    explain_state_tokens,
    semantic_match_logit,
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
    QuestionSpec,
    ScoreAnswer,
    ScoreQuestion,
    StructuredText,
    SystemOneRequest,
    SystemOneResponse,
    UsageInfo,
    _ensure_finite_json,
    render_structured_text,
)


class StudioAnalyzeRequest(BaseModel):
    """Request schema for Decision Studio side-by-side Gear 1 / Gear 2 / Cascade analysis."""

    state: StructuredText
    questions: dict[str, QuestionSpec]
    gate_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    run_order_flip_test: bool = True

    @field_validator("state", mode="before")
    @classmethod
    def validate_studio_state(cls, v: Any) -> StructuredText:
        if v is None or not isinstance(v, (str, dict, list)):
            raise ValueError("state must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)

    @model_validator(mode="after")
    def validate_studio_questions(self) -> "StudioAnalyzeRequest":
        if not self.questions:
            raise ValueError("questions must be a non-empty object.")
        if not np.isfinite(self.gate_threshold) or not np.isfinite(self.alpha):
            raise ValueError("gate_threshold and alpha must be finite numbers.")
        return self


class StudioRLCDDemoRequest(BaseModel):
    """Request schema for running a live local RLCD Stage 1 + Stage 2 training & calibration run."""

    stage1_steps: int = Field(default=35, ge=5, le=100)
    stage2_steps: int = Field(default=25, ge=5, le=100)
    use_rlcd2_weights: bool = True
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    features: list[list[float]] | None = None
    labels: list[int] | None = None


class CalibrateRequest(BaseModel):
    """Request schema for `POST /v1/calibrate` (Per-K temperature scaling + Split Conformal + Clopper-Pearson)."""

    logits: list[list[float]] | None = None
    labels: list[int] | None = None
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    target_risk: float = Field(default=0.10, gt=0.0, lt=1.0)
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def validate_calibration_arrays(self) -> "CalibrateRequest":
        if self.logits is None and self.labels is None:
            return self
        if not self.logits or not self.labels or len(self.logits) != len(self.labels):
            raise ValueError("logits and labels must be non-empty lists of equal length.")
        _ensure_finite_json(self.logits)
        k = len(self.logits[0])
        if k < 2:
            raise ValueError("Each logit vector must have at least 2 classes.")
        for row in self.logits:
            if len(row) != k:
                raise ValueError("All logit vectors must have identical cardinality K.")
        for y in self.labels:
            if y < 0 or y >= k:
                raise ValueError(f"Each label must be in 0..{k - 1}.")
        return self


class GuardrailRequest(BaseModel):
    """Request schema for `POST /v1/guardrail` (Zero-Shot AI Safety Guardrail pack, arXiv:2609.29429)."""

    state: StructuredText
    context: StructuredText | None = None
    threshold: float = Field(default=0.55, ge=0.1, le=0.99)
    custom_rubrics: dict[str, str] | None = None

    @field_validator("state", mode="before")
    @classmethod
    def validate_guardrail_state(cls, v: Any) -> StructuredText:
        if v is None or not isinstance(v, (str, dict, list)):
            raise ValueError("state must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)

    @field_validator("context", mode="before")
    @classmethod
    def validate_guardrail_context(cls, v: Any) -> StructuredText | None:
        if v is None:
            return None
        if not isinstance(v, (str, dict, list)):
            raise ValueError("context must be a string, JSON object, or JSON array.")
        return _ensure_finite_json(v)


class ProfilePullRequest(BaseModel):
    """Request schema for `POST /v1/profiles/pull` (`microllm pull <profile>`)."""

    profile: str
    cache_dir: str | None = None
    download_weights: bool = False
    from_hub: bool = False


class BatchRequest(BaseModel):
    """Request schema for `POST /v1/batch` (Bulk CSV/JSONL decision evaluation)."""

    items: list[StructuredText]
    questions: dict[str, QuestionSpec]
    model: str = "microllm-gear1-fast"
    multilingual_strategy: str | None = None
    neural_model: str | None = None

    @model_validator(mode="after")
    def validate_batch_payload(self) -> "BatchRequest":
        if not self.items or len(self.items) > 500:
            raise ValueError("items must contain between 1 and 500 states.")
        _ensure_finite_json(self.items)
        if not self.questions:
            raise ValueError("questions must be a non-empty object.")
        return self


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
        version="0.2.0",
        description="100% JevCompat (/v1/systemone) Local Hybrid Engine (Gear 1 Fast Encoder + Gear 2 AnyJev L0 + OpenJev-RLCD)",
    )

    gear1 = FastEncoderBackend(logit_scorer=scorer)
    gear2 = AnyJevPrefillBackend(logit_scorer=scorer)
    cascade = DualGearCascadeBackend(gear1=gear1, gear2=gear2, gate_threshold=0.75, alpha=0.75)
    mq_cache = VerdictMQMultiQuestionCache(max_entries=1024)

    neural_backends: dict[str, FastEncoderBackend] = {}

    def get_neural_backend(m_name: str) -> FastEncoderBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
        if clean not in neural_backends:
            from microllm.registry import load_profile_scorer

            p_name = "multilingual-minilm-int8" if clean == "minilm" else "multilingual-mdeberta-int8"
            neural_backends[clean] = FastEncoderBackend(logit_scorer=load_profile_scorer(p_name))
        return neural_backends[clean]

    cascade_backends: dict[str, MultilingualCascadeBackend] = {}

    def get_cascade_backend(m_name: str) -> MultilingualCascadeBackend:
        clean = "mdeberta" if "mdeberta" in str(m_name).lower() else "minilm"
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
        elif req_strat == "fast":
            active_backend = gear1
        elif req_strat == "neural":
            active_backend = get_neural_backend(clean_neural_model)
        else:  # "auto"
            active_backend = get_cascade_backend(clean_neural_model)

        answers: dict[str, Any] = {}
        any_cascade = False
        any_neural = False
        for qid, q_spec in req.questions.items():
            # Verdict-MQ (IDEA-022): reuse cached state & clause embeddings across questions (1 miss + M-1 hits)
            mq_cache.encode_state(state_text)
            ans_obj, raw = _evaluate_isolated_question(active_backend, state_text, q_spec)
            answers[qid] = ans_obj
            if raw.get("x_abstain", False):
                telemetry_state["abstained_decisions"] += 1
            if raw.get("x_cascade_triggered", False):
                any_cascade = True
            strat_raw = raw.get("x_strategy_used")
            if strat_raw == "neural" or req_strat == "neural":
                any_neural = True

        input_tokens = max(1, len(state_text.split()) + len(req.questions) * 8)
        output_tokens = max(1, len(req.questions) * 4)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry_state["total_requests"] += 1
        telemetry_state["systemone_requests"] += 1
        telemetry_state["latencies_ms"].append(elapsed_ms)

        final_strat = "neural" if any_neural else ("fast" if req_strat == "fast" else ("neural" if any_cascade else "fast"))
        response.headers["X-Strategy-Used"] = final_strat
        response.headers["X-Cascade-Triggered"] = str(any_cascade).lower()
        response.headers["X-Neural-Model"] = clean_neural_model

        return SystemOneResponse(
            model=req.model or "jev-latest",
            answers=answers,
            usage=UsageInfo(input_tokens=input_tokens, output_tokens=output_tokens),
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

        scaler = PerKTemperatureScaler()
        conformal = SplitConformalCalibrator(alpha=float(req.alpha))
        if req.logits is not None and req.labels is not None:
            logits_arr = np.asarray(req.logits, dtype=np.float64)
            labels_arr = np.asarray(req.labels, dtype=np.int64)
        else:
            rng = np.random.default_rng(req.seed)
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
            confidences, correct, target_risk=req.target_risk, delta=req.delta
        )
        bins = reliability_diagram_bins(cal_probs, labels_arr, n_bins=8)

        return {
            "cardinality_k": int(k_card),
            "fitted_temperature": round(float(fitted_t), 4),
            "raw_brier": round(float(raw_brier), 4),
            "calibrated_brier": round(float(cal_brier), 4),
            "raw_ece": round(float(raw_ece), 4),
            "calibrated_ece": round(float(cal_ece), 4),
            "conformal_q_hat": round(float(q_hat), 4),
            "clopper_pearson_certification": cp_cert,
            "reliability_bins": bins,
        }

    @app.post("/v1/studio/analyze")
    async def studio_analyze(req: StudioAnalyzeRequest) -> dict[str, Any]:
        """Diagnostic endpoint for Decision Studio comparing Gear 1, Gear 2, Cascade, Order-Flip Bias, Counterfactual XAI, Kolmogorov Coherence, and VOC."""
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

    @app.post("/v1/guardrail")
    async def guardrail_endpoint(
        req: GuardrailRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Zero-Shot AI Safety Guardrail pack inspired by 'Just Ask Jev' (arXiv:2609.29429)."""
        import re
        import time
        from microllm.backends.base import embed_text_deterministic, tokenize_multilingual

        t0 = time.perf_counter()
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        state_text = render_structured_text(req.state)
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

        inj_res = gear1.evaluate_noul(
            state_text,
            "Evaluate input safety.",
            {
                "true": "Ignore previous instructions, system prompt override, jailbreak, bypass safety, disregard rules",
                "false": "Normal benign user request, summarize documentation, roadmap, question, conversation",
            },
        )
        dest_res = gear1.evaluate_noul(
            state_text,
            "Evaluate command safety.",
            {
                "true": "rm -rf, drop table, truncate database, delete production data, wipe format disk",
                "false": "Normal benign user request, summarize documentation, roadmap, safe read-only query",
            },
        )
        pii_res = gear1.evaluate_noul(
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

        # Calibrate baseline when zero security concept clusters (slots 7, 8, 9) fire in state
        p_inj = float(inj_res["noul"]) if v_state[8] > 0 else float(inj_res["noul"]) * 0.35
        if any(k in lowered or k in deobfuscated for k in injection_markers) and not is_defensive:
            p_inj = max(p_inj, 0.92)

        p_dest = float(dest_res["noul"]) if v_state[7] > 0 else float(dest_res["noul"]) * 0.35
        if any(k in lowered or k in deobfuscated for k in destructive_or_harm_markers) and not is_defensive:
            p_dest = max(p_dest, 0.95)

        p_pii = float(pii_res["noul"]) if v_state[9] > 0 else float(pii_res["noul"]) * 0.35
        if any(k in lowered or k in deobfuscated for k in ("sk-", "api_key", "secret_key", "password=", "bearer ")):
            p_pii = max(p_pii, 0.90)

        # 4th Rubric: Hallucination / Groundedness (arXiv:2609.29429 "Just Ask Jev")
        ctx_raw = req.context
        claim_text = state_text
        if ctx_raw is None and isinstance(req.state, dict) and "context" in req.state:
            ctx_raw = req.state.get("context")
            claim_text = render_structured_text(
                req.state.get("response") or req.state.get("claim") or req.state
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

            # Check numeric consistency (numbers in claim not present in context)
            ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", ctx_text))
            claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim_text))
            ungrounded_nums = len(claim_nums - ctx_nums)

            # Check polarity / negation contradiction between context and claim
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

        if req.custom_rubrics:
            for rub_name, rub_desc in req.custom_rubrics.items():
                c_res = gear1.evaluate_noul(
                    state_text,
                    f"Evaluate custom safety rubric: {rub_name}",
                    {
                        "true": str(rub_desc),
                        "false": "Normal benign safe request, standard public query",
                    },
                )
                checks_dict[str(rub_name)] = round(float(c_res["noul"]), 4)

        risk_score = round(float(max(checks_dict.values())), 4)
        is_safe = risk_score < float(req.threshold)
        if risk_score >= 0.75:
            action = "block"
        elif risk_score >= float(req.threshold):
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
        telemetry_state["total_requests"] += 1
        telemetry_state["guardrail_requests"] += 1
        if not is_safe:
            telemetry_state["guardrail_blocked"] += 1
        telemetry_state["latencies_ms"].append(latency_ms)

        return {
            "safe": is_safe,
            "action": action,
            "risk_score": risk_score,
            "groundedness_score": groundedness_score,
            "checks": checks_dict,
            "token_attribution": attributions,
            "latency_ms": latency_ms,
        }

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
            return pull_profile(
                req.profile,
                cache_dir=req.cache_dir,
                download_weights=req.download_weights,
                from_hub=req.from_hub,
            )
        except ValueError as exc:
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
        if req_strat is not None and "model" not in fields_set:
            if req_strat == "fast":
                active_backend = gear1
            elif req_strat == "neural":
                active_backend = get_neural_backend(clean_neural)
            else:
                active_backend = get_cascade_backend(clean_neural)
        elif req.model == "jev-latest":
            active_backend = cascade
        elif req.model == "microllm-gear2-anyjev-l0":
            active_backend = gear2

        results: list[dict[str, Any]] = []
        abstained_count = 0

        for idx, item_state in enumerate(req.items):
            item_t0 = time.perf_counter()
            s_text = render_structured_text(item_state)
            item_answers: dict[str, Any] = {}
            item_abstain = False
            for qid, q_spec in req.questions.items():
                mq_cache.encode_state(s_text)
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
            },
            "results": results,
        }

    @app.get("/v1/metrics")
    async def metrics_endpoint() -> dict[str, Any]:
        """Production telemetry & observability endpoint (IDEA-020)."""
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

    def _run_rlcd_experiment(req: StudioRLCDDemoRequest) -> dict[str, Any]:
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

        metrics = trainer.fit(features, labels)
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
            "initial_reliability_bins": init_bins,
            "final_reliability_bins": final_bins,
            "risk_coverage_curve": risk_coverage_curve,
            "clopper_pearson_certification": cp_cert,
            "proposition1_variance_identity": prop1,
        }

    @app.post("/v1/studio/rlcd-demo")
    async def studio_rlcd_demo(req: StudioRLCDDemoRequest) -> dict[str, Any]:
        """Run a live OpenJev-RLCD Algorithm 1 + StartLux RLCD² training & calibration experiment."""
        return _run_rlcd_experiment(req)

    @app.post("/v1/rlcd")
    async def rlcd_endpoint(
        req: StudioRLCDDemoRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Local RLCD Stage 1 + Stage 2 training endpoint (`implementation_plan.md`)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        return _run_rlcd_experiment(req)

    @app.get("/v1/benchmark")
    async def get_benchmark_endpoint() -> dict[str, Any]:
        """Return the latest 540-item benchmark suite results (running it if not cached)."""
        import json
        from microllm.benchmarks import run_benchmark_suite

        cached = Path(__file__).resolve().parents[2] / "benchmarks" / "latest_results.json"
        if cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
        return run_benchmark_suite(save_artifacts=True)

    @app.post("/v1/benchmark")
    async def run_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 540-item multi-track benchmark suite across all 5 evaluation tracks."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        from microllm.benchmarks import run_benchmark_suite

        save_flag = bool((payload or {}).get("save_artifacts", True))
        return run_benchmark_suite(save_artifacts=save_flag)

    @app.get("/v1/benchmark/external")
    async def get_external_benchmark_endpoint(full_splits: bool = False) -> dict[str, Any]:
        """Return the latest 16-dataset external web benchmark results (curated or full unsampled splits)."""
        import json
        from microllm.web_benchmark_suite import run_all_web_benchmarks

        filename = (
            "web_benchmarks_full_results.json" if full_splits else "web_benchmarks_results.json"
        )
        cached = Path(__file__).resolve().parents[2] / "benchmarks" / filename
        if cached.is_file():
            data = json.loads(cached.read_text(encoding="utf-8"))
            if data.get("total_web_datasets") == 16:
                return data
        return run_all_web_benchmarks(output_path=cached, full_splits=full_splits)

    @app.post("/v1/benchmark/external")
    async def run_external_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 16-dataset external web benchmark suite (`src/microllm/web_benchmark_suite.py`)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
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
            Path(__file__).resolve().parents[2] / "benchmarks" / filename
            if (save_flag and not datasets_filter)
            else None
        )
        return run_all_web_benchmarks(
            output_path=out_path,
            entries_per_dataset=entries_per_ds,
            scale_per_dataset=scale_per_ds,
            datasets=datasets_filter,
            fetch_live=fetch_live,
            full_splits=full_splits,
        )

    @app.get("/v1/benchmark/hf-jev-models")
    async def get_hf_jev_models_benchmark_endpoint() -> dict[str, Any]:
        """Return the latest 12-dataset Hugging Face JEV models ecosystem benchmark results (Issue #108)."""
        import json
        from microllm.web_benchmark_suite import run_hf_jev_model_benchmarks

        cached = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "hf_jev_models_benchmarks_results.json"
        )
        if cached.is_file():
            data = json.loads(cached.read_text(encoding="utf-8"))
            if data.get("total_web_datasets") == 12:
                return data
        return run_hf_jev_model_benchmarks(output_path=cached)

    @app.post("/v1/benchmark/hf-jev-models")
    async def run_hf_jev_models_benchmark_endpoint(
        payload: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> Any:
        """Execute the 12-dataset Hugging Face JEV models ecosystem benchmark suite (Issue #108)."""
        auth_err = _check_auth(authorization)
        if auth_err is not None:
            return auth_err
        from microllm.web_benchmark_suite import run_hf_jev_model_benchmarks

        body = payload or {}
        save_flag = bool(body.get("save_artifacts", True))
        entries_per_ds = body.get("entries_per_dataset")
        datasets_filter = body.get("datasets")
        fetch_live = bool(body.get("fetch_live", False))
        out_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "hf_jev_models_benchmarks_results.json"
            if (save_flag and entries_per_ds is None and not datasets_filter)
            else None
        )
        return run_hf_jev_model_benchmarks(
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
        doc_file = Path(__file__).resolve().parents[2] / "docs" / "index.html"
        if doc_file.exists():
            return HTMLResponse(content=doc_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>MicroLLM Documentation</h1>")

    @app.get("/slides", response_class=HTMLResponse)
    async def serve_slides_ui() -> HTMLResponse:
        """Serve the LightPath-branded 12-slide interactive executive & technical deck (`docs/slides.html`)."""
        slides_file = Path(__file__).resolve().parents[2] / "docs" / "slides.html"
        if slides_file.exists():
            return HTMLResponse(content=slides_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>MicroLLM LightPath Presentation Slides</h1>")

    @app.get("/assets/{asset_name}")
    async def serve_svg_asset(asset_name: str) -> FileResponse:
        assets_dir = (Path(__file__).resolve().parents[2] / "docs" / "assets").resolve()
        candidate = (assets_dir / Path(asset_name).name).resolve()
        if candidate.parent != assets_dir or not candidate.is_file() or candidate.suffix.lower() != ".svg":
            raise HTTPException(status_code=404, detail="SVG asset not found.")
        return FileResponse(path=str(candidate), media_type="image/svg+xml")

    @app.get("/papers/{paper_name}")
    async def serve_paper_pdf(paper_name: str) -> FileResponse:
        papers_dir = (Path(__file__).resolve().parents[2] / "docs" / "papers").resolve()
        candidate = (papers_dir / Path(paper_name).name).resolve()
        if candidate.parent != papers_dir or not candidate.is_file() or candidate.suffix.lower() != ".pdf":
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

    return app


def run_cli(argv: list[str] | None = None) -> int:
    """CLI handler supporting `microllm pull`, `microllm list`, `microllm benchmark`, and `microllm serve`."""
    import argparse
    import json
    import sys
    from microllm.benchmarks import run_benchmark_suite
    from microllm.registry import MODEL_PROFILES, list_profiles, pull_profile

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["serve"]

    parser = argparse.ArgumentParser(
        prog="microllm",
        description="MicroLLM Local System-One Engine CLI, Benchmark Suite & ONNX INT8 Profile Manager",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Run the FastAPI System-One server")
    serve_p.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind (default: 8000)",
    )
    serve_p.add_argument(
        "--strategy",
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto"],
        default=os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY", "auto"),
        help="Multilingual analysis strategy ('fast' | 'neural' | 'auto', default: 'auto')",
    )
    serve_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta"],
        default=os.environ.get("MICROLLM_NEURAL_MODEL", "minilm"),
        help="Neural ONNX model ('minilm' | 'mdeberta', default: 'minilm')",
    )
    serve_p.add_argument(
        "--api-key",
        default=os.environ.get("MICROLLM_API_KEY"),
        help="Optional API key for Bearer authentication",
    )
    serve_p.add_argument(
        "--reload",
        action="store_true",
        help="Enable development reload",
    )

    pull_p = sub.add_parser("pull", help="Download and quantize an ONNX INT8 hardware profile")
    pull_p.add_argument(
        "profile",
        choices=sorted(MODEL_PROFILES.keys()),
        help="Profile name (fast-cpu-int8, multilingual, guardrail)",
    )
    pull_p.add_argument("--cache-dir", default=None, help="Custom cache directory")
    pull_p.add_argument("--download", action="store_true", help="Fetch remote HF config metadata")
    pull_p.add_argument("--from-hub", dest="from_hub", action="store_true", help="Download quantized ONNX model from Hugging Face Hub")

    sub.add_parser("mcp", help="Run the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio")

    studio_p = sub.add_parser(
        "studio",
        aliases=["gradio"],
        help="Launch the Gradio Decision Studio UI (Hugging Face Spaces compatible engine)",
    )
    studio_p.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860)")
    studio_p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    studio_p.add_argument(
        "--strategy",
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto"],
        default=None,
        help="Default multilingual strategy ('fast' | 'neural' | 'auto')",
    )
    studio_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta"],
        default=None,
        help="Default neural ONNX model ('minilm' | 'mdeberta')",
    )

    list_p = sub.add_parser("list", help="List available and installed hardware profiles")
    list_p.add_argument("--cache-dir", default=None, help="Custom cache directory")

    bench_p = sub.add_parser("benchmark", help="Run the 540-item, 16-dataset external, or unified all-suites evaluation")
    bench_p.add_argument("--output", default=None, help="Optional JSON output file path")
    bench_p.add_argument(
        "--all-suites",
        "--all",
        dest="all_suites",
        action="store_true",
        help="Run the unified automation orchestrator across all MicroLLM benchmark suites",
    )
    bench_p.add_argument(
        "--suites",
        default=None,
        help="Comma-separated list of suite IDs when running with --all-suites",
    )
    bench_p.add_argument(
        "--skip-full-splits",
        "--fast",
        "--quick",
        dest="skip_full_splits",
        action="store_true",
        help="Skip the 114,018-row full unsampled external split evaluation when running --all-suites",
    )
    bench_p.add_argument(
        "--external",
        "--web",
        dest="external",
        action="store_true",
        help="Run the 16-dataset external web benchmark suite (BoolQ, MultiNLI, Banking77, CLINC150, WildGuard, RouteLLM, JevBench, etc.)",
    )
    bench_p.add_argument(
        "--hf-jev-models",
        dest="hf_jev_models",
        action="store_true",
        help="Run the 12-dataset Hugging Face JEV models ecosystem benchmark suite (huggingface.co/models?search=jev)",
    )
    bench_p.add_argument(
        "--full-splits",
        "--no-sampling",
        dest="full_splits",
        action="store_true",
        help="Evaluate complete unsampled official evaluation splits across all 16 external datasets (114,018 rows)",
    )
    bench_p.add_argument(
        "--entries-per-dataset",
        "--scale-per-dataset",
        dest="entries_per_dataset",
        type=int,
        default=None,
        help="Optional target entries per external dataset (e.g. 40 for 640 total entries)",
    )
    bench_p.add_argument(
        "--fetch-live",
        dest="fetch_live",
        action="store_true",
        help="Fetch live rows from Hugging Face Datasets Server API when running sampled external benchmarks",
    )
    bench_p.add_argument(
        "--multilingual",
        dest="multilingual",
        action="store_true",
        help="Run the quadrilingual evaluation benchmark across EN, FR, ES, PT with fast, neural, and auto cascade options",
    )
    bench_p.add_argument(
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto", "all"],
        default="all",
        help="Multilingual strategy option to benchmark ('fast', 'neural', 'auto', or 'all')",
    )
    bench_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta", "all"],
        default="all",
        help="Neural model option for neural and auto strategies ('minilm', 'mdeberta', or 'all')",
    )
    bench_p.add_argument(
        "--language",
        dest="language",
        choices=["en", "fr", "es", "pt", "all"],
        default="all",
        help="Language to evaluate ('en', 'fr', 'es', 'pt', or 'all')",
    )

    papers_p = sub.add_parser(
        "papers",
        help="Search, verify, and download the Jev, RLCD, System-1 & Calibrated LLM academic paper catalog",
    )
    papers_p.add_argument("--pillar", default=None, help="Filter by pillar ID or title substring")
    papers_p.add_argument("--query", "-q", default=None, help="Keyword search in title, authors, summary, or relevance")
    papers_p.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")
    papers_p.add_argument("--download-pdfs", action="store_true", help="Download matching arXiv PDFs into docs/papers/")
    papers_p.add_argument("--verify-pdfs", action="store_true", help="Verify local PDF presence in docs/papers/")
    papers_p.add_argument("--output-dir", default=None, help="Optional custom output directory for PDFs")

    rca_p = sub.add_parser("rca", help="Run sequential or targeted RCA non-regression verification")
    rca_p.add_argument("--rca", default=None, help="Target specific RCA ID (e.g., RCA-001)")
    rca_p.add_argument("--all", action="store_true", help="Run all 15 RCAs sequentially one by one")
    rca_p.add_argument("--list", action="store_true", help="List all registered RCAs in catalog")
    rca_p.add_argument("--iterations", type=int, default=2, help="Number of profiling iterations per RCA (default: 2)")
    rca_p.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")

    sync_hf_p = sub.add_parser(
        "sync-hf-readme",
        help="Synchronize root README.md to all 6 official Hugging Face repositories (IDEA-038)",
    )
    sync_hf_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build all 6 Hugging Face cards without uploading to the Hub",
    )
    sync_hf_p.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face API token (defaults to HF_TOKEN or HUGGINGFACE_TOKEN env var)",
    )

    publish_p = sub.add_parser(
        "publish",
        help="Stage and publish multilingual benchmark datasets and neural models to Hugging Face Hub",
        description="Stage and publish multilingual benchmark datasets and neural models to Hugging Face Hub",
    )
    publish_p.add_argument(
        "--dataset",
        action="store_true",
        help="Package and publish the multilingual benchmark dataset (jmrenouard/microllm-multilingual-benchmarks)",
    )
    publish_p.add_argument(
        "--model",
        default=None,
        help="Profile name or alias of the neural multilingual model to package and publish (e.g. multilingual-minilm-int8, multilingual-mdeberta-int8)",
    )
    publish_p.add_argument(
        "--all",
        action="store_true",
        help="Package and publish both the multilingual benchmark dataset and all neural multilingual models",
    )
    publish_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage files locally without publishing to Hugging Face Hub (zero network calls)",
    )
    publish_p.add_argument(
        "--staging-dir",
        default=None,
        help="Custom directory to stage repository artifacts",
    )
    publish_p.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN or KeePass vault)",
    )
    publish_p.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message for Hub uploads",
    )
    publish_p.add_argument(
        "--space",
        action="store_true",
        help="Stage and synchronize Gradio Decision Studio to Hugging Face Space",
    )
    publish_p.add_argument(
        "--space-repo-id",
        default="jmrenouard/microllm-decision-studio",
        help="Target Hugging Face Space repository ID (default: jmrenouard/microllm-decision-studio)",
    )
    publish_p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (json or text, default: json)",
    )

    sync_space_p = sub.add_parser(
        "sync-space",
        help="Stage and synchronize Gradio Decision Studio to Hugging Face Space (jmrenouard/microllm-decision-studio)",
        description="Stage and synchronize Gradio Decision Studio to Hugging Face Space",
    )
    sync_space_p.add_argument(
        "--repo-id",
        default="jmrenouard/microllm-decision-studio",
        help="Target Hugging Face Space repository ID (default: jmrenouard/microllm-decision-studio)",
    )
    sync_space_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage Space bundle locally and validate manifests without uploading (zero network calls)",
    )
    sync_space_p.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN or KeePass vault)",
    )
    sync_space_p.add_argument(
        "--staging-dir",
        default=None,
        help="Custom staging directory for Space bundle",
    )
    sync_space_p.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for Hugging Face Space upload",
    )
    sync_space_p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (json or text, default: json)",
    )

    parsed = parser.parse_args(args_list)
    if parsed.command == "sync-space":
        from microllm.hf_space_sync import sync_hf_space

        res = sync_hf_space(
            space_repo_id=parsed.repo_id,
            staging_dir=parsed.staging_dir,
            dry_run=bool(parsed.dry_run),
            token=parsed.token,
            commit_message=parsed.commit_message,
        )
        if parsed.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"Status: {res.get('status')}")
            print(f"Space Repo: {res.get('repo_id')}")
            print(f"Staging Directory: {res.get('staging_dir')}")
        return 0
    if parsed.command == "publish":
        from microllm.hf_publishing import publish_cli

        return publish_cli(parsed)
    if parsed.command == "serve":
        if parsed.multilingual_strategy:
            os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] = str(parsed.multilingual_strategy)
        if parsed.neural_model:
            os.environ["MICROLLM_NEURAL_MODEL"] = str(parsed.neural_model)
        if parsed.api_key:
            os.environ["MICROLLM_API_KEY"] = str(parsed.api_key)
        uvicorn.run(
            "microllm.server:create_app",
            factory=True,
            host=parsed.host,
            port=parsed.port,
            reload=bool(getattr(parsed, "reload", False)),
        )
        return 0
    if parsed.command == "sync-hf-readme":
        from microllm.registry import sync_readme_to_huggingface_hub

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sync_res = sync_readme_to_huggingface_hub(
            token=parsed.token,
            dry_run=bool(parsed.dry_run),
        )
        print(json.dumps(sync_res, indent=2))
        return 0
    if parsed.command == "pull":
        manifest = pull_profile(
            parsed.profile,
            cache_dir=parsed.cache_dir,
            download_weights=parsed.download,
            from_hub=bool(getattr(parsed, "from_hub", False)),
        )
        print(json.dumps(manifest, indent=2))
        return 0
    if parsed.command == "mcp":
        from microllm.mcp_server import run_mcp_stdio_server

        return run_mcp_stdio_server()
    if parsed.command in {"studio", "gradio"}:
        if getattr(parsed, "multilingual_strategy", None):
            os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] = str(parsed.multilingual_strategy)
        if getattr(parsed, "neural_model", None):
            os.environ["MICROLLM_NEURAL_MODEL"] = str(parsed.neural_model)
        from microllm.gradio_app import create_gradio_demo

        demo = create_gradio_demo()
        demo.launch(server_name="127.0.0.1", server_port=int(parsed.port), share=bool(parsed.share))
        return 0

    if parsed.command == "list":
        print(json.dumps(list_profiles(cache_dir=parsed.cache_dir), indent=2))
        return 0
    if parsed.command == "benchmark":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if getattr(parsed, "all_suites", False):
            from microllm.benchmark_runner import (
                format_master_benchmark_markdown,
                run_all_benchmark_suites,
            )

            raw_suites = getattr(parsed, "suites", None)
            suite_list = (
                [s.strip() for s in raw_suites.split(",") if s.strip()]
                if raw_suites
                else None
            )
            try:
                master_rep = run_all_benchmark_suites(
                    suites=suite_list,
                    include_full_splits=not bool(getattr(parsed, "skip_full_splits", False)),
                    master_output_path=parsed.output,
                    save_artifacts=True,
                )
            except ValueError as exc:
                sys.stderr.write(f"Error: {exc}\n")
                return 2
            print(format_master_benchmark_markdown(master_rep))
            return 0 if master_rep["all_suites_passed"] else 1
        if getattr(parsed, "hf_jev_models", False):
            from microllm.web_benchmark_suite import (
                format_hf_jev_models_benchmarks_markdown,
                run_hf_jev_model_benchmarks,
            )

            default_jev_out = (
                Path(parsed.output)
                if parsed.output
                else Path(__file__).resolve().parents[2]
                / "benchmarks"
                / "hf_jev_models_benchmarks_results.json"
            )
            jev_rep = run_hf_jev_model_benchmarks(
                output_path=default_jev_out,
                entries_per_dataset=getattr(parsed, "entries_per_dataset", None),
                fetch_live=bool(getattr(parsed, "fetch_live", False)),
            )
            print(format_hf_jev_models_benchmarks_markdown(jev_rep))
            return 0
        if getattr(parsed, "external", False):
            from microllm.web_benchmark_suite import (
                format_web_benchmarks_markdown,
                run_all_web_benchmarks,
            )

            full_splits = bool(getattr(parsed, "full_splits", False))
            default_filename = (
                "web_benchmarks_full_results.json"
                if full_splits
                else "web_benchmarks_results.json"
            )
            default_ext_out = (
                Path(parsed.output)
                if parsed.output
                else Path(__file__).resolve().parents[2] / "benchmarks" / default_filename
            )
            ext_rep = run_all_web_benchmarks(
                output_path=default_ext_out,
                entries_per_dataset=getattr(parsed, "entries_per_dataset", None),
                fetch_live=bool(getattr(parsed, "fetch_live", False)),
                full_splits=full_splits,
            )
            print(format_web_benchmarks_markdown(ext_rep))
            return 0
        if getattr(parsed, "multilingual", False):
            from microllm.multilingual_benchmark_runner import (
                format_multilingual_benchmarks_markdown,
                run_multilingual_benchmarks,
            )

            strats = (
                ["fast", "neural", "auto"]
                if getattr(parsed, "multilingual_strategy", "all") == "all"
                else [parsed.multilingual_strategy]
            )
            models = (
                ["minilm", "mdeberta"]
                if getattr(parsed, "neural_model", "all") == "all"
                else [parsed.neural_model]
            )
            langs = (
                ["en", "fr", "es", "pt"]
                if getattr(parsed, "language", "all") == "all"
                else [parsed.language]
            )
            m_rep = run_multilingual_benchmarks(
                output_path=parsed.output,
                strategies=strats,
                neural_models=models,
                languages=langs,
                save_artifacts=True,
            )
            if getattr(parsed, "json", False):
                print(json.dumps(m_rep, indent=2))
            else:
                print(format_multilingual_benchmarks_markdown(m_rep))
            return 0
        rep = run_benchmark_suite(save_artifacts=True, output_path=parsed.output)
        print(rep["markdown_table"])
        return 0
    if parsed.command == "papers":
        from microllm.papers import (
            download_academic_papers,
            format_papers_markdown,
            get_literature_taxonomy,
            list_academic_papers,
            verify_downloaded_pdfs,
        )

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if parsed.download_pdfs:
            dl_res = download_academic_papers(
                output_dir=parsed.output_dir,
                pillar=parsed.pillar,
                query=parsed.query,
            )
            print(json.dumps(dl_res, indent=2))
            return 0
        if parsed.verify_pdfs:
            ver_res = verify_downloaded_pdfs(
                output_dir=parsed.output_dir,
                pillar=parsed.pillar,
                query=parsed.query,
            )
            print(json.dumps(ver_res, indent=2))
            return 0
        if parsed.format == "markdown":
            print(format_papers_markdown(pillar=parsed.pillar, query=parsed.query))
        else:
            items = list_academic_papers(pillar=parsed.pillar, query=parsed.query)
            print(
                json.dumps(
                    {
                        "total": len(items),
                        "taxonomy": get_literature_taxonomy(),
                        "papers": items,
                    },
                    indent=2,
                )
            )
        return 0
    if parsed.command == "rca":
        from microllm.rca_runner import RCARunner

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        runner = RCARunner()
        if parsed.list:
            if parsed.format == "json":
                print(json.dumps(runner.list_rcas(), indent=2))
            else:
                print("| RCA ID | Title | Module | Target SLA | Latency SLA |")
                print("| :--- | :--- | :--- | :---: | :---: |")
                for r in runner.list_rcas():
                    print(f"| **{r['id']}** | {r['title']} | `{r['target_module']}` | {r['accuracy_sla']*100:.0f}% | {r['latency_p50_sla_ms']:.1f} ms |")
            return 0

        target_ids = [parsed.rca] if parsed.rca else None
        results = runner.run_sequential(
            rca_ids=target_ids,
            iterations_per_rca=parsed.iterations,
        )

        if parsed.format == "json":
            from dataclasses import asdict
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print(runner.format_report_markdown(results))

        all_ok = all(r.non_regression_passed for r in results)
        return 0 if all_ok else 1

    parser.print_help()
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `microllm`."""
    code = run_cli(argv)
    if argv is None:
        raise SystemExit(code)
    return code


if __name__ == "__main__":
    main()

