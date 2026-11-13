"""Backend evaluation handlers and experiment runners for Gradio Decision Studio."""

from __future__ import annotations

import functools
import json
import time
from typing import Any

from microllm.backends.base import explain_state_tokens
from microllm.backends.fast_encoder import format_nli_hypothesis
from microllm.gradio_helpers import (
    compute_deep_studio_diagnostics,
    generate_code_export,
    upsert_visual_question_into_json,
)
from microllm.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    MultiAnswer,
    MultiQuestion,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    render_structured_text,
)


@functools.lru_cache(maxsize=32)
def _get_cached_studio_backend(
    strategy: str,
    neural_model: str,
    margin_threshold: float = 0.12,
    fusion_alpha: float = 0.75,
) -> Any:
    """Return a cached multilingual/cascade backend instance so UI clicks avoid reloading `.npz` weights."""
    from microllm.backends.cascade import create_multilingual_backend

    return create_multilingual_backend(
        strategy=strategy,  # type: ignore[arg-type]
        neural_model=neural_model,
        margin_threshold=float(margin_threshold),
        entropy_threshold=0.88,
        fusion_alpha=float(fusion_alpha),
    )


_STUDIO_TEST_CLIENT: Any = None


def _get_studio_test_client() -> Any:
    """Return a lazy in-memory FastAPI `TestClient` singleton for Guardrail and RLCD Studio tabs."""
    global _STUDIO_TEST_CLIENT
    if _STUDIO_TEST_CLIENT is None:
        from fastapi.testclient import TestClient
        from microllm.server import create_app

        _STUDIO_TEST_CLIENT = TestClient(create_app())
    return _STUDIO_TEST_CLIENT


def evaluate_studio_multi_mode(
    input_mode: str,
    state_input: str,
    questions_json: str,
    visual_qid: str,
    visual_type: str,
    visual_instructions: str,
    visual_criteria: str,
    visual_threshold: float,
    engine_choice: str,
    gate_threshold: float,
    alpha_val: float,
    export_lang: str = "Python SDK",
    multilingual_strategy: str = "auto",
    neural_model: str = "minilm",
    target_language: str = "auto",
) -> tuple[str, str, dict[str, float], dict[str, Any], str]:
    """Evaluate single or multi-question payload from either Visual Builder Mode or Raw JSON Mode."""
    from microllm.lexicons.normalizer import detect_language_fast
    from microllm.server import StudioAnalyzeRequest, _evaluate_isolated_question

    t0 = time.perf_counter()
    raw_state = state_input.strip() or "Empty state"
    state_val: Any = raw_state
    if raw_state.startswith("{") or raw_state.startswith("["):
        try:
            state_val = json.loads(raw_state)
        except Exception:
            state_val = raw_state

    # Determine active questions dictionary depending on mode
    questions_dict: dict[str, Any] = {}
    if input_mode == "Visual Builder Mode":
        try:
            parsed = json.loads(questions_json) if questions_json.strip() else {}
            if isinstance(parsed, dict):
                questions_dict = dict(parsed)
        except Exception:
            questions_dict = {}
        if visual_qid.strip():
            updated_str, _ = upsert_visual_question_into_json(
                json.dumps(questions_dict),
                visual_qid,
                visual_type,
                visual_instructions,
                visual_criteria,
                visual_threshold,
            )
            questions_dict = json.loads(updated_str)
    else:
        try:
            questions_dict = json.loads(questions_json)
        except Exception as exc:
            err_msg = f"### ❌ JSON Syntax Error\n`{exc}`\nSwitch to **Visual Builder Mode** to build questions without writing raw JSON."
            return err_msg, err_msg, {}, {"error": str(exc)}, ""

    try:
        req = StudioAnalyzeRequest(
            state=state_val,
            questions=questions_dict,
            gate_threshold=float(gate_threshold),
            alpha=float(alpha_val),
            run_order_flip_test=True,
        )
    except Exception as exc:
        err_msg = f"### ❌ Validation Error\n`{exc}`"
        return err_msg, err_msg, {}, {"error": str(exc)}, ""

    clean_strat = (multilingual_strategy or "auto").lower().strip()
    clean_model = "mdeberta" if "mdeberta" in str(neural_model).lower() else "minilm"

    if "Gear 1" in engine_choice:
        active_strat = "fast"
    elif "Gear 2" in engine_choice:
        active_strat = "neural"
    else:
        active_strat = clean_strat

    gear1 = _get_cached_studio_backend("fast", clean_model, round(float(req.gate_threshold), 4), round(float(req.alpha), 4))
    gear2 = _get_cached_studio_backend("neural", clean_model, round(float(req.gate_threshold), 4), round(float(req.alpha), 4))
    custom_cascade = _get_cached_studio_backend(
        active_strat,
        clean_model,
        round(float(req.gate_threshold), 4),
        round(float(req.alpha), 4),
    )
    state_text = render_structured_text(req.state)
    detected_lang = detect_language_fast(state_text)
    effective_lang = detected_lang if target_language == "auto" else target_language
    if effective_lang not in ("en", "fr", "es", "pt"):
        effective_lang = "en"

    g1_out: dict[str, Any] = {}
    g2_out: dict[str, Any] = {}
    cas_out: dict[str, Any] = {}
    flip_audit: dict[str, Any] = {}
    token_attribution: dict[str, list[dict[str, Any]]] = {}
    counterfactual_xai: dict[str, dict[str, Any]] = {}
    coherence_audit: dict[str, dict[str, Any]] = {}
    voc_diagnostics: dict[str, dict[str, Any]] = {}

    summary_lines: list[str] = []
    xai_lines: list[str] = []
    primary_probs: dict[str, float] = {}

    for qid, q_spec in req.questions.items():
        a1, raw1 = _evaluate_isolated_question(gear1, state_text, q_spec)
        a2, raw2 = _evaluate_isolated_question(gear2, state_text, q_spec)
        ac, rawc = _evaluate_isolated_question(custom_cascade, state_text, q_spec)

        active_ans = ac
        active_raw = rawc
        if "Gear 1" in engine_choice:
            active_ans, active_raw = a1, raw1
        elif "Gear 2" in engine_choice:
            active_ans, active_raw = a2, raw2

        g1_out[qid] = {**a1.model_dump(), "meta": raw1}
        g2_out[qid] = {**a2.model_dump(), "meta": raw2}
        cas_out[qid] = {**active_ans.model_dump(), "meta": active_raw}

        instr_text = render_structured_text(q_spec.instructions, fallback="")
        is_focused_visual_q = (
            input_mode == "Visual Builder Mode" and visual_qid.strip() == qid
        )
        if isinstance(q_spec, ChoiceQuestion) and isinstance(active_ans, ChoiceAnswer):
            win_desc = render_structured_text(q_spec.criteria.get(active_ans.choice), fallback="")
            win_hyp = format_nli_hypothesis(active_ans.choice, win_desc)
            if not primary_probs or is_focused_visual_q:
                primary_probs = {str(k): float(v) for k, v in active_ans.probabilities.items()}
            verdict_str = f"`{active_ans.choice}` (Confidence `{active_ans.confidence * 100:.1f}%`)"
        elif isinstance(q_spec, MultiQuestion) and isinstance(active_ans, MultiAnswer):
            best_lbl = max(active_ans.probabilities, key=lambda k: active_ans.probabilities[k])
            win_desc = render_structured_text(q_spec.criteria.get(best_lbl), fallback="")
            win_hyp = format_nli_hypothesis(best_lbl, win_desc)
            if not primary_probs or is_focused_visual_q:
                primary_probs = {str(k): float(v) for k, v in active_ans.probabilities.items()}
            verdict_str = f"`{active_ans.labels}` (Certainty `{active_ans.confidence * 100:.1f}%`)"
        elif isinstance(q_spec, ScoreQuestion) and isinstance(active_ans, ScoreAnswer):
            best_idx = max(active_ans.probabilities, key=lambda k: active_ans.probabilities[k])
            win_hyp = format_nli_hypothesis(best_idx, render_structured_text(active_ans.legend[best_idx]))
            if not primary_probs or is_focused_visual_q:
                primary_probs = {
                    f"{k} ({active_ans.legend.get(k, k)})": float(v)
                    for k, v in active_ans.probabilities.items()
                }
            verdict_str = f"Score `{active_ans.score:.2f}` (Confidence `{active_ans.confidence * 100:.1f}%`)"
        else:
            assert isinstance(active_ans, NoulAnswer)
            win_hyp = instr_text or "True affirmative proposition"
            p_t = float(active_ans.noul)
            if not primary_probs or is_focused_visual_q:
                primary_probs = {"true": round(p_t, 4), "false": round(1.0 - p_t, 4)}
            verdict_str = f"`{'TRUE' if p_t >= 0.5 else 'FALSE'}` (`P(True) = {p_t * 100:.1f}%`)"

        attrs = explain_state_tokens(state_text, instr_text, win_hyp, top_k=6)
        token_attribution[qid] = attrs

        cf_info, coh_info, voc_info = compute_deep_studio_diagnostics(state_text, q_spec, active_ans)
        counterfactual_xai[qid] = cf_info
        coherence_audit[qid] = coh_info
        voc_diagnostics[qid] = voc_info

        abstain_flag = bool(active_raw.get("x_abstain", False) or voc_info.get("should_abstain", False))
        conflict_flag = bool(active_raw.get("x_rubric_conflict_detected", False))
        abstain_badge = "⚠️ **ABSTAIN (Escalate)**" if abstain_flag else "✅ **COMMIT (System-1)**"
        conflict_badge = " · 🚨 **Rubric Key/Definition Conflict Detected**" if conflict_flag else ""

        strat_used = active_raw.get("x_strategy_used", active_strat)
        cascade_triggered = bool(active_raw.get("x_cascade_triggered", False))
        cascade_reason = str(active_raw.get("x_cascade_reason", "direct"))
        top2_margin = float(active_raw.get("x_top2_margin", 0.0))
        norm_entropy = float(active_raw.get("x_norm_entropy", 0.0))
        n_model = active_raw.get("x_neural_model", clean_model)

        if strat_used == "fast":
            routing_badge = "⚡ **Strategy 1: Fast Lexicon Direct** (<0.20 ms CPU)"
        else:
            escalation_note = f" (Escalated: `{cascade_reason}`, Margin `{top2_margin:.3f}`)" if cascade_triggered else ""
            routing_badge = f"🧠 **Strategy 2: Neural ONNX ({n_model})**{escalation_note}"

        if isinstance(q_spec, ChoiceQuestion) and isinstance(a2, ChoiceAnswer):
            rev_q = ChoiceQuestion(
                type="choice",
                instructions=q_spec.instructions,
                criteria=dict(reversed(list(q_spec.criteria.items()))),
            )
            a2_rev, _ = _evaluate_isolated_question(gear2, state_text, rev_q)
            if isinstance(a2_rev, ChoiceAnswer):
                max_diff = max(
                    abs(a2.probabilities[k] - a2_rev.probabilities[k]) for k in a2.probabilities
                )
                flip_audit[qid] = {
                    "forward_choice": a2.choice,
                    "reverse_choice": a2_rev.choice,
                    "order_flipped": a2.choice != a2_rev.choice,
                    "max_probability_drift": round(float(max_diff), 6),
                }

        summary_lines.append(
            f"- **`{qid}`** (`{q_spec.type}`): {verdict_str} · {routing_badge} · Lang: `{effective_lang.upper()}` · {abstain_badge} · Regime: `{voc_info['uncertainty_regime']}`{conflict_badge}"
        )


        top_toks = ", ".join(
            f"`{a['token']}` ({a['normalized_weight'] * 100:.0f}%)" for a in attrs[:4]
        ) or "None"
        flip_toks = ", ".join(f"`{t}`" for t in cf_info["flip_candidates"][:3]) or "None (Robust)"
        top_impacts = ", ".join(
            f"`-{ imp['token'] }` (ΔP={imp['probability_drop']:+.3f})"
            for imp in cf_info["token_impacts"][:3]
        )
        xai_lines.append(
            f"#### 🔍 Question `{qid}` ({q_spec.type})\n"
            f"- **Top Driving State Tokens**: {top_toks}\n"
            f"- **Counterfactual Minimal Flip Tokens (`IDEA-019`)**: {flip_toks} · Leave-One-Out Shifts: {top_impacts}\n"
            f"- **Kolmogorov Probability Coherence (`IDEA-023`)**: Raw Noul Sum = `{coh_info['raw_noul_sum']:.3f}` (Dev `{coh_info['raw_coherence_deviation']:.3f}`) → Bregman Simplex Dev = **`{coh_info['projected_coherence_deviation']:.6f}`** · Complement Sum = `{coh_info['complement_sum']:.4f}`\n"
            f"- **Value-of-Computation Regime (`HYP-005`)**: `{voc_info['uncertainty_regime']}` (`E[VOC] = {voc_info['expected_voc']:+.4f}`, Max Logit = `{voc_info['max_pre_softmax_logit']:.3f}`)"
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    summary_md = (
        f"### 🏆 Calibrated System-1 Decisions ({len(req.questions)} Question(s) · `{elapsed_ms:.2f} ms` CPU)\n"
        + "\n".join(summary_lines)
    )
    xai_md = "### 🧠 Deep Counterfactual XAI & Probabilistic Coherence Audit\n" + "\n\n".join(xai_lines)

    telemetry = {
        "latency_ms": elapsed_ms,
        "cascade": cas_out,
        "gear1": g1_out,
        "gear2": g2_out,
        "multilingual_routing": {
            "strategy": active_strat,
            "neural_model": clean_model,
            "detected_language": detected_lang,
            "effective_language": effective_lang,
            "cascade_triggered": any(bool(cas_out[q].get("meta", {}).get("x_cascade_triggered", False)) for q in cas_out),
            "cascade_reason": next((str(cas_out[q].get("meta", {}).get("x_cascade_reason", "direct")) for q in cas_out if cas_out[q].get("meta", {}).get("x_cascade_reason")), "direct"),
            "strategy_used": next((str(cas_out[q].get("meta", {}).get("x_strategy_used", active_strat)) for q in cas_out if cas_out[q].get("meta", {}).get("x_strategy_used")), active_strat),
            "top2_margin": max((float(cas_out[q].get("meta", {}).get("x_top2_margin", 0.0)) for q in cas_out), default=0.0),
            "norm_entropy": max((float(cas_out[q].get("meta", {}).get("x_norm_entropy", 0.0)) for q in cas_out), default=0.0),
        },
        "order_flip_audit": flip_audit,
        "token_attribution": token_attribution,
        "counterfactual_xai": counterfactual_xai,
        "coherence_audit": coherence_audit,
        "voc_diagnostics": voc_diagnostics,
    }

    code_snippet = generate_code_export(state_text, questions_dict, export_lang)
    return summary_md, xai_md, primary_probs, telemetry, code_snippet



def evaluate_consumer_yes_no(
    text_input: str,
    question_prompt: str,
    yes_hint: str = "",
    no_hint: str = "",
    engine_choice: str = "Dual-Gear Cascade (StartLux) [Recommended]",
    gate_threshold: float = 0.75,
    alpha_val: float = 0.75,
    multilingual_strategy: str = "auto",
    neural_model: str = "minilm",
    language: str = "auto",
) -> tuple[str, dict[str, float], str, dict[str, Any], str]:
    """Consumer-friendly evaluator for the dedicated Yes / No (`noul`) tab."""
    q_clean = question_prompt.strip() or "Is this statement affirmative?"
    y_desc = yes_hint.strip() or f"Yes, affirmative: {q_clean}"
    n_desc = no_hint.strip() or f"No, negative or not applicable: {q_clean}"
    crit_text = f"true: {y_desc}\nfalse: {n_desc}"

    _, xai_md, probs_raw, telemetry, code_snippet = evaluate_studio_multi_mode(
        input_mode="Visual Builder Mode",
        state_input=text_input,
        questions_json="{}",
        visual_qid="yes_no_decision",
        visual_type="noul",
        visual_instructions=q_clean,
        visual_criteria=crit_text,
        visual_threshold=0.5,
        engine_choice=engine_choice,
        gate_threshold=gate_threshold,
        alpha_val=alpha_val,
        export_lang="Python SDK",
        multilingual_strategy=multilingual_strategy,
        neural_model=neural_model,
        target_language=language,
    )
    p_yes = float(probs_raw.get("true", 0.5))
    p_no = round(1.0 - p_yes, 4)
    is_yes = p_yes >= 0.5
    conf = max(p_yes, p_no)
    badge = "✅ YES (Affirmative)" if is_yes else "❌ NO (Negative)"
    certainty_pill = (
        "🟢 **High Certainty** — Ready for automatic decision"
        if conf >= 0.75
        else "🟡 **Moderate Certainty** — Human review recommended"
    )
    attrs = telemetry.get("token_attribution", {}).get("yes_no_decision", [])
    key_words = ", ".join(f"**`{a['token']}`**" for a in attrs[:4]) or "None"
    verdict_md = (
        f"## {badge} — `{conf * 100:.1f}%` Confidence\n"
        f"- **Question**: *{q_clean}*\n"
        f"- **Probabilities**: **YES** = `{p_yes * 100:.1f}%` · **NO** = `{p_no * 100:.1f}%`\n"
        f"- **Certainty Guide**: {certainty_pill}\n"
        f"- **Key Words Driving Decision**: {key_words} · *(CPU Latency: `{telemetry.get('latency_ms', 0.0):.2f} ms`)*"
    )
    consumer_probs = {"YES (True)": round(p_yes, 4), "NO (False)": round(p_no, 4)}
    return verdict_md, consumer_probs, xai_md, telemetry, code_snippet


def evaluate_consumer_choice(
    text_input: str,
    question_prompt: str,
    options_text: str,
    engine_choice: str = "Dual-Gear Cascade (StartLux) [Recommended]",
    gate_threshold: float = 0.75,
    alpha_val: float = 0.75,
    multilingual_strategy: str = "auto",
    neural_model: str = "minilm",
    language: str = "auto",
) -> tuple[str, dict[str, float], str, dict[str, Any], str]:
    """Consumer-friendly evaluator for the dedicated Single Choice (`choice`) tab."""
    q_clean = question_prompt.strip() or "Select the best matching category."
    _, xai_md, probs_raw, telemetry, code_snippet = evaluate_studio_multi_mode(
        input_mode="Visual Builder Mode",
        state_input=text_input,
        questions_json="{}",
        visual_qid="category_choice",
        visual_type="choice",
        visual_instructions=q_clean,
        visual_criteria=options_text,
        visual_threshold=0.5,
        engine_choice=engine_choice,
        gate_threshold=gate_threshold,
        alpha_val=alpha_val,
        export_lang="Python SDK",
        multilingual_strategy=multilingual_strategy,
        neural_model=neural_model,
        target_language=language,
    )
    cas_ans = telemetry.get("cascade", {}).get("category_choice", {})
    winner = str(cas_ans.get("choice", next(iter(probs_raw.keys()), "N/A")))
    conf = float(cas_ans.get("confidence", probs_raw.get(winner, 0.0)))
    conformal_set = cas_ans.get("meta", {}).get("x_conformal_set", [winner])
    attrs = telemetry.get("token_attribution", {}).get("category_choice", [])
    key_words = ", ".join(f"**`{a['token']}`**" for a in attrs[:4]) or "None"
    certainty_note = (
        f"🟢 **Clear Winner** (`{winner}`)"
        if len(conformal_set) <= 1
        else f"🟡 **Plausible Alternatives**: `{', '.join(str(x) for x in conformal_set)}`"
    )
    verdict_md = (
        f"## 🎯 Selected Category: `{winner}` (`{conf * 100:.1f}%` Confidence)\n"
        f"- **Question**: *{q_clean}*\n"
        f"- **Decision Certainty**: {certainty_note}\n"
        f"- **Key Words Driving Decision**: {key_words} · *(CPU Latency: `{telemetry.get('latency_ms', 0.0):.2f} ms`)*"
    )
    return verdict_md, probs_raw, xai_md, telemetry, code_snippet


def evaluate_consumer_multi(
    text_input: str,
    question_prompt: str,
    tags_text: str,
    sensitivity_threshold: float = 0.5,
    engine_choice: str = "Dual-Gear Cascade (StartLux) [Recommended]",
    gate_threshold: float = 0.75,
    alpha_val: float = 0.75,
    multilingual_strategy: str = "auto",
    neural_model: str = "minilm",
    language: str = "auto",
) -> tuple[str, dict[str, float], str, dict[str, Any], str]:
    """Consumer-friendly evaluator for the dedicated Multi-Tagging (`multi`) tab."""
    q_clean = question_prompt.strip() or "Select all tags that apply to this text."
    _, xai_md, probs_raw, telemetry, code_snippet = evaluate_studio_multi_mode(
        input_mode="Visual Builder Mode",
        state_input=text_input,
        questions_json="{}",
        visual_qid="multi_tags",
        visual_type="multi",
        visual_instructions=q_clean,
        visual_criteria=tags_text,
        visual_threshold=float(sensitivity_threshold),
        engine_choice=engine_choice,
        gate_threshold=gate_threshold,
        alpha_val=alpha_val,
        export_lang="Python SDK",
        multilingual_strategy=multilingual_strategy,
        neural_model=neural_model,
        target_language=language,
    )
    cas_ans = telemetry.get("cascade", {}).get("multi_tags", {})
    active_labels: list[str] = list(cas_ans.get("labels", []))
    active_badges = (
        ", ".join(f"✅ **`{lbl}`** (`{probs_raw.get(lbl, 0.0) * 100:.0f}%`)" for lbl in active_labels)
        if active_labels
        else "⚪ *No tag exceeded the sensitivity threshold*"
    )
    inactive_labels = [k for k in probs_raw if k not in active_labels]
    inactive_badges = (
        ", ".join(f"⚪ `{lbl}` (`{probs_raw.get(lbl, 0.0) * 100:.0f}%`)" for lbl in inactive_labels)
        or "None"
    )
    verdict_md = (
        f"## 🏷️ Matched Tags ({len(active_labels)} / {len(probs_raw)} Active)\n"
        f"- **Selected Tags (`≥ {sensitivity_threshold:.2f}`)**: {active_badges}\n"
        f"- **Other Candidate Tags**: {inactive_badges}\n"
        f"- **Overall Multi-Label Certainty**: `{float(cas_ans.get('confidence', 0.0)) * 100:.1f}%` · *(CPU Latency: `{telemetry.get('latency_ms', 0.0):.2f} ms`)*"
    )
    return verdict_md, probs_raw, xai_md, telemetry, code_snippet


def evaluate_consumer_score(
    text_input: str,
    question_prompt: str,
    scale_levels_text: str,
    engine_choice: str = "Dual-Gear Cascade (StartLux) [Recommended]",
    gate_threshold: float = 0.75,
    alpha_val: float = 0.75,
    multilingual_strategy: str = "auto",
    neural_model: str = "minilm",
    language: str = "auto",
) -> tuple[str, dict[str, float], str, dict[str, Any], str]:
    """Consumer-friendly evaluator for the dedicated Rating & Scale (`score`) tab."""
    q_clean = question_prompt.strip() or "Rate the intensity level on this scale."
    _, xai_md, probs_raw, telemetry, code_snippet = evaluate_studio_multi_mode(
        input_mode="Visual Builder Mode",
        state_input=text_input,
        questions_json="{}",
        visual_qid="intensity_rating",
        visual_type="score",
        visual_instructions=q_clean,
        visual_criteria=scale_levels_text,
        visual_threshold=0.5,
        engine_choice=engine_choice,
        gate_threshold=gate_threshold,
        alpha_val=alpha_val,
        export_lang="Python SDK",
        multilingual_strategy=multilingual_strategy,
        neural_model=neural_model,
        target_language=language,
    )

    cas_ans = telemetry.get("cascade", {}).get("intensity_rating", {})
    score_val = float(cas_ans.get("score", 0.0))
    legend: dict[str, str] = cas_ans.get("legend", {})
    max_idx = max(1, len(legend) - 1)
    nearest_idx = str(min(max_idx, max(0, round(score_val))))
    nearest_label = legend.get(nearest_idx, f"Level {nearest_idx}")
    pct = min(100, max(0, int(round((score_val / max_idx) * 100))))
    filled_blocks = int(round(pct / 10))
    thermometer = "🟦" * filled_blocks + "⬜" * (10 - filled_blocks)
    verdict_md = (
        f"## 📊 Rating: `{score_val:.2f} / {max_idx:.0f}` — **{nearest_label}**\n"
        f"- **Intensity Meter (`{pct}%`)**: {thermometer}\n"
        f"- **Scale Levels**: {' → '.join(f'`{k}: {v}`' for k, v in legend.items())}\n"
        f"- **Confidence**: `{float(cas_ans.get('confidence', 0.0)) * 100:.1f}%` · *(CPU Latency: `{telemetry.get('latency_ms', 0.0):.2f} ms`)*"
    )
    return verdict_md, probs_raw, xai_md, telemetry, code_snippet


def run_guardrail_with_custom_rubrics(
    text_input: str,
    context_input: str = "",
    threshold: float = 0.55,
    custom_rubrics_text: str = "",
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    """Run the 4-rubric AI Safety Guardrail + optional custom enterprise policy rubrics."""
    from microllm.server import evaluate_guardrail_in_process

    custom_map: dict[str, str] = {}
    for line in custom_rubrics_text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if ":" in line_clean:
            k, v = line_clean.split(":", 1)
            custom_map[k.strip()] = v.strip()
        else:
            custom_map[line_clean] = line_clean

    backend = _get_cached_studio_backend("fast", "minilm", 0.75, 0.75)
    gear1 = getattr(backend, "gear1", backend)
    d = evaluate_guardrail_in_process(
        state=text_input.strip() or "Empty query",
        threshold=float(threshold),
        context=context_input.strip() or None,
        custom_rubrics=custom_map or None,
        gear1=gear1,
    )
    checks = {str(k): float(v) for k, v in d.get("checks", {}).items()}
    latency_str = f"{d.get('latency_ms', 0.0):.2f} ms"
    if d.get("safe", True):
        verdict_md = (
            f"## 🟢 SAFE / ALLOW (`Risk = {d['risk_score'] * 100:.1f}%` · Groundedness `{d['groundedness_score'] * 100:.1f}%`)\n"
            f"All {len(checks)} safety rubrics passed below threshold `{threshold:.2f}`."
        )
    else:
        worst_rubric = max(checks, key=lambda k: checks[k])
        verdict_md = (
            f"## 🔴 {str(d.get('action', 'block')).upper()} — FLAGGED (`Risk = {d['risk_score'] * 100:.1f}%`)\n"
            f"Highest policy risk in **`{worst_rubric}`** (`{checks[worst_rubric] * 100:.1f}%`)."
        )
    return verdict_md, latency_str, checks, d


def run_rlcd_lab_gradio(
    stage1_steps: int = 35,
    stage2_steps: int = 25,
    use_rlcd2_weights: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Execute OpenJev-RLCD Stage 1 (Pathwise Brier) + Stage 2 (LOO REINFORCE + KL) in Gradio."""
    from microllm.server import run_rlcd_experiment_in_process

    d = run_rlcd_experiment_in_process(
        stage1_steps=int(stage1_steps),
        stage2_steps=int(stage2_steps),
        use_rlcd2_weights=bool(use_rlcd2_weights),
    )
    cp = d["clopper_pearson_certification"]
    p1 = d["proposition1_variance_identity"]
    md = (
        "### 🧪 OpenJev-RLCD² Two-Stage Calibration Lab Results\n"
        f"- **Brier Score Reduction**: `{d['initial_brier']:.4f}` → **`{d['final_brier']:.4f}`**\n"
        f"- **Expected Calibration Error (ECE)**: `{d['initial_ece'] * 100:.2f}%` → **`{d['final_ece'] * 100:.2f}%`**\n"
        f"- **Decision Accuracy**: `{d['initial_accuracy'] * 100:.1f}%` → **`{d['final_accuracy'] * 100:.1f}%`**\n"
        f"- **Certified Clopper-Pearson Coverage (`δ=0.05`)**: **`{cp['coverage'] * 100:.1f}%`** at threshold `τ = {cp['threshold']:.3f}` (Upper Risk Bound `{cp['upper_risk_bound']:.4f}`)\n"
        f"- **Proposition 1 Variance Identity**: `Score(Mean) = {p1['score_of_mean']:.4f} = Mean(Scores) {p1['mean_of_scores']:.4f} + Diversity Bonus {p1['diversity_bonus']:.4f}`"
    )
    return md, d



__all__ = [
    "_get_cached_studio_backend",
    "_get_studio_test_client",
    "evaluate_studio_multi_mode",
    "evaluate_consumer_yes_no",
    "evaluate_consumer_choice",
    "evaluate_consumer_multi",
    "evaluate_consumer_score",
    "run_guardrail_with_custom_rubrics",
    "run_rlcd_lab_gradio",
]
