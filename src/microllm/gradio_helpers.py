"""UI helper utilities and criteria parsers for Gradio Decision Studio."""

from __future__ import annotations

import json
from typing import Any

from microllm.backends.base import explain_counterfactual
from microllm.gradio_presets import STUDIO_DOMAIN_PRESETS
from microllm.research_challenge import (
    evaluate_probability_coherence,
    value_of_computation_gate,
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
    render_structured_text,
)

def parse_criteria_text(question_type: str, criteria_text: str) -> Any:
    """Convert visual builder criteria text into a JevCompat `criteria` dict or list."""
    raw = criteria_text.strip()
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except Exception:
            pass

    q_norm = question_type.lower().strip()
    if q_norm.startswith("score"):
        lines = [line.strip() for line in raw.replace("\n", ",").split(",") if line.strip()]
        if len(lines) < 2:
            return ["Low", "Medium", "High"]
        return lines

    # Parse key: description lines (or comma-separated entries)
    entries: dict[str, str] = {}
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) == 1 and ":" not in lines[0] and "," in lines[0]:
        parts = [p.strip() for p in lines[0].split(",") if p.strip()]
        if q_norm.startswith("noul"):
            return {
                "true": parts[0] if len(parts) > 0 else "Affirmative, Yes, True",
                "false": parts[1] if len(parts) > 1 else "Negative, No, False",
            }
        return {p: p for p in parts}

    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            entries[k.strip()] = v.strip() or k.strip()
        else:
            entries[line.strip()] = line.strip()

    if q_norm.startswith("noul"):
        return {
            "true": entries.get("true", next(iter(entries.values()), "Affirmative, Yes, True")),
            "false": entries.get("false", "Negative, No, False"),
        }
    if len(entries) < 2:
        return {"option_a": "First candidate description", "option_b": "Second candidate description"}
    return entries


def format_criteria_for_visual(q_spec: dict[str, Any]) -> str:
    """Format a question's `criteria` into human-editable Visual Builder text."""
    q_type = str(q_spec.get("type", "choice"))
    crit = q_spec.get("criteria")
    if q_type == "score" and isinstance(crit, list):
        return ", ".join(str(x) for x in crit)
    if isinstance(crit, dict):
        return "\n".join(f"{k}: {v}" for k, v in crit.items())
    return ""


def questions_json_to_summary_rows(questions_json_str: str) -> list[list[str]]:
    """Parse `questions` JSON into tabular summary rows `[question_id, type, instructions, options_summary]`."""
    try:
        q_obj = json.loads(questions_json_str)
        if not isinstance(q_obj, dict):
            return []
    except Exception:
        return []

    rows: list[list[str]] = []
    for qid, spec in q_obj.items():
        if not isinstance(spec, dict):
            continue
        q_type = str(spec.get("type", "choice"))
        inst = str(spec.get("instructions", ""))
        crit = spec.get("criteria")
        if isinstance(crit, dict):
            summary = ", ".join(f"{k}" for k in crit.keys())
        elif isinstance(crit, list):
            summary = " -> ".join(str(x) for x in crit)
        else:
            summary = "-"
        rows.append([str(qid), q_type, inst, summary])
    return rows


def upsert_visual_question_into_json(
    current_json_str: str,
    question_id: str,
    question_type: str,
    instructions: str,
    criteria_text: str,
    multi_threshold: float = 0.5,
) -> tuple[str, str]:
    """Add or update a single question from the Visual Question Builder into the JSON specification."""
    try:
        q_obj = json.loads(current_json_str) if current_json_str.strip() else {}
        if not isinstance(q_obj, dict):
            q_obj = {}
    except Exception:
        q_obj = {}

    clean_id = (question_id or "question_1").strip().replace(" ", "_")
    q_norm = question_type.split()[0].lower().strip()
    if q_norm not in {"noul", "choice", "score", "multi"}:
        q_norm = "choice"

    parsed_crit = parse_criteria_text(q_norm, criteria_text)
    spec: dict[str, Any] = {
        "type": q_norm,
        "instructions": instructions.strip() or f"Evaluate {clean_id}.",
        "criteria": parsed_crit,
    }
    if q_norm == "multi":
        spec["threshold"] = round(float(multi_threshold), 2)

    q_obj[clean_id] = spec
    updated_json = json.dumps(q_obj, indent=2)
    return updated_json, f"✅ Added/updated question `{clean_id}` ({q_norm}) — {len(q_obj)} active question(s)."


def remove_question_from_json(current_json_str: str, question_id: str) -> tuple[str, str]:
    """Remove a question by ID from the JSON specification."""
    try:
        q_obj = json.loads(current_json_str) if current_json_str.strip() else {}
        if not isinstance(q_obj, dict):
            q_obj = {}
    except Exception:
        return current_json_str, "⚠️ Invalid JSON; could not remove question."

    clean_id = question_id.strip()
    if clean_id in q_obj:
        del q_obj[clean_id]
        return json.dumps(q_obj, indent=2), f"🗑️ Removed question `{clean_id}` — {len(q_obj)} remaining."
    return json.dumps(q_obj, indent=2), f"ℹ️ Question `{clean_id}` not found in active set."


def compute_deep_studio_diagnostics(
    state_text: str,
    q_spec: NoulQuestion | ChoiceQuestion | ScoreQuestion | MultiQuestion,
    ac: NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute Counterfactual XAI, Kolmogorov Probability Coherence, and VOC Uncertainty Regime for a question."""
    instr_text = render_structured_text(q_spec.instructions, fallback="")

    if isinstance(q_spec, ChoiceQuestion):
        opt_map: dict[str, str | None] = {
            k: (render_structured_text(v) if v is not None else None)
            for k, v in q_spec.criteria.items()
        }
    elif isinstance(q_spec, MultiQuestion):
        opt_map = {
            k: (render_structured_text(v) if v is not None else None)
            for k, v in q_spec.criteria.items()
        }
    elif isinstance(q_spec, ScoreQuestion):
        opt_map = {
            str(i): render_structured_text(lvl) for i, lvl in enumerate(q_spec.criteria)
        }
    else:
        crit_raw = q_spec.criteria or {"true": "Affirmative, Yes, True", "false": "Negative, No, False"}
        opt_map = {
            "true": render_structured_text(crit_raw.get("true"), fallback="Affirmative, Yes, True"),
            "false": render_structured_text(crit_raw.get("false"), fallback="Negative, No, False"),
        }

    # 1. Counterfactual Token Sensitivity (IDEA-019)
    cf_raw = explain_counterfactual(state_text, instr_text, opt_map, top_k=5)
    counterfactual_info = {
        **cf_raw,
        "baseline_winner": cf_raw["original_choice"],
        "flip_candidates": cf_raw["decision_flip_tokens"],
    }

    # 2. Kolmogorov Probability Coherence Audit (IDEA-023)
    coh_raw = evaluate_probability_coherence(state_text, instr_text, opt_map)
    raw_noul_sum = round(float(sum(coh_raw["raw_noul_probabilities"].values())), 6)
    coherence_info = {
        **coh_raw,
        "raw_noul_sum": raw_noul_sum,
        "raw_coherence_deviation": coh_raw["raw_noul_sum_deviation"],
    }

    # 3. StartLux Proposition 5 Value-of-Computation (VOC) Gate (HYP-005)
    voc_raw = value_of_computation_gate(state_text, instr_text, opt_map)
    voc_info = {
        **voc_raw,
        "expected_voc": voc_raw["expected_voc_gain"],
        "max_pre_softmax_logit": voc_raw["max_semantic_logit"],
        "should_escalate": voc_raw["should_escalate_gear2"],
    }

    # Ensure winner consistency with actual answer for Noul
    if isinstance(ac, NoulAnswer):
        counterfactual_info["baseline_winner"] = "true" if ac.noul >= 0.5 else "false"

    return counterfactual_info, coherence_info, voc_info


def load_preset_for_gradio(
    preset_key: str,
) -> tuple[str, str, list[list[str]], str, str, str, str, float]:
    """Load one of the 10 enterprise domain presets into both Visual Builder and Raw JSON states."""
    preset = STUDIO_DOMAIN_PRESETS.get(preset_key, STUDIO_DOMAIN_PRESETS["support"])
    raw_state = preset["state"]
    state_str = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, indent=2)
    q_dict = preset["questions"]
    q_json_str = json.dumps(q_dict, indent=2)
    rows = questions_json_to_summary_rows(q_json_str)

    first_qid = next(iter(q_dict.keys()), "question_1")
    first_spec = q_dict.get(first_qid, {"type": "choice", "instructions": "", "criteria": {}})
    first_type = str(first_spec.get("type", "choice"))
    first_inst = str(first_spec.get("instructions", ""))
    first_crit = format_criteria_for_visual(first_spec)
    first_thresh = float(first_spec.get("threshold", 0.5))
    return (
        state_str,
        q_json_str,
        rows,
        first_qid,
        first_type,
        first_inst,
        first_crit,
        first_thresh,
    )


def generate_code_export(state_text: str, questions_obj: dict[str, Any], lang: str) -> str:
    """Generate a ready-to-run Python SDK, TypeScript, or cURL snippet."""
    q_formatted = json.dumps(questions_obj, indent=2)
    if lang.lower().startswith("python"):
        return (
            "from microllm import MicroJevClient\n\n"
            'client = MicroJevClient(base_url="http://127.0.0.1:8000")\n'
            "response = client.ask(\n"
            f"    state={json.dumps(state_text)},\n"
            f"    questions={q_formatted}\n"
            ")\n"
            "for qid, ans in response.items():\n"
            '    print(qid, "->", ans.value)'
        )
    if lang.lower().startswith("typescript"):
        return (
            'const resp = await fetch("http://127.0.0.1:8000/v1/systemone", {\n'
            '  method: "POST",\n'
            '  headers: { "Content-Type": "application/json" },\n'
            "  body: JSON.stringify({\n"
            '    model: "jev-latest",\n'
            f"    state: {json.dumps(state_text)},\n"
            f"    questions: {q_formatted}\n"
            "  })\n"
            "});\n"
            "const data = await resp.json();\n"
            "console.log(data.answers);"
        )
    body_str = json.dumps({"model": "jev-latest", "state": state_text, "questions": questions_obj})
    return (
        "curl -X POST http://127.0.0.1:8000/v1/systemone \\\n"
        '  -H "Content-Type: application/json" \\\n'
        f"  -d '{body_str}'"
    )


def load_preset_for_all_tabs(preset_key: str) -> tuple[Any, ...]:
    """Load an enterprise domain preset across all 6 Decision Studio tabs atomically.

    Returns 26 values in exact order:
      - Tab 1: yn_text, yn_question, yn_yes_hint, yn_no_hint
      - Tab 2: ch_text, ch_question, ch_options
      - Tab 3: mt_text, mt_question, mt_tags, mt_thresh
      - Tab 4: sc_text, sc_question, sc_levels
      - Tab 5: guard_input, guard_custom
      - Tab 6: state_input, questions_json, active_questions_table, visual_qid, visual_type,
               visual_instructions, visual_criteria, visual_threshold, visual_status_md, code_export_box
    """
    preset = STUDIO_DOMAIN_PRESETS.get(preset_key, STUDIO_DOMAIN_PRESETS["support"])
    raw_state = preset["state"]
    state_str = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, indent=2)
    q_dict = preset["questions"]
    domain = preset.get("domain", "scenario")
    title = preset.get("title", preset_key)

    # Search for question specs by primitive type
    noul_q = next((spec for spec in q_dict.values() if spec.get("type") == "noul"), None)
    choice_q = next((spec for spec in q_dict.values() if spec.get("type") == "choice"), None)
    multi_q = next((spec for spec in q_dict.values() if spec.get("type") == "multi"), None)
    score_q = next((spec for spec in q_dict.values() if spec.get("type") == "score"), None)

    # Tab 1: Yes / No (noul)
    yn_text = state_str
    if noul_q:
        yn_q = str(noul_q.get("instructions", ""))
        crit = noul_q.get("criteria", {})
        yn_yes = crit.get("true", "") if isinstance(crit, dict) else ""
        yn_no = crit.get("false", "") if isinstance(crit, dict) else ""
    else:
        yn_q = f"Does this {domain} situation require immediate operational escalation?"
        yn_yes = "Requires immediate escalation, rollback, or manual intervention"
        yn_no = "Nominal or routine processing under standard monitoring"

    # Tab 2: Single Choice (choice)
    ch_text = state_str
    if choice_q:
        ch_q = str(choice_q.get("instructions", ""))
        ch_opts = format_criteria_for_visual(choice_q)
    else:
        ch_q = f"Select the primary {domain} operational pathway:"
        ch_opts = (
            "escalate: Escalate immediately to Tier-2 specialist\n"
            "standard_sop: Execute standard operating procedure\n"
            "log_and_monitor: Log diagnostic telemetry and monitor"
        )

    # Tab 3: Multi-Tagging (multi)
    mt_text = state_str
    if multi_q:
        mt_q = str(multi_q.get("instructions", ""))
        mt_tags = format_criteria_for_visual(multi_q)
        mt_thresh = float(multi_q.get("threshold", 0.50))
    else:
        mt_q = f"Select all applicable {domain} risk indicators:"
        mt_tags = (
            f"{preset_key}_active: Active scenario indicator\n"
            "sla_breach_risk: Risk of violating client SLA\n"
            "compliance_audit: Requires regulatory compliance logging"
        )
        mt_thresh = 0.50

    # Tab 4: Rating & Scale (score)
    sc_text = state_str
    if score_q:
        sc_q = str(score_q.get("instructions", ""))
        sc_levels = format_criteria_for_visual(score_q)
    else:
        sc_q = f"Rate the operational urgency or severity of this {domain}:"
        sc_levels = "SEV-4 Minor, SEV-3 Moderate, SEV-2 Major Degradation, SEV-1 Critical Outage"

    # Tab 5: Safety Guardrail (guardrail)
    guard_input = state_str
    if preset_key == "guardrail":
        guard_custom = (
            "destructive_command: Permanently delete or wipe production database\n"
            "prompt_injection: Ignore previous instructions or override policy"
        )
    elif preset_key == "cyber_soc":
        guard_custom = (
            "credential_dump: LSASS memory dump, Mimikatz, password theft\n"
            "lateral_movement: SMB or WMI movement across domain"
        )
    elif preset_key == "fintech_aml":
        guard_custom = (
            "structuring_evasion: Cash deposits below reporting threshold\n"
            "offshore_wire: Unverified wire to foreign shell entity"
        )
    else:
        guard_custom = (
            "compliance_violation: Action violates corporate data policy\n"
            "unauthorized_escalation: Exceeds designated role permissions"
        )

    # Tab 6: Advanced Studio
    q_json_str = json.dumps(q_dict, indent=2)
    rows = questions_json_to_summary_rows(q_json_str)
    first_qid = next(iter(q_dict.keys()), "question_1")
    first_spec = q_dict.get(first_qid, {"type": "choice", "instructions": "", "criteria": {}})
    first_type = str(first_spec.get("type", "choice"))
    first_inst = str(first_spec.get("instructions", ""))
    first_crit = format_criteria_for_visual(first_spec)
    first_thresh = float(first_spec.get("threshold", 0.5))
    visual_status = f"ℹ️ *Visual Builder loaded '{title}' ({len(q_dict)} active questions).*"
    code_export = generate_code_export(state_str, q_dict, "Python SDK")

    return (
        # Tab 1
        yn_text,
        yn_q,
        yn_yes,
        yn_no,
        # Tab 2
        ch_text,
        ch_q,
        ch_opts,
        # Tab 3
        mt_text,
        mt_q,
        mt_tags,
        mt_thresh,
        # Tab 4
        sc_text,
        sc_q,
        sc_levels,
        # Tab 5
        guard_input,
        guard_custom,
        # Tab 6
        state_str,
        q_json_str,
        rows,
        first_qid,
        first_type,
        first_inst,
        first_crit,
        first_thresh,
        visual_status,
        code_export,
    )



__all__ = [
    "parse_criteria_text",
    "format_criteria_for_visual",
    "questions_json_to_summary_rows",
    "upsert_visual_question_into_json",
    "remove_question_from_json",
    "compute_deep_studio_diagnostics",
    "load_preset_for_gradio",
    "generate_code_export",
    "load_preset_for_all_tabs",
]
