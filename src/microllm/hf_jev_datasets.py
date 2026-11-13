"""Hugging Face JEV Models Ecosystem Catalog & Dataset Adapters (`huggingface.co/models?search=jev`).

Catalogs 25 real System-One / JEV models published on the Hugging Face Hub (`https://huggingface.co/models?search=jev`)
and implements 12 canonical `BaseDatasetAdapter` classes for the training, distillation, safety, and evaluation
datasets referenced across those models:
1. `SargeDev/jev-distill-corpus-v3` (`JevDistillCorpusV3Adapter` — `autotrust/JEV-27B`, `JEV-9B`, `Bonsai-JEV`)
2. `ZefanCai/Open-Jev` (`OpenJevCorpusAdapter` — `Open-Jev-9B`, `TokenRhythm/NeoHorse-Jev-4B`)
3. `tasksource/tasksource-jev-typed-decisions` (`TasksourceJevTypedDecisionsAdapter` — `scairesearch/kieve-jev`)
4. `JevAgentHarness/AgentDecisionBench` (`AgentDecisionBenchAdapter` — `JevAgentHarness/AgentDecision-4B`)
5. `LocalLLaMA/typed-decisions` (`LocalLLaMATypedDecisionsAdapter` — `J33-AI/j33-jev`)
6. `3nesdeniz/agentic-prompt-injection-5k` (`AgenticPromptInjection5kAdapter` — `patronus-studio/wolf-defender-jev`)
7. `MaziyarPanahi/Jev-RAG-Claim-Verification-130K` (`JevRAGClaimVerificationAdapter` — JEV 1.13 RAG verification)
8. `MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K` (`JevSafetyRefusalDecisionsAdapter` — JEV 1.13 Aegis safety)
9. `SetFit/sst5` (`SST5OrdinalAdapter` — `com-kotobalabs/open-jev-deberta-v3-large`, `Jev-Style-Qwen3.5-2B`)
10. `google-research-datasets/paws` (`PAWSParaphraseAdapter` — `shibing624/train-jev-4b-lora`)
11. `akhilaaa3/decision-bench` (`DecisionBenchAdapter` — `akhilaaa3/Jev-Omni`)
12. `Hangtao/JevAdvBench` (`JevAdvBenchAdapter` — `arXiv:2609.31142` adversarial & option-order robustness)
"""

from __future__ import annotations

import functools
import importlib.resources
import json
from typing import Any

from microllm.dataset_adapters import BaseDatasetAdapter


ALL_12_HF_JEV_MODEL_DATASETS: list[str] = [
    "SargeDev/jev-distill-corpus-v3",
    "ZefanCai/Open-Jev",
    "tasksource/tasksource-jev-typed-decisions",
    "JevAgentHarness/AgentDecisionBench",
    "LocalLLaMA/typed-decisions",
    "3nesdeniz/agentic-prompt-injection-5k",
    "MaziyarPanahi/Jev-RAG-Claim-Verification-130K",
    "MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K",
    "SetFit/sst5",
    "google-research-datasets/paws",
    "akhilaaa3/decision-bench",
    "Hangtao/JevAdvBench",
]

@functools.lru_cache(maxsize=1)
def _load_hf_jev_models_catalog() -> list[dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("hf_jev_models_catalog.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


HF_JEV_ECOSYSTEM_MODELS_CATALOG: list[dict[str, Any]] = _load_hf_jev_models_catalog()
HF_JEV_MODELS_CATALOG = HF_JEV_ECOSYSTEM_MODELS_CATALOG


def list_hf_jev_ecosystem_models(
    query: str | None = None,
    primitive: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Filter and return models from `HF_JEV_ECOSYSTEM_MODELS_CATALOG`."""
    results = list(HF_JEV_ECOSYSTEM_MODELS_CATALOG)
    if query:
        q = query.strip().lower()
        results = [
            m
            for m in results
            if q in m["model_id"].lower()
            or q in m["architecture"].lower()
            or q in m["focus"].lower()
            or any(q in d.lower() for d in m["datasets"])
        ]
    if primitive:
        p = primitive.strip().lower()
        results = [m for m in results if p in [x.lower() for x in m["primitives"]]]
    if dataset:
        ds_q = dataset.strip().lower()
        results = [m for m in results if any(ds_q in d.lower() for d in m["datasets"])]
    return results


@functools.lru_cache(maxsize=1)
def _load_hf_jev_server_config_map() -> dict[str, dict[str, str]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("hf_jev_server_config_map.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


HF_JEV_DATASET_SERVER_CONFIG_MAP: dict[str, dict[str, str]] = _load_hf_jev_server_config_map()
HF_JEV_SERVER_CONFIG_MAP = HF_JEV_DATASET_SERVER_CONFIG_MAP


def _parse_json_or_val(val: Any) -> Any:
    """Safely parse a JSON string or return the value as-is."""
    if isinstance(val, str):
        stripped = val.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ) or (stripped.startswith('"') and stripped.endswith('"')):
            try:
                return json.loads(stripped)
            except Exception:
                return val
    return val


def _parse_options_and_target(
    raw_options: Any, raw_target: Any, kind: str
) -> tuple[dict[str, str] | list[str], Any]:
    """Normalize `options` and `target` from `jev-distill-corpus-v3`, `Open-Jev`, or `tasksource`."""
    if kind == "noul":
        rules: dict[str, str] | list[str] = {
            "true": "The statement or condition is confirmed, satisfied, true, and supported by the state.",
            "false": "The statement or condition is contradicted, missing, prohibited, denied, or false.",
        }
        if isinstance(raw_target, list) and len(raw_target) >= 2:
            expected: Any = bool(float(raw_target[1]) >= float(raw_target[0]))
        elif isinstance(raw_target, bool):
            expected = raw_target
        else:
            expected = str(raw_target).strip().lower() in ("true", "1", "yes")
        return rules, expected

    if kind == "score":
        if isinstance(raw_options, list) and raw_options and len(str(raw_options[0])) > 3:
            levels = [str(x) for x in raw_options]
        else:
            levels = [
                "0 - Minimal / None: calm, routine, negligible impact, very negative or lowest rating",
                "1 - Low: minor issue, slight inconvenience, negative or low rating",
                "2 - Moderate: standard priority, mixed or neutral impact",
                "3 - High: significant operational impact, positive or high priority",
                "4 - Critical / Maximum: urgent outage, severe security risk, or outstanding top rating",
            ]
        if isinstance(raw_target, list) and raw_target:
            idx = int(max(range(len(raw_target)), key=lambda i: float(raw_target[i])))
            expected = min(idx, len(levels) - 1)
        else:
            expected = max(0, min(len(levels) - 1, int(raw_target or 0)))
        return levels, expected

    # choice
    options_dict: dict[str, str] = {}
    keys_order: list[str] = []
    if isinstance(raw_options, dict):
        for k, v in raw_options.items():
            options_dict[str(k)] = str(v)
            keys_order.append(str(k))
    elif isinstance(raw_options, list):
        for i, item in enumerate(raw_options):
            s = str(item).strip()
            if ":" in s and len(s.split(":", 1)[0].strip()) <= 40 and " " not in s.split(":", 1)[0].strip():
                k_part, v_part = s.split(":", 1)
                key = k_part.strip()
                options_dict[key] = v_part.strip() or key
            else:
                key = s if len(s) <= 32 and " " not in s else f"opt_{i}"
                options_dict[key] = s
            keys_order.append(key)
    if not options_dict:
        options_dict = {"option_a": "First candidate option", "option_b": "Second candidate option"}
        keys_order = ["option_a", "option_b"]

    if isinstance(raw_target, list) and raw_target and keys_order:
        best_i = int(max(range(min(len(raw_target), len(keys_order))), key=lambda i: float(raw_target[i])))
        expected = keys_order[best_i]
    elif isinstance(raw_target, int) and 0 <= raw_target < len(keys_order):
        expected = keys_order[raw_target]
    else:
        expected = str(raw_target) if raw_target in options_dict else keys_order[0]
    return options_dict, expected


# =============================================================================
# 17. SargeDev/jev-distill-corpus-v3 Adapter (`noul`, `choice`, `score`)
# =============================================================================
class JevDistillCorpusV3Adapter(BaseDatasetAdapter):
    """Adapter for `SargeDev/jev-distill-corpus-v3` (used by `autotrust/JEV-27B`, `JEV-9B`, `Bonsai-JEV`)."""

    @property
    def dataset_name(self) -> str:
        return "SargeDev/jev-distill-corpus-v3"

    @property
    def target_decision_type(self) -> str:
        return "jevcompat"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        kind = str(raw_record.get("kind", raw_record.get("decision_type", "noul"))).strip().lower()
        if kind not in ("noul", "choice", "score"):
            kind = "noul"
        state = str(raw_record.get("state", "")).strip()
        question = str(raw_record.get("question", raw_record.get("instructions", "Evaluate the state."))).strip()
        if "rules" in raw_record and "expected" in raw_record:
            rules = raw_record["rules"]
            expected = raw_record["expected"]
        else:
            rules, expected = _parse_options_and_target(
                raw_record.get("options"), raw_record.get("target"), kind
            )
        return {
            "id": f"JEVDIST-{record_idx:04d}",
            "domain": str(raw_record.get("domain", "jev_distill_v3")),
            "message": state,
            "decision_type": kind,
            "instructions": question,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "SargeDev/jev-distill-corpus-v3",
                "upstream_id": raw_record.get("id", f"v3_{record_idx}"),
                "stream": raw_record.get("source", "yuri_v3"),
            },
        }


# =============================================================================
# 18. ZefanCai/Open-Jev Adapter (`noul`, `choice`, `score`)
# =============================================================================
class OpenJevCorpusAdapter(BaseDatasetAdapter):
    """Adapter for `ZefanCai/Open-Jev` (used by `Open-Jev-9B`, `autotrust/JEV-27B`, `NeoHorse-Jev-4B`)."""

    @property
    def dataset_name(self) -> str:
        return "ZefanCai/Open-Jev"

    @property
    def target_decision_type(self) -> str:
        return "jevcompat"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        kind = str(raw_record.get("kind", raw_record.get("decision_type", "choice"))).strip().lower()
        if kind not in ("noul", "choice", "score"):
            kind = "choice"
        raw_state = raw_record.get("state", raw_record.get("state_json", ""))
        parsed_state = _parse_json_or_val(raw_state)
        state = parsed_state if isinstance(parsed_state, str) else json.dumps(parsed_state, ensure_ascii=False)
        question = str(raw_record.get("question", raw_record.get("instructions", "Evaluate the state."))).strip()
        if "rules" in raw_record and "expected" in raw_record:
            rules = raw_record["rules"]
            expected = raw_record["expected"]
        else:
            rules, expected = _parse_options_and_target(
                raw_record.get("options"), raw_record.get("target"), kind
            )
        return {
            "id": f"OPENJEV-{record_idx:04d}",
            "domain": str(raw_record.get("source", "open_jev_control")),
            "message": str(state).strip(),
            "decision_type": kind,
            "instructions": question,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "ZefanCai/Open-Jev",
                "upstream_id": raw_record.get("id", f"openjev_{record_idx}"),
                "split": raw_record.get("split", "test"),
            },
        }


# =============================================================================
# 19. tasksource/tasksource-jev-typed-decisions Adapter (`noul`, `choice`, `score`)
# =============================================================================
class TasksourceJevTypedDecisionsAdapter(BaseDatasetAdapter):
    """Adapter for `tasksource/tasksource-jev-typed-decisions` (multi-task typed decisions)."""

    @property
    def dataset_name(self) -> str:
        return "tasksource/tasksource-jev-typed-decisions"

    @property
    def target_decision_type(self) -> str:
        return "jevcompat"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        kind = str(raw_record.get("kind", raw_record.get("decision_type", "noul"))).strip().lower()
        if kind not in ("noul", "choice", "score"):
            kind = "noul"
        state = str(raw_record.get("state", "")).strip()
        question = str(raw_record.get("question", raw_record.get("instructions", "Answer the question."))).strip()
        if "rules" in raw_record and "expected" in raw_record:
            rules = raw_record["rules"]
            expected = raw_record["expected"]
        else:
            rules, expected = _parse_options_and_target(
                raw_record.get("options"), raw_record.get("target"), kind
            )
        return {
            "id": f"TASKSOURCE-{record_idx:04d}",
            "domain": str(raw_record.get("source", "tasksource_jev")),
            "message": state,
            "decision_type": kind,
            "instructions": question,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "tasksource/tasksource-jev-typed-decisions",
                "upstream_id": raw_record.get("id", f"ts_{record_idx}"),
            },
        }


# =============================================================================
# 20. JevAgentHarness/AgentDecisionBench Adapter (`choice`)
# =============================================================================
class AgentDecisionBenchAdapter(BaseDatasetAdapter):
    """Adapter for `JevAgentHarness/AgentDecisionBench` (`AgentDecision-4B`, `arXiv:2610.21412`)."""

    @property
    def dataset_name(self) -> str:
        return "JevAgentHarness/AgentDecisionBench"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        state = str(raw_record.get("state", "")).strip()
        raw_opts = raw_record.get("options", [])
        raw_descs = raw_record.get("descriptions", [])
        rules: dict[str, str] = {}
        if isinstance(raw_opts, dict):
            rules = {str(k): str(v) for k, v in raw_opts.items()}
        elif isinstance(raw_opts, list):
            for i, opt in enumerate(raw_opts):
                desc = str(raw_descs[i]) if isinstance(raw_descs, list) and i < len(raw_descs) else str(opt)
                rules[str(opt)] = desc
        if not rules:
            rules = {
                "ask_user": "Ask the customer for missing required information before acting.",
                "find_user": "Look up the customer account profile by name or email.",
            }
        gold = raw_record.get("gold", raw_record.get("expected"))
        if not gold or str(gold) not in rules:
            gold = next(iter(rules.keys()))
        return {
            "id": f"AGENTDEC-{record_idx:04d}",
            "domain": "agent_tool_selection_harness",
            "message": state,
            "decision_type": "choice",
            "instructions": str(
                raw_record.get(
                    "instructions",
                    "Select the optimal next agent tool call or ask_user action for this state.",
                )
            ),
            "rules": rules,
            "expected_result": str(gold),
            "metadata": {
                "source": "JevAgentHarness/AgentDecisionBench",
                "qhash": raw_record.get("qhash", f"qhash_{record_idx}"),
            },
        }


# =============================================================================
# 21. LocalLLaMA/typed-decisions Adapter (`choice`, `noul`, `score`)
# =============================================================================
class LocalLLaMATypedDecisionsAdapter(BaseDatasetAdapter):
    """Adapter for `LocalLLaMA/typed-decisions` (used by `J33-AI/j33-jev`)."""

    @property
    def dataset_name(self) -> str:
        return "LocalLLaMA/typed-decisions"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        raw_state = _parse_json_or_val(raw_record.get("state", ""))
        state_str = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, ensure_ascii=False)
        if "rules" in raw_record and "expected" in raw_record:
            dtype = str(raw_record.get("decision_type", "choice"))
            return {
                "id": f"LLTYPED-{record_idx:04d}",
                "domain": str(raw_record.get("workflow", "agent_trace_observability")),
                "message": state_str,
                "decision_type": dtype,
                "instructions": str(raw_record.get("instructions", "Evaluate the operational workflow trace.")),
                "rules": raw_record["rules"],
                "expected_result": raw_record["expected"],
                "metadata": {"source": "LocalLLaMA/typed-decisions", "workflow": raw_record.get("workflow", "all")},
            }

        questions = _parse_json_or_val(raw_record.get("questions", {}))
        gold = _parse_json_or_val(raw_record.get("gold", {}))
        if isinstance(questions, dict) and questions:
            q_key = "action" if "action" in questions else next(iter(questions.keys()))
            q_spec = questions[q_key] if isinstance(questions[q_key], dict) else {}
            criteria = q_spec.get("criteria", {"continue": "Continue execution", "stop": "Halt execution"})
            instructions = str(q_spec.get("instructions", f"Evaluate {q_key} for this workflow state."))
            gold_entry = gold.get(q_key, {}) if isinstance(gold, dict) else {}
            expected = (
                gold_entry.get("label")
                if isinstance(gold_entry, dict)
                else raw_record.get(f"{q_key}__label", next(iter(criteria.keys())))
            )
        else:
            criteria = {"continue": "Let the agent proceed without interruption.", "stop": "Halt the agent."}
            instructions = "What action should the supervisor take?"
            expected = raw_record.get("action__label", "continue")

        return {
            "id": f"LLTYPED-{record_idx:04d}",
            "domain": str(raw_record.get("workflow", "agent_trace_observability")),
            "message": state_str,
            "decision_type": "choice",
            "instructions": instructions,
            "rules": criteria,
            "expected_result": str(expected),
            "metadata": {
                "source": "LocalLLaMA/typed-decisions",
                "workflow": raw_record.get("workflow", "agent_trace_observability"),
            },
        }


# =============================================================================
# 22. 3nesdeniz/agentic-prompt-injection-5k Adapter (`guardrail`)
# =============================================================================
class AgenticPromptInjection5kAdapter(BaseDatasetAdapter):
    """Adapter for `3nesdeniz/agentic-prompt-injection-5k` (`patronus-studio/wolf-defender-jev`)."""

    @property
    def dataset_name(self) -> str:
        return "3nesdeniz/agentic-prompt-injection-5k"

    @property
    def target_decision_type(self) -> str:
        return "guardrail"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", raw_record.get("prompt", ""))).strip()
        raw_lbl = raw_record.get("label", raw_record.get("is_injection", 0))
        cls_str = str(raw_record.get("class", "")).strip().lower()
        if isinstance(raw_lbl, bool):
            is_attack = raw_lbl
        elif isinstance(raw_lbl, int):
            is_attack = raw_lbl == 1
        else:
            is_attack = str(raw_lbl).strip().lower() in ("1", "true", "injection", "harmful")
        if cls_str == "injection":
            is_attack = True
        elif cls_str == "benign":
            is_attack = False

        return {
            "id": f"AGPI5K-{record_idx:04d}",
            "domain": "agentic_prompt_injection_guardrail",
            "message": text,
            "decision_type": "guardrail",
            "instructions": "Evaluate whether this agentic input contains a direct or indirect prompt injection.",
            "rules": {
                "prompt_injection": "Direct or indirect instruction override, memory poisoning, or tool hijacking",
                "destructive_operation": "Unauthorized system or database modification command",
            },
            "expected_result": is_attack,
            "metadata": {
                "source": "3nesdeniz/agentic-prompt-injection-5k",
                "attack_family": raw_record.get("attack_family", "benign" if not is_attack else "injection"),
                "severity": raw_record.get("severity", "none" if not is_attack else "high"),
            },
        }


# =============================================================================
# 23. MaziyarPanahi/Jev-RAG-Claim-Verification-130K Adapter (`noul`)
# =============================================================================
class JevRAGClaimVerificationAdapter(BaseDatasetAdapter):
    """Adapter for `MaziyarPanahi/Jev-RAG-Claim-Verification-130K` -> MicroLLM `noul` groundedness check."""

    @property
    def dataset_name(self) -> str:
        return "MaziyarPanahi/Jev-RAG-Claim-Verification-130K"

    @property
    def target_decision_type(self) -> str:
        return "noul"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        state_obj = _parse_json_or_val(raw_record.get("state_json", raw_record.get("state", "")))
        if isinstance(state_obj, dict):
            ctx = state_obj.get("authoritative_context", state_obj.get("context", ""))
            resp = state_obj.get("assistant_response", state_obj.get("claim", ""))
            message = f"Authoritative Context: {ctx}\nAssistant Response: {resp}".strip()
        else:
            message = str(state_obj).strip()

        question = str(
            raw_record.get(
                "question_text",
                "Is every factual claim in `assistant_response` supported by `authoritative_context`?",
            )
        ).strip()
        gold_lbl = raw_record.get("gold_label", raw_record.get("expected", True))
        if isinstance(gold_lbl, bool):
            expected = gold_lbl
        else:
            expected = str(gold_lbl).strip().lower() in ("true", "1", "yes", "good", "supported", "grounded")

        rules = {
            "true": "Supported and grounded: every factual claim in the response is confirmed by the authoritative context with no hallucination.",
            "false": "Hallucinated, unsupported, contradicted, fabricated, missing, prohibited, or ungrounded claim.",
        }
        return {
            "id": f"JEVRAG-{record_idx:04d}",
            "domain": "jev_rag_claim_verification",
            "message": message,
            "decision_type": "noul",
            "instructions": question,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "MaziyarPanahi/Jev-RAG-Claim-Verification-130K",
                "row_id": raw_record.get("row_id", f"rag_{record_idx}"),
                "task_family": raw_record.get("task_family", "response_grounding"),
            },
        }


# =============================================================================
# 24. MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K Adapter (`guardrail`)
# =============================================================================
class JevSafetyRefusalDecisionsAdapter(BaseDatasetAdapter):
    """Adapter for `MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K` -> MicroLLM `guardrail` safety check."""

    @property
    def dataset_name(self) -> str:
        return "MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K"

    @property
    def target_decision_type(self) -> str:
        return "guardrail"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        state_obj = _parse_json_or_val(raw_record.get("state_json", raw_record.get("prompt", "")))
        if isinstance(state_obj, dict):
            text = str(
                state_obj.get("user_request", state_obj.get("prompt", json.dumps(state_obj, ensure_ascii=False)))
            )
        else:
            text = str(state_obj)

        gold_lbl = raw_record.get("gold_label", raw_record.get("is_unsafe", False))
        if isinstance(gold_lbl, bool):
            is_unsafe = gold_lbl
        else:
            is_unsafe = str(gold_lbl).strip().lower() in ("unsafe", "true", "1", "harmful", "refuse")

        return {
            "id": f"JEVSAFE-{record_idx:04d}",
            "domain": "jev_safety_refusal_decisions",
            "message": text.strip(),
            "decision_type": "guardrail",
            "instructions": str(raw_record.get("question_text", "Classify the safety of the user request.")),
            "rules": {
                "prompt_injection": "Jailbreak, safety override, or adversarial refusal bypass",
                "destructive_operation": "Harmful, dangerous, illegal, or destructive action under Aegis taxonomy",
                "pii_or_secret_leak": "Credential, secret key, or private data exfiltration",
            },
            "expected_result": is_unsafe,
            "metadata": {
                "source": "MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K",
                "row_id": raw_record.get("row_id", f"safe_{record_idx}"),
                "task_family": raw_record.get("task_family", "prompt_safety"),
            },
        }


# =============================================================================
# 25. SetFit/sst5 Adapter (`score`)
# =============================================================================
SST5_ORDINAL_LEVELS: list[str] = [
    "0 - Very negative (1 star): awful, terrible, rancid, unwatchable, worst, garbage, boring, disaster, dreadful, waste",
    "1 - Negative (2 stars): disappointing, dull, no movement, mediocre, flat, weak, poorly written, uninspired, slow",
    "2 - Neutral (3 stars): average, mixed feelings, okay, neither good nor bad, standard, passable, fair",
    "3 - Positive (4 stars): good, enjoyable, entertaining, charming, solid performance, pleasant, well-crafted, funny",
    "4 - Very positive (5 stars): masterpiece, brilliant, stirring, transporting, phenomenal, best film, extraordinary, flawless",
]


class SST5OrdinalAdapter(BaseDatasetAdapter):
    """Adapter for `SetFit/sst5` (`open-jev-deberta-v3-large`, `Jev-Style-Qwen3.5-2B`) -> 5-level `score`."""

    @property
    def dataset_name(self) -> str:
        return "SetFit/sst5"

    @property
    def target_decision_type(self) -> str:
        return "score"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        text = str(raw_record.get("text", "")).strip()
        label = max(0, min(4, int(raw_record.get("label", 2))))
        return {
            "id": f"SST5-{record_idx:04d}",
            "domain": "sentiment_ordinal_sst5",
            "message": text,
            "decision_type": "score",
            "instructions": "Rate the fine-grained sentiment of this review sentence on the 5-level SST-5 scale (0=very negative to 4=very positive).",
            "rules": list(SST5_ORDINAL_LEVELS),
            "expected_result": label,
            "metadata": {
                "source": "SetFit/sst5",
                "label_text": raw_record.get("label_text", ""),
            },
        }


# =============================================================================
# 26. google-research-datasets/paws Adapter (`noul`)
# =============================================================================
class PAWSParaphraseAdapter(BaseDatasetAdapter):
    """Adapter for `google-research-datasets/paws` (`shibing624/train-jev-4b-lora`) -> `noul` equivalence."""

    @property
    def dataset_name(self) -> str:
        return "google-research-datasets/paws"

    @property
    def target_decision_type(self) -> str:
        return "noul"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        s1 = str(raw_record.get("sentence1", "")).strip()
        s2 = str(raw_record.get("sentence2", "")).strip()
        raw_lbl = raw_record.get("label", 0)
        is_paraphrase = bool(raw_lbl == 1 if isinstance(raw_lbl, int) else raw_lbl)
        message = raw_record.get("message") or f"Sentence 1: {s1}\nSentence 2: {s2}"
        return {
            "id": f"PAWS-{record_idx:04d}",
            "domain": "paraphrase_identification_paws",
            "message": str(message).strip(),
            "decision_type": "noul",
            "instructions": "Do Sentence 1 and Sentence 2 have the exact same semantic meaning (paraphrase)?",
            "rules": {
                "true": f"Paraphrase: both sentences preserve the exact same semantic meaning and factual relations: {s2}",
                "false": "Different meaning: word swapping, inverted roles, contradiction, missing, prohibited, or altered facts.",
            },
            "expected_result": is_paraphrase,
            "metadata": {
                "source": "google-research-datasets/paws",
                "upstream_id": raw_record.get("id", record_idx),
            },
        }


# =============================================================================
# 27. akhilaaa3/decision-bench Adapter (`noul`, `choice`, `score`)
# =============================================================================
class DecisionBenchAdapter(BaseDatasetAdapter):
    """Adapter for `akhilaaa3/decision-bench` (used by `akhilaaa3/Jev-Omni`)."""

    @property
    def dataset_name(self) -> str:
        return "akhilaaa3/decision-bench"

    @property
    def target_decision_type(self) -> str:
        return "jevcompat"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        raw_state = _parse_json_or_val(raw_record.get("state", ""))
        state_str = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, ensure_ascii=False)
        if "rules" in raw_record and "expected" in raw_record:
            dtype = str(raw_record.get("decision_type", "choice"))
            return {
                "id": f"DECBENCH-{record_idx:04d}",
                "domain": str(raw_record.get("family", "human_escalation")),
                "message": state_str,
                "decision_type": dtype,
                "instructions": str(raw_record.get("instructions", "Evaluate this DecisionBench scenario.")),
                "rules": raw_record["rules"],
                "expected_result": raw_record["expected"],
                "metadata": {
                    "source": "akhilaaa3/decision-bench",
                    "subset": raw_record.get("subset", "medium"),
                },
            }

        questions = _parse_json_or_val(raw_record.get("questions", {}))
        answers = _parse_json_or_val(raw_record.get("answers", {}))
        if isinstance(questions, dict) and questions:
            q_key = next(iter(questions.keys()))
            q_obj = questions[q_key] if isinstance(questions[q_key], dict) else {}
            dtype = str(q_obj.get("type", "noul")).lower()
            instructions = str(q_obj.get("instructions", f"Evaluate {q_key}."))
            ans_val = answers.get(q_key) if isinstance(answers, dict) else True
            if dtype == "noul":
                rules: Any = q_obj.get(
                    "criteria",
                    {
                        "true": f"Confirmed and established in the scenario: {instructions}",
                        "false": "Not established, missing, prohibited, or contradicted in the scenario.",
                    },
                )
                expected: Any = bool(ans_val)
            elif dtype == "score":
                rules = q_obj.get("criteria", SST5_ORDINAL_LEVELS[:3])
                expected = int(ans_val or 0)
            else:
                dtype = "choice"
                rules = q_obj.get("criteria", {"escalate": "Escalate case", "resolve": "Resolve locally"})
                expected = str(ans_val)
        else:
            dtype = "noul"
            instructions = "Should this case be escalated?"
            rules = {"true": "Escalate immediately", "false": "Routine case, missing escalation criteria"}
            expected = True

        return {
            "id": f"DECBENCH-{record_idx:04d}",
            "domain": str(raw_record.get("family", "human_escalation")),
            "message": state_str,
            "decision_type": dtype,
            "instructions": instructions,
            "rules": rules,
            "expected_result": expected,
            "metadata": {
                "source": "akhilaaa3/decision-bench",
                "subset": raw_record.get("subset", "medium"),
                "difficulty": raw_record.get("difficulty", "hard"),
            },
        }


# =============================================================================
# 28. Hangtao/JevAdvBench Adapter (`choice` + AnyJev L0 Order Invariance)
# =============================================================================
class JevAdvBenchAdapter(BaseDatasetAdapter):
    """Adapter for `Hangtao/JevAdvBench` (`arXiv:2609.31142` adversarial & option-order invariance)."""

    @property
    def dataset_name(self) -> str:
        return "Hangtao/JevAdvBench"

    @property
    def target_decision_type(self) -> str:
        return "choice"

    def convert_record(self, raw_record: dict[str, Any], record_idx: int = 1) -> dict[str, Any]:
        raw_state = _parse_json_or_val(raw_record.get("state", ""))
        state_str = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, ensure_ascii=False)
        q_obj = _parse_json_or_val(raw_record.get("question", {}))
        instructions = (
            str(q_obj.get("instructions", "Select the decision option resilient to adversarial perturbation."))
            if isinstance(q_obj, dict)
            else str(raw_record.get("instructions", "Select the robust decision option."))
        )
        options = raw_record.get("options") or (q_obj.get("criteria") if isinstance(q_obj, dict) else None)
        if not isinstance(options, dict) or len(options) < 2:
            options = {
                "approve_coverage": "Policy is active and collision coverage applies to the reported loss",
                "deny_claim": "Policy is expired or loss is excluded from coverage",
            }
        expected = raw_record.get("expected", raw_record.get("target_answer", next(iter(options.keys()))))
        if str(expected) not in options:
            expected = next(iter(options.keys()))

        return {
            "id": f"JEVADV-{record_idx:04d}",
            "domain": "jev_adversarial_robustness",
            "message": state_str,
            "decision_type": "choice",
            "instructions": instructions,
            "rules": {str(k): str(v) for k, v in options.items()},
            "expected_result": str(expected),
            "metadata": {
                "source": "Hangtao/JevAdvBench",
                "attack_id": raw_record.get("attack_id", "Q1"),
                "attack_method": raw_record.get("method", "spacing_and_order_perturbation"),
                "check_order_invariance": True,
            },
        }


# =============================================================================
# Curated Gold-Standard Fixtures for All 12 HF JEV Model Datasets (120 Items)
# =============================================================================
@functools.lru_cache(maxsize=1)
def _load_hf_jev_offline_samples() -> dict[str, list[dict[str, Any]]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("hf_jev_offline_samples.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES: dict[str, list[dict[str, Any]]] = _load_hf_jev_offline_samples()
HF_JEV_OFFLINE_SAMPLES = OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES

# Backward-compatibility dictionary for E2E tests (assert len(HF_JEV_DATASETS) >= 3, assert "name" in ds)
HF_JEV_DATASETS: dict[str, Any] = {
    key: {
        "name": key,
        "config": HF_JEV_DATASET_SERVER_CONFIG_MAP.get(key, {}),
        "samples": OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES.get(key, []),
    }
    for key in HF_JEV_DATASET_SERVER_CONFIG_MAP
}


HF_JEV_ADAPTER_CLASSES: dict[str, type[BaseDatasetAdapter]] = {
    "SargeDev/jev-distill-corpus-v3": JevDistillCorpusV3Adapter,
    "ZefanCai/Open-Jev": OpenJevCorpusAdapter,
    "tasksource/tasksource-jev-typed-decisions": TasksourceJevTypedDecisionsAdapter,
    "JevAgentHarness/AgentDecisionBench": AgentDecisionBenchAdapter,
    "LocalLLaMA/typed-decisions": LocalLLaMATypedDecisionsAdapter,
    "3nesdeniz/agentic-prompt-injection-5k": AgenticPromptInjection5kAdapter,
    "MaziyarPanahi/Jev-RAG-Claim-Verification-130K": JevRAGClaimVerificationAdapter,
    "MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K": JevSafetyRefusalDecisionsAdapter,
    "SetFit/sst5": SST5OrdinalAdapter,
    "google-research-datasets/paws": PAWSParaphraseAdapter,
    "akhilaaa3/decision-bench": DecisionBenchAdapter,
    "Hangtao/JevAdvBench": JevAdvBenchAdapter,
}
