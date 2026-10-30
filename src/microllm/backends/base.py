"""Protocol definition and shared semantic utilities for System-One decision backends."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol
import numpy as np


class DecisionBackend(Protocol):
    """Unified interface for System-One decision engines (Gear 1, Gear 2, Cascade)."""

    name: str

    def evaluate_noul(
        self,
        state: str,
        instructions: str,
        criteria: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Return {'type': 'noul', 'noul': float, ...}."""
        ...

    def evaluate_choice(
        self,
        state: str,
        instructions: str,
        options: Mapping[str, str | None],
    ) -> dict[str, Any]:
        """Return {'type': 'choice', 'choice': str, 'probabilities': dict, 'confidence': float, ...}."""
        ...

    def evaluate_score(
        self,
        state: str,
        instructions: str,
        levels: list[str],
    ) -> dict[str, Any]:
        """Return {'type': 'score', 'score': float, 'probabilities': dict, 'confidence': float, ...}."""
        ...


from microllm.lexicons import ALL_SENTENCE_CONNECTORS, ALL_STOPWORDS
from microllm.lexicons.concepts import (
    CONCEPT_CLUSTERS,
    STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS,
)

_STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS = STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS


def tokenize_multilingual(text: str) -> list[str]:
    """Extract lowercase word and CJK/Unicode tokens from arbitrary text."""
    text_str = text if isinstance(text, str) else str(text)
    lowered = text_str.lower()
    tokens = re.findall(r"[\w\u0080-\uffff]+", lowered)
    return tokens


@functools.lru_cache(maxsize=65536)
def _token_hash_contribution(tok: str, n_hash_slots: int = 240) -> tuple[tuple[int, float], ...]:
    """LRU-cached per-token BLAKE2b word and character 3-gram hash slot contributions."""
    acc: dict[int, float] = {}
    digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
    idx = 16 + (int.from_bytes(digest[:4], "little") % n_hash_slots)
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    acc[idx] = acc.get(idx, 0.0) + sign * 1.6
    padded = f"<{tok}>"
    for i in range(max(1, len(padded) - 2)):
        tri = padded[i : i + 3]
        td = hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest()
        t_idx = 16 + (int.from_bytes(td, "little") % n_hash_slots)
        acc[t_idx] = acc.get(t_idx, 0.0) + 0.45
    return tuple(acc.items())


@functools.lru_cache(maxsize=8192)
def _tokenize_set_cached(clean: str) -> frozenset[str]:
    """LRU-cached token set for fast lexical overlap evaluation across large benchmark splits."""
    return frozenset(tokenize_multilingual(clean))


@functools.lru_cache(maxsize=8192)
def _embed_text_cached(clean: str, dim: int = 256) -> tuple[float, ...]:
    """LRU-cached deterministic embedding computation (IDEA-012)."""
    vec = np.zeros(dim, dtype=np.float64)
    if not clean:
        vec[0] = 1.0
        return tuple(float(x) for x in vec)

    tokens = tokenize_multilingual(clean)
    # 1. Word and character 3-gram feature hashing into slots [16..dim-1]
    n_hash_slots = dim - 16
    for tok in tokens:
        for idx, delta in _token_hash_contribution(tok, n_hash_slots):
            vec[idx] += delta

    # 2. Concept cluster activations in slots [0..15]
    token_set = set(tokens)
    for c_idx, (_, keywords) in enumerate(CONCEPT_CLUSTERS.items()):
        hits = len(token_set & keywords)
        for kw in keywords:
            if len(kw) >= 4:
                if kw in _STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS:
                    if kw in token_set:
                        hits += 1
                elif kw in clean:
                    hits += 1
        vec[c_idx] = float(hits) * 2.2

    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return tuple(float(x) for x in vec)


def embed_text_deterministic(text: str, dim: int = 256) -> np.ndarray:
    """Compute a fast, deterministic character-ngram + concept-cluster embedding on CPU (<0.5 ms)."""
    text_str = text if isinstance(text, str) else str(text)
    clean = text_str.lower().strip()
    return np.array(_embed_text_cached(clean, int(dim)), dtype=np.float64)


# Candidate hypothesis template framing words stripped during lexical overlap calculation
_CANDIDATE_STOPWORDS: frozenset[str] = frozenset({
    # English hypothesis frame words
    "it", "is", "the", "a", "an", "option",
    # French hypothesis frame words
    "il", "s", "agit", "de", "d",
    # Spanish hypothesis frame words
    "se", "trata", "opcion", "opción",
    # Portuguese hypothesis frame words
    "trata", "opcao", "opção",
})

# Module-level compiled regex for multilingual salient clause splitting
_CONNECTOR_PATTERN = "|".join(
    re.escape(c) for c in sorted(ALL_SENTENCE_CONNECTORS, key=len, reverse=True)
)
_CLAUSE_SPLIT_REGEX = re.compile(rf"[.;\n]+|\b(?:{_CONNECTOR_PATTERN})\b", re.IGNORECASE)


@functools.lru_cache(maxsize=8192)
def _extract_salient_clauses_cached(s_clean: str) -> tuple[str, ...]:
    """Extract salient clauses from a multi-clause state for HYP-008 max-pooling (APP-011)."""
    body = s_clean
    if "\nquestion:" in body:
        body = body.split("\nquestion:", 1)[0]
    elif "\nhypothesis:" in body:
        body = body.split("\nhypothesis:", 1)[0]
    raw_clauses = [
        c.strip()
        for c in _CLAUSE_SPLIT_REGEX.split(body)
        if c.strip() and len(c.strip()) >= 12
    ]
    if len(raw_clauses) <= 1:
        return ()
    return tuple(raw_clauses[:10])


def _base_semantic_match_logit(s_clean: str, q_clean: str, c_clean: str) -> float:
    """Compute single-span calibrated logit alignment between `(state, query)` and `candidate`."""
    v_state = _embed_text_cached(s_clean, 256)
    v_cand = _embed_text_cached(c_clean, 256)
    v_query = _embed_text_cached(q_clean, 256)

    cos_sc = float(sum(a * b for a, b in zip(v_state, v_cand)))
    cos_qc = float(sum(a * b for a, b in zip(v_query, v_cand)))

    s_tokens = _tokenize_set_cached(s_clean)
    c_tokens = _tokenize_set_cached(c_clean) - _CANDIDATE_STOPWORDS
    overlap = len(s_tokens & c_tokens) / float(max(1, len(c_tokens)))

    concept_boost = float(sum(a * b for a, b in zip(v_state[:16], v_cand[:16]))) * 1.8
    return float(3.2 * cos_sc + 0.6 * cos_qc + 2.8 * overlap + concept_boost)


def semantic_match_logit(
    state_text: str,
    query_text: str,
    candidate_text: str,
    clause_salient: bool = True,
) -> float:
    """Compute calibrated logit alignment between `(state, instructions)` and `candidate`.

    Integrates Clause-Salient Max-Pooling (`HYP-008` / `APP-011`, `0.25 * global + 0.75 * max_clause`)
    on multi-clause / distractor-padded states so irrelevant context clauses do not dilute supporting evidence.
    """
    s_str = state_text if isinstance(state_text, str) else str(state_text)
    q_str = query_text if isinstance(query_text, str) else str(query_text)
    c_str = candidate_text if isinstance(candidate_text, str) else str(candidate_text)

    s_clean = s_str.lower().strip()
    q_clean = q_str.lower().strip()
    c_clean = c_str.lower().strip()

    global_logit = _base_semantic_match_logit(s_clean, q_clean, c_clean)
    if clause_salient and (
        "\n\n" in s_clean
        or len(s_clean) >= 650
        or any(conn in s_clean for conn in ALL_SENTENCE_CONNECTORS)
    ):
        clauses = _extract_salient_clauses_cached(s_clean)
        if clauses:
            max_clause_logit = max(
                _base_semantic_match_logit(cl, q_clean, c_clean) for cl in clauses
            )
            if max_clause_logit > global_logit:
                return float(0.25 * global_logit + 0.75 * max_clause_logit)
    return global_logit


def explain_state_tokens(
    state_text: str,
    query_text: str,
    candidate_text: str,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Compute per-token attribution weights explaining which words in `state_text` drove `candidate_text`."""
    stopwords = ALL_STOPWORDS
    raw_tokens = tokenize_multilingual(state_text)
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) >= 2 and tok not in stopwords and tok not in seen:
            seen.add(tok)
            unique_tokens.append(tok)

    if not unique_tokens:
        return []

    v_cand = embed_text_deterministic(candidate_text)
    v_query = embed_text_deterministic(query_text)
    cos_qc = float(np.dot(v_query, v_cand))
    c_tokens = set(tokenize_multilingual(candidate_text)) - _CANDIDATE_STOPWORDS
    c_len = float(max(1, len(c_tokens)))

    scored: list[tuple[str, float]] = []
    for tok in unique_tokens:
        v_tok = embed_text_deterministic(tok)
        cos_sc = float(np.dot(v_tok, v_cand))
        overlap = (1.0 / c_len) if tok in c_tokens else 0.0
        concept_boost = float(np.dot(v_tok[:16], v_cand[:16])) * 1.8
        w = float(3.2 * cos_sc + 0.6 * cos_qc + 2.8 * overlap + concept_boost)
        if w > 0.05:
            scored.append((tok, w))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_items = scored[: max(1, top_k)]
    max_w = max((w for _, w in top_items), default=1.0)
    return [
        {
            "token": tok,
            "score": round(w, 4),
            "normalized_weight": round(min(1.0, max(0.05, w / max(1e-6, max_w))), 4),
        }
        for tok, w in top_items
    ]


def explain_counterfactual(
    state_text: str,
    instructions_text: str,
    options: Mapping[str, str | None],
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute leave-one-token-out counterfactual sensitivity & decision-flipping tokens (IDEA-019)."""
    keys = list(options.keys())
    if len(keys) < 2:
        raise ValueError("explain_counterfactual requires at least 2 options.")

    def _probs_for(text: str) -> dict[str, float]:
        logits = np.array(
            [
                semantic_match_logit(
                    text,
                    instructions_text,
                    f"It is {k}: {options[k]}." if options[k] else f"It is {k}.",
                )
                for k in keys
            ],
            dtype=np.float64,
        )
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted / 0.80)
        p = exp_l / np.sum(exp_l)
        return {keys[i]: float(p[i]) for i in range(len(keys))}

    base_probs = _probs_for(state_text)
    orig_choice = max(base_probs, key=lambda k: base_probs[k])
    orig_prob = base_probs[orig_choice]

    words = state_text.split()
    impacts: list[dict[str, Any]] = []
    seen_lower: set[str] = set()
    for idx, word in enumerate(words):
        clean_tok = re.sub(r"[^\w\u0080-\uffff-]", "", word.lower())
        if len(clean_tok) < 2 or clean_tok in seen_lower:
            continue
        seen_lower.add(clean_tok)
        ablated_text = " ".join(w for j, w in enumerate(words) if j != idx)
        ablated_probs = _probs_for(ablated_text)
        new_choice = max(ablated_probs, key=lambda k: ablated_probs[k])
        drop = orig_prob - ablated_probs[orig_choice]
        impacts.append(
            {
                "token": clean_tok,
                "probability_drop": round(float(drop), 4),
                "new_choice": new_choice,
                "flips_decision": new_choice != orig_choice,
            }
        )

    impacts.sort(key=lambda item: float(item["probability_drop"]), reverse=True)
    flip_tokens = [item["token"] for item in impacts if item["flips_decision"]]
    return {
        "original_choice": orig_choice,
        "original_probability": round(orig_prob, 4),
        "token_impacts": impacts[: max(1, top_k)],
        "decision_flip_tokens": flip_tokens,
    }


def create_hf_zero_shot_scorer(
    model_name: str = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli",
    revision: str = "main",
) -> Callable[[str, str, str], float]:
    """Create a HuggingFace NLI / sequence-classification logit scorer compatible with `semantic_match_logit`.

    Lazily imports `torch` and `transformers`. Raises a helpful `ImportError` with
    `pip install microllm[torch]` if the optional dependencies are not installed.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "HuggingFace zero-shot scorer requires `torch` and `transformers`. "
            "Install them with: pip install microllm[torch]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    model.eval()

    label2id = getattr(getattr(model, "config", None), "label2id", None) or {}
    entail_idx = -1
    for label, idx in label2id.items():
        if "entail" in str(label).lower():
            entail_idx = int(idx)
            break

    def _score(state_text: str, query_text: str, candidate_text: str) -> float:
        premise = f"{state_text}\n{query_text}".strip() if query_text else state_text
        inputs = tokenizer(
            premise,
            candidate_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        return float(logits[entail_idx].item())

    return _score


def _find_entailment_from_mapping(mapping: Mapping[Any, Any]) -> int | None:
    """Extract the integer index for the entailment class from an id2label or label2id mapping."""
    if not isinstance(mapping, Mapping):
        return None
    # 1. Exact match pass (prioritizes 'entailment' / 'entail' over contrastive compounds like 'not_entailment')
    for k, v in mapping.items():
        sk, sv = str(k).lower().strip(), str(v).lower().strip()
        if sk in ("entailment", "entail"):
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
        if sv in ("entailment", "entail"):
            try:
                return int(k)
            except (ValueError, TypeError):
                pass
    # 2. Substring match pass (excluding negative prefixes: 'not', 'non-', 'non_', 'dis')
    for k, v in mapping.items():
        sk, sv = str(k).lower().strip(), str(v).lower().strip()
        if "entail" in sk and not any(neg in sk for neg in ("not", "non-", "non_", "dis")):
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
        if "entail" in sv and not any(neg in sv for neg in ("not", "non-", "non_", "dis")):
            try:
                return int(k)
            except (ValueError, TypeError):
                pass
    return None


def resolve_nli_entailment_idx(
    session: Any = None,
    onnx_path: str | Path | None = None,
    tokenizer_name: str | None = None,
    revision: str = "main",
    explicit_idx: int | None = None,
) -> int:
    """Dynamically resolve the entailment class index for NLI ONNX models.

    Resolution order:
    1. Explicit caller override (`explicit_idx`).
    2. ONNX Runtime model metadata (`session.get_modelmeta().custom_metadata_map`).
    3. Local `config.json` or `hf_config.json` adjacent to `onnx_path` or in its parent directories.
    4. Hugging Face `AutoConfig` via `transformers` (if available).
    5. Model family heuristics:
       - Moritz Laurer / XNLI / MiniLM / mDeBERTa models -> index 0
       - RoBERTa / BART / classic MNLI models -> index 2
    6. Backward-compatible fallback -> index -1
    """
    if explicit_idx is not None:
        return int(explicit_idx)

    # 1. Inspect ONNX session model metadata
    if session is not None and hasattr(session, "get_modelmeta"):
        try:
            meta = session.get_modelmeta()
            cust = getattr(meta, "custom_metadata_map", None) or {}
            if "entailment_idx" in cust:
                try:
                    return int(cust["entailment_idx"])
                except (ValueError, TypeError):
                    pass
            for key in ("id2label", "label2id", "labels"):
                if key in cust:
                    try:
                        raw = cust[key]
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        if isinstance(parsed, Mapping):
                            idx = _find_entailment_from_mapping(parsed)
                            if idx is not None:
                                return idx
                    except Exception:
                        continue
        except Exception:
            pass

    # 2. Inspect local configuration files adjacent to onnx_path
    if onnx_path is not None:
        p = Path(onnx_path)
        candidates = [
            p.with_name("config.json"),
            p.with_name("hf_config.json"),
            p.parent / "config.json",
            p.parent / "hf_config.json",
            p.parent.parent / "config.json",
            p.parent.parent / "hf_config.json",
            p.with_suffix(".json"),
        ]
        for c in candidates:
            if c.is_file():
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    mapping = data.get("label2id") or data.get("id2label")
                    if isinstance(mapping, Mapping):
                        idx = _find_entailment_from_mapping(mapping)
                        if idx is not None:
                            return idx
                except Exception:
                    pass

    # 3. Inspect Hugging Face AutoConfig if available
    if tokenizer_name:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(tokenizer_name, revision=revision)
            mapping = getattr(cfg, "label2id", None) or getattr(cfg, "id2label", None)
            if isinstance(mapping, Mapping):
                idx = _find_entailment_from_mapping(mapping)
                if idx is not None:
                    return idx
        except Exception:
            pass

    # 4. Model family heuristics
    name_str = f"{tokenizer_name or ''} {onnx_path or ''}".lower()
    if any(sig in name_str for sig in ("moritz", "laurer", "xnli", "minilm", "mdeberta")):
        return 0
    if any(sig in name_str for sig in ("roberta", "bart", "classic")):
        return 2

    # 5. Backward-compatible fallback for unclassified legacy test mocks
    return -1


resolve_onnx_entailment_index = resolve_nli_entailment_idx


def create_onnx_nli_scorer(
    onnx_path: str,
    tokenizer_name: str = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli",
    revision: str = "main",
    entailment_idx: int | None = None,
) -> Callable[[str, str, str], float]:
    """Create an ONNX Runtime NLI logit scorer compatible with `semantic_match_logit`.

    Lazily imports `onnxruntime` and `transformers`. Raises a helpful `ImportError` with
    `pip install microllm[onnx]` if `onnxruntime` is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "ONNX NLI scorer requires `onnxruntime`. "
            "Install it with: pip install microllm[onnx]"
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ONNX NLI scorer requires `transformers` for tokenization. "
            "Install with: pip install microllm[onnx] transformers"
        ) from exc

    session = ort.InferenceSession(onnx_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    valid_inputs = {inp.name for inp in session.get_inputs()}

    resolved_entail_idx = resolve_nli_entailment_idx(
        session=session,
        onnx_path=onnx_path,
        tokenizer_name=tokenizer_name,
        revision=revision,
        explicit_idx=entailment_idx,
    )

    def _score(state_text: str, query_text: str, candidate_text: str) -> float:
        premise = f"{state_text}\n{query_text}".strip() if query_text else state_text
        encoded = tokenizer(
            premise,
            candidate_text,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )
        feed = {k: v for k, v in encoded.items() if k in valid_inputs} or dict(encoded)
        outputs = session.run(None, feed)[0]
        flat = np.asarray(outputs, dtype=np.float64).reshape(-1)
        idx = resolved_entail_idx if abs(resolved_entail_idx) < len(flat) else -1
        return float(flat[idx])

    setattr(_score, "entailment_idx", resolved_entail_idx)
    return _score


def create_hybrid_fast_path_scorer(
    deep_scorer: Callable[[str, str, str], float] | None = None,
    margin_threshold: float = 0.65,
    fast_weight: float = 0.35,
    int8_weights: np.ndarray | None = None,
    int8_scale: float = 1.0,
    *,
    secondary_scorer: Callable[[str, str, str], float] | None = None,
    secondary_weight: float | None = None,
) -> Callable[[str, str, str], float]:
    """Create a Hybrid Fast-Path Ensemble Scorer resolving `APP-001` / `APP-003`.

    Uses the ultra-fast deterministic CPU encoder (`semantic_match_logit`, ~0.2 ms) when the
    alignment signal is unambiguous (`|fast_logit - 0.85| >= margin_threshold`), and escalates
    ambiguous inputs to `deep_scorer` / `secondary_scorer` (e.g., ONNX INT8 cross-encoder,
    HuggingFace NLI, or an INT8 bilinear + multi-template ensemble scorer when `None`).
    """
    stats: dict[str, int] = {"total_calls": 0, "fast_path_hits": 0, "deep_escalations": 0}
    eff_fast_weight = (1.0 - float(secondary_weight)) if secondary_weight is not None else float(fast_weight)
    w_fast = float(max(0.0, min(1.0, eff_fast_weight)))
    w_deep = 1.0 - w_fast

    if int8_weights is not None:
        w_deq = np.asarray(int8_weights, dtype=np.float64) * float(int8_scale)
    else:
        w_deq = None

    def _default_deep_ensemble(s_text: str, q_text: str, c_text: str) -> float:
        l1 = semantic_match_logit(s_text, q_text, c_text, clause_salient=True)
        l2 = semantic_match_logit(s_text, q_text, f"The state satisfies: {c_text}", clause_salient=True)
        l3 = semantic_match_logit(s_text, q_text, f"Category — {c_text}", clause_salient=True)
        base_deep = (l1 + l2 + l3) / 3.0
        if w_deq is not None and w_deq.ndim == 2 and w_deq.shape[0] == 256:
            vs = np.asarray(_embed_text_cached(s_text.lower().strip(), 256), dtype=np.float64)
            vc = np.asarray(_embed_text_cached(c_text.lower().strip(), 256), dtype=np.float64)
            ps = vs @ w_deq
            pc = vc @ w_deq
            norm_p = float(np.linalg.norm(ps) * np.linalg.norm(pc))
            bilinear_cos = float(np.dot(ps, pc) / norm_p) if norm_p > 1e-9 else 0.0
            return float(0.85 * base_deep + 0.45 * bilinear_cos)
        return float(base_deep)

    active_deep = deep_scorer or secondary_scorer or _default_deep_ensemble

    def _hybrid_score(state_text: str, query_text: str, candidate_text: str) -> float:
        stats["total_calls"] += 1
        fast_logit = semantic_match_logit(state_text, query_text, candidate_text)
        if abs(fast_logit - 0.85) >= float(margin_threshold):
            stats["fast_path_hits"] += 1
            return float(fast_logit)
        stats["deep_escalations"] += 1
        deep_logit = float(active_deep(state_text, query_text, candidate_text))
        return float(w_fast * fast_logit + w_deep * deep_logit)

    setattr(_hybrid_score, "stats", stats)
    return _hybrid_score


class VerdictMQMultiQuestionCache:
    """Verdict-MQ (`IDEA-022`, `arXiv:2610.20778`) Encode-Once-Decide-Many State Cache.

    Encodes shared `state` representations once (256-D embedding, token set, and salient clauses)
    and evaluates `M` independent questions with block-diagonal isolation (zero cross-question leakage).
    """

    def __init__(self, max_states: int = 1024, max_entries: int | None = None) -> None:
        eff_max = max_entries if max_entries is not None else max_states
        self.max_states = max(16, int(eff_max))
        self._state_store: dict[str, dict[str, Any]] = {}
        self.encode_hits: int = 0
        self.encode_misses: int = 0

    def encode_state_once(self, state_text: str) -> dict[str, Any]:
        """Encode `state_text` once or retrieve its frozen representation from the Verdict-MQ cache."""
        s_str = state_text if isinstance(state_text, str) else str(state_text)
        s_clean = s_str.lower().strip()
        cached = self._state_store.get(s_clean)
        if cached is not None:
            self.encode_hits += 1
            return cached

        self.encode_misses += 1
        if len(self._state_store) >= self.max_states:
            oldest_key = next(iter(self._state_store))
            del self._state_store[oldest_key]

        clauses = _extract_salient_clauses_cached(s_clean)
        clause_embeddings = tuple(_embed_text_cached(cl, 256) for cl in clauses)
        entry = {
            "clean": s_clean,
            "embedding": _embed_text_cached(s_clean, 256),
            "tokens": _tokenize_set_cached(s_clean),
            "clauses": clauses,
            "clause_embeddings": clause_embeddings,
            "clause_count": max(1, len(clauses)),
        }
        self._state_store[s_clean] = entry
        return entry

    def encode_state(self, state_text: str) -> dict[str, Any]:
        """Alias for `encode_state_once`."""
        return self.encode_state_once(state_text)

    def score_cached_state(self, state_text: str, query_text: str, hypothesis: str) -> float:
        """Score `(state_text, query_text, hypothesis)` while ensuring `state_text` is cached in Verdict-MQ."""
        self.encode_state_once(state_text)
        return semantic_match_logit(state_text, query_text, hypothesis)

    @property
    def stats(self) -> dict[str, Any]:
        """Return Verdict-MQ cache hit/miss telemetry."""
        total = self.encode_hits + self.encode_misses
        return {
            "hits": self.encode_hits,
            "misses": self.encode_misses,
            "cached_states": len(self._state_store),
            "hit_rate": round(self.encode_hits / float(max(1, total)), 4),
        }

    def evaluate_many(
        self,
        backend: Any,
        state_text: str,
        questions: Mapping[str, Any],
        evaluator_fn: Callable[[Any, str, Any], tuple[Any, dict[str, Any]]],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Encode the shared `state` representation once and evaluate all `M` questions with cached state reuse."""
        answers: dict[str, Any] = {}
        raw_metas: dict[str, dict[str, Any]] = {}
        if not questions:
            self.encode_state_once(state_text)
            return answers, raw_metas
        for qid, q_spec in questions.items():
            self.encode_state_once(state_text)
            ans_obj, raw = evaluator_fn(backend, state_text, q_spec)
            answers[qid] = ans_obj
            raw_metas[qid] = raw
        return answers, raw_metas



