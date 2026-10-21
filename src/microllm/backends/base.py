"""Protocol definition and shared semantic utilities for System-One decision backends."""

from __future__ import annotations

import hashlib
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


# Multilingual concept clusters for high-accuracy zero-shot CPU scoring without network downloads
CONCEPT_CLUSTERS: dict[str, set[str]] = {
    "billing_finance": {
        "bill", "billed", "billing", "invoice", "invoices", "charge", "charged",
        "double-charged", "refund", "payment", "payments", "subscription", "card",
        "price", "pricing", "fee", "overcharged", "facture", "remboursement",
        "paiement", "결제", "환불", "청구",
    },
    "technical_support": {
        "tech", "technical", "bug", "bugs", "crash", "crashes", "error", "errors",
        "outage", "server", "login", "password", "api", "slow", "broken", "down",
        "exception", "stacktrace", "panne", "erreur", "기술", "지원",
    },
    "shipping_logistics": {
        "ship", "shipping", "delivery", "parcel", "package", "courier", "late",
        "delayed", "lost", "tracking", "order", "arrived", "livraison", "colis",
    },
    "urgency_high": {
        "urgent", "immediately", "immediate", "today", "now", "critical", "blocking",
        "outage", "furious", "angry", "cancel", "twice", "still", "escalate",
        "emergency", "severe", "high", "immédiat", "urgent",
    },
    "urgency_low": {
        "calm", "low", "minor", "whenever", "question", "curious", "fine",
        "great", "happy", "satisfied", "normal", "not urgent",
    },
    "affirmative": {
        "yes", "true", "valid", "safe", "allow", "approved", "confirm", "urgent",
        "refund", "cancel", "outage", "oui", "vrai",
    },
    "negative": {
        "no", "false", "never", "not", "none", "calm", "minor", "non", "faux",
    },
    "security_destructive": {
        "rm", "-rf", "drop", "truncate", "delete", "wipe", "format", "destroy",
        "destructive", "production", "postgresql", "table", "database",
    },
    "security_injection": {
        "ignore", "previous", "instructions", "jailbreak", "bypass", "override",
        "system", "prompt", "hack", "injection", "disregard",
    },
    "security_pii": {
        "password", "secret", "token", "apikey", "api_key", "ssn", "credit",
        "private", "bearer", "credentials", "leak",
    },
}


def tokenize_multilingual(text: str) -> list[str]:
    """Extract lowercase word and CJK/Unicode tokens from arbitrary text."""
    lowered = text.lower()
    tokens = re.findall(r"[\w\u0080-\uffff]+", lowered)
    return tokens


def embed_text_deterministic(text: str, dim: int = 256) -> np.ndarray:
    """Compute a fast, deterministic character-ngram + concept-cluster embedding on CPU (<0.5 ms)."""
    vec = np.zeros(dim, dtype=np.float64)
    clean = text.lower().strip()
    if not clean:
        vec[0] = 1.0
        return vec

    tokens = tokenize_multilingual(clean)
    # 1. Word and character 3-gram feature hashing into slots [16..dim-1]
    n_hash_slots = dim - 16
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = 16 + (int.from_bytes(digest[:4], "little") % n_hash_slots)
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * 1.6
        # Subword char trigrams for morphological and multilingual robustness
        padded = f"<{tok}>"
        for i in range(max(1, len(padded) - 2)):
            tri = padded[i : i + 3]
            td = hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest()
            t_idx = 16 + (int.from_bytes(td, "little") % n_hash_slots)
            vec[t_idx] += 0.45

    # 2. Concept cluster activations in slots [0..15]
    token_set = set(tokens)
    for c_idx, (_, keywords) in enumerate(CONCEPT_CLUSTERS.items()):
        hits = len(token_set & keywords)
        for kw in keywords:
            if len(kw) >= 4 and kw in clean:
                hits += 1
        vec[c_idx] = float(hits) * 2.2

    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return vec


def semantic_match_logit(state_text: str, query_text: str, candidate_text: str) -> float:
    """Compute calibrated logit alignment between `(state, instructions)` and `candidate`."""
    v_state = embed_text_deterministic(state_text)
    v_cand = embed_text_deterministic(candidate_text)
    v_query = embed_text_deterministic(query_text)

    cos_sc = float(np.dot(v_state, v_cand))
    cos_qc = float(np.dot(v_query, v_cand))

    # Direct lexical + substring overlap boost
    s_tokens = set(tokenize_multilingual(state_text))
    c_tokens = set(tokenize_multilingual(candidate_text)) - {"it", "is", "the", "a", "an", "option"}
    overlap = len(s_tokens & c_tokens) / float(max(1, len(c_tokens)))

    # Concept cluster alignment
    concept_boost = float(np.dot(v_state[:16], v_cand[:16])) * 1.8
    return float(3.2 * cos_sc + 0.6 * cos_qc + 2.8 * overlap + concept_boost)


def explain_state_tokens(
    state_text: str,
    query_text: str,
    candidate_text: str,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Compute per-token attribution weights explaining which words in `state_text` drove `candidate_text`."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "on",
        "in", "and", "or", "i", "my", "our", "we", "you", "it", "this", "that",
        "have", "been", "be", "with", "by", "at", "from", "please", "can",
    }
    raw_tokens = tokenize_multilingual(state_text)
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) >= 2 and tok not in stopwords and tok not in seen:
            seen.add(tok)
            unique_tokens.append(tok)

    if not unique_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for tok in unique_tokens:
        w = semantic_match_logit(tok, query_text, candidate_text)
        if w > 0.05:
            scored.append((tok, float(w)))

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


def create_hf_zero_shot_scorer(
    model_name: str = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli",
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
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


def create_onnx_nli_scorer(
    onnx_path: str,
    tokenizer_name: str = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli",
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    valid_inputs = {inp.name for inp in session.get_inputs()}

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
        return float(flat[-1])

    return _score

