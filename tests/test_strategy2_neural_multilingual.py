"""Comprehensive Unit and Integration Tests for Strategy 2: Neural Multilingual Models.

Validates:
1. Dynamic entailment index resolution (Moritz Laurer index 0 vs RoBERTa index 2) resolving RCA-056.
2. ONNX session metadata inspection (custom_metadata_map, adjacent config.json, and heuristics).
3. Localized NLI hypothesis template formatting across EN, FR, ES, PT with automatic language detection.
4. Model registry entries and metadata integrity for multilingual-minilm-int8 and multilingual-mdeberta-int8.
5. End-to-end customer support ticket classification using mock ONNX inference sessions (zero network calls).
6. Graceful error handling and informative ImportError messages when optional packages are missing.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
from typing import Any

import numpy as np
import pytest

from microllm.backends.base import (
    create_onnx_nli_scorer,
    resolve_nli_entailment_idx,
    resolve_onnx_entailment_index,
)
from microllm.backends.fast_encoder import format_nli_hypothesis
from microllm.lexicons.normalizer import detect_language_fast
from microllm.registry import (
    MODEL_PROFILES,
    OFFICIAL_MICROLLM_HF_MODELS,
    PROFILE_ALIASES,
    load_profile_scorer,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# In-Memory Mock/Fixture Infrastructure for ONNX Runtime & Tokenizers
# ---------------------------------------------------------------------------


class MockOnnxModelMetadata:
    """Mock ONNX model metadata providing custom_metadata_map."""

    def __init__(self, custom_metadata: dict[str, str] | None = None) -> None:
        self.custom_metadata_map: dict[str, str] = custom_metadata or {}


class MockOnnxInput:
    """Mock ONNX input tensor descriptor."""

    def __init__(self, name: str = "input_ids") -> None:
        self.name: str = name


class MockOnnxInferenceSession:
    """Configurable mock ONNX InferenceSession for deterministic unit testing.

    Simulates outputs for Moritz Laurer or RoBERTa class ordering without downloading weights.
    """

    def __init__(
        self,
        model_path: str,
        custom_metadata: dict[str, str] | None = None,
        label_order: str = "moritz_laurer",  # "moritz_laurer" (0=entail) or "roberta" (2=entail)
    ) -> None:
        self.model_path = model_path
        if custom_metadata is None:
            custom_metadata = {"id2label": json.dumps({"0": "entailment", "1": "neutral", "2": "contradiction"})}
        self._metadata = MockOnnxModelMetadata(custom_metadata)
        self.label_order = label_order
        self.call_history: list[dict[str, Any]] = []

    def get_inputs(self) -> list[MockOnnxInput]:
        return [
            MockOnnxInput("input_ids"),
            MockOnnxInput("attention_mask"),
            MockOnnxInput("premise_text"),
            MockOnnxInput("hyp_text"),
        ]

    def get_modelmeta(self) -> MockOnnxModelMetadata:
        return self._metadata

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.call_history.append(input_feed)
        premise_text = str(input_feed.get("premise_text", "")).lower()
        hyp_text = str(input_feed.get("hyp_text", "")).lower()

        billing_kws = [
            "billing", "charged", "card", "refund", "factur", "rembours", "reembols",
            "fatur", "paiement", "payment", "cobrad", "abonnement", "suscripción", "assinatura",
        ]
        shipping_kws = [
            "shipping", "colis", "livraison", "envío", "envio", "encomenda",
            "douane", "customs", "aduana", "alfândega", "tracking", "suivi",
            "rastreamento", "parcels", "paquetes",
        ]
        tech_kws = [
            "tech", "technical", "login", "crash", "500", "server", "servidor", "serveur",
            "down", "pannes", "fallas", "falhas", "erreur", "error",
        ]

        premise_cat = None
        if any(kw in premise_text for kw in billing_kws):
            premise_cat = "billing"
        elif any(kw in premise_text for kw in shipping_kws):
            premise_cat = "shipping"
        elif any(kw in premise_text for kw in tech_kws):
            premise_cat = "tech"

        hyp_cat = None
        if any(kw in hyp_text for kw in billing_kws):
            hyp_cat = "billing"
        elif any(kw in hyp_text for kw in shipping_kws):
            hyp_cat = "shipping"
        elif any(kw in hyp_text for kw in tech_kws):
            hyp_cat = "tech"

        is_entailed = (premise_cat is not None and premise_cat == hyp_cat)

        entail_logit = 3.85 if is_entailed else -2.50
        neutral_logit = 0.10
        contra_logit = -2.50 if is_entailed else 3.85

        if self.label_order == "moritz_laurer":
            # 0: entailment, 1: neutral, 2: contradiction
            logits = np.array([[entail_logit, neutral_logit, contra_logit]], dtype=np.float64)
        else:
            # 0: contradiction, 1: neutral, 2: entailment (RoBERTa convention)
            logits = np.array([[contra_logit, neutral_logit, entail_logit]], dtype=np.float64)

        return [logits]


class MockTokenizer:
    """Deterministic mock tokenizer returning numpy arrays."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(
        self,
        premise: str,
        hypothesis: str,
        return_tensors: str = "np",
        truncation: bool = True,
        max_length: int = 512,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": np.array([[101, 2054, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
            "premise_text": premise,
            "hyp_text": hypothesis,
        }


@pytest.fixture
def mock_onnx_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture injecting mock onnxruntime and transformers modules into sys.modules."""
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = MockOnnxInferenceSession  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kwargs: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


# ---------------------------------------------------------------------------
# 1. Dynamic Entailment Index Resolution & RCA-056 Verification
# ---------------------------------------------------------------------------


def test_onnx_scorer_moritz_laurer_entailment_index_fixed(mock_onnx_environment: None) -> None:
    """RCA-056: Verify Moritz Laurer models extract entailment from index 0, NOT index 2."""
    scorer = create_onnx_nli_scorer(
        onnx_path="dummy_moritz_laurer.onnx",
        tokenizer_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    )
    assert getattr(scorer, "entailment_idx") == 0

    # Positive entailment pair: ticket asking for refund with billing hypothesis
    score_entail = scorer("I was charged twice on my card.", "Route ticket", "It is billing: refunds.")
    assert score_entail > 1.0, f"Entailing pair must receive positive score, got {score_entail}"

    # Negative/contradicting pair: ticket asking for refund with shipping hypothesis
    score_contra = scorer("I was charged twice on my card.", "Route ticket", "It is shipping: physical parcels.")
    assert score_contra < 0.0, f"Contradicting pair must receive negative score, got {score_contra}"


def test_onnx_scorer_roberta_entailment_index_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RoBERTa models extract entailment from index 2."""
    fake_ort = types.ModuleType("onnxruntime")

    class RoBERTaSession(MockOnnxInferenceSession):
        def __init__(self, path: str) -> None:
            meta = {"id2label": json.dumps({"0": "contradiction", "1": "neutral", "2": "entailment"})}
            super().__init__(path, custom_metadata=meta, label_order="roberta")

    fake_ort.InferenceSession = RoBERTaSession  # type: ignore[attr-defined]
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = create_onnx_nli_scorer(
        onnx_path="roberta_model.onnx",
        tokenizer_name="roberta-large-mnli",
    )
    assert getattr(scorer, "entailment_idx") == 2
    score_entail = scorer("The server is completely down.", "Route", "It is tech: server down.")
    assert score_entail > 1.0


def test_onnx_scorer_adjacent_config_json_resolution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify resolution of id2label from adjacent config.json when ONNX metadata is absent."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy onnx content")
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"label2id": {"entailment": 0, "neutral": 1, "contradiction": 2}}),
        encoding="utf-8",
    )

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda path: MockOnnxInferenceSession(path, custom_metadata={})  # type: ignore[attr-defined]
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = create_onnx_nli_scorer(str(onnx_file), tokenizer_name="custom-unlabeled-tokenizer")
    assert getattr(scorer, "entailment_idx") == 0
    score = scorer("Payment problem: card charged", "Route", "It is billing.")
    assert score > 1.0


def test_onnx_scorer_explicit_entailment_index_override(mock_onnx_environment: None) -> None:
    """Verify caller can supply explicit entailment_idx parameter."""
    scorer_0 = create_onnx_nli_scorer("model.onnx", "dummy-tok", entailment_idx=0)
    scorer_2 = create_onnx_nli_scorer("model.onnx", "dummy-tok", entailment_idx=2)

    assert getattr(scorer_0, "entailment_idx") == 0
    assert getattr(scorer_2, "entailment_idx") == 2

    score_0 = scorer_0("Test state", "Query", "It is billing.")
    score_2 = scorer_2("Test state", "Query", "It is billing.")
    assert score_0 != score_2


def test_resolve_nli_entailment_idx_direct() -> None:
    """Verify direct resolution across metadata, configs, and family heuristics."""
    # Moritz family heuristic
    assert resolve_nli_entailment_idx(tokenizer_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") == 0
    assert resolve_nli_entailment_idx(tokenizer_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli") == 0
    # RoBERTa family heuristic
    assert resolve_nli_entailment_idx(tokenizer_name="roberta-large-mnli") == 2
    assert resolve_nli_entailment_idx(tokenizer_name="facebook/bart-large-mnli") == 2
    # Unclassified fallback
    assert resolve_nli_entailment_idx(tokenizer_name="custom-mock-unknown") == -1
    # Alias verification
    assert resolve_onnx_entailment_index(explicit_idx=1) == 1


# ---------------------------------------------------------------------------
# 2. Localized Hypothesis Templates across EN, FR, ES, PT
# ---------------------------------------------------------------------------


def test_format_nli_hypothesis_quadrilingual_templates() -> None:
    """Verify localized hypothesis prefixes and structures for EN, FR, ES, PT."""
    assert format_nli_hypothesis("billing", "refunds", language="en") == "It is billing: refunds."
    assert format_nli_hypothesis("billing", None, language="en") == "It is billing."

    assert format_nli_hypothesis("facturation", "remboursements", language="fr") == "Il s'agit de facturation: remboursements."
    assert format_nli_hypothesis("facturation", None, language="fr") == "Il s'agit de facturation."

    assert format_nli_hypothesis("facturación", "reembolsos", language="es") == "Se trata de facturación: reembolsos."
    assert format_nli_hypothesis("facturación", None, language="es") == "Se trata de facturación."

    assert format_nli_hypothesis("faturamento", "reembolsos", language="pt") == "Trata-se de faturamento: reembolsos."
    assert format_nli_hypothesis("faturamento", None, language="pt") == "Trata-se de faturamento."


def test_format_nli_hypothesis_styles() -> None:
    """Verify format_nli_hypothesis template styles ('about', 'inquiry', 'default')."""
    assert format_nli_hypothesis("facturation", "demande", language="fr", template_type="about") == (
        "Ce texte concerne facturation: demande."
    )
    assert format_nli_hypothesis("facturación", "consulta", language="es", template_type="about") == (
        "Este texto trata sobre facturación: consulta."
    )
    assert format_nli_hypothesis("cobrança", "dúvida", language="pt", template_type="inquiry") == (
        "Esta consulta é sobre cobrança: dúvida."
    )
    assert format_nli_hypothesis("billing", "inquiry", language="en", template_type="inquiry") == (
        "This inquiry is about billing: inquiry."
    )


def test_format_nli_hypothesis_automatic_language_selection() -> None:
    """Verify integration of detect_language_fast for automated hypothesis selection."""
    en_premise = "I need an urgent refund for my cancelled subscription."
    fr_premise = "J'ai besoin d'un remboursement urgent pour mon abonnement annulé."
    es_premise = "Necesito un reembolso urgente por mi suscripción cancelada."
    pt_premise = "Preciso de um reembolso urgente para minha assinatura cancelada."

    assert detect_language_fast(en_premise) == "en"
    assert detect_language_fast(fr_premise) == "fr"
    assert detect_language_fast(es_premise) == "es"
    assert detect_language_fast(pt_premise) == "pt"

    # Hypotheses formatted with detected language
    assert format_nli_hypothesis("remboursement", "demande", language=detect_language_fast(fr_premise)) == (
        "Il s'agit de remboursement: demande."
    )
    assert format_nli_hypothesis("reembolso", "solicitud", language=detect_language_fast(es_premise)) == (
        "Se trata de reembolso: solicitud."
    )
    assert format_nli_hypothesis("reembolso", "pedido", language=detect_language_fast(pt_premise)) == (
        "Trata-se de reembolso: pedido."
    )
    # Auto selection via premise parameter
    assert format_nli_hypothesis("facture", "remboursement", premise=fr_premise) == (
        "Il s'agit de facture: remboursement."
    )


def test_format_nli_hypothesis_backward_compatibility() -> None:
    """Verify 100% backward compatibility when language argument is omitted."""
    assert format_nli_hypothesis("billing", "invoices and charges") == "It is billing: invoices and charges."
    assert format_nli_hypothesis("It is already formatted.", None) == "It is already formatted."


# ---------------------------------------------------------------------------
# 3. Model Registry Specifications & Integrity
# ---------------------------------------------------------------------------


def test_model_registry_contains_multilingual_profiles() -> None:
    """Verify MODEL_PROFILES entries for MiniLM and mDeBERTa."""
    assert "multilingual" in MODEL_PROFILES
    assert "multilingual-minilm-int8" in MODEL_PROFILES
    assert "multilingual-mdeberta-int8" in MODEL_PROFILES

    # Check MiniLM profile
    minilm_prof = MODEL_PROFILES["multilingual-minilm-int8"]
    assert "minilm" in minilm_prof["hf_repo"].lower()
    assert minilm_prof["pipeline_tag"] == "zero-shot-classification"
    assert minilm_prof["quantization"] == "INT8_DYNAMIC"
    assert minilm_prof["entailment_idx"] == 0

    # Check mDeBERTa profile
    mdeberta_prof = MODEL_PROFILES["multilingual-mdeberta-int8"]
    assert "mdeberta" in mdeberta_prof["hf_repo"].lower()
    assert mdeberta_prof["pipeline_tag"] == "zero-shot-classification"
    assert mdeberta_prof["entailment_idx"] == 0

    # Check aliases
    assert PROFILE_ALIASES["minilm"] == "multilingual-minilm-int8"
    assert PROFILE_ALIASES["mdeberta"] == "multilingual-mdeberta-int8"


def test_official_hf_models_catalog_metadata() -> None:
    """Verify OFFICIAL_MICROLLM_HF_MODELS catalog contains multilingual entries and retains 4 entries."""
    assert len(OFFICIAL_MICROLLM_HF_MODELS) == 4
    hf_repos = {m["repo_id"] for m in OFFICIAL_MICROLLM_HF_MODELS}
    assert "jmrenouard/microllm-multilingual-int8" in hf_repos

    for entry in OFFICIAL_MICROLLM_HF_MODELS:
        assert entry["license"] == "apache-2.0"
        assert entry["size_mb"] < 200.0


def test_load_profile_scorer_offline_fallback(tmp_path: pathlib.Path) -> None:
    """Verify load_profile_scorer operates offline safely with synthetic INT8 weights."""
    scorer = load_profile_scorer("multilingual", cache_dir=tmp_path)
    score = scorer("Fui cobrado duas vezes", "Qual departamento?", "Trata-se de faturamento.")
    assert isinstance(score, float)
    assert not np.isnan(score)

    scorer_minilm = load_profile_scorer("multilingual-minilm-int8", cache_dir=tmp_path)
    score_minilm = scorer_minilm("Mon colis est bloqué", "Quel service?", "Il s'agit de livraison.")
    assert isinstance(score_minilm, float)
    assert not np.isnan(score_minilm)


# ---------------------------------------------------------------------------
# 4. End-to-End Multilingual Customer Support Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "ticket", "expected_category"),
    [
        ("en", "I was charged twice on my credit card. Please process a full refund.", "billing"),
        ("en", "My package is delayed and tracking number shows stuck in customs.", "shipping"),
        ("en", "I cannot log in to my account, getting HTTP 500 internal server error.", "technical"),
        ("fr", "J'ai été prélevé deux fois sur ma carte. Je demande un remboursement immédiat.", "billing"),
        ("fr", "Mon colis a du retard et le numéro de suivi indique bloqué en douane.", "shipping"),
        ("fr", "Impossible de me connecter à mon compte, j'obtiens une erreur HTTP 500.", "technical"),
        ("es", "Me cobraron dos veces en mi tarjeta. Solicito un reembolso completo.", "billing"),
        ("es", "Mi paquete está retrasado y el número de seguimiento indica retenido en aduana.", "shipping"),
        ("es", "No puedo iniciar sesión en mi cuenta, recibo un error 500 del servidor.", "technical"),
        ("pt", "Fui cobrado duas vezes no cartão. Solicito o reembolso imediato do valor.", "billing"),
        ("pt", "Minha encomenda está atrasada e o rastreamento mostra parada na alfândega.", "shipping"),
        ("pt", "Não consigo fazer login na minha conta, aparece erro 500 do servidor.", "technical"),
    ],
)
def test_e2e_multilingual_routing_with_mock_onnx(
    mock_onnx_environment: None,
    lang: str,
    ticket: str,
    expected_category: str,
) -> None:
    """Verify that simulated ONNX zero-shot scorer routes quadrilingual inquiries accurately."""
    scorer = create_onnx_nli_scorer(
        onnx_path="multilingual_model.onnx",
        tokenizer_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    )

    categories = {
        "billing": {"en": "billing: refunds", "fr": "facturation: remboursements", "es": "facturación: reembolsos", "pt": "faturamento: reembolsos"},
        "shipping": {"en": "shipping: parcels", "fr": "livraison: colis", "es": "envío: paquetes", "pt": "envio: encomendas"},
        "technical": {"en": "technical: crashes", "fr": "technique: pannes", "es": "técnico: errores", "pt": "técnico: falhas"},
    }

    scores = {}
    for cat_name, localized_desc in categories.items():
        hyp = format_nli_hypothesis(cat_name, localized_desc[lang], language=lang)
        scores[cat_name] = scorer(ticket, "Route ticket", hyp)

    winning_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
    assert winning_cat == expected_category, (
        f"Failed for {lang}: expected {expected_category}, got {winning_cat}. Scores: {scores}"
    )


# ---------------------------------------------------------------------------
# 5. Error Handling & Missing Dependencies Verification
# ---------------------------------------------------------------------------


def test_onnx_scorer_raises_helpful_importerror_when_ort_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify helpful error message when onnxruntime is absent."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"pip install microllm\[onnx\]"):
        create_onnx_nli_scorer("dummy.onnx", "dummy-tok")


def test_onnx_scorer_raises_helpful_importerror_when_transformers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify helpful error message when transformers is absent."""
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = MockOnnxInferenceSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match=r"pip install microllm\[onnx\] transformers"):
        create_onnx_nli_scorer("dummy.onnx", "dummy-tok")
