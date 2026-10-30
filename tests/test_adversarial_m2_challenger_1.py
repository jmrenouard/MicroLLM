"""Adversarial Challenge Test Suite for Milestone 2 (Dynamic ONNX Entailment Index Resolution).

Empirically challenges:
1. Dynamic entailment index resolution tiers under malformed/corrupted metadata and configs.
2. Binary models (2-class) with 'not_entailment' vs 'entailment' (GLUE RTE / QNLI conventions).
3. N-class models (1-class, 3-class, 5-class) and out-of-bounds indexing.
4. Moritz Laurer vs RoBERTa logit polarity (strictly positive for entailment, negative for contradiction).
5. Legacy fallback behavior when metadata and heuristics are completely absent.
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
    _find_entailment_from_mapping,
    create_onnx_nli_scorer,
    resolve_nli_entailment_idx,
    resolve_onnx_entailment_index,
)
from microllm.backends.fast_encoder import FastEncoderBackend, format_nli_hypothesis
from microllm.registry import MODEL_PROFILES, PROFILE_ALIASES


# ---------------------------------------------------------------------------
# Test Infrastructure / Mocks
# ---------------------------------------------------------------------------


class MockOnnxModelMetadata:
    def __init__(self, custom_metadata: dict[str, Any] | None = None) -> None:
        self.custom_metadata_map: dict[str, Any] = custom_metadata or {}


class MockOnnxInput:
    def __init__(self, name: str = "input_ids") -> None:
        self.name = name


class DynamicMockOnnxSession:
    """Mock ONNX session capable of returning arbitrary tensor shapes and logits."""

    def __init__(
        self,
        output_logits: np.ndarray | list[float],
        custom_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_logits = np.asarray(output_logits, dtype=np.float64)
        self._metadata = MockOnnxModelMetadata(custom_metadata)
        self.call_history: list[dict[str, Any]] = []

    def get_inputs(self) -> list[MockOnnxInput]:
        return [MockOnnxInput("input_ids"), MockOnnxInput("attention_mask")]

    def get_modelmeta(self) -> MockOnnxModelMetadata:
        return self._metadata

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.call_history.append(input_feed)
        # Ensure 2D shape (1, num_classes)
        if self.output_logits.ndim == 1:
            out = self.output_logits.reshape(1, -1)
        else:
            out = self.output_logits
        return [out]


class MockTokenizer:
    def __init__(self, name: str = "mock-tok") -> None:
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
            "input_ids": np.array([[101, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
            "premise_text": premise,
            "hyp_text": hypothesis,
        }


# ===========================================================================
# 1. Stress-Testing _find_entailment_from_mapping
# ===========================================================================


def test_find_entailment_corrupted_and_scalar_inputs() -> None:
    """Verify _find_entailment_from_mapping does not crash on non-mapping or invalid entries."""
    assert _find_entailment_from_mapping(None) is None  # type: ignore[arg-type]
    assert _find_entailment_from_mapping([]) is None  # type: ignore[arg-type]
    assert _find_entailment_from_mapping("not a mapping") is None  # type: ignore[arg-type]
    assert _find_entailment_from_mapping(123) is None  # type: ignore[arg-type]
    assert _find_entailment_from_mapping({}) is None

    # Mapping with non-convertible integer values
    assert _find_entailment_from_mapping({"entailment": "NOT_AN_INT"}) is None
    assert _find_entailment_from_mapping({"NOT_AN_INT": "entailment"}) is None
    assert _find_entailment_from_mapping({None: "entailment"}) is None


def test_find_entailment_standard_mnli_mappings() -> None:
    """Verify standard Moritz Laurer (0) and RoBERTa (2) mappings."""
    # Moritz Laurer id2label & label2id
    assert _find_entailment_from_mapping({"0": "entailment", "1": "neutral", "2": "contradiction"}) == 0
    assert _find_entailment_from_mapping({"entailment": 0, "neutral": 1, "contradiction": 2}) == 0
    assert _find_entailment_from_mapping({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0

    # RoBERTa id2label & label2id
    assert _find_entailment_from_mapping({"0": "contradiction", "1": "neutral", "2": "entailment"}) == 2
    assert _find_entailment_from_mapping({"contradiction": 0, "neutral": 1, "entailment": 2}) == 2
    assert _find_entailment_from_mapping({0: "contradiction", 1: "neutral", 2: "entailment"}) == 2


def test_find_entailment_binary_rte_qnli_vulnerability() -> None:
    """Empirical demonstration of binary NLI mapping vulnerability with 'not_entailment'.

    In GLUE RTE and QNLI, labels are often:
      0: 'not_entailment', 1: 'entailment'
    Because _find_entailment_from_mapping searches for substring 'entail' without word boundaries
    or negative prefix checks, 'not_entailment' matches first if listed at index 0.
    """
    mapping_id2label = {"0": "not_entailment", "1": "entailment"}
    mapping_label2id = {"not_entailment": 0, "entailment": 1}

    res_id2label = _find_entailment_from_mapping(mapping_id2label)
    res_label2id = _find_entailment_from_mapping(mapping_label2id)

    # EXPECTED CONTRACT: Binary models with {'not_entailment', 'entailment'} MUST resolve index 1 ('entailment').
    # FAILURE: Current implementation extracts index 0 ('not_entailment') due to naive substring match.
    assert res_id2label == 1, (
        f"Binary id2label failed: expected index 1 ('entailment'), got {res_id2label} ('not_entailment')"
    )
    assert res_label2id == 1, (
        f"Binary label2id failed: expected index 1 ('entailment'), got {res_label2id} ('not_entailment')"
    )


# ===========================================================================
# 2. Corrupted / Malformed Metadata & File Handling
# ===========================================================================


def test_resolve_nli_entailment_corrupted_session_metadata() -> None:
    """Verify resolve_nli_entailment_idx does not crash when ONNX metadata is malformed."""
    # 1. Malformed JSON string in id2label
    session_bad_json = DynamicMockOnnxSession(
        [1.0, 2.0, 3.0],
        custom_metadata={"id2label": "{malformed json:"},
    )
    assert resolve_nli_entailment_idx(session=session_bad_json) == -1

    # 2. Non-numeric entailment_idx in metadata
    session_bad_idx = DynamicMockOnnxSession(
        [1.0, 2.0, 3.0],
        custom_metadata={"entailment_idx": "INVALID_NUMBER"},
    )
    assert resolve_nli_entailment_idx(session=session_bad_idx) == -1

    # 3. Session where get_modelmeta raises RuntimeError
    class CrashingSession:
        def get_modelmeta(self) -> Any:
            raise RuntimeError("Engine fault")

    assert resolve_nli_entailment_idx(session=CrashingSession()) == -1

    # 4. Session where get_modelmeta returns None
    class NoneMetaSession:
        def get_modelmeta(self) -> Any:
            return None

    assert resolve_nli_entailment_idx(session=NoneMetaSession()) == -1


def test_resolve_nli_entailment_corrupted_adjacent_configs(tmp_path: pathlib.Path) -> None:
    """Verify resolve_nli_entailment_idx does not crash on corrupt adjacent JSON configs."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy")

    cfg_file = tmp_path / "config.json"

    # Malformed JSON
    cfg_file.write_text("{\"label2id\": {broken json", encoding="utf-8")
    assert resolve_nli_entailment_idx(onnx_path=onnx_file) == -1

    # Binary garbage
    cfg_file.write_bytes(b"\x00\xff\xfe\x01\x02\x03")
    assert resolve_nli_entailment_idx(onnx_path=onnx_file) == -1

    # Empty file
    cfg_file.write_text("", encoding="utf-8")
    assert resolve_nli_entailment_idx(onnx_path=onnx_file) == -1

    # JSON with list instead of mapping
    cfg_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert resolve_nli_entailment_idx(onnx_path=onnx_file) == -1


def test_resolve_nli_entailment_corrupted_id2label_skips_valid_label2id() -> None:
    """Empirical observation: If id2label contains corrupted JSON, json.loads aborts the session loop.

    Even if label2id has valid mapping, the outer try-except catches the DecodeError and exits.
    """
    session = DynamicMockOnnxSession(
        [1.0, 2.0, 3.0],
        custom_metadata={
            "id2label": "{malformed json",
            "label2id": json.dumps({"neutral": 0, "entailment": 1}),
        },
    )
    idx = resolve_nli_entailment_idx(session=session)
    # EXPECTED CONTRACT: If id2label is corrupted but label2id has valid metadata, the resolver should recover index 1.
    # FAILURE: The outer try-except surrounding the key loop catches JSONDecodeError and aborts without checking label2id.
    assert idx == 1, (
        f"Corrupted id2label aborted key loop: expected index 1 from label2id, got {idx}"
    )


# ===========================================================================
# 3. Model Classes (1-Class, 2-Class, 3-Class, 5-Class) and Bounds Handling
# ===========================================================================


def test_create_onnx_scorer_1class_regression_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 1-class output models (e.g. single logit) handle clamping gracefully."""
    fake_ort = types.ModuleType("onnxruntime")
    mock_session = DynamicMockOnnxSession(output_logits=[2.45])
    fake_ort.InferenceSession = lambda path: mock_session  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # If explicit_idx=0 is passed: abs(0) < 1 -> idx = 0 -> returns 2.45
    scorer = create_onnx_nli_scorer("model.onnx", entailment_idx=0)
    score = scorer("Premise", "Query", "Candidate")
    assert score == pytest.approx(2.45)

    # If explicit_idx=2 is passed on 1-class model: abs(2) < 1 is False -> clamps to -1 -> flat[-1] is 2.45
    scorer_clamp = create_onnx_nli_scorer("model.onnx", entailment_idx=2)
    score_clamp = scorer_clamp("Premise", "Query", "Candidate")
    assert score_clamp == pytest.approx(2.45)


def test_create_onnx_scorer_2class_binary_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 2-class binary model with [contradiction_logit, entailment_logit]."""
    fake_ort = types.ModuleType("onnxruntime")
    # Logits: [-1.5, 3.2] -> class 0 is negative, class 1 is positive
    mock_session = DynamicMockOnnxSession(
        output_logits=[-1.5, 3.2],
        custom_metadata={"id2label": json.dumps({"0": "contradiction", "1": "entailment"})},
    )
    fake_ort.InferenceSession = lambda path: mock_session  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = create_onnx_nli_scorer("model.onnx")
    assert getattr(scorer, "entailment_idx") == 1
    score = scorer("Premise", "Query", "Candidate")
    assert score == pytest.approx(3.2)


def test_create_onnx_scorer_5class_finegrained_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 5-class model extracts correct entailment logit."""
    fake_ort = types.ModuleType("onnxruntime")
    # Logits for 5 classes: [-2.0, -1.0, 0.5, 4.25, 1.2]
    # Class 3 is 'entailment'
    mock_session = DynamicMockOnnxSession(
        output_logits=[-2.0, -1.0, 0.5, 4.25, 1.2],
        custom_metadata={
            "id2label": json.dumps({
                "0": "strongly_contradictory",
                "1": "contradictory",
                "2": "neutral",
                "3": "entailment",
                "4": "possible",
            })
        },
    )
    fake_ort.InferenceSession = lambda path: mock_session  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = create_onnx_nli_scorer("model.onnx")
    assert getattr(scorer, "entailment_idx") == 3
    score = scorer("Premise", "Query", "Candidate")
    assert score == pytest.approx(4.25)


def test_create_onnx_scorer_out_of_bounds_clamping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify out-of-bounds explicit indices clamp to -1 without crashing."""
    fake_ort = types.ModuleType("onnxruntime")
    # Logits: [10.0, 20.0, 30.0]
    mock_session = DynamicMockOnnxSession(output_logits=[10.0, 20.0, 30.0])
    fake_ort.InferenceSession = lambda path: mock_session  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # Index 99 is >= len(flat) -> clamped to -1 (last element: 30.0)
    scorer_oob_pos = create_onnx_nli_scorer("model.onnx", entailment_idx=99)
    assert scorer_oob_pos("P", "Q", "C") == pytest.approx(30.0)

    # Index -99 is < -len(flat) -> clamped to -1 (last element: 30.0)
    scorer_oob_neg = create_onnx_nli_scorer("model.onnx", entailment_idx=-99)
    assert scorer_oob_neg("P", "Q", "C") == pytest.approx(30.0)


# ===========================================================================
# 4. Moritz Laurer vs RoBERTa Strict Polarity Verification
# ===========================================================================


def test_moritz_laurer_polarity_across_all_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empirically assert that Moritz Laurer models strictly return positive scores

    for entailed pairs and negative scores for contradictory pairs across EN, FR, ES, PT.
    """
    fake_ort = types.ModuleType("onnxruntime")

    class MoritzSession:
        def __init__(self, path: str) -> None:
            self._meta = MockOnnxModelMetadata({
                "id2label": json.dumps({"0": "entailment", "1": "neutral", "2": "contradiction"})
            })

        def get_inputs(self) -> list[MockOnnxInput]:
            return [
                MockOnnxInput("input_ids"),
                MockOnnxInput("attention_mask"),
                MockOnnxInput("premise_text"),
                MockOnnxInput("hyp_text"),
            ]

        def get_modelmeta(self) -> MockOnnxModelMetadata:
            return self._meta

        def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
            # Mock logit output: index 0 is entailment (pos), index 2 is contradiction (neg)
            # Check candidate text
            candidate = input_feed.get("hyp_text", "").lower()
            premise = input_feed.get("premise_text", "").lower()

            # Billing match
            if ("factur" in premise or "bill" in premise or "cobrad" in premise or "fatur" in premise):
                if ("factur" in candidate or "bill" in candidate or "cobrad" in candidate or "fatur" in candidate):
                    # Entailment
                    return [np.array([[3.50, 0.10, -2.80]], dtype=np.float64)]
                else:
                    # Contradiction
                    return [np.array([[-2.80, 0.10, 3.50]], dtype=np.float64)]
            # Default
            return [np.array([[0.0, 0.0, 0.0]], dtype=np.float64)]

    fake_ort.InferenceSession = MoritzSession  # type: ignore[attr-defined]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda name, **kw: MockTokenizer(name)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = create_onnx_nli_scorer(
        onnx_path="dummy_moritz.onnx",
        tokenizer_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    )
    assert getattr(scorer, "entailment_idx") == 0

    pairs = [
        # (premise, correct_hyp, wrong_hyp)
        ("Double billing issue on credit card.", "It is billing: charges.", "It is shipping: parcels."),
        ("J'ai une question sur ma facturation.", "Il s'agit de facturation.", "Il s'agit de livraison."),
        ("Cobro indebido en mi factura.", "Se trata de facturación.", "Se trata de soporte técnico."),
        ("Problema com faturamento da conta.", "Trata-se de faturamento.", "Trata-se de entrega."),
    ]

    for premise, pos_hyp, neg_hyp in pairs:
        score_pos = scorer(premise, "Route", pos_hyp)
        score_neg = scorer(premise, "Route", neg_hyp)

        assert score_pos > 0.0, f"Entailed pair must be strictly positive, got {score_pos}"
        assert score_neg < 0.0, f"Contradictory pair must be strictly negative, got {score_neg}"
        assert score_pos > score_neg, f"Entailed score ({score_pos}) must exceed contradiction score ({score_neg})"


# ===========================================================================
# 5. Legacy Fallback Behavior (Metadata and Heuristics Absent)
# ===========================================================================


def test_legacy_fallback_when_all_signals_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify legacy fallback returns -1 when metadata, files, transformers, and heuristics are missing."""
    # Complete absence of signals
    idx = resolve_nli_entailment_idx(
        session=None,
        onnx_path=None,
        tokenizer_name=None,
    )
    assert idx == -1

    # Unrecognized names
    idx_unrec = resolve_nli_entailment_idx(
        session=None,
        onnx_path="custom_weights.onnx",
        tokenizer_name="custom_vendor/bert_custom",
    )
    assert idx_unrec == -1

    # Explicit override takes precedence over everything
    idx_override = resolve_nli_entailment_idx(
        session=None,
        onnx_path="roberta_model.onnx",
        tokenizer_name="roberta-large-mnli",
        explicit_idx=1,
    )
    assert idx_override == 1
