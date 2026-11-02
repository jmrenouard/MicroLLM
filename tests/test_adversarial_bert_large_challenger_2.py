"""Adversarial Challenge Suite for BERT Large Profile, Aliases, Entailment, and Offline Invariants.

Challenger 2 Empirical Verification Suite:
1. Alias Resolution: Asserts ALL aliases (`bert-large`, `bert-large-int8`, `bert-large-uncased`,
   `bert-large-mnli`, `bert-large-nli`, `bert-large-classifier`) resolve to `bert-large-int8`.
2. Dynamic Entailment Resolution: Tests classical BERT MNLI (index 2), Moritz Laurer (index 0),
   adjacent config resolution, negative prefix filtering, and unrecognized model fallback (-1).
3. Offline Bundle Determinism: Tests seed 506 mathematical invariance, SHA-256 manifest digest,
   and anti-tampering verification.
4. Offline Socket Isolation: Tests 100% offline execution under `guard_sockets()` with zero network calls,
   and verifies re-entrancy and socket interception.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import numpy as np
import pytest

from microllm.backends.base import (
    resolve_nli_entailment_idx,
    _find_entailment_from_mapping,
)
from microllm.hf_publishing import guard_sockets
from microllm.registry import (
    MODEL_PROFILES,
    PROFILE_ALIASES,
    load_profile_scorer,
    pull_profile,
    quantize_matrix_int8,
    verify_profile,
)


# ==============================================================================
# Challenge 1: Alias Resolution Invariants
# ==============================================================================

REQUIRED_BERT_LARGE_ALIASES = [
    "bert-large",
    "bert-large-int8",
    "bert-large-uncased",
    "bert-large-mnli",
    "bert-large-nli",
    "bert-large-classifier",
]


def test_adversarial_bert_large_all_aliases_registered() -> None:
    """Challenge: Verify that ALL 6 required aliases resolve to 'bert-large-int8'."""
    missing_aliases: list[str] = []
    mismatched_aliases: dict[str, str] = {}

    for alias in REQUIRED_BERT_LARGE_ALIASES:
        resolved = PROFILE_ALIASES.get(alias)
        if resolved is None:
            missing_aliases.append(alias)
        elif resolved != "bert-large-int8":
            mismatched_aliases[alias] = resolved

    assert not missing_aliases, (
        f"Missing aliases in PROFILE_ALIASES: {missing_aliases}. "
        f"Required: {REQUIRED_BERT_LARGE_ALIASES}"
    )
    assert not mismatched_aliases, (
        f"Mismatched aliases in PROFILE_ALIASES: {mismatched_aliases}"
    )


def test_adversarial_pull_profile_via_all_aliases(tmp_path: Path) -> None:
    """Challenge: Verify pull_profile succeeds for each required alias."""
    for alias in REQUIRED_BERT_LARGE_ALIASES:
        cache_subdir = tmp_path / f"cache_{alias}"
        manifest = pull_profile(alias, cache_dir=cache_subdir, download_weights=False)
        assert manifest["profile"] == "bert-large-int8", (
            f"Alias '{alias}' resolved to unexpected profile '{manifest.get('profile')}'"
        )
        assert (cache_subdir / "bert-large-int8" / "model_quantized_int8.npz").is_file()


def test_adversarial_load_profile_scorer_via_all_aliases(tmp_path: Path) -> None:
    """Challenge: Verify load_profile_scorer returns callable for each required alias."""
    for alias in REQUIRED_BERT_LARGE_ALIASES:
        cache_subdir = tmp_path / f"scorer_cache_{alias}"
        scorer = load_profile_scorer(alias, cache_dir=cache_subdir)
        assert callable(scorer), f"load_profile_scorer failed for alias '{alias}'"
        res = scorer("State prompt", "Query prompt", "Hypothesis candidate")
        assert isinstance(res, float)


# ==============================================================================
# Challenge 2: Dynamic Entailment Index Resolution Invariants
# ==============================================================================


def test_adversarial_dynamic_entailment_classical_bert_mnli() -> None:
    """Challenge: Classical BERT MNLI models must resolve to index 2."""
    classical_models = [
        "bert-large-uncased",
        "bert-large-uncased-mnli",
        "yoshitomo-matsubara/bert-large-uncased-mnli",
        "bert-base-uncased",
        "roberta-large-mnli",
        "facebook/bart-large-mnli",
    ]
    for model_name in classical_models:
        idx = resolve_nli_entailment_idx(tokenizer_name=model_name)
        assert idx == 2, f"Model '{model_name}' expected index 2, got {idx}"


def test_adversarial_dynamic_entailment_moritz_laurer() -> None:
    """Challenge: Moritz Laurer zero-shot models must resolve to index 0."""
    moritz_models = [
        "MoritzLaurer/DeBERTa-v3-base-mnli-xnli",
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    ]
    for model_name in moritz_models:
        idx = resolve_nli_entailment_idx(tokenizer_name=model_name)
        assert idx == 0, f"Model '{model_name}' expected index 0, got {idx}"


def test_adversarial_dynamic_entailment_unrecognized_models() -> None:
    """Challenge: Unrecognized models must fall back to -1 without false-positive collision."""
    unrecognized_models = [
        "random-vendor/custom-nli-head",
        "custom_vendor/bert_custom",
        "google/flan-t5-base",
        "meta-llama/Llama-3-8B",
        "gpt2",
        "",
        None,
    ]
    for model_name in unrecognized_models:
        idx = resolve_nli_entailment_idx(tokenizer_name=model_name)
        assert idx == -1, f"Model '{model_name}' expected index -1, got {idx}"


def test_adversarial_dynamic_entailment_explicit_override() -> None:
    """Challenge: Explicit index override must always take precedence over heuristics."""
    assert resolve_nli_entailment_idx(tokenizer_name="bert-large-uncased", explicit_idx=0) == 0
    assert resolve_nli_entailment_idx(tokenizer_name="MoritzLaurer/DeBERTa-v3-base-mnli-xnli", explicit_idx=2) == 2
    assert resolve_nli_entailment_idx(tokenizer_name="random-vendor/custom", explicit_idx=1) == 1


def test_adversarial_dynamic_entailment_adjacent_config(tmp_path: Path) -> None:
    """Challenge: Adjacent config.json mapping takes priority over model name heuristics."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.touch()
    cfg_path = tmp_path / "config.json"

    # Config specifies entailment at index 1
    cfg_path.write_text(
        json.dumps({"label2id": {"contradiction": 0, "entailment": 1, "neutral": 2}}),
        encoding="utf-8",
    )
    # Even though tokenizer name suggests bert-large (heuristic=2), config mapping should yield 1
    idx = resolve_nli_entailment_idx(onnx_path=fake_onnx, tokenizer_name="bert-large-uncased")
    assert idx == 1, f"Expected adjacent config index 1, got {idx}"


def test_adversarial_dynamic_entailment_negative_prefix_filtering() -> None:
    """Challenge: Contrastive labels like 'not_entailment' must not trigger false positive entailment."""
    mapping = {"0": "not_entailment", "1": "entailment"}
    idx = _find_entailment_from_mapping(mapping)
    assert idx == 1, f"Expected index 1 for pure entailment, got {idx}"

    mapping_reverse = {"0": "entailment", "1": "non_entailment"}
    idx2 = _find_entailment_from_mapping(mapping_reverse)
    assert idx2 == 0, f"Expected index 0, got {idx2}"


# ==============================================================================
# Challenge 3: Offline Bundle Generation (Seed 506 and SHA-256)
# ==============================================================================


def test_adversarial_pull_profile_seed_506_mathematical_determinism(tmp_path: Path) -> None:
    """Challenge: Verify pull_profile generates exact seed 506 projection weights."""
    manifest = pull_profile("bert-large-int8", cache_dir=tmp_path, download_weights=False)
    assert manifest["profile"] == "bert-large-int8"
    assert manifest["installed"] is True
    assert manifest["entailment_idx"] == 2

    npz_path = Path(manifest["quantized_bundle"])
    assert npz_path.is_file()

    # Reconstruct ground-truth array from seed 506
    rng = np.random.default_rng(506)
    expected_fp32 = rng.normal(0.0, 0.35, size=(256, 64))
    expected_q = quantize_matrix_int8(expected_fp32)

    with np.load(npz_path) as data:
        weights_int8 = data["weights_int8"]
        scale = data["scale"]
        per_k_temps = data["per_k_temperatures"]

        assert weights_int8.shape == (256, 64)
        assert weights_int8.dtype == np.int8
        np.testing.assert_array_equal(weights_int8, expected_q["weights_int8"])
        assert np.isclose(scale[0], expected_q["scale"])
        np.testing.assert_array_equal(per_k_temps, np.array([0.75, 0.80, 0.85, 0.88, 0.95]))


def test_adversarial_pull_profile_sha256_manifest_and_tamper_detection(tmp_path: Path) -> None:
    """Challenge: SHA-256 digest in manifest must match actual bundle and detect tampering."""
    manifest = pull_profile("bert-large-int8", cache_dir=tmp_path, download_weights=False)
    npz_path = Path(manifest["quantized_bundle"])

    actual_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert manifest["sha256_checksum"] == actual_sha

    # Verification function passes
    res = verify_profile("bert-large-int8", cache_dir=tmp_path)
    assert res["verified"] is True
    assert res["sha256_match"] is True

    # Tamper with 1 byte in the .npz bundle
    raw_bytes = bytearray(npz_path.read_bytes())
    raw_bytes[100] = (raw_bytes[100] + 1) % 256
    npz_path.write_bytes(bytes(raw_bytes))

    # Verification function MUST detect tampering
    tampered_res = verify_profile("bert-large-int8", cache_dir=tmp_path)
    assert tampered_res["verified"] is False
    assert tampered_res["sha256_match"] is False


# ==============================================================================
# Challenge 4: Offline Socket Isolation under guard_sockets()
# ==============================================================================


def test_adversarial_offline_socket_isolation(tmp_path: Path) -> None:
    """Challenge: 100% offline workflow executes with zero socket calls under guard_sockets()."""
    with guard_sockets():
        manifest = pull_profile("bert-large-int8", cache_dir=tmp_path, download_weights=False)
        assert manifest["installed"] is True

        v = verify_profile("bert-large-int8", cache_dir=tmp_path)
        assert v["verified"] is True

        scorer = load_profile_scorer("bert-large-int8", cache_dir=tmp_path)
        assert callable(scorer)

        # Evaluate various ticket scenarios
        score_billing = scorer("Customer duplicate billing transaction", "Classify", "Billing issue")
        score_tech = scorer("Customer duplicate billing transaction", "Classify", "Hardware defect")
        assert score_billing > score_tech

        # Edge cases: empty strings, long strings
        empty_score = scorer("", "", "")
        assert isinstance(empty_score, float)

        long_state = "Invoice line item " * 200
        long_score = scorer(long_state, "Question", "Billing")
        assert isinstance(long_score, float)


def test_adversarial_guard_sockets_blocks_network() -> None:
    """Challenge: guard_sockets() intercepts and blocks real socket connection attempts."""
    with pytest.raises(RuntimeError, match="Socket connection blocked by 100% offline isolation"):
        with guard_sockets():
            sock = socket.socket()
            sock.connect(("1.1.1.1", 80))


def test_adversarial_guard_sockets_reentrancy() -> None:
    """Challenge: guard_sockets() supports nested re-entrant contexts cleanly."""
    orig_connect = socket.socket.connect
    with guard_sockets():
        # First level
        with pytest.raises(RuntimeError):
            socket.socket().connect(("1.1.1.1", 80))

        with guard_sockets():
            # Nested second level
            with pytest.raises(RuntimeError):
                socket.socket().connect(("1.1.1.1", 80))

        # Back to first level
        with pytest.raises(RuntimeError):
            socket.socket().connect(("1.1.1.1", 80))

    # After exit, socket.socket.connect must be restored to original
    assert socket.socket.connect is orig_connect
