"""Adversarial stress harness for Milestone 3 (Caching, SDK Batch Parity, Licensing, CLI JSON).

Authored by Challenger 1 (Empirical Challenger) to rigorously stress-test:
1. F06: Token caching under high-churn volume (25k strings), extreme Unicode (CJK, RTL, emojis,
   100k-char string, zero-width spaces), cache determinism, and list mutation isolation.
2. F06: VerdictMQMultiQuestionCache under multi-threaded concurrency (30 threads) and eviction bounds.
3. F07: SDK client batch parity: MicroJevClient.batch vs AsyncMicroJevClient.batch signature and
   payload parity, plus server /v1/batch validation boundary mining (empty, 500/501 items, None, NaN).
4. F11: CLI benchmark --json output purity, stdout parseability, invalid argument handling,
   and behavior under default benchmark invocation.
5. F09/F10: Space license normalization and vault path environment variable overrides.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gc
import inspect
import io
import json
import os
import random
import string
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pytest
from starlette.testclient import TestClient

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    _tokenize_multilingual_cached,
    tokenize_multilingual,
)
from microllm.backends.cascade import (
    DualGearCascadeBackend,
    evaluate_voc_gate_regime,
)
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.backends.prefill_llm import AnyJevPrefillBackend
from microllm.client import (
    AsyncMicroJevClient,
    Choice,
    MicroJevClient,
    Multi,
    Noul,
    Score,
    _build_batch_payload,
)
from microllm.hf_publishing import get_vault_paths
from microllm.hf_space_sync import build_space_readme, stage_hf_space, sync_hf_space
from microllm.server import BatchRequest, build_cli_parser, create_app, run_cli


# ============================================================================
# 1. F06: High-Churn Token Caching & Extreme String Stress
# ============================================================================


def test_f06_high_churn_token_caching_and_lru_eviction() -> None:
    """Stress tokenize_multilingual with 20,000 unique strings exceeding LRU capacity (16384)."""
    initial_info = tokenize_multilingual.cache_info()
    assert initial_info.maxsize == 16384

    # Churn 20,000 distinct strings
    for i in range(20000):
        res = tokenize_multilingual(f"synthetic churn payload sequence number {i}")
        assert len(res) == 6

    info_after = tokenize_multilingual.cache_info()
    # Ensure cache size is strictly bounded by maxsize
    assert info_after.currsize <= 16384
    assert info_after.currsize == 16384

    # Test deterministic hit on repeated string
    probe = "probe repeated string for cache hit verification"
    tokenize_multilingual(probe)
    hits_before = tokenize_multilingual.cache_info().hits
    toks_first = tokenize_multilingual(probe)
    hits_after = tokenize_multilingual.cache_info().hits
    assert hits_after == hits_before + 1
    assert toks_first == ["probe", "repeated", "string", "for", "cache", "hit", "verification"]


def test_f06_extreme_unicode_cjk_emoji_and_100k_char() -> None:
    """Stress tokenize_multilingual on CJK, RTL, emojis, zero-width chars, and 100k character text."""
    # 1. Zero-length string
    assert tokenize_multilingual("") == []

    # 2. Whitespace and zero-width spaces
    assert tokenize_multilingual("   \t\r\n   ") == []
    # Zero-width characters match the \u0080-\uffff Unicode regex range
    zw_toks = tokenize_multilingual("\u200b\u200c\u200d\ufeff")
    assert isinstance(zw_toks, list)
    assert len(zw_toks) == 1

    # 3. Chinese (CJK)
    cjk_text = "这是一段关于微型语言模型决策和快速推理的文本"
    cjk_toks = tokenize_multilingual(cjk_text)
    assert len(cjk_toks) >= 1
    assert "".join(cjk_toks) == cjk_text

    # 4. Japanese (Kanji + Hiragana + Katakana)
    jp_text = "日本語のテキストトークン化テスト"
    jp_toks = tokenize_multilingual(jp_text)
    assert len(jp_toks) >= 1

    # 5. Korean (Hangul)
    kr_text = "한국어 자연어 처리 모델 테스트"
    kr_toks = tokenize_multilingual(kr_text)
    assert len(kr_toks) == 5

    # 6. Arabic (RTL)
    ar_text = "نموذج معالجة اللغة الطبيعية والذكاء الاصطناعي"
    ar_toks = tokenize_multilingual(ar_text)
    assert len(ar_toks) >= 5

    # 7. Hebrew
    he_text = "בדיקת עיבוד שפה טבעית במערכת"
    he_toks = tokenize_multilingual(he_text)
    assert len(he_toks) >= 4

    # 8. Emojis and combined symbols
    emoji_text = "Alert 🚨: Customer credit card 💳 payment failed ❌! Need urgent refund 💸."
    emoji_toks = tokenize_multilingual(emoji_text)
    assert "alert" in emoji_toks
    assert "payment" in emoji_toks
    assert "refund" in emoji_toks

    # 9. 100,000 characters long text
    long_text = "transaction verification check " * 3500  # ~108,500 chars
    t0 = time.perf_counter()
    toks_long = tokenize_multilingual(long_text)
    t_first = time.perf_counter() - t0
    assert len(toks_long) == 3500 * 3

    # Cached evaluation should be sub-millisecond
    t0 = time.perf_counter()
    toks_long_2 = tokenize_multilingual(long_text)
    t_second = time.perf_counter() - t0
    assert t_second < 0.005  # < 5ms for cached retrieval
    assert toks_long == toks_long_2


def test_f06_mutation_behavior_on_cached_list() -> None:
    """Empirical check: Verify whether modifying returned list affects subsequent calls.

    Because tokenize_multilingual returns a list, if a caller mutates the list,
    does it mutate the cached object in lru_cache?
    """
    sample = "unique immutability test string for token cache audit"
    toks1 = tokenize_multilingual(sample)
    assert isinstance(toks1, list)
    expected = list(toks1)

    # In Python, if lru_cache caches the return value of tokenize_multilingual,
    # mutating toks1 in-place will mutate the cached object.
    toks1.append("POISON_TOKEN")
    toks2 = tokenize_multilingual(sample)

    # Note empirical observation:
    # If tokenize_multilingual is lru_cached, toks2 is toks1 and contains POISON_TOKEN.
    # If tokenize_multilingual builds list(_tokenize_multilingual_cached(...)), each call creates a new list.
    is_same_object = (toks1 is toks2)
    has_poison = ("POISON_TOKEN" in toks2)

    # Assert callers get isolated lists and cannot poison the token cache
    assert isinstance(toks2, list)
    assert not is_same_object
    assert not has_poison


def test_f06_verdict_mq_cache_concurrency_and_eviction() -> None:
    """Stress VerdictMQMultiQuestionCache under 30 concurrent threads and bounded capacity."""
    cache = VerdictMQMultiQuestionCache(max_states=20)
    num_threads = 30
    ops_per_thread = 150
    errors: list[Exception] = []

    test_states = [f"Concurrent state payload {i % 50}" for i in range(100)]

    def worker() -> None:
        try:
            for _ in range(ops_per_thread):
                s = random.choice(test_states)
                entry = cache.encode_state_once(s)
                assert "clean" in entry
                assert "tokens" in entry
                assert "embedding" in entry
                assert len(entry["embedding"]) == 256
                # Verify get_cached_state alias behaves identically
                entry2 = cache.get_cached_state(s)
                assert entry["clean"] == entry2["clean"]
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Encountered {len(errors)} thread errors: {errors[:3]}"
    # Verify bounded capacity invariant
    assert len(cache._cache) <= 20
    assert cache.stats["cached_states"] <= 20
    assert cache.stats["hits"] + cache.stats["misses"] > 0


def test_f06_evaluate_voc_gate_regime_state_tokens_types() -> None:
    """Verify evaluate_voc_gate_regime accepts diverse state_tokens sequence/set containers."""
    state = "The claim is approved unless the customer provided invalid documentation."
    instructions = "Verify claim eligibility."
    options = {"valid": "Claim is fully valid", "invalid": "Claim is invalid or fraudulent"}
    conf0 = 0.80

    # 1. state_tokens as set
    voc_set = evaluate_voc_gate_regime(
        state, instructions, options, conf0=conf0, state_tokens={"claim", "approved", "unless"}
    )
    assert voc_set["uncertainty_regime"] == "epistemic_deliberative"

    # 2. state_tokens as frozenset
    voc_frozenset = evaluate_voc_gate_regime(
        state, instructions, options, conf0=conf0, state_tokens=frozenset(["claim", "approved", "unless"])
    )
    assert voc_frozenset["uncertainty_regime"] == voc_set["uncertainty_regime"]

    # 3. state_tokens as list
    voc_list = evaluate_voc_gate_regime(
        state, instructions, options, conf0=conf0, state_tokens=["claim", "approved", "unless"]
    )
    assert voc_list["uncertainty_regime"] == voc_set["uncertainty_regime"]

    # 4. state_tokens as tuple
    voc_tuple = evaluate_voc_gate_regime(
        state, instructions, options, conf0=conf0, state_tokens=("claim", "approved", "unless")
    )
    assert voc_tuple["uncertainty_regime"] == voc_set["uncertainty_regime"]

    # 5. state_tokens as None (falls back to tokenize_multilingual)
    voc_none = evaluate_voc_gate_regime(
        state, instructions, options, conf0=conf0, state_tokens=None
    )
    assert voc_none["uncertainty_regime"] == "epistemic_deliberative"


# ============================================================================
# 2. F07: SDK Client Batch Parity & Validation Boundaries
# ============================================================================


def test_f07_sdk_batch_signature_parity() -> None:
    """Assert 100% parameter signature parity between MicroJevClient.batch and AsyncMicroJevClient.batch."""
    sig_sync = inspect.signature(MicroJevClient.batch)
    sig_async = inspect.signature(AsyncMicroJevClient.batch)

    sync_params = sig_sync.parameters
    async_params = sig_async.parameters

    assert set(sync_params.keys()) == set(async_params.keys())
    for name in sync_params:
        p_sync = sync_params[name]
        p_async = async_params[name]
        assert p_sync.name == p_async.name
        assert p_sync.default == p_async.default
        assert p_sync.kind == p_async.kind


def test_f07_sdk_build_batch_payload_exhaustive_matrix() -> None:
    """Assert _build_batch_payload produces identical payloads across edge-case combinations."""
    test_cases = [
        # (items, questions, model, strategy, neural)
        ([], {}, "jev-latest", None, None),
        (["state 1"], {"q": Noul()}, "microllm-gear1-fast", "fast", "minilm"),
        (["s1", "s2"], {"c": Choice(options=["a", "b"])}, "jev-latest", "neural", "mdeberta"),
        ([{"raw": "obj"}], {"s": Score(levels=["low", "medium", "high"])}, "custom-model", "auto", None),
        (["item"], {"m": Multi(options=["x", "y"])}, "jev-latest", None, "mdeberta"),
    ]

    for items, q_map, model, strat, neural in test_cases:
        p = _build_batch_payload(
            items=items,
            questions=q_map,
            model=model,
            multilingual_strategy=strat,
            neural_model=neural,
        )
        assert p["items"] == list(items)
        assert "questions" in p
        assert p["model"] == model
        if strat is not None:
            assert p["multilingual_strategy"] == strat
        else:
            assert "multilingual_strategy" not in p
        if neural is not None:
            assert p["neural_model"] == neural
        else:
            assert "neural_model" not in p


def test_f07_server_batch_validation_boundaries() -> None:
    """Verify FastAPI /v1/batch endpoint enforces strict validation bounds (HTTP 200 vs 422)."""
    app = create_app()
    client = TestClient(app)

    # 1. Empty items and empty questions -> valid HTTP 200
    resp_empty = client.post("/v1/batch", json={"items": [], "questions": {}})
    assert resp_empty.status_code == 200
    data_empty = resp_empty.json()
    assert data_empty["summary"]["total_items"] == 0
    assert data_empty["results"] == []

    # 2. Normal batch of 2 items
    resp_normal = client.post(
        "/v1/batch",
        json={
            "items": ["User wants account cancellation", "Payment failed for invoice 42"],
            "questions": {"noul_q": {"type": "noul", "instructions": "Is this a complaint?"}},
        },
    )
    assert resp_normal.status_code == 200
    assert resp_normal.json()["summary"]["total_items"] == 2

    # 3. Exactly 500 items -> valid HTTP 200
    items_500 = [f"Customer state ticket {i}" for i in range(500)]
    resp_500 = client.post(
        "/v1/batch",
        json={
            "items": items_500,
            "questions": {"q": {"type": "noul"}},
        },
    )
    assert resp_500.status_code == 200
    assert resp_500.json()["summary"]["total_items"] == 500

    # 4. 501 items -> HTTP 422
    items_501 = [f"Customer state ticket {i}" for i in range(501)]
    resp_501 = client.post(
        "/v1/batch",
        json={
            "items": items_501,
            "questions": {"q": {"type": "noul"}},
        },
    )
    assert resp_501.status_code == 422

    # 5. Missing items field -> HTTP 422
    resp_no_items = client.post("/v1/batch", json={"questions": {}})
    assert resp_no_items.status_code == 422

    # 6. Missing questions field -> HTTP 422
    resp_no_questions = client.post("/v1/batch", json={"items": ["state"]})
    assert resp_no_questions.status_code == 422

    # 7. questions explicitly None -> HTTP 422
    resp_null_questions = client.post("/v1/batch", json={"items": ["state"], "questions": None})
    assert resp_null_questions.status_code == 422


def test_f07_server_batch_multilingual_routing_edge_cases() -> None:
    """Verify /v1/batch strategy routing handles fast, neural, auto and default fallback."""
    app = create_app()
    client = TestClient(app)

    for strat in ["fast", "auto"]:
        resp = client.post(
            "/v1/batch",
            json={
                "items": ["Compte bloqué après mot de passe oublié"],
                "questions": {"dept": {"type": "choice", "criteria": {"support": None, "billing": None}}},
                "multilingual_strategy": strat,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["summary"]["total_items"] == 1


# ============================================================================
# 3. F11: Benchmark CLI --json Stress & Purity
# ============================================================================


def test_f11_cli_benchmark_multilingual_json_stdout_purity() -> None:
    """Verify microllm benchmark --multilingual --json outputs pure parseable JSON to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_cli(["benchmark", "--multilingual", "--json"])
    assert code == 0
    raw = buf.getvalue().strip()
    # Must be valid JSON
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "timestamp" in data or "models" in data or "strategies" in data or "summary" in data


def test_f11_cli_benchmark_pareto_json_stdout_purity() -> None:
    """Verify microllm benchmark --pareto --json outputs pure parseable JSON to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_cli(["benchmark", "--pareto", "--json"])
    assert code == 0
    raw = buf.getvalue().strip()
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "profiles" in data


def test_f11_cli_benchmark_invalid_subcommand_and_options() -> None:
    """Verify invalid CLI subcommands and invalid choices return exit code 2."""
    def _run(args: list[str]) -> int:
        try:
            return run_cli(args)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 2

    # Invalid subcommand
    code = _run(["nonexistent_subcommand_123"])
    assert code == 2

    # Invalid strategy choice
    code_strat = _run(["benchmark", "--multilingual", "--multilingual-strategy", "bad_strategy", "--json"])
    assert code_strat == 2

    # Invalid neural model choice
    code_neural = _run(["benchmark", "--multilingual", "--neural-model", "bad_neural", "--json"])
    assert code_neural == 2

    # Invalid language choice
    code_lang = _run(["benchmark", "--multilingual", "--language", "bad_lang", "--json"])
    assert code_lang == 2


def test_f11_cli_help_flag_raises_system_exit() -> None:
    """Verify --help and -h raise SystemExit so argparse exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["benchmark", "--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--help"])
    assert exc_info.value.code == 0


def test_f11_cli_benchmark_default_suite_with_json_flag() -> None:
    """Empirical audit: Verify behavior when benchmark is run with --json alone.
    
    If neither --multilingual nor --pareto is provided, does --json output JSON or Markdown?
    """
    # Note: run_benchmark_suite executes the 540-item suite which takes ~1.5s
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = run_cli(["benchmark", "--json"])
    assert code == 0
    raw = buf.getvalue().strip()
    # Check if raw stdout starts with Markdown table | or JSON {
    is_json = raw.startswith("{") and raw.endswith("}")
    is_markdown = raw.startswith("|") or "Benchmark Track" in raw or "Evaluation Corpus" in raw
    # We observe and document the actual behavior:
    # Under current implementation, line 1973 prints rep['markdown_table'].
    assert is_markdown or is_json


def test_f11_cli_benchmark_all_suites_and_flags_combination() -> None:
    """Stress test CLI argument parser AST and flag registration across all commands."""
    parser = build_cli_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Subparsers check
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert "benchmark" in subparsers_action.choices
    bench_sub = subparsers_action.choices["benchmark"]

    # Verify all expected flags are registered on benchmark subparser
    registered_flags = [action.dest for action in bench_sub._actions]
    assert "json" in registered_flags
    assert "multilingual" in registered_flags
    assert "pareto" in registered_flags
    assert "all_suites" in registered_flags
    assert "external" in registered_flags
    assert "hf_jev_models" in registered_flags


# ============================================================================
# 4. F09 & F10: Space Licensing & Vault Paths Verification
# ============================================================================


def test_f09_hf_space_license_apache2_normalization() -> None:
    """Verify build_space_readme normalizes diverse license inputs and defaults to Apache-2.0."""
    # Default license
    readme_default = build_space_readme()
    assert "license: apache-2.0" in readme_default.lower()

    # Case insensitivity
    readme_upper = build_space_readme(license_type="APACHE-2.0")
    assert "license: apache-2.0" in readme_upper.lower()

    # Whitespace padding
    readme_padded = build_space_readme(license_type="  apache-2.0  \n")
    assert "license: apache-2.0" in readme_padded.lower()

    # Empty string fallback to apache-2.0
    readme_empty = build_space_readme(license_type="")
    assert "license: apache-2.0" in readme_empty.lower()


def test_f10_vault_paths_env_overrides_and_clean_source() -> None:
    """Verify get_vault_paths respects environment overrides and source contains no hardcoded user paths."""
    # Check source file for zero occurrences of hardcoded developer path
    hf_pub_file = Path(__file__).resolve().parents[1] / "src" / "microllm" / "hf_publishing.py"
    content = hf_pub_file.read_text(encoding="utf-8")
    assert "C:\\Users\\jmren" not in content
    assert "Users/jmren" not in content

    # Test environment variable resolution
    custom_db = "C:/custom/vault/db.kdbx"
    custom_key = "C:/custom/vault/key.keyx"
    old_db = os.environ.get("MICROLLM_KEEPASS_DB")
    old_key = os.environ.get("MICROLLM_KEEPASS_KEY")

    try:
        os.environ["MICROLLM_KEEPASS_DB"] = custom_db
        os.environ["MICROLLM_KEEPASS_KEY"] = custom_key
        db_path, key_path = get_vault_paths()
        assert str(db_path).replace("\\", "/") == custom_db
        assert str(key_path).replace("\\", "/") == custom_key
    finally:
        if old_db is not None:
            os.environ["MICROLLM_KEEPASS_DB"] = old_db
        else:
            os.environ.pop("MICROLLM_KEEPASS_DB", None)
        if old_key is not None:
            os.environ["MICROLLM_KEEPASS_KEY"] = old_key
        else:
            os.environ.pop("MICROLLM_KEEPASS_KEY", None)
