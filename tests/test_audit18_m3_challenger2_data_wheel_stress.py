"""Adversarial Data Integrity, Wheel Packaging, and Caching Concurrency Stress Suite for Milestone 3.

Authored by Empirical Challenger M3-2.
Focus:
- Static JSON Modularization & Data Integrity (Point 16 / Feature F16).
- Wheel Build & Asset Packaging Integrity (Point 16 / pyproject.toml package-data).
- Adversarial and Multi-threaded Caching Concurrency for Verdict-MQ (Point 12 & Challenger 2 Edge-Case Fix).
"""

from __future__ import annotations

import concurrent.futures
import functools
import importlib.resources
import json
from pathlib import Path
import random
import threading
from typing import Any
import zipfile

import pytest

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    semantic_match_logit,
)
import microllm.dataset_fixtures as dataset_fixtures
import microllm.gradio_presets as gradio_presets
import microllm.hf_jev_datasets as hf_jev
import microllm.papers as papers


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "microllm" / "data"

EXPECTED_JSON_METADATA: dict[str, dict[str, Any]] = {
    "academic_papers_catalog.json": {"type": list, "min_count": 100, "exact_count": 104},
    "literature_pillars.json": {"type": list, "min_count": 6, "exact_count": 6},
    "hf_jev_models_catalog.json": {"type": list, "min_count": 25, "exact_count": 25},
    "hf_jev_server_config_map.json": {"type": dict, "min_count": 12, "exact_count": 12},
    "hf_jev_offline_samples.json": {"type": dict, "min_count": 12, "exact_count": 12},
    "gradio_multilingual_presets.json": {"type": dict, "min_count": 12, "exact_count": 12},
    "gradio_domain_presets.json": {"type": dict, "min_count": 30, "exact_count": 32},
    "gradio_dataset_presets.json": {"type": dict, "min_count": 10, "exact_count": 10},
    "gradio_consumer_examples.json": {"type": dict, "min_count": 4, "exact_count": 4},
    "dataset_fixtures_offline_samples.json": {"type": dict, "min_count": 16, "exact_count": 16},
}


# =============================================================================
# 1. Static JSON Modularization & Data Integrity Challenge
# =============================================================================


class TestStaticJsonIntegrityChallenge:
    """Empirical verification of the 10 extracted JSON data files under src/microllm/data/."""

    def test_all_expected_files_exist_with_init(self) -> None:
        """Verify presence of __init__.py and all 10 JSON files."""
        assert DATA_DIR.is_dir(), f"Missing data directory: {DATA_DIR}"
        init_path = DATA_DIR / "__init__.py"
        assert init_path.is_file(), f"Missing __init__.py in {DATA_DIR}"

        for fname in EXPECTED_JSON_METADATA:
            fpath = DATA_DIR / fname
            assert fpath.is_file(), f"Missing required JSON data file: {fname}"
            assert fpath.stat().st_size > 50, f"JSON file {fname} is suspiciously small: {fpath.stat().st_size} bytes"

    def test_json_strict_utf8_decoding_and_parsing(self) -> None:
        """Verify each JSON file strictly decodes under UTF-8 and parses with valid syntax."""
        for fname, meta in EXPECTED_JSON_METADATA.items():
            fpath = DATA_DIR / fname
            raw_bytes = fpath.read_bytes()

            # Strict UTF-8 decoding test (raises UnicodeDecodeError if invalid)
            text = raw_bytes.decode("utf-8")
            assert len(text) > 0

            # Strict JSON syntax test
            parsed = json.loads(text)
            assert isinstance(parsed, meta["type"]), (
                f"{fname} parsed into {type(parsed)}, expected {meta['type']}"
            )
            assert len(parsed) == meta["exact_count"], (
                f"{fname} has length {len(parsed)}, expected {meta['exact_count']}"
            )

    def test_module_constants_match_raw_json_identically(self) -> None:
        """Verify 100% exact equality between loaded module constants and raw JSON data."""
        module_mappings = [
            ("academic_papers_catalog.json", papers.ACADEMIC_PAPERS_CATALOG),
            ("literature_pillars.json", papers.LITERATURE_PILLARS),
            ("hf_jev_models_catalog.json", hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG),
            ("hf_jev_server_config_map.json", hf_jev.HF_JEV_DATASET_SERVER_CONFIG_MAP),
            ("hf_jev_offline_samples.json", hf_jev.OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES),
            ("gradio_multilingual_presets.json", gradio_presets.MULTILINGUAL_SUPPORT_PRESETS),
            ("gradio_domain_presets.json", gradio_presets.STUDIO_DOMAIN_PRESETS),
            ("gradio_dataset_presets.json", gradio_presets.DATASET_BENCHMARK_PRESETS),
            ("gradio_consumer_examples.json", gradio_presets.CONSUMER_TYPE_EXAMPLES),
            ("dataset_fixtures_offline_samples.json", dataset_fixtures.OFFLINE_WEB_DATASET_SAMPLES),
        ]

        for fname, constant_obj in module_mappings:
            fpath = DATA_DIR / fname
            with open(fpath, "r", encoding="utf-8") as f:
                raw_json_obj = json.load(f)
            assert raw_json_obj == constant_obj, (
                f"Discrepancy detected between {fname} and module constant!"
            )

    def test_academic_papers_catalog_internal_schema(self) -> None:
        """Stress-test academic papers records for mandatory schema keys."""
        mandatory_keys = {"arxiv_id", "title", "authors", "summary", "microllm_relevance", "pillar_id"}
        for p in papers.ACADEMIC_PAPERS_CATALOG:
            missing = mandatory_keys - set(p.keys())
            assert not missing, f"Paper {p.get('arxiv_id')} missing keys: {missing}"
            assert isinstance(p["authors"], list) and len(p["authors"]) > 0
            assert len(p["title"].strip()) > 5

    def test_hf_jev_models_catalog_internal_schema(self) -> None:
        """Stress-test HF JEV models records for mandatory schema keys."""
        mandatory_keys = {"model_id", "architecture", "focus", "datasets", "primitives"}
        for m in hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG:
            missing = mandatory_keys - set(m.keys())
            assert not missing, f"Model {m.get('model_id')} missing keys: {missing}"
            assert isinstance(m["datasets"], list)
            assert isinstance(m["primitives"], list)

    def test_immutability_and_safe_copy_isolation(self) -> None:
        """Verify calling helper filter functions returns decoupled copies that do not mutate catalogs."""
        original_count = len(papers.ACADEMIC_PAPERS_CATALOG)
        papers_list = papers.list_academic_papers()
        assert len(papers_list) == original_count

        # Mutate the returned list
        papers_list.pop(0)
        assert len(papers_list) == original_count - 1
        assert len(papers.ACADEMIC_PAPERS_CATALOG) == original_count, (
            "Mutating list_academic_papers() return value corrupted papers.ACADEMIC_PAPERS_CATALOG!"
        )

        # Mutate an entry's field in the returned list
        paper_copy = papers.get_academic_paper("2609.38850")
        assert paper_copy is not None
        original_title = paper_copy["title"]
        paper_copy["title"] = "MUTATED ADVERSARIAL TITLE"
        fresh_copy = papers.get_academic_paper("2609.38850")
        assert fresh_copy is not None
        assert fresh_copy["title"] == original_title, (
            "Mutating get_academic_paper() return value corrupted internal entry!"
        )

        # Same for HF JEV models list
        orig_hf_count = len(hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG)
        hf_models_list = hf_jev.list_hf_jev_ecosystem_models()
        hf_models_list.clear()
        assert len(hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG) == orig_hf_count, (
            "Mutating list_hf_jev_ecosystem_models() return value corrupted HF_JEV_ECOSYSTEM_MODELS_CATALOG!"
        )


# =============================================================================
# 2. Wheel Packaging Challenge
# =============================================================================


class TestWheelPackagingChallenge:
    """Empirical verification that all 10 data files and __init__.py are included in the wheel."""

    def test_wheel_archive_contains_all_ten_json_data_files(self) -> None:
        """Inspect generated wheel in dist/ and verify presence of all 10 JSON files."""
        import subprocess
        import sys

        dist_dir = REPO_ROOT / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            try:
                cmd = [sys.executable, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "-w", str(dist_dir)]
                subprocess.run(cmd, check=True, capture_output=True)
                wheels = list(dist_dir.glob("*.whl"))
            except Exception:
                pass
        if not wheels:
            pytest.skip("No wheel distribution found in dist/! Run python -m build --wheel.")

        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        assert latest_wheel.is_file()
        assert latest_wheel.stat().st_size > 100_000, f"Wheel size too small: {latest_wheel.stat().st_size}"

        found_json: set[str] = set()
        found_init: bool = False

        with zipfile.ZipFile(latest_wheel, "r") as zf:
            namelist = zf.namelist()
            for name in namelist:
                if name == "microllm/data/__init__.py":
                    found_init = True
                elif name.startswith("microllm/data/") and name.endswith(".json"):
                    fname = name.split("microllm/data/")[1]
                    found_json.add(fname)
                    # Verify content can be read and parsed directly from inside the wheel
                    raw = zf.read(name)
                    assert len(raw) > 50, f"File {name} inside wheel is suspiciously empty"
                    data = json.loads(raw.decode("utf-8"))
                    expected_meta = EXPECTED_JSON_METADATA.get(fname)
                    assert expected_meta is not None, f"Unexpected JSON file in wheel: {fname}"
                    assert len(data) == expected_meta["exact_count"], (
                        f"Data count mismatch inside wheel for {fname}"
                    )

        assert found_init, "microllm/data/__init__.py missing from wheel archive!"
        missing = set(EXPECTED_JSON_METADATA.keys()) - found_json
        assert not missing, f"Missing JSON data files inside wheel archive: {missing}"
        assert len(found_json) == 10, f"Expected 10 JSON data files in wheel, found {len(found_json)}"


# =============================================================================
# 3. Adversarial Verdict-MQ State Caching & Multi-threaded Concurrency
# =============================================================================


class TestVerdictMqStateCachingStress:
    """Stress-test VerdictMQMultiQuestionCache with adversarial inputs and high concurrency."""

    @pytest.mark.parametrize(
        "whitespace_state",
        [
            "",
            " ",
            "   ",
            "\t",
            "\r\n",
            "   \t\r\n\t   ",
            "\u00A0",  # Non-breaking space
            "\u2000\u2001\u2002\u2003\u2009\u200A\u202F\u205F\u3000",  # Unicode blank characters
            " \n \t \r ",
            "\u3000\u3000",  # Ideographic space
        ],
    )
    def test_score_cached_state_with_exhaustive_whitespace(self, whitespace_state: str) -> None:
        """Verify score_cached_state returns 0.0 without error on all whitespace variants."""
        cache = VerdictMQMultiQuestionCache()
        score = cache.score_cached_state(
            state_text=whitespace_state,
            query_text="Is account locked?",
            hypothesis="Account security lock",
        )
        assert score == 0.0

        # Verify cached entry contains frozenset tokens and clean empty string
        cached_entry = cache.get_cached_state(whitespace_state)
        assert cached_entry["clean"] == ""
        assert isinstance(cached_entry["tokens"], frozenset)
        assert len(cached_entry["tokens"]) == 0

    def test_score_cached_state_extreme_lengths_and_symbols(self) -> None:
        """Stress-test score_cached_state with extreme string lengths and unusual character sets."""
        cache = VerdictMQMultiQuestionCache(max_states=50)

        # 1. 20k characters repeated text
        large_state = "The customer requested a password reset. " * 500
        score_large = cache.score_cached_state(large_state, "password reset", "security")
        assert isinstance(score_large, float)

        # 2. Control characters & emojis
        emoji_state = "🚨 Emergency alert! 🤖 AI system triggered high temperature warning: 95°C. 🔥"
        score_emoji = cache.score_cached_state(emoji_state, "Alert level", "critical alert")
        assert isinstance(score_emoji, float)

        # 3. Punctuation only
        score_punct = cache.score_cached_state("?!.,;;:---***", "question", "topic")
        assert isinstance(score_punct, float)

    def test_multithreaded_concurrent_caching_and_eviction_stress(self) -> None:
        """Run 40 concurrent threads performing 2,000 queries across a small cache (max_states=10).

        Verifies:
        - Thread safety under heavy eviction and simultaneous insert.
        - Zero race condition / KeyError / RuntimeError on cache dictionary.
        - Consistent stats telemetry.
        """
        small_cache = VerdictMQMultiQuestionCache(max_states=10)

        test_states = [
            "",
            "   ",
            "\u00A0\u2000",
            "Customer wants billing refund.",
            "Technical error code 500 on checkout.",
            "Account locked due to multiple login attempts.",
            "Inquire about shipping delays.",
            "Update home address in profile.",
            "Request cancelation of premium subscription.",
            "Security vulnerability report.",
            "Feature request for dark mode.",
            "API rate limit exceeded.",
            "Webhook delivery failure.",
            "GDPR data export request.",
            "Partner integration inquiry.",
        ]

        queries = [
            "What is the customer issue?",
            "Categorize ticket",
            "Urgency level",
            "Department destination",
        ]

        hypotheses = [
            "billing refund",
            "technical support",
            "security team",
            "general inquiry",
        ]

        errors: list[Exception] = []

        def worker_task(worker_id: int, iterations: int) -> int:
            local_success = 0
            for i in range(iterations):
                s = random.choice(test_states)
                q = random.choice(queries)
                h = random.choice(hypotheses)
                try:
                    score = small_cache.score_cached_state(s, q, h)
                    if not s.strip():
                        assert score == 0.0
                    else:
                        assert isinstance(score, float)
                    local_success += 1
                except Exception as ex:
                    errors.append(ex)
            return local_success

        num_threads = 40
        iterations_per_thread = 50
        total_operations = num_threads * iterations_per_thread

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_task, idx, iterations_per_thread)
                for idx in range(num_threads)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert not errors, f"Concurrent stress test encountered exceptions: {errors[:5]}"
        assert sum(results) == total_operations

        # Verify cache integrity
        assert len(small_cache._cache) <= small_cache.max_states, (
            f"Cache exceeded max_states: {len(small_cache._cache)} > {small_cache.max_states}"
        )

        stats = small_cache.stats
        assert "hits" in stats and "misses" in stats
        total_encodes = stats["hits"] + stats["misses"]
        assert total_encodes == total_operations
        assert stats["cached_states"] <= small_cache.max_states
        assert 0.0 <= stats["hit_rate"] <= 1.0
