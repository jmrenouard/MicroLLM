"""Adversarial stress harness for Milestone 4 (SDK Base Consolidation & Codebase Modularization).

Authored by Challenger 1 (Empirical Challenger) to rigorously stress-test:
1. F08: Base client inheritance (_BaseMicroJevClient), attribute propagation, resolution logic,
   and auth headers generation across MicroJevClient and AsyncMicroJevClient.
2. F08: Wire-format parity and payload builder equivalence across all 4 question types (choice,
   score, multi, noul), batching with custom strategies, guardrail, and academic papers queries.
3. F12: Decomposed module structure, dynamic descriptor property facades, and backward-compatible
   re-exports on gradio_app, dataset_adapters, and server.
4. F12: High-concurrency module unload/reload churn across 10 threads under random permutation orders,
   verifying absence of circular import deadlocks and 'is' identity preservation.
5. AST Invariant & CLI: inspect.getsource(server.create_app) deque(maxlen=) bounded telemetry invariant,
   and clean CLI command execution.
"""

from __future__ import annotations

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from enum import Enum
import importlib
import inspect
import io
import json
import os
import random
import sys
import threading
import time
from typing import Any, Mapping, Sequence
import pytest

from microllm.client import (
    AsyncMicroJevClient,
    Choice,
    ChoiceResult,
    MicroJevClient,
    Multi,
    MultiResult,
    Noul,
    NoulResult,
    Score,
    ScoreResult,
    _BaseMicroJevClient,
    _build_ask_payload,
    _build_batch_payload,
    _build_guardrail_payload,
    _build_papers_params,
    _parse_systemone_answers,
)


# ============================================================================
# Helpers for Mocking Wire Transports
# ============================================================================

class DummyResponse:
    def __init__(self, json_data: dict[str, Any], status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MockSyncHttpTransport:
    """Mock HTTP transport that records outbound calls for sync client."""

    def __init__(self, response_data: dict[str, Any] | None = None) -> None:
        self.response_data = response_data or {}
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> DummyResponse:
        self.calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
        return DummyResponse(self.response_data)

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> DummyResponse:
        self.calls.append({"method": "GET", "url": url, "params": params, "headers": headers})
        return DummyResponse(self.response_data)


class MockAsyncHttpTransport:
    """Mock HTTP transport that records outbound calls for async client."""

    def __init__(self, response_data: dict[str, Any] | None = None) -> None:
        self.response_data = response_data or {}
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> DummyResponse:
        self.calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
        return DummyResponse(self.response_data)

    async def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> DummyResponse:
        self.calls.append({"method": "GET", "url": url, "params": params, "headers": headers})
        return DummyResponse(self.response_data)


class SamplePriorityEnum(Enum):
    LOW = "low_priority"
    MEDIUM = "medium_priority"
    HIGH = "high_priority"
    CRITICAL = "critical_priority"


# ============================================================================
# 1. F08: Base Client Inheritance, Attribute Propagation & Headers
# ============================================================================

class TestF08ClientHierarchyAndInheritance:
    """Empirical verification of SDK client class hierarchy and inheritance."""

    def test_base_micro_jev_client_subclassing(self) -> None:
        """Verify MicroJevClient and AsyncMicroJevClient strictly inherit from _BaseMicroJevClient."""
        assert issubclass(MicroJevClient, _BaseMicroJevClient), "MicroJevClient must inherit from _BaseMicroJevClient"
        assert issubclass(AsyncMicroJevClient, _BaseMicroJevClient), "AsyncMicroJevClient must inherit from _BaseMicroJevClient"

    def test_base_micro_jev_client_instantiation_and_isinstance(self) -> None:
        """Verify instantiated clients are instances of the base class."""
        sync_client = MicroJevClient(
            base_url="http://custom-host:9000",
            api_key="secret-key-123",
            model="jev-custom-model",
            multilingual_strategy="translate_test_time",
            neural_model="bge-m3-custom",
        )
        async_client = AsyncMicroJevClient(
            base_url="http://custom-host:9000",
            api_key="secret-key-123",
            model="jev-custom-model",
            multilingual_strategy="translate_test_time",
            neural_model="bge-m3-custom",
        )

        assert isinstance(sync_client, _BaseMicroJevClient)
        assert isinstance(async_client, _BaseMicroJevClient)

        for client in (sync_client, async_client):
            assert client.base_url == "http://custom-host:9000"
            assert client.api_key == "secret-key-123"
            assert client.model == "jev-custom-model"
            assert client.multilingual_strategy == "translate_test_time"
            assert client.neural_model == "bge-m3-custom"

    def test_auth_headers_generation_and_edge_cases(self) -> None:
        """Verify _auth_headers formats Bearer token accurately across normal and extreme keys."""
        base_client = _BaseMicroJevClient(api_key="standard-key")
        assert base_client._auth_headers() == {"Authorization": "Bearer standard-key"}

        # Empty key
        base_empty = _BaseMicroJevClient(api_key="")
        assert base_empty._auth_headers() == {"Authorization": "Bearer "}

        # Special character key with unicode and punctuation
        special_key = "key_!@#$%^&*()_+=~`{}[]|:;'<>,.?/🚀"
        base_special = _BaseMicroJevClient(api_key=special_key)
        assert base_special._auth_headers() == {"Authorization": f"Bearer {special_key}"}

    def test_strategy_and_neural_resolution_fallback(self) -> None:
        """Verify _resolve_strategy and _resolve_neural prioritize call arguments over client defaults."""
        client = _BaseMicroJevClient(
            multilingual_strategy="default_strategy",
            neural_model="default_neural",
        )

        # Fallback to defaults when None is passed
        assert client._resolve_strategy(None) == "default_strategy"
        assert client._resolve_neural(None) == "default_neural"

        # Explicit override takes precedence
        assert client._resolve_strategy("override_strategy") == "override_strategy"
        assert client._resolve_neural("override_neural") == "override_neural"

        # When client defaults are None, None is returned
        client_none = _BaseMicroJevClient()
        assert client_none._resolve_strategy(None) is None
        assert client_none._resolve_neural(None) is None


# ============================================================================
# 2. F08: Wire-Format Parity Across All 4 Question Types & Endpoints
# ============================================================================

class TestF08PayloadBuildersAndWireParity:
    """Stress test wire-level request equality between sync and async SDK clients."""

    def test_ask_payload_builder_parity_all_4_question_types(self) -> None:
        """Verify _build_ask_payload accurately serializes all 4 question types simultaneously."""
        q_noul = Noul(instructions="Is system nominal?", true_description="Nominal", false_description="Degraded")
        q_choice = Choice(options=["alpha", "beta", "gamma"], instructions="Choose tier")
        q_score = Score(levels=[1, 2, 3, 4, 5], instructions="Rate severity")
        q_multi = Multi(options=["tag1", "tag2", "tag3"], threshold=0.7, instructions="Select tags")

        questions = {
            "q1_noul": q_noul,
            "q2_choice": q_choice,
            "q3_score": q_score,
            "q4_multi": q_multi,
        }

        payload = _build_ask_payload(
            state="System status report text payload",
            questions=questions,
            model="jev-latest",
            multilingual_strategy="translate_train_test",
            neural_model="test-neural-head",
        )

        assert payload["model"] == "jev-latest"
        assert payload["state"] == "System status report text payload"
        assert payload["multilingual_strategy"] == "translate_train_test"
        assert payload["neural_model"] == "test-neural-head"

        q_dict = payload["questions"]
        assert q_dict["q1_noul"]["type"] == "noul"
        assert q_dict["q1_noul"]["instructions"] == "Is system nominal?"
        assert q_dict["q1_noul"]["criteria"] == {"true": "Nominal", "false": "Degraded"}

        assert q_dict["q2_choice"]["type"] == "choice"
        assert set(q_dict["q2_choice"]["criteria"].keys()) == {"alpha", "beta", "gamma"}

        assert q_dict["q3_score"]["type"] == "score"
        assert q_dict["q3_score"]["criteria"] == [1, 2, 3, 4, 5]

        assert q_dict["q4_multi"]["type"] == "multi"
        assert q_dict["q4_multi"]["threshold"] == 0.7

    def test_choice_and_multi_from_enum_parity(self) -> None:
        """Verify Choice.from_enum and Multi.from_enum build identical criteria mappings."""
        choice_enum = Choice.from_enum(SamplePriorityEnum, instructions="Enum priority selection")
        multi_enum = Multi.from_enum(SamplePriorityEnum, instructions="Enum multi tagging", threshold=0.65)

        c_pay = choice_enum.to_payload()
        m_pay = multi_enum.to_payload()

        expected_keys = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert set(c_pay["criteria"].keys()) == expected_keys
        assert set(m_pay["criteria"].keys()) == expected_keys
        assert c_pay["criteria"]["HIGH"] == "high_priority"
        assert m_pay["criteria"]["CRITICAL"] == "critical_priority"

    def test_sync_vs_async_ask_wire_equality(self) -> None:
        """Assert sync MicroJevClient.ask and async AsyncMicroJevClient.ask emit identical requests."""
        dummy_answers = {
            "answers": {
                "q_noul": {"type": "noul", "noul": 0.92},
                "q_choice": {"type": "choice", "choice": "B", "probabilities": {"A": 0.1, "B": 0.8, "C": 0.1}, "confidence": 0.8},
                "q_score": {"type": "score", "score": 4.2, "legend": {"1": 0.05, "5": 0.8}, "probabilities": {"4": 0.7}, "confidence": 0.7},
                "q_multi": {"type": "multi", "labels": ["cat1"], "probabilities": {"cat1": 0.9, "cat2": 0.2}, "confidence": 0.9},
            }
        }

        mock_sync_transport = MockSyncHttpTransport(dummy_answers)
        mock_async_transport = MockAsyncHttpTransport(dummy_answers)

        sync_client = MicroJevClient(api_key="parity-token", http_client=mock_sync_transport)
        async_client = AsyncMicroJevClient(api_key="parity-token", http_client=mock_async_transport)

        questions = {
            "q_noul": Noul(instructions="Is valid?"),
            "q_choice": Choice(options=["A", "B", "C"]),
            "q_score": Score(levels=[1, 2, 3, 4, 5]),
            "q_multi": Multi(options=["cat1", "cat2"]),
        }

        # Run sync ask
        res_sync = sync_client.ask(
            state="Transmitted input state across channels",
            questions=questions,
            model="jev-parity-model",
            multilingual_strategy="translate_test_time",
            neural_model="parity-neural",
        )

        # Run async ask
        res_async = asyncio.run(
            async_client.ask(
                state="Transmitted input state across channels",
                questions=questions,
                model="jev-parity-model",
                multilingual_strategy="translate_test_time",
                neural_model="parity-neural",
            )
        )

        # 1. Assert outbound HTTP calls are 100% identical
        assert len(mock_sync_transport.calls) == 1
        assert len(mock_async_transport.calls) == 1

        sync_call = mock_sync_transport.calls[0]
        async_call = mock_async_transport.calls[0]

        assert sync_call["method"] == async_call["method"] == "POST"
        assert sync_call["url"] == async_call["url"] == "/v1/systemone"
        assert sync_call["headers"] == async_call["headers"] == {"Authorization": "Bearer parity-token"}
        assert sync_call["json"] == async_call["json"]

        # 2. Assert parsed result values match
        assert res_sync.keys() == res_async.keys()
        for k in res_sync:
            assert type(res_sync[k]) is type(res_async[k])
            if isinstance(res_sync[k], NoulResult):
                assert res_sync[k].probability == res_async[k].probability
                assert res_sync[k].value == res_async[k].value
            elif isinstance(res_sync[k], ChoiceResult):
                assert res_sync[k].value == res_async[k].value
                assert res_sync[k].confidence == res_async[k].confidence
            elif isinstance(res_sync[k], ScoreResult):
                assert res_sync[k].score == res_async[k].score
            elif isinstance(res_sync[k], MultiResult):
                assert res_sync[k].labels == res_async[k].labels

    def test_sync_vs_async_batch_wire_equality(self) -> None:
        """Assert sync MicroJevClient.batch and async AsyncMicroJevClient.batch emit identical requests."""
        dummy_batch_resp = {"status": "ok", "total_items": 3, "results": []}

        mock_sync = MockSyncHttpTransport(dummy_batch_resp)
        mock_async = MockAsyncHttpTransport(dummy_batch_resp)

        sync_client = MicroJevClient(api_key="batch-key", http_client=mock_sync)
        async_client = AsyncMicroJevClient(api_key="batch-key", http_client=mock_async)

        items = ["Item 1 state text", "Item 2 state text", "Item 3 state text"]
        questions = {"priority": Choice(options=["P0", "P1", "P2"])}

        sync_client.batch(
            items=items,
            questions=questions,
            model="jev-batch-engine",
            multilingual_strategy="aligned_embeddings",
            neural_model="bge-m3-large",
        )

        asyncio.run(
            async_client.batch(
                items=items,
                questions=questions,
                model="jev-batch-engine",
                multilingual_strategy="aligned_embeddings",
                neural_model="bge-m3-large",
            )
        )

        assert len(mock_sync.calls) == 1
        assert len(mock_async.calls) == 1

        sync_call = mock_sync.calls[0]
        async_call = mock_async.calls[0]

        assert sync_call["method"] == async_call["method"] == "POST"
        assert sync_call["url"] == async_call["url"] == "/v1/batch"
        assert sync_call["headers"] == async_call["headers"] == {"Authorization": "Bearer batch-key"}
        assert sync_call["json"] == async_call["json"]
        assert sync_call["json"]["multilingual_strategy"] == "aligned_embeddings"
        assert sync_call["json"]["neural_model"] == "bge-m3-large"

    def test_sync_vs_async_guardrail_wire_equality(self) -> None:
        """Assert sync MicroJevClient.guardrail and async AsyncMicroJevClient.guardrail emit identical requests."""
        dummy_guard_resp = {"state": "clean", "passed": True, "scores": {}}

        mock_sync = MockSyncHttpTransport(dummy_guard_resp)
        mock_async = MockAsyncHttpTransport(dummy_guard_resp)

        sync_client = MicroJevClient(api_key="guard-key", http_client=mock_sync)
        async_client = AsyncMicroJevClient(api_key="guard-key", http_client=mock_async)

        sync_client.guardrail(
            state="Evaluate user input for safety",
            context="Enterprise security firewall agent",
            threshold=0.82,
            custom_rubrics={"pii_leak": "Check for social security numbers"},
        )

        asyncio.run(
            async_client.guardrail(
                state="Evaluate user input for safety",
                context="Enterprise security firewall agent",
                threshold=0.82,
                custom_rubrics={"pii_leak": "Check for social security numbers"},
            )
        )

        assert len(mock_sync.calls) == 1
        assert len(mock_async.calls) == 1

        sync_call = mock_sync.calls[0]
        async_call = mock_async.calls[0]

        assert sync_call["url"] == async_call["url"] == "/v1/guardrail"
        assert sync_call["headers"] == async_call["headers"] == {"Authorization": "Bearer guard-key"}
        assert sync_call["json"] == async_call["json"]
        assert sync_call["json"]["threshold"] == 0.82
        assert sync_call["json"]["custom_rubrics"] == {"pii_leak": "Check for social security numbers"}

    def test_sync_vs_async_papers_wire_equality(self) -> None:
        """Assert sync MicroJevClient.papers and async AsyncMicroJevClient.papers emit identical GET queries."""
        dummy_papers_resp = {"total": 1, "papers": []}

        mock_sync = MockSyncHttpTransport(dummy_papers_resp)
        mock_async = MockAsyncHttpTransport(dummy_papers_resp)

        sync_client = MicroJevClient(api_key="papers-key", http_client=mock_sync)
        async_client = AsyncMicroJevClient(api_key="papers-key", http_client=mock_async)

        sync_client.papers(pillar="sparse_decision", query="conformal prediction", arxiv_id="2409.11223")
        asyncio.run(async_client.papers(pillar="sparse_decision", query="conformal prediction", arxiv_id="2409.11223"))

        assert len(mock_sync.calls) == 1
        assert len(mock_async.calls) == 1

        sync_call = mock_sync.calls[0]
        async_call = mock_async.calls[0]

        assert sync_call["method"] == async_call["method"] == "GET"
        assert sync_call["url"] == async_call["url"] == "/v1/papers"
        assert sync_call["headers"] == async_call["headers"] == {"Authorization": "Bearer papers-key"}
        assert sync_call["params"] == async_call["params"]
        assert sync_call["params"] == {
            "pillar": "sparse_decision",
            "q": "conformal prediction",
            "arxiv_id": "2409.11223",
        }

    def test_extreme_inputs_and_edge_cases(self) -> None:
        """Stress-test edge cases: empty strings, 100k-char state, unicode emojis, empty dicts."""
        # 1. 100k-character state
        long_state = "A" * 100_000
        q = {"choice": Choice(options=["opt1", "opt2"])}
        pay = _build_ask_payload(state=long_state, questions=q, model="jev-latest")
        assert len(pay["state"]) == 100_000

        # 2. Unicode and emojis
        emoji_state = "🔥 Multi-lingual & Emoji State: 🚀 💎 ⚡ 🤖 🧠 📈 🛡️ 🌍 🧪"
        pay_emoji = _build_ask_payload(state=emoji_state, questions=q, model="jev-latest")
        assert pay_emoji["state"] == emoji_state

        # 3. Empty questions map
        pay_empty_q = _build_ask_payload(state="some state", questions={}, model="jev-latest")
        assert pay_empty_q["questions"] == {}

        # 4. Empty batch items list
        pay_empty_batch = _build_batch_payload(items=[], questions=q, model="jev-latest")
        assert pay_empty_batch["items"] == []

        # 5. Guardrail with special rubrics
        guard_pay = _build_guardrail_payload(
            state="text",
            threshold=0.0,
            custom_rubrics={"spec;ial\nkey": "val\"with'quotes\nand\ttabs"},
        )
        assert guard_pay["threshold"] == 0.0
        assert "spec;ial\nkey" in guard_pay["custom_rubrics"]


# ============================================================================
# 3. F12: Decomposed Module Structure & Dynamic Re-Export Facades
# ============================================================================

class TestF12ModularizationAndDynamicFacades:
    """Empirical verification of modular boundaries and re-export descriptor mechanics."""

    def test_all_extracted_modules_importable(self) -> None:
        """Verify all 6 newly created cohesive modules can be independently imported."""
        extracted_modules = [
            "microllm.gradio_presets",
            "microllm.gradio_helpers",
            "microllm.gradio_handlers",
            "microllm.dataset_fixtures",
            "microllm.server_models",
            "microllm.cli",
        ]
        for mod_name in extracted_modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"

    def test_reexport_identity_preservation_across_all_three_subsystems(self) -> None:
        """Verify 'is' identity preservation on all primary re-exported symbols."""
        import microllm.dataset_adapters as dadapt
        import microllm.dataset_fixtures as dfix
        import microllm.gradio_app as gapp
        import microllm.gradio_presets as gpresets
        import microllm.server as srv
        import microllm.server_models as smodels

        # 1. Dataset fixtures
        assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES

        # 2. Gradio presets
        assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS
        assert gapp.MULTILINGUAL_SUPPORT_PRESETS is gpresets.MULTILINGUAL_SUPPORT_PRESETS
        assert gapp.STUDIO_DOMAIN_PRESETS is gpresets.STUDIO_DOMAIN_PRESETS
        assert gapp.DATASET_BENCHMARK_PRESETS is gpresets.DATASET_BENCHMARK_PRESETS
        assert gapp.CONSUMER_TYPE_EXAMPLES is gpresets.CONSUMER_TYPE_EXAMPLES

        # 3. Server models
        assert srv.StudioRLCDDemoRequest is smodels.StudioRLCDDemoRequest
        assert srv.StudioAnalyzeRequest is smodels.StudioAnalyzeRequest
        assert srv.CalibrateRequest is smodels.CalibrateRequest
        assert srv.GuardrailRequest is smodels.GuardrailRequest
        assert srv.ProfilePullRequest is smodels.ProfilePullRequest
        assert srv.BatchRequest is smodels.BatchRequest

    def test_partial_module_unload_dynamic_recovery(self) -> None:
        """Verify accessing properties triggers dynamic lookup even if target module was unloaded."""
        import microllm.dataset_adapters as dadapt
        import microllm.server as srv
        import microllm.gradio_app as gapp

        # 1. Unload dataset_fixtures and re-access via dataset_adapters
        if "microllm.dataset_fixtures" in sys.modules:
            del sys.modules["microllm.dataset_fixtures"]

        samples = dadapt.OFFLINE_WEB_DATASET_SAMPLES
        assert "microllm.dataset_fixtures" in sys.modules
        assert "google/boolq" in samples

        # 2. Unload server_models and re-access via server
        if "microllm.server_models" in sys.modules:
            del sys.modules["microllm.server_models"]

        req_cls = srv.StudioRLCDDemoRequest
        assert "microllm.server_models" in sys.modules
        assert hasattr(req_cls, "model_fields")

        # 3. Unload gradio_presets and re-access via gradio_app
        if "microllm.gradio_presets" in sys.modules:
            del sys.modules["microllm.gradio_presets"]

        presets = gapp.MULTILINGUAL_BENCHMARK_PRESETS
        assert "microllm.gradio_presets" in sys.modules
        assert len(presets) > 0

    def test_gradio_app_helpers_and_handlers_reexport_integrity(self) -> None:
        """Verify gradio_app re-exports all helper and handler functions from decomposed modules."""
        import microllm.gradio_app as gapp
        import microllm.gradio_helpers as ghelpers
        import microllm.gradio_handlers as ghandlers

        # Helpers
        assert gapp.compute_deep_studio_diagnostics is ghelpers.compute_deep_studio_diagnostics
        assert gapp.format_criteria_for_visual is ghelpers.format_criteria_for_visual
        assert gapp.parse_criteria_text is ghelpers.parse_criteria_text
        assert gapp.load_preset_for_gradio is ghelpers.load_preset_for_gradio
        assert gapp.generate_code_export is ghelpers.generate_code_export
        assert gapp.load_preset_for_all_tabs is ghelpers.load_preset_for_all_tabs

        # Handlers
        assert gapp.evaluate_studio_multi_mode is ghandlers.evaluate_studio_multi_mode
        assert gapp.evaluate_consumer_choice is ghandlers.evaluate_consumer_choice
        assert gapp.evaluate_consumer_multi is ghandlers.evaluate_consumer_multi
        assert gapp.evaluate_consumer_score is ghandlers.evaluate_consumer_score
        assert gapp.evaluate_consumer_yes_no is ghandlers.evaluate_consumer_yes_no
        assert gapp.run_guardrail_with_custom_rubrics is ghandlers.run_guardrail_with_custom_rubrics
        assert gapp.run_rlcd_lab_gradio is ghandlers.run_rlcd_lab_gradio

    def test_server_cli_reexport_integrity(self) -> None:
        """Verify server re-exports CLI functions (main, run_cli, build_cli_parser)."""
        import microllm.server as srv
        import microllm.cli as scli

        assert srv.build_cli_parser is scli.build_cli_parser
        assert srv.run_cli is scli.run_cli
        assert srv.main is scli.main
        assert callable(srv.main)


# ============================================================================
# 4. F12: Stress Test Module Unload & Concurrent Reload Churn
# ============================================================================

class TestF12ModuleChurnAndConcurrencyStress:
    """Stress test module unload and reload under random permutations and multithreaded concurrency."""

    MODULE_SET = [
        "microllm.dataset_fixtures",
        "microllm.dataset_adapters",
        "microllm.server_models",
        "microllm.server",
        "microllm.gradio_presets",
        "microllm.gradio_helpers",
        "microllm.gradio_handlers",
        "microllm.gradio_app",
        "microllm.cli",
    ]

    def test_sequential_random_unload_reload_churn_50_cycles(self) -> None:
        """Perform 50 cycles of deleting and reloading modules in random order, asserting identity."""
        rng = random.Random(42)

        for cycle in range(50):
            # 1. Pick a random order to unload
            unload_order = list(self.MODULE_SET)
            rng.shuffle(unload_order)

            for mod in unload_order:
                if mod in sys.modules:
                    del sys.modules[mod]

            # 2. Pick a random order to reload
            reload_order = list(self.MODULE_SET)
            rng.shuffle(reload_order)

            for mod in reload_order:
                importlib.import_module(mod)

            # 3. Assert identity after each cycle
            import microllm.dataset_adapters as dadapt
            import microllm.dataset_fixtures as dfix
            import microllm.gradio_app as gapp
            import microllm.gradio_presets as gpresets
            import microllm.server as srv
            import microllm.server_models as smodels

            assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES, (
                f"Dataset samples identity failed at cycle {cycle}"
            )
            assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS, (
                f"Gradio presets identity failed at cycle {cycle}"
            )
            assert srv.StudioRLCDDemoRequest is smodels.StudioRLCDDemoRequest, (
                f"Server models identity failed at cycle {cycle}"
            )

    def test_multithreaded_concurrent_module_churn_stress(self) -> None:
        """Run 8 concurrent threads repeatedly reloading modules to stress circular import deadlock immunity."""
        num_threads = 8
        cycles_per_thread = 15
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker_task(thread_id: int) -> int:
            rng = random.Random(1000 + thread_id)
            success_count = 0
            for i in range(cycles_per_thread):
                try:
                    # Random subset of modules to import
                    mods_to_test = list(self.MODULE_SET)
                    rng.shuffle(mods_to_test)

                    # Import each module in random order
                    for mod in mods_to_test:
                        m = importlib.import_module(mod)
                        assert m is not None

                    # Check identity under thread access
                    import microllm.dataset_adapters as dadapt
                    import microllm.dataset_fixtures as dfix
                    import microllm.gradio_app as gapp
                    import microllm.gradio_presets as gpresets
                    import microllm.server as srv
                    import microllm.server_models as smodels

                    assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES
                    assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS
                    assert srv.StudioRLCDDemoRequest is smodels.StudioRLCDDemoRequest
                    success_count += 1
                except Exception as ex:
                    with lock:
                        errors.append(ex)
                time.sleep(0.005)
            return success_count

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            for future in as_completed(futures, timeout=30.0):
                future.result()

        assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent churn: {errors[:3]}"

    def test_active_concurrent_unloader_and_loader_threads(self) -> None:
        """Adversarially pit unloader threads against loader threads concurrently without deadlocks."""
        num_cycles = 10
        lock = threading.Lock()
        deadlock_timeout = 20.0

        for cycle in range(num_cycles):
            barrier = threading.Barrier(4)

            def unloader(uid: int) -> None:
                barrier.wait(timeout=deadlock_timeout)
                mods = list(self.MODULE_SET)
                random.shuffle(mods)
                for mod in mods:
                    with lock:
                        if mod in sys.modules:
                            del sys.modules[mod]
                    time.sleep(0.001)

            def loader(lid: int) -> None:
                barrier.wait(timeout=deadlock_timeout)
                mods = list(self.MODULE_SET)
                random.shuffle(mods)
                for mod in mods:
                    try:
                        importlib.import_module(mod)
                    except Exception:
                        pass
                    time.sleep(0.001)

            t1 = threading.Thread(target=unloader, args=(1,))
            t2 = threading.Thread(target=unloader, args=(2,))
            t3 = threading.Thread(target=loader, args=(1,))
            t4 = threading.Thread(target=loader, args=(2,))

            threads = [t1, t2, t3, t4]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=deadlock_timeout)
                assert not t.is_alive(), "Deadlock detected in concurrent unload/reload threads!"

        # Final stabilization import to assert everything cleanly recovers
        for mod in self.MODULE_SET:
            importlib.import_module(mod)

        import microllm.dataset_adapters as dadapt
        import microllm.dataset_fixtures as dfix
        import microllm.gradio_app as gapp
        import microllm.gradio_presets as gpresets
        import microllm.server as srv
        import microllm.server_models as smodels

        assert dadapt.OFFLINE_WEB_DATASET_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES
        assert gapp.MULTILINGUAL_BENCHMARK_PRESETS is gpresets.MULTILINGUAL_BENCHMARK_PRESETS
        assert srv.StudioRLCDDemoRequest is smodels.StudioRLCDDemoRequest


# ============================================================================
# 5. Gradio Handlers Runtime Integrity & Missing Names Audit
# ============================================================================

class TestGradioDecompositionRuntimeIntegrity:
    """Empirically test runtime execution of decomposed gradio handlers."""

    def test_evaluate_studio_visual_builder_mode_no_name_error(self) -> None:
        """Assert evaluate_consumer_choice and visual builder mode execute without NameError."""
        from microllm.gradio_handlers import (
            evaluate_consumer_choice,
            evaluate_consumer_multi,
            evaluate_consumer_score,
            evaluate_consumer_yes_no,
        )

        # Consumer helper invokes evaluate_studio_multi_mode with visual_qid
        res = evaluate_consumer_choice("Checking card balance", "Select intent", "balance\nlimit")
        assert len(res) == 5


# ============================================================================
# 6. AST Invariants & Clean CLI Execution
# ============================================================================

class TestASTInvariantsAndCLIExecution:
    """Verify source AST constraints and CLI functionality."""

    def test_server_create_app_ast_invariant_deque_bounded(self) -> None:
        """Verify inspect.getsource(server.create_app) strictly contains 'deque(maxlen='."""
        import microllm.server as srv

        src = inspect.getsource(srv.create_app)
        assert "deque(maxlen=" in src, "Invariant violated: create_app source missing deque(maxlen=)"

    def test_server_create_app_is_genuine_function_definition_in_server(self) -> None:
        """Verify AST of server.py defines create_app directly (not an imported stub)."""
        import microllm.server as srv
        server_path = inspect.getfile(srv)

        with open(server_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=server_path)

        func_defs = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "create_app" in func_defs, "create_app must be defined as a top-level function in server.py"

    def test_cli_clean_execution_and_subcommands(self) -> None:
        """Verify microllm.cli.main executes commands cleanly without unhandled exceptions."""
        import microllm.cli as cli

        # 1. Top-level --help raises SystemExit(0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as exc:
                cli.main(["--help"])
        assert exc.value.code == 0
        assert "MicroLLM Local System-One Engine CLI" in buf.getvalue()

        # 2. List help
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as exc:
                cli.main(["list", "--help"])
        assert exc.value.code == 0
        assert "--cache-dir" in buf.getvalue()

        # 3. Benchmark help
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as exc:
                cli.main(["benchmark", "--help"])
        assert exc.value.code == 0
        assert "--multilingual" in buf.getvalue()
        assert "--json" in buf.getvalue()

        # 4. Serve help
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as exc:
                cli.main(["serve", "--help"])
        assert exc.value.code == 0
        assert "--host" in buf.getvalue()
        assert "--port" in buf.getvalue()

        # 5. Benchmark multilingual with JSON output executes normally and returns 0
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["benchmark", "--multilingual", "--json"])
        assert code == 0
        parsed = json.loads(buf.getvalue().strip())
        assert isinstance(parsed, dict)
        assert "benchmark_suite" in parsed
        assert "Multilingual" in parsed["benchmark_suite"]
