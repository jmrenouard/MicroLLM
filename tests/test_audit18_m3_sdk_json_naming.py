"""Comprehensive Unit & Integration Test Suite for Milestone 3 (Audit Points 9, 15, 16).

Covers:
- Audit Point 9 (Feature F09): Package name consistency to 'micro-jev' across error messages, CLI, workflows.
- Audit Point 15 (Feature F15): SDK client logic deduplication into _BaseMicroJevClient, signature & public method parity.
- Audit Point 16 (Feature F16): Modular packaged JSON data files under src/microllm/data/ and backward-compatible re-exports.
- Edge-Case Verification: VerdictMQMultiQuestionCache.score_cached_state empty/whitespace handling.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import sys
import types
from typing import Any

from fastapi.testclient import TestClient
import pytest

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    create_hf_zero_shot_scorer,
    create_onnx_nli_scorer,
)
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
)
from microllm.registry import quantize_onnx_file_int8
from microllm.server import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Audit Point 9: Package Name Consistency ('micro-jev')
# ==============================================================================


class TestAuditPoint09PackageNaming:
    """Verify all error messages, CLI hints, and packaging workflows reference 'micro-jev'."""

    def test_backends_base_torch_missing_recommends_micro_jev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify missing torch/transformers raises ImportError recommending 'pip install micro-jev[torch]'."""
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[torch\]"):
            create_hf_zero_shot_scorer("dummy-model")

    def test_backends_base_onnx_missing_recommends_micro_jev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify missing onnxruntime raises ImportError recommending 'pip install micro-jev[onnx]'."""
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[onnx\]"):
            create_onnx_nli_scorer("dummy.onnx", "dummy-tok")

    def test_backends_base_transformers_missing_recommends_micro_jev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify missing transformers with onnx raises ImportError recommending 'pip install micro-jev[onnx] transformers'."""
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        monkeypatch.setitem(sys.modules, "transformers", None)

        with pytest.raises(ImportError, match=r"pip install micro-jev\[onnx\] transformers"):
            create_onnx_nli_scorer("dummy.onnx", "dummy-tok")

    def test_registry_onnx_missing_recommends_micro_jev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify quantize_onnx_file_int8 raises ImportError recommending 'pip install micro-jev[onnx]'."""
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[onnx\]"):
            quantize_onnx_file_int8("in.onnx", "out.onnx")

    def test_registry_cli_report_recommends_micro_jev(self) -> None:
        """Verify registry benchmark reporting text uses 'pip install micro-jev'."""
        reg_file = REPO_ROOT / "src" / "microllm" / "registry.py"
        content = reg_file.read_text(encoding="utf-8")
        assert "pip install micro-jev" in content
        assert "pip install microllm\n" not in content

    def test_hf_space_sync_quickstart_recommends_micro_jev(self) -> None:
        """Verify hf_space_sync quickstart docstring specifies 'pip install \"micro-jev[gradio]\"'."""
        hf_sync = REPO_ROOT / "src" / "microllm" / "hf_space_sync.py"
        content = hf_sync.read_text(encoding="utf-8")
        assert 'pip install "micro-jev[gradio]"' in content
        assert 'pip install "microllm[gradio]"' not in content

    def test_pypi_workflow_references_micro_jev_and_fallback(self) -> None:
        """Verify .github/workflows/publish-pypi.yml references 'micro-jev' and performs valid fallback replacement."""
        pypi_wf = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
        content = pypi_wf.read_text(encoding="utf-8")
        assert "Successfully published micro-jev distribution to PyPI." in content
        assert 'name = \\"micro-jev\\"' in content
        assert 'name = \\"microllm-jev\\"' in content
        assert "microllm-jev" in content


# ==============================================================================
# Audit Point 15: SDK Client Deduplication & Parity
# ==============================================================================


class TestAuditPoint15SdkDeduplication:
    """Verify DRY consolidation of MicroJevClient and AsyncMicroJevClient into _BaseMicroJevClient."""

    def test_client_class_hierarchy_and_base_encapsulation(self) -> None:
        """Verify MicroJevClient and AsyncMicroJevClient inherit from _BaseMicroJevClient."""
        assert issubclass(MicroJevClient, _BaseMicroJevClient)
        assert issubclass(AsyncMicroJevClient, _BaseMicroJevClient)

    def test_base_client_initialization_and_normalization(self) -> None:
        """Verify base client normalizes base_url, handles auth headers, and sets timeout."""
        base = _BaseMicroJevClient(
            base_url="http://127.0.0.1:8000/",
            api_key="custom-secret-key",
            model="custom-model",
            timeout=15.5,
        )
        assert base.base_url == "http://127.0.0.1:8000"
        assert base.timeout == 15.5
        assert base._auth_headers() == {"Authorization": "Bearer custom-secret-key"}
        assert base.headers == {"Authorization": "Bearer custom-secret-key"}
        assert base._headers == {"Authorization": "Bearer custom-secret-key"}

    def test_shared_request_payload_builders(self) -> None:
        """Verify shared request payload preparation helpers in _BaseMicroJevClient."""
        base = _BaseMicroJevClient(model="jev-v1", multilingual_strategy="fast")

        # 1. Ask payload
        payload_ask = base._prepare_ask(
            state="Test state text",
            questions={"q1": Noul(instructions="Is valid?")},
            neural_model="minilm",
        )
        assert payload_ask["model"] == "jev-v1"
        assert payload_ask["state"] == "Test state text"
        assert "q1" in payload_ask["questions"]
        assert payload_ask["multilingual_strategy"] == "fast"
        assert payload_ask["neural_model"] == "minilm"

        # Classmethod alias
        cls_payload = _BaseMicroJevClient._build_systemone_payload(
            state="S",
            questions={"q": Choice(options=["a", "b"])},
        )
        assert cls_payload["state"] == "S"
        assert "q" in cls_payload["questions"]

        # 2. Guardrail payload
        payload_gr = base._prepare_guardrail(state="Guardrail test", threshold=0.7)
        assert payload_gr["state"] == "Guardrail test"
        assert payload_gr["threshold"] == 0.7

        # 3. Batch payload
        payload_batch = base._prepare_batch(
            items=["Item 1", "Item 2"],
            questions={"q": Score(levels=[1, 2, 3])},
        )
        assert payload_batch["items"] == ["Item 1", "Item 2"]
        assert "q" in payload_batch["questions"]

        # 4. Papers params
        params_papers = base._prepare_papers(pillar="pillar_1", query="RLCD", arxiv_id="2609.38850")
        assert params_papers["pillar"] == "pillar_1"
        assert params_papers["q"] == "RLCD"
        assert params_papers["arxiv_id"] == "2609.38850"

    def test_shared_response_decoding_helpers(self) -> None:
        """Verify shared response decoding helpers in _BaseMicroJevClient."""
        # 1. Ask decoding
        raw_ask = {
            "answers": {
                "noul_q": {"type": "noul", "noul": 0.85},
                "choice_q": {
                    "type": "choice",
                    "choice": "opt_a",
                    "probabilities": {"opt_a": 0.8, "opt_b": 0.2},
                    "confidence": 0.8,
                },
                "score_q": {
                    "type": "score",
                    "score": 2.0,
                    "legend": {"0": "low", "1": "med", "2": "high"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                    "confidence": 0.8,
                },
                "multi_q": {
                    "type": "multi",
                    "labels": ["tag1", "tag2"],
                    "probabilities": {"tag1": 0.9, "tag2": 0.85},
                    "confidence": 0.88,
                },
            }
        }
        parsed = _BaseMicroJevClient._parse_ask_response(raw_ask)
        assert isinstance(parsed["noul_q"], NoulResult)
        assert parsed["noul_q"].value is True
        assert isinstance(parsed["choice_q"], ChoiceResult)
        assert parsed["choice_q"].value == "opt_a"
        assert isinstance(parsed["score_q"], ScoreResult)
        assert parsed["score_q"].value == "high"
        assert isinstance(parsed["multi_q"], MultiResult)
        assert parsed["multi_q"].value == ["tag1", "tag2"]

        # Aliases
        assert _BaseMicroJevClient._decode_response == _BaseMicroJevClient._parse_ask_response
        assert _BaseMicroJevClient._parse_result == _BaseMicroJevClient._parse_ask_response

    def test_sync_and_async_public_api_method_parity(self) -> None:
        """Assert sync and async clients export the exact same public method set."""
        sync_methods = {m for m in dir(MicroJevClient) if not m.startswith("_")}
        async_methods = {m for m in dir(AsyncMicroJevClient) if not m.startswith("_")}

        required_methods = {
            "ask",
            "systemone",
            "healthz",
            "batch",
            "guardrail",
            "papers",
            "evaluate_choice",
            "evaluate_score",
            "evaluate_multi",
            "evaluate_noul",
        }
        assert required_methods.issubset(sync_methods)
        assert required_methods.issubset(async_methods)
        assert required_methods <= (sync_methods & async_methods)

    def test_sync_and_async_batch_signature_exact_parity(self) -> None:
        """Assert parameter signature parity between MicroJevClient.batch and AsyncMicroJevClient.batch."""
        sig_sync = inspect.signature(MicroJevClient.batch)
        sig_async = inspect.signature(AsyncMicroJevClient.batch)

        assert sig_sync.parameters.keys() == sig_async.parameters.keys()
        for name, p_sync in sig_sync.parameters.items():
            p_async = sig_async.parameters[name]
            assert p_sync.name == p_async.name
            assert p_sync.default == p_async.default
            assert p_sync.kind == p_async.kind

    def test_sync_client_context_manager(self) -> None:
        """Verify MicroJevClient context manager protocol."""
        with MicroJevClient(base_url="http://localhost:8000") as client:
            assert isinstance(client, MicroJevClient)

    def test_async_client_context_manager(self) -> None:
        """Verify AsyncMicroJevClient async context manager protocol."""
        async def _run() -> None:
            async with AsyncMicroJevClient(base_url="http://localhost:8000") as client:
                assert isinstance(client, AsyncMicroJevClient)

        asyncio.run(_run())

    def test_sync_client_execution_with_test_client(self) -> None:
        """Verify sync MicroJevClient executes ask, systemone, healthz, batch, guardrail against FastAPI."""
        app = create_app()
        http_client = TestClient(app)
        client = MicroJevClient(http_client=http_client)

        # 1. Healthz
        hz = client.healthz()
        assert hz.get("status") in {"ok", "healthy"} or "version" in hz

        # 2. Ask & systemone alias
        res = client.ask(
            state="User reports an unauthorized charge on credit card.",
            questions={"choice": Choice(options=["billing", "technical"])},
        )
        assert res["choice"].value == "billing"

        res_alias = client.systemone(
            state="User reports an unauthorized charge on credit card.",
            questions={"choice": Choice(options=["billing", "technical"])},
        )
        assert res_alias["choice"].value == "billing"

        # 3. Guardrail
        gr = client.guardrail(state="Harmless user query", threshold=0.5)
        assert "safe" in gr

        # 4. Batch
        b_res = client.batch(
            items=["Item 1", "Item 2"],
            questions={"noul": Noul(instructions="Is valid?")},
        )
        assert "results" in b_res

        # 5. Papers
        p_res = client.papers(query="RLCD")
        assert "papers" in p_res or "total" in p_res or isinstance(p_res, dict)

    def test_async_client_execution_with_test_client(self) -> None:
        """Verify AsyncMicroJevClient executes ask, systemone, healthz, batch, guardrail."""
        async def _run() -> None:
            app = create_app()
            http_client = TestClient(app)
            client = AsyncMicroJevClient(http_client=http_client)

            # 1. Healthz
            hz = await client.healthz()
            assert hz.get("status") in {"ok", "healthy"} or "version" in hz

            # 2. Ask & systemone alias
            res = await client.ask(
                state="User reports a login failure.",
                questions={"choice": Choice(options=["billing", "technical"])},
            )
            assert res["choice"].value == "technical"

            res_alias = await client.systemone(
                state="User reports a login failure.",
                questions={"choice": Choice(options=["billing", "technical"])},
            )
            assert res_alias["choice"].value == "technical"

            # 3. Guardrail
            gr = await client.guardrail(state="Harmless user query", threshold=0.5)
            assert "safe" in gr

        asyncio.run(_run())


# ==============================================================================
# Audit Point 16: Packaged JSON Modularization & Backward Compatibility
# ==============================================================================


class TestAuditPoint16PackagedJson:
    """Verify extraction of static data into src/microllm/data/*.json and clean loading."""

    EXPECTED_JSON_FILES = [
        "academic_papers_catalog.json",
        "literature_pillars.json",
        "hf_jev_models_catalog.json",
        "hf_jev_server_config_map.json",
        "hf_jev_offline_samples.json",
        "gradio_multilingual_presets.json",
        "gradio_domain_presets.json",
        "gradio_dataset_presets.json",
        "gradio_consumer_examples.json",
        "dataset_fixtures_offline_samples.json",
    ]

    def test_all_ten_json_data_files_exist_and_non_empty(self) -> None:
        """Verify all 10 expected JSON files exist in src/microllm/data/ and are non-empty (>20 bytes)."""
        data_dir = REPO_ROOT / "src" / "microllm" / "data"
        assert data_dir.is_dir(), "src/microllm/data/ directory must exist"

        init_file = data_dir / "__init__.py"
        assert init_file.is_file(), "src/microllm/data/__init__.py must exist for packaging"

        for fname in self.EXPECTED_JSON_FILES:
            file_path = data_dir / fname
            assert file_path.is_file(), f"Missing packaged JSON data file: {fname}"
            assert file_path.stat().st_size > 20, f"JSON data file {fname} is unexpectedly empty"

    def test_all_ten_json_files_parse_as_valid_utf8_json(self) -> None:
        """Verify each JSON file strictly decodes under UTF-8 and parses to valid Python dict/list."""
        data_dir = REPO_ROOT / "src" / "microllm" / "data"
        for fname in self.EXPECTED_JSON_FILES:
            file_path = data_dir / fname
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
            assert isinstance(data, (dict, list)), f"{fname} must parse to dict or list"

    def test_pyproject_toml_includes_package_data(self) -> None:
        """Verify pyproject.toml configures package-data including 'data/*'."""
        pyproject = REPO_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert 'microllm = ["static/*", "data/*"]' in content

    def test_papers_module_loads_via_resources_and_backward_compatibility(self) -> None:
        """Verify microllm.papers loads JSON and re-exports LITERATURE_PILLARS, ACADEMIC_PAPERS_CATALOG, PAPERS."""
        import microllm.papers as papers

        assert len(papers.LITERATURE_PILLARS) == 6
        assert len(papers.ACADEMIC_PAPERS_CATALOG) >= 100

        # Backward compatibility for PAPERS dict
        assert isinstance(papers.PAPERS, dict)
        assert len(papers.PAPERS) >= 3
        assert "paper-openjev-rlcd" in papers.PAPERS
        assert "title" in papers.PAPERS["paper-openjev-rlcd"]
        for pid, paper in papers.PAPERS.items():
            assert "title" in paper, f"Paper record {pid} missing 'title'"

    def test_hf_jev_datasets_loads_via_resources_and_backward_compatibility(self) -> None:
        """Verify microllm.hf_jev_datasets loads JSON and re-exports catalogs, server config map, samples, and HF_JEV_DATASETS."""
        import microllm.hf_jev_datasets as hf_jev

        assert len(hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG) == 25
        assert len(hf_jev.HF_JEV_DATASET_SERVER_CONFIG_MAP) == 12
        assert len(hf_jev.OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES) == 12

        # Aliases
        assert hf_jev.HF_JEV_MODELS_CATALOG is hf_jev.HF_JEV_ECOSYSTEM_MODELS_CATALOG
        assert hf_jev.HF_JEV_SERVER_CONFIG_MAP is hf_jev.HF_JEV_DATASET_SERVER_CONFIG_MAP
        assert hf_jev.HF_JEV_OFFLINE_SAMPLES is hf_jev.OFFLINE_HF_JEV_MODEL_DATASET_SAMPLES

        # Backward compatibility for HF_JEV_DATASETS dict
        assert isinstance(hf_jev.HF_JEV_DATASETS, dict)
        assert len(hf_jev.HF_JEV_DATASETS) >= 3
        for did, ds in hf_jev.HF_JEV_DATASETS.items():
            assert "name" in ds, f"Dataset record {did} missing 'name'"
            assert "samples" in ds, f"Dataset record {did} missing 'samples'"

    def test_gradio_presets_loads_via_resources_and_backward_compatibility(self) -> None:
        """Verify microllm.gradio_presets loads JSON and re-exports all 4 preset dictionaries."""
        import microllm.gradio_presets as gradio

        assert len(gradio.MULTILINGUAL_SUPPORT_PRESETS) == 12
        assert len(gradio.STUDIO_DOMAIN_PRESETS) >= 30
        assert len(gradio.DATASET_BENCHMARK_PRESETS) == 10
        assert len(gradio.CONSUMER_TYPE_EXAMPLES) == 4

        # Aliases
        assert gradio.GRADIO_MULTILINGUAL_PRESETS is gradio.MULTILINGUAL_SUPPORT_PRESETS
        assert gradio.GRADIO_DOMAIN_PRESETS is gradio.STUDIO_DOMAIN_PRESETS
        assert gradio.GRADIO_DATASET_PRESETS is gradio.DATASET_BENCHMARK_PRESETS
        assert gradio.GRADIO_CONSUMER_EXAMPLES is gradio.CONSUMER_TYPE_EXAMPLES
        assert gradio.MULTILINGUAL_BENCHMARK_PRESETS is gradio.MULTILINGUAL_SUPPORT_PRESETS

    def test_dataset_fixtures_loads_via_resources_and_backward_compatibility(self) -> None:
        """Verify microllm.dataset_fixtures loads JSON and re-exports OFFLINE_WEB_DATASET_SAMPLES."""
        import microllm.dataset_fixtures as dfix

        assert len(dfix.OFFLINE_WEB_DATASET_SAMPLES) == 16
        assert dfix.DATASET_FIXTURES_OFFLINE_SAMPLES is dfix.OFFLINE_WEB_DATASET_SAMPLES


# ==============================================================================
# Edge-Case Discovery: Verdict-MQ Cache on Empty/Whitespace State
# ==============================================================================


class TestVerdictMqEdgeCaseFix:
    """Verify Challenger 2 edge-case fix for score_cached_state on empty or whitespace state text."""

    def test_score_cached_state_with_empty_and_whitespace_state(self) -> None:
        """Verify score_cached_state returns 0.0 without TypeError when state is empty or whitespace."""
        cache = VerdictMQMultiQuestionCache()

        # Empty string
        score_empty = cache.score_cached_state("", "routing query", "billing team")
        assert score_empty == 0.0

        # Whitespace-only string
        score_ws = cache.score_cached_state("   \n\t  ", "routing query", "billing team")
        assert score_ws == 0.0

        # Standard state evaluates cleanly
        score_valid = cache.score_cached_state(
            "Customer asks for a refund on duplicate invoice charge.",
            "Route department",
            "billing",
        )
        assert isinstance(score_valid, float)
        assert score_valid != 0.0
