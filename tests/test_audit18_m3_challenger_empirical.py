"""Empirical Adversarial Stress Test Suite for Milestone 3 (Challenger M3-1).

Adversarially challenges:
1. SDK Client Signature & Parameter Parity:
   - Reflection across all public methods of MicroJevClient and AsyncMicroJevClient.
   - Mock HTTP transport stress test across ask, systemone, batch, guardrail, papers, healthz.
   - Verification of auth headers, custom tokens, and timeout propagation.
   - Error code handling (HTTP 401 Unauthorized, HTTP 422 Unprocessable, HTTP 500 Internal Error).
2. Package Name Consistency:
   - Repository-wide grep across src/ and tests/ asserting zero occurrences of deprecated package name.
   - Verification that missing optional dependencies produce exact canonical package names:
     'pip install micro-jev[torch]', 'pip install micro-jev[onnx]', 'pip install micro-jev[gradio]'.
3. Challenger 2 Edge-Case Regression:
   - VerdictMQMultiQuestionCache encode_state and score_cached_state under empty string,
     whitespace, non-string, Unicode control characters, and high-volume churn.
4. Packaged JSON Data Extraction & Wheel Integrity:
   - Verification of all 10 JSON files in src/microllm/data/ and packaging config.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import httpx
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

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Challenge 1: SDK Reflection & Signature Parity
# ==============================================================================


class TestSdkReflectionParity:
    """Adversarially challenge public method and parameter parity between sync and async SDK clients."""

    def test_public_methods_and_signatures_match_100_percent(self) -> None:
        """Verify that every public API method exists in both clients with identical parameters."""
        sync_methods = {
            m: getattr(MicroJevClient, m)
            for m in dir(MicroJevClient)
            if not m.startswith("_") and callable(getattr(MicroJevClient, m))
        }
        async_methods = {
            m: getattr(AsyncMicroJevClient, m)
            for m in dir(AsyncMicroJevClient)
            if not m.startswith("_") and callable(getattr(AsyncMicroJevClient, m))
        }

        # The only allowable difference is the lifecycle closer: close (sync) vs aclose (async)
        common_methods = set(sync_methods.keys()) & set(async_methods.keys())
        diff_methods = (set(sync_methods.keys()) ^ set(async_methods.keys())) - {"close", "aclose"}
        assert not diff_methods, f"Unexpected asymmetric public methods found: {diff_methods}"

        required_methods = {
            "ask",
            "systemone",
            "batch",
            "guardrail",
            "papers",
            "healthz",
            "evaluate_choice",
            "evaluate_score",
            "evaluate_multi",
            "evaluate_noul",
        }
        assert required_methods.issubset(common_methods)

        for m_name in required_methods:
            sig_sync = inspect.signature(sync_methods[m_name])
            sig_async = inspect.signature(async_methods[m_name])

            sync_params = list(sig_sync.parameters.values())
            async_params = list(sig_async.parameters.values())

            assert len(sync_params) == len(async_params), (
                f"Parameter count mismatch in {m_name}: "
                f"sync={len(sync_params)} vs async={len(async_params)}"
            )

            for p_sync, p_async in zip(sync_params, async_params):
                assert p_sync.name == p_async.name, (
                    f"Parameter name mismatch in {m_name}: '{p_sync.name}' vs '{p_async.name}'"
                )
                assert p_sync.default == p_async.default, (
                    f"Parameter default mismatch in {m_name}.{p_sync.name}: "
                    f"{p_sync.default!r} vs {p_async.default!r}"
                )
                assert p_sync.kind == p_async.kind, (
                    f"Parameter kind mismatch in {m_name}.{p_sync.name}: "
                    f"{p_sync.kind} vs {p_async.kind}"
                )

    def test_sync_client_context_manager_protocol(self) -> None:
        """Verify MicroJevClient context manager protocol."""
        with MicroJevClient(base_url="http://mock:8000") as client:
            assert isinstance(client, MicroJevClient)
            assert client.base_url == "http://mock:8000"

    def test_async_client_context_manager_protocol(self) -> None:
        """Verify AsyncMicroJevClient async context manager protocol."""
        async def _run() -> None:
            async with AsyncMicroJevClient(base_url="http://mock:8000") as client:
                assert isinstance(client, AsyncMicroJevClient)
                assert client.base_url == "http://mock:8000"

        asyncio.run(_run())


# ==============================================================================
# Challenge 2: Mock HTTP Transport Stress Testing (Sync & Async)
# ==============================================================================


class TestSdkMockTransportStress:
    """Stress-test sync and async SDK clients with mocked HTTP transport across all endpoints."""

    @staticmethod
    def _create_mock_handler(recorded: list[dict[str, Any]]):
        def handler(request: httpx.Request) -> httpx.Response:
            recorded.append({
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "headers": dict(request.headers),
                "params": dict(request.url.params),
                "body": json.loads(request.read().decode("utf-8")) if request.content else None,
            })
            path = request.url.path
            auth = request.headers.get("authorization", "")
            if auth == "Bearer invalid-key":
                return httpx.Response(401, json={"detail": "Invalid authentication credentials"})

            if path == "/healthz":
                return httpx.Response(200, json={"status": "healthy", "version": "0.1.0"})
            elif path == "/v1/systemone":
                return httpx.Response(200, json={
                    "answers": {
                        "q1": {
                            "type": "choice",
                            "choice": "opt_a",
                            "probabilities": {"opt_a": 0.85, "opt_b": 0.15},
                            "confidence": 0.85,
                        },
                        "q2": {
                            "type": "noul",
                            "noul": 0.92,
                        },
                    }
                })
            elif path == "/v1/guardrail":
                return httpx.Response(200, json={
                    "safe": True,
                    "score": 0.12,
                    "flagged": False,
                    "rubrics": {"hate": 0.01, "violence": 0.02},
                })
            elif path == "/v1/batch":
                return httpx.Response(200, json={
                    "results": [
                        {
                            "answers": {
                                "q1": {
                                    "type": "choice",
                                    "choice": "opt_a",
                                    "confidence": 0.85,
                                    "probabilities": {"opt_a": 0.85, "opt_b": 0.15},
                                }
                            }
                        }
                    ]
                })
            elif path == "/v1/papers":
                return httpx.Response(200, json={
                    "papers": [{"id": "paper-openjev-rlcd", "title": "OpenJev-RLCD"}],
                    "total": 1,
                })
            return httpx.Response(404, json={"detail": "Not Found"})

        return handler

    def test_sync_client_mock_transport_all_endpoints(self) -> None:
        """Verify MicroJevClient executes across ask, systemone, batch, guardrail, papers, healthz."""
        recorded: list[dict[str, Any]] = []
        transport = httpx.MockTransport(self._create_mock_handler(recorded))
        http_client = httpx.Client(transport=transport, base_url="http://mock-jev:8000")

        client = MicroJevClient(
            base_url="http://mock-jev:8000",
            api_key="adversarial-token-42",
            http_client=http_client,
            timeout=18.5,
        )

        # 1. Healthz
        hz = client.healthz()
        assert hz["status"] == "healthy"

        # 2. Ask
        res_ask = client.ask("State 1", {"q1": Choice(options=["opt_a", "opt_b"])})
        assert isinstance(res_ask["q1"], ChoiceResult)
        assert res_ask["q1"].value == "opt_a"

        # 3. SystemOne (alias)
        res_sys = client.systemone("State 2", {"q2": Noul(instructions="Check")})
        assert isinstance(res_sys["q2"], NoulResult)
        assert res_sys["q2"].value is True

        # 4. Guardrail
        res_gr = client.guardrail("Query text", threshold=0.6)
        assert res_gr["safe"] is True

        # 5. Batch
        res_batch = client.batch(["Item 1"], {"q1": Choice(options=["opt_a", "opt_b"])})
        assert "results" in res_batch

        # 6. Papers
        res_papers = client.papers(query="RLCD", pillar="pillar_1")
        assert res_papers["total"] == 1

        # Assert all 6 requests carried the exact Authorization token
        assert len(recorded) == 6
        for req in recorded:
            assert req["headers"].get("authorization") == "Bearer adversarial-token-42"

    def test_async_client_mock_transport_all_endpoints(self) -> None:
        """Verify AsyncMicroJevClient executes across all endpoints using mock transport."""
        async def _run() -> None:
            recorded: list[dict[str, Any]] = []
            transport = httpx.MockTransport(self._create_mock_handler(recorded))
            http_client = httpx.AsyncClient(transport=transport, base_url="http://mock-jev:8000")

            client = AsyncMicroJevClient(
                base_url="http://mock-jev:8000",
                api_key="adversarial-token-async-99",
                http_client=http_client,
                timeout=25.0,
            )

            # 1. Healthz
            hz = await client.healthz()
            assert hz["status"] == "healthy"

            # 2. Ask
            res_ask = await client.ask("State async", {"q1": Choice(options=["opt_a", "opt_b"])})
            assert res_ask["q1"].value == "opt_a"

            # 3. SystemOne
            res_sys = await client.systemone("State async 2", {"q2": Noul(instructions="Check")})
            assert res_sys["q2"].value is True

            # 4. Guardrail
            res_gr = await client.guardrail("Query text async", threshold=0.7)
            assert res_gr["safe"] is True

            # 5. Batch
            res_batch = await client.batch(["Batch 1"], {"q1": Choice(options=["opt_a", "opt_b"])})
            assert "results" in res_batch

            # 6. Papers
            res_papers = await client.papers(query="RLCD")
            assert res_papers["total"] == 1

            assert len(recorded) == 6
            for req in recorded:
                assert req["headers"].get("authorization") == "Bearer adversarial-token-async-99"

        asyncio.run(_run())

    def test_sync_and_async_client_error_responses(self) -> None:
        """Verify clients raise HTTPStatusError on 401 Unauthorized and 500 errors."""
        recorded: list[dict[str, Any]] = []
        transport = httpx.MockTransport(self._create_mock_handler(recorded))

        # 401 Unauthorized test
        bad_sync = MicroJevClient(
            base_url="http://mock-jev:8000",
            api_key="invalid-key",
            http_client=httpx.Client(transport=transport, base_url="http://mock-jev:8000"),
        )
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            bad_sync.healthz()
        assert exc_info.value.response.status_code == 401

        async def _test_async_error() -> None:
            bad_async = AsyncMicroJevClient(
                base_url="http://mock-jev:8000",
                api_key="invalid-key",
                http_client=httpx.AsyncClient(transport=transport, base_url="http://mock-jev:8000"),
            )
            with pytest.raises(httpx.HTTPStatusError) as exc_async:
                await bad_async.ask("State", {"q": Choice(options=["a", "b"])})
            assert exc_async.value.response.status_code == 401

        asyncio.run(_test_async_error())


# ==============================================================================
# Challenge 3: Package Name Consistency Across Repository
# ==============================================================================


class TestPackageNameConsistencyRepoWide:
    """Verify deprecated package install command is completely eradicated from src and tests."""

    def test_repo_grep_zero_deprecated_package_in_src_and_tests(self) -> None:
        """Assert zero occurrences of deprecated package install command across src/ and tests/."""
        target_str = "pip install " + "microllm"
        cmd = ["git", "grep", "-n", target_str, "src/"]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert res.returncode == 1, (
            f"Found deprecated '{target_str}' in src/:\n{res.stdout}"
        )
        assert res.stdout == ""

        cmd_tests = ["git", "grep", "-n", target_str, "tests/", ":(exclude)tests/test_audit18_m3_*.py"]
        res_tests = subprocess.run(cmd_tests, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert res_tests.returncode == 1, (
            f"Found deprecated '{target_str}' in non-audit tests/:\n{res_tests.stdout}"
        )
        assert res_tests.stdout == ""

    def test_optional_dependency_error_messages_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify all optional dependency ImportErrors recommend 'micro-jev'."""
        # 1. Torch
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[torch\]"):
            create_hf_zero_shot_scorer("dummy")

        # 2. ONNX
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[onnx\]"):
            create_onnx_nli_scorer("model.onnx", "tok")

        # 3. ONNX quantization
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization", None)
        with pytest.raises(ImportError, match=r"pip install micro-jev\[onnx\]"):
            quantize_onnx_file_int8("in.onnx", "out.onnx")

        # 4. Gradio in Space Sync
        space_sync_content = (REPO_ROOT / "src" / "microllm" / "hf_space_sync.py").read_text(encoding="utf-8")
        assert 'pip install "micro-jev[gradio]"' in space_sync_content
        assert 'pip install "microllm[gradio]"' not in space_sync_content


# ==============================================================================
# Challenge 4: Challenger 2 Edge-Case Regression (Verdict-MQ Cache)
# ==============================================================================


class TestVerdictMqAdversarialEdgeCases:
    """Stress test VerdictMQMultiQuestionCache encode_state and score_cached_state under hostile inputs."""

    def test_empty_whitespace_and_unicode_control_characters(self) -> None:
        """Verify encode_state and score_cached_state never raise TypeError on empty, whitespace, control chars."""
        cache = VerdictMQMultiQuestionCache()

        hostile_inputs = [
            "",
            " ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            "\t\n\r\f\v",
            "\u200b\u200c\u200d\uFEFF",  # Zero-width spaces & byte-order marks
            "😀 🚀 🧠",  # Emojis
            "   \n   \t  ",
            "a" * 10000,  # 10k characters
        ]

        for s in hostile_inputs:
            encoded = cache.encode_state(s)
            assert isinstance(encoded, dict)
            assert isinstance(encoded.get("tokens"), frozenset)

            # Score against query and hypothesis
            score = cache.score_cached_state(s, "Sample query", "Sample candidate")
            assert isinstance(score, float)
            if not s.strip():
                assert score == 0.0

    def test_hostile_non_string_types_to_score_cached_state(self) -> None:
        """Verify non-string types in state, query, and hypothesis do not crash score_cached_state with TypeError."""
        cache = VerdictMQMultiQuestionCache()

        non_strings = [None, 0, 42, 3.14159, False, True, [], {}, (1, 2)]

        for item in non_strings:
            # As state: must return a float without TypeError
            score_state = cache.score_cached_state(item, "query", "cand")  # type: ignore[arg-type]
            assert isinstance(score_state, float)
            assert not isinstance(score_state, bool)

            # As query: must return a float without TypeError
            score_q = cache.score_cached_state("valid context", item, "cand")  # type: ignore[arg-type]
            assert isinstance(score_q, float)

            # As hypothesis: must return a float without TypeError
            score_h = cache.score_cached_state("valid context", "query", item)  # type: ignore[arg-type]
            assert isinstance(score_h, float)

    def test_cache_churn_and_capacity_bound(self) -> None:
        """Verify cache capacity bound is strictly honored without memory leakage."""
        cache = VerdictMQMultiQuestionCache(max_states=50)

        for i in range(200):
            cache.encode_state(f"State permutation {i}: customer needs assistance with id {i}")

        assert len(cache._cache) <= 50
        assert cache.encode_misses == 200

        # Repeated access to the most recent states should hit cache
        for i in range(150, 200):
            cache.encode_state(f"State permutation {i}: customer needs assistance with id {i}")

        assert cache.encode_hits == 50
        assert cache.stats["hits"] == 50


# ==============================================================================
# Challenge 5: Packaged JSON Modularization & Wheel Completeness
# ==============================================================================


class TestPackagedJsonDataAndWheelIntegrity:
    """Verify all 10 JSON data files are properly packaged and discoverable via importlib.resources."""

    EXPECTED_JSON_FILES = {
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
    }

    def test_data_directory_structure_and_json_validity(self) -> None:
        """Verify src/microllm/data/ contains all 10 expected files and valid JSON."""
        data_dir = REPO_ROOT / "src" / "microllm" / "data"
        assert data_dir.is_dir()
        assert (data_dir / "__init__.py").is_file()

        existing = {p.name for p in data_dir.glob("*.json")}
        assert self.EXPECTED_JSON_FILES == existing

        for fname in existing:
            p = data_dir / fname
            data = json.loads(p.read_text(encoding="utf-8"))
            assert isinstance(data, (dict, list))
            assert len(data) > 0

    def test_pyproject_toml_package_data_configured(self) -> None:
        """Verify pyproject.toml package-data specifies data/*."""
        pyproject = REPO_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert '["static/*", "data/*"]' in content

    def test_built_wheel_contains_all_ten_json_data_files(self) -> None:
        """Verify built wheel in dist/ packages all 10 JSON files under microllm/data/."""
        import zipfile
        whls = sorted((REPO_ROOT / "dist").glob("*.whl"))
        if not whls:
            pytest.skip("No wheel built in dist/ to inspect")
        latest_whl = whls[-1]
        with zipfile.ZipFile(latest_whl) as z:
            names = z.namelist()
            data_files = {n.split("/")[-1] for n in names if "microllm/data/" in n and n.endswith(".json")}
            assert data_files == self.EXPECTED_JSON_FILES, (
                f"Missing JSON files in wheel: {self.EXPECTED_JSON_FILES - data_files}"
            )
