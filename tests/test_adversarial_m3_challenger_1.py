"""Empirical Adversarial Challenge Test Suite for Milestone 3 (m3_challenger_1).

Audits and stress-tests:
1. CLI flag matrix, precedence, aliases, and exit code 2 on invalid arguments.
2. SystemOneRequest schema validation under malformed, extra, missing, and invalid types.
3. JevCompat §6 error envelope compliance on HTTP 422 responses.
4. MicroJevClient and AsyncMicroJevClient payload serialization, precedence, and typed helpers.
5. Telemetry HTTP headers (X-Strategy-Used, X-Cascade-Triggered, X-Neural-Model) under various routing modes.
6. Structured text support (string, list, dict) under fast, neural, and auto strategies.
7. All 4 question types (choice, score, multi, noul) under strategy selection.
8. Absolute network isolation (zero outbound network connections).
9. Empirical exposure of the factory create_app() environment variable fallback bug.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from microllm.client import (
    AsyncMicroJevClient,
    Choice,
    ChoiceResult,
    MicroJevClient,
    MultiResult,
    NoulResult,
    ScoreResult,
)
from microllm.schemas import (
    ChoiceQuestion,
    SystemOneRequest,
)
from microllm.server import create_app, run_cli


@pytest.fixture(autouse=True)
def clean_environment() -> Any:
    """Ensure no leftover environment variables pollute between test cases."""
    os.environ.pop("MICROLLM_API_KEY", None)
    os.environ.pop("MICROLLM_MULTILINGUAL_STRATEGY", None)
    os.environ.pop("MICROLLM_NEURAL_MODEL", None)
    yield
    os.environ.pop("MICROLLM_API_KEY", None)
    os.environ.pop("MICROLLM_MULTILINGUAL_STRATEGY", None)
    os.environ.pop("MICROLLM_NEURAL_MODEL", None)


# ==============================================================================
# 1. CLI Strategy Flags, Precedence, and Exit Codes
# ==============================================================================


class TestCliAdversarialChallenges:
    """Stress-tests microllm serve CLI options, error codes, and env var overrides."""

    def test_cli_invalid_strategy_exits_code_2(self) -> None:
        """CLI must exit with code 2 when an invalid strategy is passed."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["serve", "--strategy", "quantum"])
        assert exc_info.value.code == 2

    def test_cli_invalid_multilingual_strategy_alias_exits_code_2(self) -> None:
        """CLI must exit with code 2 when an invalid --multilingual-strategy is passed."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["serve", "--multilingual-strategy", "nonexistent"])
        assert exc_info.value.code == 2

    def test_cli_invalid_neural_model_exits_code_2(self) -> None:
        """CLI must exit with code 2 when an invalid neural model is passed."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["serve", "--neural-model", "bert-large"])
        assert exc_info.value.code == 2

    def test_cli_missing_flag_arguments_exits_code_2(self) -> None:
        """CLI must exit with code 2 when arguments are missing values."""
        with pytest.raises(SystemExit) as exc1:
            run_cli(["serve", "--strategy"])
        assert exc1.value.code == 2

        with pytest.raises(SystemExit) as exc2:
            run_cli(["serve", "--neural-model"])
        assert exc2.value.code == 2

    def test_cli_serve_flag_matrix_export_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies CLI properly populates environment variables before uvicorn.run."""
        matrix = [
            ("fast", "minilm"),
            ("fast", "mdeberta"),
            ("neural", "minilm"),
            ("neural", "mdeberta"),
            ("auto", "minilm"),
            ("auto", "mdeberta"),
        ]
        for strat, model in matrix:
            recorded: dict[str, Any] = {}
            monkeypatch.setattr("uvicorn.run", lambda *a, **k: recorded.update(k))
            code = run_cli([
                "serve",
                "--strategy", strat,
                "--neural-model", model,
                "--host", "0.0.0.0",
                "--port", "9090",
                "--api-key", "secret-test-key",
            ])
            assert code == 0
            assert os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY") == strat
            assert os.environ.get("MICROLLM_NEURAL_MODEL") == model
            assert os.environ.get("MICROLLM_API_KEY") == "secret-test-key"
            assert recorded["host"] == "0.0.0.0"
            assert recorded["port"] == 9090
        os.environ.pop("MICROLLM_API_KEY", None)
        os.environ.pop("MICROLLM_MULTILINGUAL_STRATEGY", None)
        os.environ.pop("MICROLLM_NEURAL_MODEL", None)

    def test_cli_precedence_over_existing_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI flags must take precedence over pre-existing environment variables."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "fast")
        monkeypatch.setenv("MICROLLM_NEURAL_MODEL", "minilm")
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        run_cli(["serve", "--strategy", "neural", "--neural-model", "mdeberta"])
        assert os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] == "neural"
        assert os.environ["MICROLLM_NEURAL_MODEL"] == "mdeberta"

    def test_cli_default_fallback_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no CLI flags are given, defaults should be read from environment variables."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "neural")
        monkeypatch.setenv("MICROLLM_NEURAL_MODEL", "mdeberta")
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        run_cli(["serve"])
        assert os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] == "neural"
        assert os.environ["MICROLLM_NEURAL_MODEL"] == "mdeberta"


# ==============================================================================
# 2. SystemOneRequest Schema Adversarial Probing & JevCompat §6 Errors
# ==============================================================================


class TestSchemaValidationAdversarialChallenges:
    """Stress-tests SystemOneRequest edge cases, malformed payloads, and 422 structures."""

    def test_system_one_request_extra_fields_allowed(self) -> None:
        """Extra metadata fields (trace_id, client_id, tags) must be permitted."""
        req = SystemOneRequest(
            state="Incoming payment dispute.",
            questions={"q1": ChoiceQuestion(type="choice", criteria={"refund": None, "charge": None})},
            trace_id="req-12345",  # type: ignore[call-arg]
            tags=["billing", "tier-1"],  # type: ignore[call-arg]
        )
        assert req.state == "Incoming payment dispute."
        assert getattr(req, "trace_id", None) == "req-12345" or req.model_extra.get("trace_id") == "req-12345"

    def test_system_one_request_missing_required_fields(self) -> None:
        """Missing state or questions must raise ValidationError."""
        with pytest.raises(ValidationError) as exc1:
            SystemOneRequest.model_validate({"questions": {"q1": {"type": "choice", "criteria": {"a": None, "b": None}}}})
        assert any("state" in str(err["loc"]) for err in exc1.value.errors())

        with pytest.raises(ValidationError) as exc2:
            SystemOneRequest.model_validate({"state": "hello world"})
        assert any("questions" in str(err["loc"]) for err in exc2.value.errors())

    def test_system_one_request_malformed_state_types(self) -> None:
        """Non-string/dict/list states or infinite/nan floats must raise ValidationError."""
        with pytest.raises(ValidationError):
            SystemOneRequest(
                state=None,  # type: ignore[arg-type]
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
            )

        with pytest.raises(ValidationError):
            SystemOneRequest(
                state=12345,  # type: ignore[arg-type]
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
            )

        with pytest.raises(ValidationError):
            SystemOneRequest(
                state={"nan_key": float("nan")},
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
            )

    def test_system_one_request_empty_questions_or_empty_question_id(self) -> None:
        """Empty questions dictionary or empty string key must be rejected."""
        with pytest.raises(ValidationError) as exc1:
            SystemOneRequest(state="State text", questions={})
        assert "questions must be a non-empty object" in str(exc1.value)

        with pytest.raises(ValidationError) as exc2:
            SystemOneRequest(
                state="State text",
                questions={"": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
            )
        assert "Every question id must be a non-empty string" in str(exc2.value)

    def test_system_one_request_case_sensitivity(self) -> None:
        """Strategies and neural models are case-sensitive; uppercase raises ValidationError."""
        with pytest.raises(ValidationError):
            SystemOneRequest(
                state="State text",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                multilingual_strategy="FAST",  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationError):
            SystemOneRequest(
                state="State text",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                neural_model="MINILM",  # type: ignore[arg-type]
            )

    def test_jevcompat_error_envelope_422_on_invalid_strategy(self) -> None:
        """POST /v1/systemone with invalid strategy must return 422 with JevCompat §6 loc format."""
        app = create_app(api_key=None)
        client = TestClient(app)
        payload = {
            "state": "Bonjour, je voudrais un remboursement.",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {"remboursement": None, "livraison": None},
                }
            },
            "multilingual_strategy": "quantum",
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0
        for err in body["detail"]:
            assert "loc" in err
            assert "msg" in err
            assert "type" in err
            # JevCompat §6: first element of loc must be 'body'
            assert err["loc"][0] == "body"

    def test_jevcompat_error_envelope_422_on_invalid_neural_model(self) -> None:
        """POST /v1/systemone with invalid neural model must return 422 starting with 'body'."""
        app = create_app(api_key=None)
        client = TestClient(app)
        payload = {
            "state": "Bonjour, je voudrais un remboursement.",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {"remboursement": None, "livraison": None},
                }
            },
            "neural_model": "llama-3-8b",
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body
        assert body["detail"][0]["loc"][0] == "body"
        assert "neural_model" in body["detail"][0]["loc"]


# ==============================================================================
# 3. SDK Client Forwarding and Typed Helpers
# ==============================================================================


class TestMicroJevClientAdversarialChallenges:
    """Stress-tests MicroJevClient and AsyncMicroJevClient payload composition and typed methods."""

    def test_client_defaults_vs_per_call_overrides(self) -> None:
        """Per-call ask() parameters must override client-level defaults in outgoing payloads."""
        captured_payloads: list[dict[str, Any]] = []

        class MockResponse:
            def raise_for_status(self) -> None:
                pass
            def json(self) -> dict[str, Any]:
                return {"answers": {"q1": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.9}}}

        class MockHttpClient:
            def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> MockResponse:
                captured_payloads.append(json)
                return MockResponse()

        # Case 1: Client defaults used
        client = MicroJevClient(
            http_client=MockHttpClient(),
            multilingual_strategy="fast",
            neural_model="mdeberta",
        )
        client.ask("test state", {"q1": Choice(options=["a", "b"])})
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["multilingual_strategy"] == "fast"
        assert captured_payloads[0]["neural_model"] == "mdeberta"

        # Case 2: Per-call override
        client.ask(
            "test state",
            {"q1": Choice(options=["a", "b"])},
            multilingual_strategy="neural",
            neural_model="minilm",
        )
        assert len(captured_payloads) == 2
        assert captured_payloads[1]["multilingual_strategy"] == "neural"
        assert captured_payloads[1]["neural_model"] == "minilm"

    def test_client_typed_helpers_payload_and_return_types(self) -> None:
        """Typed helpers evaluate_choice, evaluate_score, evaluate_multi, evaluate_noul work correctly."""
        captured_payloads: list[dict[str, Any]] = []

        class DynamicMockHttpClient:
            def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> Any:
                captured_payloads.append(json)
                q_dict = json["questions"]
                answers: dict[str, Any] = {}
                for qid, q_spec in q_dict.items():
                    q_type = q_spec["type"]
                    if q_type == "choice":
                        answers[qid] = {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.85, "tech": 0.15}, "confidence": 0.85}
                    elif q_type == "score":
                        answers[qid] = {"type": "score", "score": 4.0, "legend": {"1": "low", "4": "high"}, "probabilities": {"1": 0.1, "4": 0.9}, "confidence": 0.9}
                    elif q_type == "multi":
                        answers[qid] = {"type": "multi", "labels": ["refund", "urgent"], "probabilities": {"refund": 0.9, "urgent": 0.8}, "confidence": 0.85}
                    elif q_type == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.88}

                class Resp:
                    def raise_for_status(self) -> None:
                        pass
                    def json(self) -> dict[str, Any]:
                        return {"answers": answers}
                return Resp()

        client = MicroJevClient(http_client=DynamicMockHttpClient())

        # 1. evaluate_choice
        c_res = client.evaluate_choice("Refund please", ["billing", "tech"], multilingual_strategy="fast")
        assert isinstance(c_res, ChoiceResult)
        assert c_res.value == "billing"
        assert c_res.confidence == 0.85
        assert captured_payloads[-1]["multilingual_strategy"] == "fast"

        # 2. evaluate_score
        s_res = client.evaluate_score("Rating test", [1, 2, 3, 4, 5], neural_model="mdeberta")
        assert isinstance(s_res, ScoreResult)
        assert s_res.score == 4.0
        assert s_res.value == "high"
        assert captured_payloads[-1]["neural_model"] == "mdeberta"

        # 3. evaluate_multi
        m_res = client.evaluate_multi("Multiple tags", ["refund", "urgent", "other"], threshold=0.7)
        assert isinstance(m_res, MultiResult)
        assert m_res.labels == ["refund", "urgent"]
        assert m_res.value == ["refund", "urgent"]

        # 4. evaluate_noul
        n_res = client.evaluate_noul("Is proposition true?", instructions="Check validity")
        assert isinstance(n_res, NoulResult)
        assert n_res.probability == 0.88
        assert n_res.value is True

    def test_async_client_typed_helpers_and_overrides(self) -> None:
        """AsyncMicroJevClient typed helpers asynchronously forward strategies and return typed objects."""
        captured_payloads: list[dict[str, Any]] = []

        class AsyncMockHttpClient:
            async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> Any:
                captured_payloads.append(json)
                q_dict = json["questions"]
                answers: dict[str, Any] = {}
                for qid, q_spec in q_dict.items():
                    q_type = q_spec["type"]
                    if q_type == "choice":
                        answers[qid] = {"type": "choice", "choice": "es_opt", "probabilities": {"es_opt": 0.95}, "confidence": 0.95}
                    elif q_type == "score":
                        answers[qid] = {"type": "score", "score": 3.0, "legend": {"3": "mid"}, "probabilities": {"3": 1.0}, "confidence": 1.0}
                    elif q_type == "multi":
                        answers[qid] = {"type": "multi", "labels": ["alpha"], "probabilities": {"alpha": 0.9}, "confidence": 0.9}
                    elif q_type == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.35}

                class Resp:
                    def raise_for_status(self) -> None:
                        pass
                    def json(self) -> dict[str, Any]:
                        return {"answers": answers}
                return Resp()

        client = AsyncMicroJevClient(
            http_client=AsyncMockHttpClient(),
            multilingual_strategy="auto",
            neural_model="minilm",
        )

        async def _run() -> None:
            c_res = await client.evaluate_choice("Ticket ES", ["es_opt"], multilingual_strategy="neural")
            assert isinstance(c_res, ChoiceResult)
            assert c_res.value == "es_opt"
            assert captured_payloads[-1]["multilingual_strategy"] == "neural"

            s_res = await client.evaluate_score("Rating", [1, 2, 3])
            assert isinstance(s_res, ScoreResult)
            assert s_res.score == 3.0

            m_res = await client.evaluate_multi("Multi test", ["alpha", "beta"])
            assert isinstance(m_res, MultiResult)
            assert m_res.labels == ["alpha"]

            n_res = await client.evaluate_noul("Noul test")
            assert isinstance(n_res, NoulResult)
            assert n_res.probability == 0.35
            assert n_res.value is False

        asyncio.run(_run())

    def test_sdk_error_propagation_on_server_error(self) -> None:
        """SDK must raise httpx.HTTPStatusError when server returns an error code."""
        class ErrorHttpClient:
            def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> Any:
                req = httpx.Request("POST", "http://localhost:8000/v1/systemone")
                resp = httpx.Response(status_code=422, json={"detail": [{"loc": ["body", "multilingual_strategy"], "msg": "Invalid", "type": "value_error"}]}, request=req)
                resp.raise_for_status()

        client = MicroJevClient(http_client=ErrorHttpClient())
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.ask("state", {"q1": Choice(options=["a", "b"])})
        assert exc_info.value.response.status_code == 422


# ==============================================================================
# 4. HTTP Telemetry Headers & Auto-Cascade Routing
# ==============================================================================


class TestServerTelemetryHeadersAndRouting:
    """Empirically tests HTTP response headers and backend routing in server.py."""

    def test_explicit_fast_strategy_telemetry_headers(self) -> None:
        """When multilingual_strategy='fast', response headers must reflect fast path."""
        app = create_app(api_key=None)
        client = TestClient(app)
        payload = {
            "state": "I need a quick refund for my subscription payment.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "criteria": {"refund": None, "technical_support": None},
                }
            },
            "multilingual_strategy": "fast",
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Cascade-Triggered") == "false"

    def test_explicit_neural_strategy_telemetry_headers(self) -> None:
        """When multilingual_strategy='neural', headers must reflect neural model."""
        app = create_app(api_key=None)
        client = TestClient(app)
        payload = {
            "state": "Mi pedido no ha llegado y el repartidor no responde.",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {"envío": None, "facturación": None},
                }
            },
            "multilingual_strategy": "neural",
            "neural_model": "mdeberta",
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "neural"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"

    def test_auto_cascade_clear_ticket_resolves_fast(self) -> None:
        """Clear query with dominant localized keywords must stay on fast gear in auto mode."""
        app = create_app(api_key=None, multilingual_strategy="auto")
        client = TestClient(app)
        payload = {
            "state": "Refund billing invoice charge transaction payment receipt.",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {"refund": None, "hardware_failure": None},
                }
            },
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Cascade-Triggered") == "false"

    def test_explicit_arguments_create_app_work(self) -> None:
        """create_app() respects strategy and neural_model when passed explicitly as arguments."""
        app = create_app(api_key=None, multilingual_strategy="fast", neural_model="mdeberta")
        client = TestClient(app)
        payload = {
            "state": "Bonjour tout le monde.",
            "questions": {"q1": {"type": "choice", "criteria": {"salut": None, "adieu": None}}},
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"

    def test_create_app_factory_honors_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When uvicorn invokes `create_app()` factory (0 arguments), environment variables
        MICROLLM_MULTILINGUAL_STRATEGY and MICROLLM_NEURAL_MODEL are correctly respected."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "fast")
        monkeypatch.setenv("MICROLLM_NEURAL_MODEL", "mdeberta")

        app = create_app(api_key=None)  # Invoked as uvicorn factory: 0 args
        client = TestClient(app)
        payload = {
            "state": "Unrelated ambiguous phrase with no keywords xyz 123.",
            "questions": {"q1": {"type": "choice", "criteria": {"alpha": None, "beta": None}}},
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"


# ==============================================================================
# 5. Question Types and Structured Text Across Strategies
# ==============================================================================


class TestQuestionTypesAndStructuredText:
    """Stress-tests all question types (score, multi, noul) and structured state across strategies."""

    def test_score_question_under_fast_and_neural(self) -> None:
        """ScoreQuestion rubrics evaluate correctly under fast and neural strategies."""
        app = create_app(api_key=None)
        client = TestClient(app)
        for strat in ["fast", "neural"]:
            payload = {
                "state": "Service was absolutely terrible, nothing worked.",
                "questions": {
                    "satisfaction": {
                        "type": "score",
                        "instructions": "Rate customer satisfaction",
                        "criteria": ["terrible experience", "poor service", "average", "great support"],
                    }
                },
                "multilingual_strategy": strat,
            }
            resp = client.post("/v1/systemone", json=payload)
            assert resp.status_code == 200
            data = resp.json()["answers"]["satisfaction"]
            assert data["type"] == "score"
            assert isinstance(data["score"], (int, float))

    def test_multi_question_under_fast_and_neural(self) -> None:
        """MultiQuestion independent labels evaluate correctly under fast and neural strategies."""
        app = create_app(api_key=None)
        client = TestClient(app)
        for strat in ["fast", "neural"]:
            payload = {
                "state": "I need urgent help with my billing invoice and shipping delivery.",
                "questions": {
                    "tags": {
                        "type": "multi",
                        "criteria": {"billing": None, "shipping": None, "technical": None},
                        "threshold": 0.4,
                    }
                },
                "multilingual_strategy": strat,
            }
            resp = client.post("/v1/systemone", json=payload)
            assert resp.status_code == 200
            data = resp.json()["answers"]["tags"]
            assert data["type"] == "multi"
            assert isinstance(data["labels"], list)

    def test_noul_question_under_fast_and_neural(self) -> None:
        """NoulQuestion binary proposition evaluates correctly under fast and neural strategies."""
        app = create_app(api_key=None)
        client = TestClient(app)
        for strat in ["fast", "neural"]:
            payload = {
                "state": "URGENT: system is down, all servers unresponsive!",
                "questions": {
                    "is_critical": {
                        "type": "noul",
                        "instructions": "Is this a critical emergency?",
                    }
                },
                "multilingual_strategy": strat,
            }
            resp = client.post("/v1/systemone", json=payload)
            assert resp.status_code == 200
            data = resp.json()["answers"]["is_critical"]
            assert data["type"] == "noul"
            assert isinstance(data["noul"], float)

    def test_structured_state_dict_and_list_across_strategies(self) -> None:
        """Structured state as list or dict is properly handled across strategies."""
        app = create_app(api_key=None)
        client = TestClient(app)

        # List state
        resp_list = client.post(
            "/v1/systemone",
            json={
                "state": ["Customer says: I want to cancel", "Agent says: Can I offer a discount?"],
                "questions": {"q": {"type": "choice", "criteria": {"cancellation": None, "upgrade": None}}},
                "multilingual_strategy": "fast",
            },
        )
        assert resp_list.status_code == 200

        # Dict state
        resp_dict = client.post(
            "/v1/systemone",
            json={
                "state": {"ticket_id": 999, "body": "Gostaria do reembolso.", "lang": "pt"},
                "questions": {"q": {"type": "choice", "criteria": {"reembolso": None, "suporte": None}}},
                "multilingual_strategy": "neural",
                "neural_model": "minilm",
            },
        )
        assert resp_dict.status_code == 200


# ==============================================================================
# 6. Total Offline Isolation & Zero Network Calls
# ==============================================================================


class TestOfflineNetworkIsolationAdversarialChallenge:
    """Strictly proves that no outbound external TCP connections are made."""

    def test_zero_external_network_connections_during_full_pipeline(self) -> None:
        """Asserts socket.connect raises if any outbound non-localhost IP is accessed."""
        original_connect = socket.socket.connect

        def guarded_connect(self: socket.socket, address: Any) -> Any:
            host = address[0] if isinstance(address, tuple) and len(address) > 0 else str(address)
            # Allow local loopback addresses only
            if str(host) in ("127.0.0.1", "localhost", "::1"):
                return original_connect(self, address)
            raise RuntimeError(f"FORBIDDEN: Attempted external outbound network call to {address}!")

        with patch.object(socket.socket, "connect", guarded_connect):
            # 1. Run CLI argument parsing and validation
            with pytest.raises(SystemExit):
                run_cli(["serve", "--strategy", "bad"])

            # 2. Server creation and execution across EN, FR, ES, PT
            app = create_app(api_key=None, multilingual_strategy="auto", neural_model="minilm")
            client = TestClient(app)

            multilingual_samples = [
                ("EN: Please refund my subscription charge.", {"refund": None, "shipping": None}),
                ("FR: Bonjour, je souhaite un remboursement rapide.", {"remboursement": None, "livraison": None}),
                ("ES: Hola, quiero la devolución de mi dinero.", {"devolución": None, "envío": None}),
                ("PT: Olá, gostaria do reembolso da minha compra.", {"reembolso": None, "entrega": None}),
            ]

            for state_text, criteria in multilingual_samples:
                # Fast path
                r_fast = client.post(
                    "/v1/systemone",
                    json={
                        "state": state_text,
                        "questions": {"q": {"type": "choice", "criteria": criteria}},
                        "multilingual_strategy": "fast",
                    },
                )
                assert r_fast.status_code == 200

                # Neural path
                r_neural = client.post(
                    "/v1/systemone",
                    json={
                        "state": state_text,
                        "questions": {"q": {"type": "choice", "criteria": criteria}},
                        "multilingual_strategy": "neural",
                        "neural_model": "minilm",
                    },
                )
                assert r_neural.status_code == 200

                # Auto cascade path
                r_auto = client.post(
                    "/v1/systemone",
                    json={
                        "state": state_text,
                        "questions": {"q": {"type": "choice", "criteria": criteria}},
                        "multilingual_strategy": "auto",
                    },
                )
                assert r_auto.status_code == 200
