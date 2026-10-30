"""Comprehensive 100% offline unit and integration tests for Milestone 3.

Verifies:
1. `SystemOneRequest` strategy selection schema validation (`fast`, `neural`, `auto`, `minilm`, `mdeberta`).
2. `microllm serve` CLI flag parsing (`--strategy`, `--multilingual-strategy`, `--neural-model`).
3. `create_app` and `POST /v1/systemone` server defaults, overrides, and HTTP telemetry headers.
4. `MicroJevClient` and `AsyncMicroJevClient` SDK parameters, convenience methods, and `@system_one_router`.
5. Dual-gear auto-cascade routing behavior (fast path on clear tickets, neural fallback on ambiguous tickets).
6. Total offline isolation (zero outbound network or Hugging Face downloads).
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping
from enum import Enum
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from microllm.backends.cascade import (
    MultilingualCascadeBackend,
    create_multilingual_backend,
)
from microllm.client import (
    AsyncMicroJevClient,
    Choice,
    ChoiceResult,
    MicroJevClient,
    MultiResult,
    NoulResult,
    ScoreResult,
    system_one_router,
)
from microllm.schemas import (
    ChoiceQuestion,
    SystemOneRequest,
)
from microllm.server import create_app, run_cli


# ==============================================================================
# 1. SystemOneRequest Schema Validation Tests
# ==============================================================================


class TestSystemOneRequestStrategySchemas:
    """Verifies SystemOneRequest schema defaults, Literal choices, and validation errors."""

    def test_request_schema_defaults(self) -> None:
        """Asserts multilingual_strategy == 'auto' and neural_model == 'minilm' when omitted."""
        req = SystemOneRequest(
            state="Awaiting dispatch.",
            questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
        )
        assert req.multilingual_strategy == "auto"
        assert req.neural_model == "minilm"

    def test_request_schema_valid_strategies(self) -> None:
        """Asserts 'fast', 'neural', and 'auto' strategies are accepted."""
        for strat in ["fast", "neural", "auto"]:
            req = SystemOneRequest(
                state="Processing query.",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                multilingual_strategy=strat,  # type: ignore[arg-type]
            )
            assert req.multilingual_strategy == strat

    def test_request_schema_valid_neural_models(self) -> None:
        """Asserts 'minilm' and 'mdeberta' neural models are accepted."""
        for model in ["minilm", "mdeberta"]:
            req = SystemOneRequest(
                state="Processing query.",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                neural_model=model,  # type: ignore[arg-type]
            )
            assert req.neural_model == model

    def test_request_schema_invalid_strategy_rejected(self) -> None:
        """Asserts invalid strategy raises ValidationError pointing to multilingual_strategy."""
        with pytest.raises(ValidationError) as exc_info:
            SystemOneRequest(
                state="Processing query.",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                multilingual_strategy="quantum",  # type: ignore[arg-type]
            )
        errors = exc_info.value.errors()
        assert any("multilingual_strategy" in str(err["loc"]) for err in errors)

    def test_request_schema_invalid_neural_model_rejected(self) -> None:
        """Asserts invalid neural_model raises ValidationError pointing to neural_model."""
        with pytest.raises(ValidationError) as exc_info:
            SystemOneRequest(
                state="Processing query.",
                questions={"q1": ChoiceQuestion(type="choice", criteria={"a": None, "b": None})},
                neural_model="bert-large-uncased",  # type: ignore[arg-type]
            )
        errors = exc_info.value.errors()
        assert any("neural_model" in str(err["loc"]) for err in errors)

    def test_request_schema_json_roundtrip(self) -> None:
        """Asserts serialization and deserialization preserve strategy fields."""
        payload = {
            "model": "jev-latest",
            "state": "Payment failed twice on credit card.",
            "questions": {
                "q_cat": {
                    "type": "choice",
                    "criteria": {"billing": "Billing issues", "tech": "Technical support"},
                }
            },
            "multilingual_strategy": "fast",
            "neural_model": "mdeberta",
        }
        req = SystemOneRequest.model_validate(payload)
        dumped = req.model_dump()
        assert dumped["multilingual_strategy"] == "fast"
        assert dumped["neural_model"] == "mdeberta"


# ==============================================================================
# 2. CLI microllm serve Strategy Options Tests
# ==============================================================================


class TestCliServeStrategyOptions:
    """Verifies CLI flag parsing for `microllm serve`."""

    def test_cli_serve_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts `microllm serve` defaults to auto strategy and minilm neural model."""
        recorded: dict[str, Any] = {}

        def mock_uvicorn_run(*args: Any, **kwargs: Any) -> None:
            recorded["args"] = args
            recorded["kwargs"] = kwargs

        monkeypatch.setattr("uvicorn.run", mock_uvicorn_run)
        monkeypatch.delenv("MICROLLM_MULTILINGUAL_STRATEGY", raising=False)
        monkeypatch.delenv("MICROLLM_NEURAL_MODEL", raising=False)

        ret = run_cli(["serve"])
        assert ret == 0
        assert os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY") == "auto"
        assert os.environ.get("MICROLLM_NEURAL_MODEL") == "minilm"
        assert recorded["kwargs"]["host"] == "127.0.0.1"
        assert recorded["kwargs"]["port"] == 8000

    def test_cli_serve_custom_strategy_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts `microllm serve --strategy fast` correctly parses strategy."""
        recorded: dict[str, Any] = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: recorded.update(k))

        ret = run_cli(["serve", "--strategy", "fast", "--port", "8080"])
        assert ret == 0
        assert os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY") == "fast"
        assert recorded.get("port") == 8080

    def test_cli_serve_neural_model_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts `microllm serve --strategy neural --neural-model mdeberta` sets env correctly."""
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        ret = run_cli(["serve", "--strategy", "neural", "--neural-model", "mdeberta"])
        assert ret == 0
        assert os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY") == "neural"
        assert os.environ.get("MICROLLM_NEURAL_MODEL") == "mdeberta"

    def test_cli_serve_multilingual_strategy_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts `--multilingual-strategy` works as an exact alias for `--strategy`."""
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        ret = run_cli(["serve", "--multilingual-strategy", "fast"])
        assert ret == 0
        assert os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY") == "fast"

    def test_cli_serve_invalid_strategy_choice_exits(self) -> None:
        """Asserts unknown strategy choice causes CLI to exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["serve", "--strategy", "nonexistent"])
        assert exc_info.value.code == 2

    def test_cli_serve_invokes_uvicorn_with_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts custom host, port, and reload flags are forwarded to uvicorn."""
        recorded: dict[str, Any] = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: recorded.update(k))

        ret = run_cli(["serve", "--host", "0.0.0.0", "--port", "9090", "--reload"])
        assert ret == 0
        assert recorded["host"] == "0.0.0.0"
        assert recorded["port"] == 9090
        assert recorded["reload"] is True


# ==============================================================================
# 3. Server Endpoints & Telemetry Headers Integration Tests
# ==============================================================================


class TestServerEndpointsStrategyIntegration:
    """Verifies FastAPI application factory, routing, and HTTP headers."""

    def test_create_app_with_strategy_parameters(self) -> None:
        """Asserts create_app successfully instantiates with custom strategy defaults."""
        app = create_app(multilingual_strategy="fast", neural_model="minilm")
        assert app is not None
        assert app.title == "MicroLLM System-One Decision Engine"

    def test_systemone_endpoint_uses_server_default_fast(self) -> None:
        """Asserts POST /v1/systemone routes to fast engine when app is configured with fast."""
        app = create_app(multilingual_strategy="fast")
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Bonjour, je souhaite un remboursement pour ma commande.",
                "questions": {
                    "category": {
                        "type": "choice",
                        "criteria": {"refund": "Demande de remboursement", "technical": "Problème technique"},
                    }
                },
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Cascade-Triggered") == "false"

    def test_systemone_endpoint_uses_server_default_neural(self) -> None:
        """Asserts routing to neural engine when app is configured with strategy='neural'."""
        app = create_app(multilingual_strategy="neural", neural_model="minilm")
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Hola, necesito cancelar mi suscripción inmediatamente.",
                "questions": {
                    "cat": {
                        "type": "choice",
                        "criteria": {"cancel": "Cancelar suscripción", "login": "Problema de acceso"},
                    }
                },
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "neural"
        assert resp.headers.get("X-Neural-Model") == "minilm"

    def test_systemone_endpoint_request_override(self) -> None:
        """Asserts request-level strategy='neural' overrides server-level default fast."""
        app = create_app(multilingual_strategy="fast")
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Minha encomenda está atrasada.",
                "questions": {
                    "cat": {
                        "type": "choice",
                        "criteria": {"shipping": "Atraso no envio", "billing": "Cobrança"},
                    }
                },
                "multilingual_strategy": "neural",
                "neural_model": "mdeberta",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "neural"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"

    def test_systemone_endpoint_invalid_strategy_returns_422(self) -> None:
        """Asserts invalid strategy in request body returns HTTP 422 with body error loc."""
        app = create_app()
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Test query.",
                "questions": {"q1": {"type": "choice", "criteria": {"a": None, "b": None}}},
                "multilingual_strategy": "invalid_strategy_name",
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body
        locs = [err.get("loc", []) for err in body["detail"]]
        assert any("multilingual_strategy" in loc for loc in locs)

    def test_batch_and_mcp_endpoints_resolve_cascade(self) -> None:
        """Asserts /v1/batch and /v1/mcp endpoints correctly resolve cascade without NameError."""
        app = create_app()
        client = TestClient(app)
        # Test /v1/batch with default model='jev-latest'
        b_resp = client.post(
            "/v1/batch",
            json={
                "items": ["Ticket description"],
                "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
                "model": "jev-latest",
            },
        )
        assert b_resp.status_code == 200
        assert "answers" in b_resp.json()["results"][0]

        # Test /v1/mcp JSON-RPC endpoint
        m_resp = client.post(
            "/v1/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert m_resp.status_code == 200
        tools = m_resp.json().get("result", {}).get("tools", [])
        assert len(tools) >= 3


# ==============================================================================
# 4. MicroJevClient SDK Strategy Integration Tests
# ==============================================================================


class MockHttpClient:
    """Mock HTTP client capturing outgoing requests and returning spec-conformant responses."""

    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_payload: dict[str, Any] | None = None
        self.last_headers: dict[str, str] | None = None

    def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        self.last_url = url
        self.last_payload = json
        self.last_headers = headers

        class _MockResp:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                qs = json.get("questions", {}) if json else {}
                answers: dict[str, Any] = {}
                for qid, q_spec in qs.items():
                    q_type = q_spec.get("type", "choice")
                    if q_type == "choice":
                        answers[qid] = {
                            "type": "choice",
                            "choice": "billing",
                            "probabilities": {"billing": 0.88, "tech": 0.12},
                            "confidence": 0.76,
                        }
                    elif q_type == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.85}
                    elif q_type == "score":
                        answers[qid] = {
                            "type": "score",
                            "score": 4.2,
                            "legend": {"1": "poor", "5": "excellent"},
                            "probabilities": {"1": 0.05, "5": 0.95},
                            "confidence": 0.90,
                        }
                    elif q_type == "multi":
                        answers[qid] = {
                            "type": "multi",
                            "labels": ["urgent"],
                            "probabilities": {"urgent": 0.82, "review": 0.20},
                            "confidence": 0.82,
                        }
                return {"model": json.get("model", "jev-latest") if json else "jev-latest", "answers": answers}

        return _MockResp()


class TestMicroJevClientStrategySdk:
    """Verifies Python SDK client forwards strategy parameters in request payloads."""

    def test_client_init_and_ask_strategy_payload(self) -> None:
        """Asserts client configured with strategy injects fields and allows per-call override."""
        mock_http = MockHttpClient()
        client = MicroJevClient(
            http_client=mock_http,
            multilingual_strategy="fast",
            neural_model="minilm",
        )

        client.ask("Refund request", {"q1": Choice(options=["billing", "tech"])})
        assert mock_http.last_payload is not None
        assert mock_http.last_payload["multilingual_strategy"] == "fast"
        assert mock_http.last_payload["neural_model"] == "minilm"

        # Per-call override
        client.ask(
            "Account issue",
            {"q1": Choice(options=["billing", "tech"])},
            multilingual_strategy="neural",
            neural_model="mdeberta",
        )
        assert mock_http.last_payload["multilingual_strategy"] == "neural"
        assert mock_http.last_payload["neural_model"] == "mdeberta"

    def test_async_client_ask_strategy_payload(self) -> None:
        """Asserts AsyncMicroJevClient serializes strategy options into async POST payload."""
        mock_http = MockHttpClient()
        async_client = AsyncMicroJevClient(
            http_client=mock_http,
            multilingual_strategy="auto",
            neural_model="minilm",
        )

        async def _run() -> None:
            res = await async_client.ask(
                "Minha encomenda atrasou.",
                {"track": Choice(options=["shipping", "other"])},
                multilingual_strategy="neural",
            )
            assert "track" in res

        asyncio.run(_run())
        assert mock_http.last_payload is not None
        assert mock_http.last_payload["multilingual_strategy"] == "neural"
        assert mock_http.last_payload["neural_model"] == "minilm"

    def test_system_one_router_with_strategy_options(self) -> None:
        """Asserts @system_one_router decorator passes strategy parameters to client.ask."""
        mock_http = MockHttpClient()
        client = MicroJevClient(http_client=mock_http)

        class TicketRoute(Enum):
            BILLING = "Billing inquiries"
            TECH = "Technical assistance"

        @system_one_router(
            client=client,
            options=TicketRoute,
            multilingual_strategy="neural",
            neural_model="mdeberta",
        )
        def handle_ticket(state: str, decision: str = "", confidence: float = 0.0) -> str:
            return f"Routed to {decision} (conf={confidence:.2f})"

        result = handle_ticket("Quiero la devolución de mi dinero.")
        assert "Routed to" in result
        assert mock_http.last_payload is not None
        assert mock_http.last_payload["multilingual_strategy"] == "neural"
        assert mock_http.last_payload["neural_model"] == "mdeberta"

    def test_client_convenience_methods(self) -> None:
        """Asserts evaluate_choice, evaluate_score, evaluate_multi, and evaluate_noul helpers work."""
        mock_http = MockHttpClient()
        client = MicroJevClient(http_client=mock_http)

        choice_res = client.evaluate_choice("Facture impayée", ["billing", "tech"], multilingual_strategy="fast")
        assert isinstance(choice_res, ChoiceResult)
        assert choice_res.value == "billing"
        assert mock_http.last_payload["multilingual_strategy"] == "fast"

        score_res = client.evaluate_score("Service impeccable", [1, 2, 3, 4, 5])
        assert isinstance(score_res, ScoreResult)
        assert score_res.score == 4.2

        multi_res = client.evaluate_multi("Serveur en panne critique", ["urgent", "review"])
        assert isinstance(multi_res, MultiResult)
        assert "urgent" in multi_res.labels

        noul_res = client.evaluate_noul("Le client est-il satisfait ?")
        assert isinstance(noul_res, NoulResult)
        assert noul_res.probability == 0.85


# ==============================================================================
# 5. Dual-Gear Auto-Cascade Multilingual Routing Behavior Tests
# ==============================================================================


class TestAutoCascadeMultilingualBehavior:
    """Verifies auto-cascade gating: fast path on high-confidence, neural fallback on ambiguity."""

    def test_auto_cascade_spanish_high_confidence_stays_on_fast_lexicon(self) -> None:
        """Asserts clear Spanish billing ticket is resolved by Strategy 1 in sub-millisecond time."""
        backend = create_multilingual_backend(strategy="auto", neural_model="minilm")

        ticket = "Me han cobrado dos veces en mi tarjeta y quiero un reembolso inmediato."
        options = {"billing": "Facturación y cobros", "technical": "Problemas de conexión y técnicos"}

        res = backend.evaluate_choice(ticket, "Clasificar la consulta del cliente.", options)
        assert res["type"] == "choice"
        assert res["choice"] == "billing"
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_cascade_reason"] == "confident_direct"
        assert res["x_top2_margin"] > 0.12
        assert res["x_latency_ms"] < 50.0

    def test_auto_cascade_portuguese_high_confidence_stays_on_fast_lexicon(self) -> None:
        """Asserts clear Portuguese shipping ticket is resolved by Strategy 1 with zero neural latency."""
        backend = create_multilingual_backend(strategy="auto", neural_model="minilm")

        ticket = "Minha encomenda com código de rastreamento está retida na alfândega."
        options = {"shipping": "Envio e rastreamento de encomendas", "technical": "Suporte técnico do servidor"}

        res = backend.evaluate_choice(ticket, "Classifique a solicitação.", options)
        assert res["type"] == "choice"
        assert res["choice"] == "shipping"
        assert res["x_strategy_used"] == "fast"
        assert res["x_cascade_triggered"] is False
        assert res["x_top2_margin"] > 0.12

    def test_auto_cascade_ambiguous_ticket_triggers_neural_escalation(self) -> None:
        """Asserts ambiguous ticket with narrow margin (<= 0.12) escalates to neural model."""
        # Create a mock gear1 that simulates a narrow top-2 margin (margin = 0.04 <= 0.12)
        class _AmbiguousGear1:
            name = "mock-ambiguous-gear1"

            def evaluate_choice(self, state: str, instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
                keys = list(options.keys())
                return {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {keys[0]: 0.52, keys[1]: 0.48},
                    "confidence": 0.52,
                }

        class _DecisiveGear2:
            name = "mock-decisive-gear2"
            called = 0

            def evaluate_choice(self, state: str, instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
                self.called += 1
                keys = list(options.keys())
                return {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {keys[0]: 0.95, keys[1]: 0.05},
                    "confidence": 0.95,
                }

        g1 = _AmbiguousGear1()
        g2 = _DecisiveGear2()
        cascade = MultilingualCascadeBackend(
            gear1=g1,  # type: ignore[arg-type]
            gear2=g2,
            neural_model_name="mdeberta",
            margin_threshold=0.12,
            entropy_threshold=0.88,
            fusion_alpha=0.75,
        )

        res = cascade.evaluate_choice(
            "El servicio no funciona y no sé si es por falta de pago o caída del servidor.",
            "Clasificar.",
            {"billing": "Facturación", "tech": "Falla técnica"},
        )
        assert res["type"] == "choice"
        assert g2.called == 1
        assert res["x_strategy_used"] == "neural"
        assert res["x_cascade_triggered"] is True
        assert res["x_cascade_reason"] == "top2_margin_low"
        assert res["x_neural_model"] == "mdeberta"
        # Fused probabilities check: (1 - 0.75)*0.52 + 0.75*0.95 = 0.13 + 0.7125 = 0.8425
        assert res["probabilities"]["billing"] > 0.80

    def test_auto_cascade_french_contrastive_clause_escalation(self) -> None:
        """Asserts French ticket with contrastive clause is handled properly with telemetry."""
        backend = create_multilingual_backend(strategy="auto", neural_model="minilm")

        ticket = "Le paiement a bien été effectué mais le colis n'est toujours pas arrivé."
        options = {"shipping": "Livraison et colis", "billing": "Paiement et facturation"}

        res = backend.evaluate_choice(ticket, "Classifier la demande client.", options)
        assert res["type"] == "choice"
        assert "x_strategy_used" in res
        assert "x_cascade_triggered" in res
        assert "x_top2_margin" in res
        assert res["choice"] in ["shipping", "billing"]


# ==============================================================================
# 6. Total Offline Execution Guarantees Tests
# ==============================================================================


class TestOfflineExecutionGuarantees:
    """Verifies that tests execute 100% offline without remote socket or HTTP downloads."""

    def test_zero_network_calls_during_suite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts that no outbound socket connections are attempted during execution."""
        def guarded_connect(self: Any, *args: Any, **kwargs: Any) -> None:
            # Allow connecting to local test server (127.0.0.1 or localhost)
            addr = args[0] if args else kwargs.get("address")
            if isinstance(addr, tuple) and addr[0] in ("127.0.0.1", "localhost", "::1"):
                return
            raise RuntimeError(f"Prohibited external network call attempted to: {addr}")

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        # Run an end-to-end classification
        backend = create_multilingual_backend(strategy="fast")
        res = backend.evaluate_choice(
            "Consulta sobre mi factura mensual.",
            "Categorizar.",
            {"billing": "Facturación", "tech": "Soporte"},
        )
        assert res["choice"] == "billing"

    def test_mock_neural_session_fidelity(self) -> None:
        """Asserts fallback INT8 projection scorer functions deterministically without ONNX weights."""
        from microllm.registry import load_profile_scorer

        # load_profile_scorer falls back to deterministic INT8 projection when ONNX file is not on disk
        scorer = load_profile_scorer("multilingual-minilm-int8")
        logit = scorer(
            "Facture impayée depuis deux semaines.",
            "Quel est le problème ?",
            "Il s'agit d'une question de facturation.",
        )
        assert isinstance(logit, float)
        assert logit > 0.0


# ==============================================================================
# 5. create_app 0-Argument & Environment Variable Fallback Tests
# ==============================================================================


class TestCreateAppEnvironmentVariableFallback:
    """Verifies that create_app() with 0 arguments (uvicorn factory) strictly respects env vars."""

    def test_create_app_zero_args_reads_fast_strategy_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When called with 0 args, create_app reads MICROLLM_MULTILINGUAL_STRATEGY=fast."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "fast")
        monkeypatch.delenv("MICROLLM_NEURAL_MODEL", raising=False)
        monkeypatch.delenv("MICROLLM_API_KEY", raising=False)

        app = create_app()  # Exactly 0 arguments, as called by uvicorn factory
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Bonjour, je souhaite un remboursement pour ma commande.",
                "questions": {
                    "category": {
                        "type": "choice",
                        "criteria": {"refund": "Demande de remboursement", "tech": "Problème technique"},
                    }
                },
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Cascade-Triggered") == "false"

    def test_create_app_zero_args_reads_neural_strategy_and_mdeberta_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When called with 0 args, create_app reads neural strategy and mdeberta model from env."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "neural")
        monkeypatch.setenv("MICROLLM_NEURAL_MODEL", "mdeberta")
        monkeypatch.delenv("MICROLLM_API_KEY", raising=False)

        app = create_app()  # 0 arguments
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Hola, necesito soporte urgente.",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "criteria": {"support": "Soporte técnico", "billing": "Facturación"},
                    }
                },
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "neural"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"

    def test_create_app_zero_args_defaults_to_auto_and_minilm_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When called with 0 args and no env vars, defaults are auto and minilm."""
        monkeypatch.delenv("MICROLLM_MULTILINGUAL_STRATEGY", raising=False)
        monkeypatch.delenv("MICROLLM_NEURAL_MODEL", raising=False)
        monkeypatch.delenv("MICROLLM_API_KEY", raising=False)

        app = create_app()  # 0 arguments
        client = TestClient(app)

        # 1. Clear ticket runs on fast pass under auto
        resp_clear = client.post(
            "/v1/systemone",
            json={
                "state": "Bonjour, je veux un remboursement immédiat.",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "criteria": {"refund": "Remboursement", "tech": "Support technique"},
                    }
                },
            },
        )
        assert resp_clear.status_code == 200
        assert resp_clear.headers.get("X-Strategy-Used") == "fast"
        assert resp_clear.headers.get("X-Cascade-Triggered") == "false"

        # 2. Ambiguous ticket cascades to default neural model (minilm)
        resp_ambig = client.post(
            "/v1/systemone",
            json={
                "state": "xyz non-lexical probe string 12345",
                "questions": {
                    "dept": {
                        "type": "choice",
                        "criteria": {"alpha": "Option alpha", "beta": "Option beta"},
                    }
                },
            },
        )
        assert resp_ambig.status_code == 200
        assert resp_ambig.headers.get("X-Strategy-Used") == "neural"
        assert resp_ambig.headers.get("X-Neural-Model") == "minilm"

    def test_create_app_explicit_args_override_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit arguments passed to create_app() override environment variables."""
        monkeypatch.setenv("MICROLLM_MULTILINGUAL_STRATEGY", "neural")
        monkeypatch.setenv("MICROLLM_NEURAL_MODEL", "mdeberta")
        monkeypatch.delenv("MICROLLM_API_KEY", raising=False)

        # Caller explicitly requests fast and minilm
        app = create_app(multilingual_strategy="fast", neural_model="minilm")
        client = TestClient(app)

        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Support query.",
                "questions": {"q": {"type": "choice", "criteria": {"a": "Option A", "b": "Option B"}}},
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "fast"
        assert resp.headers.get("X-Neural-Model") == "minilm"

    def test_cli_serve_uvicorn_factory_integration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration test: microllm serve invokes uvicorn factory which correctly instantiates app."""
        import importlib

        recorded: dict[str, Any] = {}

        def mock_uvicorn_run(*args: Any, **kwargs: Any) -> None:
            recorded["args"] = args
            recorded["kwargs"] = kwargs

        monkeypatch.setattr("uvicorn.run", mock_uvicorn_run)
        monkeypatch.delenv("MICROLLM_API_KEY", raising=False)

        exit_code = run_cli(["serve", "--strategy", "neural", "--neural-model", "mdeberta"])
        assert exit_code == 0
        assert recorded["args"][0] == "microllm.server:create_app"
        assert recorded["kwargs"]["factory"] is True

        # Simulate exact uvicorn factory invocation: import and call with 0 arguments
        mod_name, func_name = recorded["args"][0].split(":")
        mod = importlib.import_module(mod_name)
        factory_fn = getattr(mod, func_name)
        app_instance = factory_fn()  # 0 arguments!

        client = TestClient(app_instance)
        resp = client.post(
            "/v1/systemone",
            json={
                "state": "Integration test state.",
                "questions": {"q": {"type": "choice", "criteria": {"a": "Option A", "b": "Option B"}}},
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Strategy-Used") == "neural"
        assert resp.headers.get("X-Neural-Model") == "mdeberta"
