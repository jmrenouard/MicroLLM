"""Automated conformance test suite covering the jevcompat 0.1 specification (32/32 MUST + SHOULDs)."""

from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient

from microllm.confidence import choice_confidence, score_confidence, score_expectation
from microllm.server import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def auth_client() -> TestClient:
    app = create_app(api_key="secret-jev-key")
    return TestClient(app)


def test_http_endpoint_and_envelope(client: TestClient) -> None:
    """Covers http.endpoint, http.json, http.content-type, response.envelope, response.usage."""
    payload = {
        "model": "jev-latest",
        "state": "Customer was billed twice for order #901 and requests an immediate refund.",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Is the customer asking for an immediate refund?",
                "criteria": {"true": "Requests immediate refund", "false": "General inquiry"},
            },
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {
                    "billing": "Invoices, double charges, refunds",
                    "technical": "Bugs, crashes, login issues",
                    "sales": None,
                },
            },
            "anger": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]

    data = resp.json()
    assert isinstance(data["model"], str)
    assert isinstance(data["answers"], dict)
    assert isinstance(data["usage"], dict)
    assert isinstance(data["usage"]["input_tokens"], int)
    assert isinstance(data["usage"]["output_tokens"], int)
    assert not isinstance(data["usage"]["input_tokens"], bool)
    assert data["usage"]["input_tokens"] >= 0
    assert data["usage"]["output_tokens"] >= 0
    assert set(data["answers"].keys()) == {"urgent", "team", "anger"}


def test_http_models_endpoint(client: TestClient) -> None:
    """Covers http.models."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    for m in data["models"]:
        assert isinstance(m["name"], str)
        assert isinstance(m["description"], str)
        assert len(m["release_date"]) == 10


def test_auth_ignored_when_off_and_enforced_when_on(
    client: TestClient, auth_client: TestClient
) -> None:
    """Covers auth.ignored-when-off, auth.bearer, auth.missing, auth.invalid."""
    minimal = {
        "state": "test state",
        "questions": {"q1": {"type": "noul", "instructions": "Is this a test?"}},
    }
    # When auth is off, Bearer header is accepted without error
    resp_off = client.post(
        "/v1/systemone",
        json=minimal,
        headers={"Authorization": "Bearer any-random-token"},
    )
    assert resp_off.status_code == 200

    # When auth is on, missing key -> 401/403 with error_type=authentication_error
    resp_missing = auth_client.post("/v1/systemone", json=minimal)
    assert resp_missing.status_code in (401, 403)
    assert resp_missing.json()["detail"]["error_type"] == "authentication_error"

    # Wrong key -> 401
    resp_wrong = auth_client.post(
        "/v1/systemone",
        json=minimal,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp_wrong.status_code == 401
    assert resp_wrong.json()["detail"]["error_type"] == "authentication_error"

    # Valid key -> 200
    resp_ok = auth_client.post(
        "/v1/systemone",
        json=minimal,
        headers={"Authorization": "Bearer secret-jev-key"},
    )
    assert resp_ok.status_code == 200


@pytest.mark.parametrize(
    "state_value",
    [
        "Plain string state",
        {"customer": "Priya", "tier": "enterprise", "items": [1, 2, 3]},
        [{"event": "login"}, {"event": "charge_failed", "amount": 49.99}],
    ],
)
def test_request_state_types_and_structured_text(
    client: TestClient, state_value: object
) -> None:
    """Covers request.state, request.structured-text, request.unknown-fields."""
    payload = {
        "model": "jev-latest",
        "state": state_value,
        "x_nonce": "nonce-12345",
        "extra_caller_field": {"nested": True},
        "questions": {
            "q_struct": {
                "type": "choice",
                "instructions": {"goal": "Classify event", "rules": ["Pick best match"]},
                "criteria": {
                    "auth": ["login", "logout"],
                    "payment": {"desc": "charge or invoice"},
                },
            }
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    ans = resp.json()["answers"]["q_struct"]
    assert ans["type"] == "choice"
    assert set(ans["probabilities"].keys()) == {"auth", "payment"}


def test_question_ids_and_unicode_byte_for_byte(client: TestClient) -> None:
    """Covers request.question-ids, request.unicode, choice.probability-keys."""
    qid = "q-1.a 🚀 한국어 Decision"
    opt_a = "결제_오류_💳"
    opt_b = "기술_지원_🛠️"
    payload = {
        "state": "고객이 두 번 결제되었습니다. 환불해 주세요.",
        "questions": {
            qid: {
                "type": "choice",
                "instructions": "어느 부서로 보내야 합니까?",
                "criteria": {opt_a: "청구 및 환불", opt_b: None},
            }
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert qid in answers
    ans = answers[qid]
    assert set(ans["probabilities"].keys()) == {opt_a, opt_b}
    assert ans["choice"] in {opt_a, opt_b}


@pytest.mark.parametrize(
    "noul_question",
    [
        {"type": "noul", "instructions": "Is it urgent?"},
        {"type": "noul", "instructions": "Is it urgent?", "criteria": None},
        {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "Yes urgent"}},
        {"type": "noul", "instructions": "Is it urgent?", "criteria": {"false": "Not urgent"}},
        {"type": "noul"},
    ],
)
def test_noul_variants_and_answer_shape(
    client: TestClient, noul_question: dict[str, object]
) -> None:
    """Covers noul.criteria, noul.no-instructions, noul.answer, response.extensions."""
    resp = client.post(
        "/v1/systemone",
        json={"state": "Server outage in prod!", "questions": {"n": noul_question}},
    )
    assert resp.status_code == 200
    ans = resp.json()["answers"]["n"]
    assert ans["type"] == "noul"
    assert 0.0 <= ans["noul"] <= 1.0
    assert math.isfinite(ans["noul"])
    # Noul MUST NOT have an unprefixed 'confidence' key
    assert "confidence" not in ans


@pytest.mark.parametrize("num_options", [2, 5, 26, 65, 130, 255])
def test_choice_option_counts_up_to_255_and_confidence(
    client: TestClient, num_options: int
) -> None:
    """Covers choice.options (2..255 tournament), choice.distribution, choice.argmax, confidence.formula."""
    criteria = {f"opt_{i:03d}": (f"Option number {i}" if i % 2 == 0 else None) for i in range(num_options)}
    resp = client.post(
        "/v1/systemone",
        json={
            "state": "Select option number 4 or 1.",
            "questions": {
                "big_choice": {
                    "type": "choice",
                    "instructions": "Pick the matching option",
                    "criteria": criteria,
                }
            },
        },
    )
    assert resp.status_code == 200
    ans = resp.json()["answers"]["big_choice"]
    probs = ans["probabilities"]
    assert len(probs) == num_options
    assert set(probs.keys()) == set(criteria.keys())
    assert abs(sum(probs.values()) - 1.0) <= 1e-4
    assert all(0.0 <= p <= 1.0 and math.isfinite(p) for p in probs.values())

    max_p = max(probs.values())
    assert probs[ans["choice"]] >= max_p - 0.005

    expected_conf = choice_confidence(probs)
    assert 0.0 <= ans["confidence"] <= 1.0
    assert abs(ans["confidence"] - expected_conf) <= 0.02


@pytest.mark.parametrize("num_levels", [2, 3, 5, 10])
def test_score_levels_expectation_and_confidence(
    client: TestClient, num_levels: int
) -> None:
    """Covers score.levels (2..10), score.legend, score.expectation, confidence.formula."""
    levels: list[object] = [f"Level_{i}" for i in range(num_levels)]
    if num_levels == 3:
        levels[1] = {"label": "Medium", "detail": "Moderate severity"}
    resp = client.post(
        "/v1/systemone",
        json={
            "state": "The service is moderately degraded.",
            "questions": {
                "sev": {
                    "type": "score",
                    "instructions": "Rate the severity",
                    "criteria": levels,
                }
            },
        },
    )
    assert resp.status_code == 200
    ans = resp.json()["answers"]["sev"]
    assert ans["type"] == "score"
    expected_keys = [str(i) for i in range(num_levels)]
    assert list(ans["legend"].keys()) == expected_keys
    assert list(ans["probabilities"].keys()) == expected_keys
    assert all(v is not None for v in ans["legend"].values())

    probs = ans["probabilities"]
    assert abs(sum(probs.values()) - 1.0) <= 1e-4
    expected_score = score_expectation(probs)
    assert abs(ans["score"] - expected_score) <= 1e-4

    expected_conf = score_confidence(probs)
    assert 0.0 <= ans["confidence"] <= 1.0
    assert abs(ans["confidence"] - expected_conf) <= 0.02


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},
        {"state": "only state"},
        {"questions": {"q": {"type": "noul"}}},
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"": {"type": "noul"}}},
        {"state": "s", "questions": {"q": {"instructions": "missing type"}}},
        {"state": "s", "questions": {"q": {"type": "boolean", "instructions": "unknown type"}}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "empty", "criteria": {}}}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "one opt", "criteria": {"a": "only one"}}}},
        {
            "state": "s",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "too many",
                    "criteria": {f"k{i}": "v" for i in range(256)},
                }
            },
        },
        {
            "state": "s",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "map rejected",
                    "criteria": {"0": "low", "1": "high"},
                }
            },
        },
        {
            "state": "s",
            "questions": {
                "q": {"type": "score", "instructions": "too few", "criteria": ["only_one"]}
            },
        },
        {
            "state": "s",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "too many",
                    "criteria": [f"lvl_{i}" for i in range(11)],
                }
            },
        },
    ],
)
def test_invalid_requests_return_422_never_5xx(
    client: TestClient, bad_payload: dict[str, object]
) -> None:
    """Covers score.map-rejected, errors.no-5xx, errors.reject-invalid, errors.validation-shape."""
    resp = client.post("/v1/systemone", json=bad_payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
    assert body["detail"][0]["loc"][0] == "body"


def test_malformed_json_and_non_finite_numbers_return_4xx_json(client: TestClient) -> None:
    """Covers errors.json and errors.no-5xx on raw invalid JSON and non-finite NaN/Infinity literals."""
    resp = client.post(
        "/v1/systemone",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= resp.status_code < 500
    assert "detail" in resp.json()

    # Non-finite NaN / Infinity inside state or structured criteria must be rejected with 422
    for bad_raw in (
        b'{"state": NaN, "questions": {"q": {"type": "noul"}}}',
        b'{"state": "ok", "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": {"val": NaN}, "b": "ok"}}}}',
        b'{"state": "ok", "questions": {"q": {"type": "score", "instructions": "rate", "criteria": [{"v": Infinity}, "high"]}}}',
    ):
        r_nan = client.post(
            "/v1/systemone",
            content=bad_raw,
            headers={"Content-Type": "application/json"},
        )
        assert r_nan.status_code == 422
        assert "detail" in r_nan.json()



def test_semantics_question_isolation_invariance(client: TestClient) -> None:
    """Covers semantics.question-id, semantics.batching, semantics.question-order."""
    state = "Customer says: My invoice #402 was charged twice, please refund immediately."
    q_choice = {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {"billing": "Refunds and invoices", "tech": "Technical bugs"},
    }
    q_noul = {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?",
    }

    # 1. Base single question
    r1 = client.post(
        "/v1/systemone", json={"state": state, "questions": {"dept": q_choice}}
    ).json()["answers"]["dept"]

    # 2. Renamed question id
    r2 = client.post(
        "/v1/systemone", json={"state": state, "questions": {"completely_different_id": q_choice}}
    ).json()["answers"]["completely_different_id"]

    # 3. Batched with another question in both orders
    r3 = client.post(
        "/v1/systemone",
        json={"state": state, "questions": {"dept": q_choice, "refund": q_noul}},
    ).json()["answers"]["dept"]
    r4 = client.post(
        "/v1/systemone",
        json={"state": state, "questions": {"refund": q_noul, "dept": q_choice}},
    ).json()["answers"]["dept"]

    for variant in (r2, r3, r4):
        assert variant["choice"] == r1["choice"]
        for k in r1["probabilities"]:
            assert abs(variant["probabilities"][k] - r1["probabilities"][k]) < 1e-6
