"""Shared fixtures and utilities for the 4-tier E2E hardening test suite."""

from __future__ import annotations

import os
from typing import Any, Generator
import pytest
from fastapi.testclient import TestClient

from microllm.server import create_app


TEST_API_KEY: str = "e2e-hardening-secret-key-42"
AUTH_HEADERS: dict[str, str] = {"Authorization": f"Bearer {TEST_API_KEY}"}
INVALID_AUTH_HEADERS: dict[str, str] = {"Authorization": "Bearer wrong-invalid-key"}


@pytest.fixture
def auth_api_key() -> str:
    """Return the authoritative test API key."""
    return TEST_API_KEY


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return valid Authorization Bearer headers."""
    return dict(AUTH_HEADERS)


@pytest.fixture
def invalid_auth_headers() -> dict[str, str]:
    """Return invalid Authorization Bearer headers."""
    return dict(INVALID_AUTH_HEADERS)


@pytest.fixture
def unauth_client() -> Generator[TestClient, None, None]:
    """TestClient without API key configured (open development mode)."""
    app = create_app(api_key=None)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def auth_client() -> Generator[TestClient, None, None]:
    """TestClient with MICROLLM_API_KEY enforced."""
    app = create_app(api_key=TEST_API_KEY)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def sample_choice_question() -> dict[str, Any]:
    """Return a standard customer ticket routing question payload."""
    return {
        "dept": {
            "type": "choice",
            "instructions": "Route this customer message to the correct department.",
            "criteria": {
                "billing": "Invoices, charges, refunds, credit card issues",
                "shipping": "Order tracking, delivery dates, packages",
                "tech": "Application crash, server error, bug reports",
            },
        }
    }


@pytest.fixture
def sample_valid_rlcd_dataset() -> tuple[list[list[float]], list[int]]:
    """Return a small, valid 2D features matrix and corresponding integer labels."""
    features = [
        [0.85, 0.12, 0.03],
        [0.91, 0.05, 0.04],
        [0.10, 0.88, 0.02],
        [0.08, 0.82, 0.10],
        [0.05, 0.09, 0.86],
        [0.02, 0.11, 0.87],
    ]
    labels = [0, 0, 1, 1, 2, 2]
    return features, labels
