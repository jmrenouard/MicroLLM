"""Shared fixtures and utilities for the MicroLLM 18 Audit Points Remediation E2E test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator
import pytest
from fastapi.testclient import TestClient

from microllm.server import create_app
from microllm.mcp_server import MicroLLMMCPServer


TEST_API_KEY: str = "audit18-secret-key-99"
AUTH_HEADERS: dict[str, str] = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return valid Authorization Bearer headers."""
    return dict(AUTH_HEADERS)


@pytest.fixture
def test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """Return a TestClient instance with isolated cache directory and raise_server_exceptions=False."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(cache_dir))
    app = create_app(api_key=None)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def mcp_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MicroLLMMCPServer:
    """Return a fresh MicroLLMMCPServer instance with isolated cache directory."""
    cache_dir = tmp_path / ".mcp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MICROLLM_CACHE_DIR", str(cache_dir))
    return MicroLLMMCPServer()


@pytest.fixture
def sample_valid_rlcd_dataset() -> dict[str, Any]:
    """Return a valid, compact dataset for RLCD calibration."""
    return {
        "features": [
            [0.5, -0.2],
            [1.1, 0.8],
            [-0.5, 0.3],
            [0.2, -1.0],
            [0.9, 0.4],
            [-0.8, -0.6],
        ],
        "labels": [0, 1, 0, 1, 1, 0],
        "stage1_steps": 10,
        "stage2_steps": 10,
        "use_rlcd2_weights": True,
        "seed": 42,
    }


@pytest.fixture
def sample_valid_calibration_dataset() -> dict[str, Any]:
    """Return a valid dataset for temperature scaling & conformal prediction."""
    return {
        "logits": [
            [2.1, 0.5, 0.1],
            [0.2, 1.8, 0.4],
            [0.3, 0.2, 2.5],
            [1.9, 0.3, 0.2],
            [0.1, 2.2, 0.3],
            [0.2, 0.4, 2.0],
        ],
        "labels": [0, 1, 2, 0, 1, 2],
        "alpha": 0.10,
        "target_risk": 0.10,
        "delta": 0.05,
        "seed": 42,
    }


@pytest.fixture
def sample_choice_question() -> dict[str, Any]:
    """Return a standard customer ticket routing question payload."""
    return {
        "dept": {
            "type": "choice",
            "instructions": "Route this customer message to the correct department.",
            "criteria": {
                "billing": "Invoices, charges, refunds, credit card issues",
                "tech_support": "Broken features, errors, bug reports",
                "sales": "Pricing inquiries, custom plans, demos",
            },
        }
    }
