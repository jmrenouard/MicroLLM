"""Offline sample fixtures for external web datasets."""

from __future__ import annotations

import functools
import importlib.resources
import json
from typing import Any


@functools.lru_cache(maxsize=1)
def _load_dataset_fixtures_offline_samples() -> dict[str, list[dict[str, Any]]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("dataset_fixtures_offline_samples.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


OFFLINE_WEB_DATASET_SAMPLES: dict[str, list[dict[str, Any]]] = _load_dataset_fixtures_offline_samples()
DATASET_FIXTURES_OFFLINE_SAMPLES = OFFLINE_WEB_DATASET_SAMPLES

__all__ = ["OFFLINE_WEB_DATASET_SAMPLES", "DATASET_FIXTURES_OFFLINE_SAMPLES"]
