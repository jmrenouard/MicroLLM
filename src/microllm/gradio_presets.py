"""Static presets for MicroLLM Decision Studio."""

from __future__ import annotations

import functools
import importlib.resources
import json
from typing import Any


@functools.lru_cache(maxsize=1)
def _load_gradio_multilingual_presets() -> dict[str, dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("gradio_multilingual_presets.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


@functools.lru_cache(maxsize=1)
def _load_gradio_domain_presets() -> dict[str, dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("gradio_domain_presets.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


@functools.lru_cache(maxsize=1)
def _load_gradio_dataset_presets() -> dict[str, dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("gradio_dataset_presets.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


@functools.lru_cache(maxsize=1)
def _load_gradio_consumer_examples() -> dict[str, dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("gradio_consumer_examples.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


MULTILINGUAL_SUPPORT_PRESETS: dict[str, dict[str, Any]] = _load_gradio_multilingual_presets()
GRADIO_MULTILINGUAL_PRESETS = MULTILINGUAL_SUPPORT_PRESETS

STUDIO_DOMAIN_PRESETS: dict[str, dict[str, Any]] = _load_gradio_domain_presets()
GRADIO_DOMAIN_PRESETS = STUDIO_DOMAIN_PRESETS

DATASET_BENCHMARK_PRESETS: dict[str, dict[str, Any]] = _load_gradio_dataset_presets()
GRADIO_DATASET_PRESETS = DATASET_BENCHMARK_PRESETS

CONSUMER_TYPE_EXAMPLES: dict[str, dict[str, Any]] = _load_gradio_consumer_examples()
GRADIO_CONSUMER_EXAMPLES = CONSUMER_TYPE_EXAMPLES

# Alias for test compatibility
MULTILINGUAL_BENCHMARK_PRESETS = MULTILINGUAL_SUPPORT_PRESETS

__all__ = [
    "MULTILINGUAL_SUPPORT_PRESETS",
    "GRADIO_MULTILINGUAL_PRESETS",
    "STUDIO_DOMAIN_PRESETS",
    "GRADIO_DOMAIN_PRESETS",
    "DATASET_BENCHMARK_PRESETS",
    "GRADIO_DATASET_PRESETS",
    "MULTILINGUAL_BENCHMARK_PRESETS",
    "CONSUMER_TYPE_EXAMPLES",
    "GRADIO_CONSUMER_EXAMPLES",
]
