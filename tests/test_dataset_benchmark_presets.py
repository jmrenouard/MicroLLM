"""Tests for Curated Benchmark Dataset Presets and Sample Gallery (IDEA-048 / Issue #175).

Validates:
1. `DATASET_BENCHMARK_PRESETS` in `microllm.gradio_app` contains all 10 curated benchmark presets.
2. Preset definitions have complete, valid structures (state, questions, criteria).
3. Integration with `STUDIO_DOMAIN_PRESETS` and `load_preset_for_all_tabs`.
4. Multi-tab synchronization (exactly 26 Gradio component state outputs per preset).
5. HTML Decision Studio (`src/microllm/static/index.html`) benchmark dataset optgroup and presets.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from microllm.gradio_app import (
    DATASET_BENCHMARK_PRESETS,
    STUDIO_DOMAIN_PRESETS,
    load_preset_for_all_tabs,
)


EXPECTED_DATASET_PRESET_KEYS = [
    "ds_boolq_nli",
    "ds_multinli_entailment",
    "ds_banking77_intent",
    "ds_clinc150_abstain",
    "ds_wildguard_safety",
    "ds_toxicchat_triage",
    "ds_routellm_cascade",
    "ds_clinical_emergency",
    "ds_code_bash_guardrail",
    "ds_goemotions_multilabel",
]


def test_dataset_benchmark_presets_catalog() -> None:
    """Verify that all 10 benchmark dataset presets exist and are registered in STUDIO_DOMAIN_PRESETS."""
    assert len(DATASET_BENCHMARK_PRESETS) == 10
    for key in EXPECTED_DATASET_PRESET_KEYS:
        assert key in DATASET_BENCHMARK_PRESETS, f"Missing preset: {key}"
        assert key in STUDIO_DOMAIN_PRESETS, f"Preset not merged into STUDIO_DOMAIN_PRESETS: {key}"

        preset = DATASET_BENCHMARK_PRESETS[key]
        assert "title" in preset
        assert "domain" in preset
        assert "state" in preset
        assert "questions" in preset
        assert len(preset["questions"]) >= 1


def test_load_preset_for_all_tabs_synchronization() -> None:
    """Verify load_preset_for_all_tabs returns exactly 26 unpacked state values for all 10 presets."""
    for key in EXPECTED_DATASET_PRESET_KEYS:
        outputs = load_preset_for_all_tabs(key)
        assert isinstance(outputs, (tuple, list)), f"Preset {key} returned non-sequence output"
        assert len(outputs) == 26, f"Preset {key} output count mismatch: expected 26, got {len(outputs)}"

        # Output 0 is Tab 1 text
        assert isinstance(outputs[0], str) and len(outputs[0]) > 0
        # Output 16 is Tab 6 state_str
        assert isinstance(outputs[16], str) and len(outputs[16]) > 0
        # Output 17 is Tab 6 questions JSON
        assert isinstance(outputs[17], str) and len(outputs[17]) > 0


def test_static_html_decision_studio_presets() -> None:
    """Verify src/microllm/static/index.html includes benchmark presets and optgroup."""
    html_path = Path(__file__).resolve().parents[1] / "src" / "microllm" / "static" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    assert "📚 Benchmark Datasets (1-Click Evaluation)" in content
    for key in EXPECTED_DATASET_PRESET_KEYS:
        assert f'value="{key}"' in content or f'"{key}":' in content
