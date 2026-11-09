# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-11-09

### Added
- **Strategy 1: Lightweight Multilingual Engine (`microllm.lexicons`)**:
  - Zero-dependency localized lexicons for English, French, Spanish, and Portuguese (`en.py`, `fr.py`, `es.py`, `pt.py`).
  - Multilingual text normalizer: Unicode unaccenting (`NFKD`), rule-based light suffix stemming, fast language detection (`detect_language_fast`), and normalized tokenization.
  - Expanded `CONCEPT_CLUSTERS` across 10 categories activating the first 16 slots of the 256-D deterministic embedding space.
  - Localized stopwords, negation markers, double negations, and connectors in `backends/base.py` and `backends/fast_encoder.py`.
- **Strategy 2: Neural Multilingual Models**:
  - Dual ONNX INT8 zero-shot cross-encoder profiles in `microllm.registry`: `multilingual-minilm-int8` (~48 MB INT8) and `multilingual-mdeberta-int8` (~134 MB INT8).
  - Dynamic `label2id` / config entailment index resolution in `create_onnx_nli_scorer` (fixing hardcoded entailment index bug).
  - Localized source-language hypothesis templates for EN, FR, ES, and PT in `format_nli_hypothesis` to prevent cross-lingual transfer drift.
- **Unified Strategy Selection & Runtime Configuration**:
  - Added `multilingual_strategy` (`"fast"`, `"neural"`, `"auto"`) and `neural_model` (`"minilm"`, `"mdeberta"`) to `create_app()`, `SystemOneRequest`, and `MicroJevClient`.
  - Added CLI flags `--strategy` and `--neural-model` to `microllm serve`.
- **Multilingual Decision Studio & Space Sync**:
  - Multilingual strategy selector and preset probe tickets across EN, FR, ES, and PT in `src/microllm/gradio_app.py`.
  - Automated synchronization workflow to Hugging Face Space `jmrenouard/microllm-decision-studio`.
- **Isolated Benchmarking & Hub Publishing**:
  - Multilingual synthetic probe dataset across billing, shipping, and tech support domains in `microllm_multilingual_corpus.jsonl`.
  - Multilingual benchmark runner with telemetry isolated in `benchmarks/multilingual_benchmarks_results.json` and documented in `BENCHMARKS.md §11`.
  - Dataset and model publishing utilities for Hugging Face Hub under `jmrenouard/...` with `--dry-run` and KeePass token resolution.
- **Comprehensive E2E Opaque-Box & Adversarial Test Suite**:
  - 194-test multi-tier E2E opaque-box test suite (`tests/e2e_multilingual/`) covering Tiers 1 through 5 (contract validation, cross-lingual consistency, adversarial robustness, and latency SLA).
- **Documentation & Analysis**:
  - Detailed comparative critique and worst-case failure mode analysis in `docs/FAILURE_MODES_AND_WORST_CASES.md`.
  - Multilingual architectural overview in `docs/MULTILINGUAL_STRATEGIES_SUMMARY.md`.
  - Release notes in `docs/RELEASE_NOTES_v0.3.0.md`.

### Fixed
- Resolved hardcoded entailment index assumption in `create_onnx_nli_scorer` by dynamically inspecting `config.json` `label2id` mapping (`RCA-075`).
- Resolved mypy type narrowing in tournament fast-path options evaluation (`ACT-104`).
- Maintained sub-millisecond warm latency SLA across all localized lexicon match operations.

---

## [0.2.0] - 2026-10-23

### Added
- AnyJev $L_0$ circular rotation readout cancelling slot bias with Van der Corput order.
- StartLux Read-or-Think dual-gear cascade combining FastEncoder (Gear 1) and AnyJev (Gear 2).
- Two-Stage OpenJev-RLCD algorithm with strictly proper Brier scoring and Leave-One-Out KL-anchored policy updates.
- Split conformal prediction with bounded error rate guarantees ($x\_abstain$).
- 16 public external dataset adapters and evaluation runners.
- Interactive Decision Studio Web UI and Model Context Protocol (MCP) server.

---

## [0.1.0] - 2026-10-22

### Added
- Initial public release of MicroLLM (`micro-jev`).
- JevCompat 0.1 wire-protocol compatibility (32/32 MUST requirements).
- Core decision primitives: `noul`, `choice`, `score`, and `multi`.
- 256-D deterministic clause-salient CPU encoder.
- Initial benchmark suites and test suite.
