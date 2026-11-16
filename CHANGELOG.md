# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.1] - 2026-11-16

### Added
- **Repository Privacy & `.gitignore` Hygiene Regression Suite (`tests/test_repo_privacy_hygiene.py`)**:
  - Added automated CI validation ensuring private outreach artifacts and KeePass vault files (`*.kdbx`, `*.keyx`) stay untracked and ignored via `.gitignore`.
  - Enforced zero leaked personal email addresses, phone numbers, Gmail message/draft identifiers, or hardcoded workstation OS user paths across all tracked repository files (`#269`, `RCA-093`).
- **Cross-Surface Version Synchronization Test (`tests/test_packaging_ci.py`)**:
  - Added `test_all_version_surfaces_synchronized` verifying strict version equality across `pyproject.toml`, `microllm.__version__`, FastAPI `create_app().version`, MCP `initialize` `serverInfo.version`, and `.github/workflows/ci.yml` standalone wheel checks (`RCA-092`, `RCA-095`).

### Changed
- **Portable KeePass Vault Resolution (`src/microllm/hf_publishing.py`)**:
  - Generalized `get_vault_paths()` and `resolve_hf_token()` to resolve local KeePass credentials via environment variables (`MICROLLM_KEEPASS_DB`, `KEEPASS_DB`, `KEEPASS_DB_PATH` and `MICROLLM_KEEPASS_KEY`, `KEEPASS_KEY`, `KEEPASS_KEY_PATH`) with a portable cross-platform fallback at `~/.config/microllm/vault.kdbx` (`#269`).
- **Release Packaging & Version Synchronization**:
  - Synchronously bumped version to `0.4.1` across package metadata, FastAPI server, MCP server, standalone wheel tests, CI workflow assertions, and product/technical documentation.
  - Rebuilt and realigned PyPI (`sdist` and `wheel`), GitHub Release (`SHA256SUMS.txt`), and Docker Hub distributions with the scrubbed Git tree (`RCA-093`, `RCA-095`).

### Fixed
- **CI Standalone Wheel Version Assertion (`.github/workflows/ci.yml`)**:
  - Updated out-of-tree standalone wheel integration verification in CI to assert the synchronized package version (`#267`, `RCA-092`).
- **Standalone Wheel Fixture Ordering Isolation (`tests/test_audit18_m4_cicd_packaging.py`)**:
  - Hardened `packaged_wheel` fixture to automatically build the `0.4.1` wheel into `dist/` when executed independently of other packaging suites (`RCA-095`).

---

## [0.4.0] - 2026-11-15

### Added
- **S1-SHAP in Logit Space (`DecisionBackend.explain_counterfactual`)**:
  - Exact counterfactual attribution in unbounded logit / log-odds space avoiding probability saturation near 0.0 and 1.0.
  - Return of both `original_log_odds` and `logit_drop` alongside normalized token importance for all decision primitives.
  - Comprehensive boundary hardening in `tests/test_s1_shap_counterfactual.py` covering token deduplication, empty/whitespace inputs, multilingual text, top-k bounds, and decision flips.
- **Evidence Competition Hijack Defense**:
  - Cross-clause evidence competition detector (`detect_evidence_competition_hijack`) and certainty hijack markers (`_CERTAINTY_HIJACK_MARKERS`).
  - Automated abstain gate (`x_hijack_suspected`, `x_abstain=True`) in `FastEncoderBackend` preventing adversarial manipulation where injected certainty words override semantic evidence.
  - Hardened prompt injection detector threshold (`prompt_injection >= 0.88`) in `server.py`.
  - Comprehensive unit test suite in `tests/test_evidence_competition_hijack.py`.
- **Policy Boundary Gatekeeping & Roleplay Hardening**:
  - Explicit `policy_boundaries` (`allowed` / `forbidden`) in `SystemOneRequest`, `server.py`, `server_models.py`, `client.py`, and `mcp_server.py`.
  - Remainder-based destructive hazard inspection ensuring that allowlist rules (e.g. `rm -rf ./dist`) do not bypass destructive command detection on chained payloads (`DROP TABLE`, `rm -rf /etc`, `rm -rf $HOME`).
  - Strict non-downgradable hard hazard barriers for destructive commands (`rm -rf /`, `DROP TABLE`).
  - Unicode NFKC normalization preventing fullwidth character bypasses.
  - Interdiction of adversarial roleplay framing (`"in a novel"`, `"screenplay"`, `"pretend you are"`) coupled with harmful targets.
  - Comprehensive unit test suite in `tests/test_coding_agent_policy_gates.py`.
- **Expanded Academic Catalog & Literature Taxonomies**:
  - Added 12 new October–November 2026 arXiv papers expanding the academic catalog from 104 to 116 curated research papers (`src/microllm/data/academic_papers_catalog.json`).
  - Added research roadmaps `IDEA-052` through `IDEA-057` in `docs/RCA_REGISTER.md`.
  - Schema, URL, and metadata validation suite in `tests/test_academic_papers_catalog.py`.
- **12-Suite Comparative Benchmark Campaign**:
  - Executed full 12-suite benchmark campaign evaluating 128,713 items and 131,487 decisions in 66.73s CPU wall-clock time across 28 external datasets.
  - Master benchmark report generated in `benchmarks/all_benchmarks_master_report.json` and `benchmarks/latest_master_report.md`.
  - Confirmed p50 latency of 0.013 ms (38,917 decisions/s) on large-scale CPU evaluation (`test_cases_11000`), 0.17 ms on multilingual cross-lingual transfer, and 1.5x cascade acceleration.
  - Updated `BENCHMARKS.md` and `README.md` with verified real-world metrics.

### Changed
- Synchronously bumped version to `0.4.0` across package metadata, FastAPI server, MCP server, documentation, and packaging tests.
- Replaced stale wheel build mechanism in `test_audit18_m3_challenger2_data_wheel_stress.py` to automatically rebuild `.whl` when data catalogs are updated.

### Fixed
- Fixed allowlist rule remainder check in policy gate preventing chained destructive payloads from bypassing safety evaluation (`RCA-091`).
- Fixed hard hazard downgrade and fullwidth Unicode evasion vectors (`RCA-090`).

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
