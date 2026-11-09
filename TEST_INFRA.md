# MicroLLM Production Hardening: E2E Test Infrastructure (TEST_INFRA.md)

## 1. Executive Overview

This document specifies the test infrastructure, architecture, and verification methodology for the **MicroLLM Production Hardening (Ponytail Audit Remediation)** initiative.

The test suite provides an **opaque-box, requirement-driven, genuine verification barrier** covering all 12 Ponytail Audit findings across Requirements R1 through R5.

### Core Objectives
1. **Zero-Facade Integrity**: All tests perform authentic assertions against real server instances, SDK clients, background workers, and CLI runners. No test hardcodes mock answers to fake compliance.
2. **4-Tier Test Architecture**:
   - **Tier 1: Feature Coverage**: Direct verification of each of the 12 findings through public endpoints and public SDK APIs (>=5 tests per feature = 60 tests).
   - **Tier 2: Boundary & Corner Cases**: Empty inputs, malformed arrays, missing tokens, invalid headers, concurrent socket threads (>=5 tests per feature = 60 tests).
   - **Tier 3: Cross-Feature Combinations**: Multi-feature interactions (auth + concurrency, validation + event loop, batch + multilingual strategy, etc. = 12 tests).
   - **Tier 4: Real-World Application Scenarios**: Realistic production workloads, end-to-end client sessions, MCP tool sequences, active learning cycles (= 12 tests).
   - **Total Tests**: **144 automated tests** in `tests/test_e2e_hardening/`.
3. **Progressive Testability & Defect Escalation**: Clear separation between test code and implementation code; test suite reports precise pass/fail metrics across all 12 findings to guide milestone implementation.

---

## 2. 4-Tier Test Architecture

```
tests/test_e2e_hardening/
├── __init__.py                     # Package marker
├── conftest.py                     # Shared fixtures, auth clients, and datasets
├── test_tier1_features.py          # Tier 1: 60 Feature Coverage tests (5 * 12 features)
├── test_tier2_boundaries.py        # Tier 2: 60 Boundary & Corner Case tests (5 * 12 features)
├── test_tier3_combinations.py      # Tier 3: 12 Cross-Feature Combination tests
└── test_tier4_scenarios.py         # Tier 4: 12 Real-World Application Scenario tests
```

### Tier Breakdown

| Tier | Focus | Test Count | Description |
| :--- | :--- | :---: | :--- |
| **Tier 1** | Feature Coverage | 60 | Tests primary happy path and fundamental contract for each of F01–F12 (5 tests each). |
| **Tier 2** | Boundary & Corner Cases | 60 | Tests empty inputs, malformed arrays, extreme thresholds, missing headers, thread isolation, and error paths (5 tests each). |
| **Tier 3** | Cross-Feature Combinations | 12 | Tests interactions between auth and concurrency, validation and event loop, batch and multilingual strategies. |
| **Tier 4** | Real-World Application Scenarios | 12 | Tests full production lifecycles, MCP tool chaining, Gradio Decision Studio flows, and active learning. |
| **Total** | **Full E2E Hardening Suite** | **144** | Comprehensive requirement-driven verification across all 12 findings. |

---

## 3. 12 Ponytail Audit Findings Coverage Matrix

| Finding ID | Requirement | Feature Description | Tier 1 Tests | Tier 2 Tests | Tier 3 & 4 Tests |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **F01** | R1 | Universal Endpoint Authentication (`_check_auth` on `/v1/studio/*`, `/v1/benchmark*`, `/v1/metrics`, `/v1/profiles/pull`) | 5 | 5 | 6 |
| **F02** | R1 | In-Process Core Execution (7 MCP tools and Gradio tabs without `TestClient` loopbacks) | 5 | 5 | 4 |
| **F03** | R2 | Input Validation on RLCD (`/v1/rlcd`, `/v1/studio/rlcd-demo`, and `RLCDDecisionHeadTrainer.fit` returning 422) | 5 | 5 | 4 |
| **F04** | R3 | Thread-Isolated `guard_sockets()` (`threading.get_ident()` isolation, protecting concurrent threads) | 5 | 5 | 3 |
| **F05** | R3 | Non-Blocking Event Loop (`asyncio.to_thread`), `/healthz` probe, and cache thread safety | 5 | 5 | 3 |
| **F06** | R4 | Multi-Question Token Caching (`tokenize_multilingual` memoization, state token reuse in `evaluate_voc_gate_regime`) | 5 | 5 | 3 |
| **F07** | R4 | SDK Client Batch Parity (`MicroJevClient.batch()` accepting `multilingual_strategy` and `neural_model`) | 5 | 5 | 3 |
| **F08** | R5 | SDK Base Consolidation (`_BaseMicroJevClient`, shared auth headers, payload builders, answer parsing) | 5 | 5 | 3 |
| **F09** | R4 | Space License Alignment (Default `license_type` to `"apache-2.0"` in `hf_space_sync.py`) | 5 | 5 | 2 |
| **F10** | R4 | Workstation Paths Elimination (Removal of hardcoded `C:\Users\jmren` in favor of env vars and `Path.home()`) | 5 | 5 | 2 |
| **F11** | R4 | Benchmark CLI `--json` Flag (`--json` argument on `benchmark` command and valid JSON stdout) | 5 | 5 | 2 |
| **F12** | R5 | Modularization of Oversized Modules (`gradio_app.py`, `dataset_adapters.py`, `server.py` decomposition) | 5 | 5 | 2 |

---

## 4. Test Execution & CLI Commands

### 4.1 Running the Full E2E Hardening Suite
```powershell
pytest tests/test_e2e_hardening/ -v
```

### 4.2 Running Specific Tiers
```powershell
# Tier 1: Feature Coverage (60 tests)
pytest tests/test_e2e_hardening/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases (60 tests)
pytest tests/test_e2e_hardening/test_tier2_boundaries.py -v

# Tier 3: Cross-Feature Combinations (12 tests)
pytest tests/test_e2e_hardening/test_tier3_combinations.py -v

# Tier 4: Real-World Application Scenarios (12 tests)
pytest tests/test_e2e_hardening/test_tier4_scenarios.py -v
```

### 4.3 Running Tests for a Specific Finding
```powershell
# Run all tests for Finding F01 (Authentication)
pytest tests/test_e2e_hardening/ -k "f01" -v

# Run all tests for Finding F03 (RLCD Validation)
pytest tests/test_e2e_hardening/ -k "f03" -v

# Run all tests for Finding F04 (Socket Guard)
pytest tests/test_e2e_hardening/ -k "f04" -v
```

---

## 5. Fixtures and Environment Configuration

The test suite uses centralized fixtures defined in `tests/test_e2e_hardening/conftest.py`:
- `auth_api_key`: Provides `e2e-hardening-secret-key-42`.
- `auth_headers`: Formats `{"Authorization": "Bearer e2e-hardening-secret-key-42"}`.
- `invalid_auth_headers`: Formats `{"Authorization": "Bearer wrong-invalid-key"}`.
- `unauth_client`: FastAPI `TestClient` initialized with `api_key=None` (local development mode).
- `auth_client`: FastAPI `TestClient` initialized with `api_key=TEST_API_KEY` (production enforced authentication mode).
- `sample_choice_question`: Standard customer ticket department routing question specification.
- `sample_valid_rlcd_dataset`: Standard valid 2D feature array and labels for RLCD tests.

### Supported Environment Overrides
- `MICROLLM_API_KEY`: Configures API key authentication.
- `MICROLLM_KEEPASS_DB`: Configures custom path to KeePass database (bypassing workstation paths).
- `MICROLLM_KEEPASS_KEY`: Configures custom path to KeePass key file.
- `MICROLLM_MULTILINGUAL_STRATEGY`: Configures default multilingual routing strategy (`auto`, `lightweight`, `neural`).
- `MICROLLM_NEURAL_MODEL`: Configures default neural model (`minilm`, `mdeberta`).

---

## 6. Zero-Facade Integrity & Anti-Gaming Rules

1. **Direct Backend Invocations**: Tests make real HTTP requests via `TestClient(create_app())` and real library calls to `microllm.client`, `microllm.backends`, `microllm.rlcd`, and `microllm.hf_publishing`.
2. **No Fake Stubs**: Tests do not inject mock monkeypatches that bypass internal logic.
3. **Opaque-Box Boundary**: Tests verify observable return values, HTTP response status codes, and JSON response bodies matching the contracts specified in `PROJECT.md` and `ORIGINAL_REQUEST.md`.
4. **Verification Protocol**: Any change to tests must be audited and verified using `pytest tests/test_e2e_hardening/ -v`.
