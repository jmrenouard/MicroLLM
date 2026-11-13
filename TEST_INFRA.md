# MicroLLM 18 Audit Points Remediation: E2E Test Infrastructure (TEST_INFRA.md)

## 1. Executive Overview

This document specifies the test infrastructure, architecture, and verification methodology for the **MicroLLM 18 Audit Points Remediation & Production Hardening** initiative.

The test suite provides an **opaque-box, requirement-driven, genuine verification barrier** covering all 18 audit points identified across `ORIGINAL_REQUEST.md` and `PROJECT.md`.

### Core Objectives
1. **Zero-Facade Integrity**: All tests perform authentic assertions against real server instances, Starlette TestClients, MCP servers, SDK clients, background workers, and packaging manifests. No test uses hardcoded answers or artificial passes.
2. **4-Tier Test Architecture**:
   - **Tier 1: Feature Coverage**: Direct verification of each of the 18 audit points through public endpoints, SDK methods, MCP tools, and system configurations (5 tests per feature = 90 tests).
   - **Tier 2: Boundary & Corner Cases**: Empty inputs, malformed arrays, extreme thresholds, type mismatches (boolean injection), concurrent locks, and permission boundaries (5 tests per feature = 90 tests).
   - **Tier 3: Cross-Feature Combinations**: Pairwise and multi-feature interactions across components sharing state, event loops, or data contracts (18 combination tests).
   - **Tier 4: Real-World Application Scenarios**: End-to-end integration workflows including full active learning loops, concurrent multi-tenant inference, MCP agent sessions, and container security compliance (12 scenario tests).
   - **Total Tests**: **210 automated tests** in `tests/test_e2e_audit18/`.
3. **Progressive Testability & Gap Escalation**: Clear separation between test code and implementation code; test suite reports precise pass/fail metrics across all 18 points to guide milestone implementation and verify fixes.

---

## 2. 4-Tier Test Architecture

```
tests/test_e2e_audit18/
├── __init__.py                     # Package marker
├── conftest.py                     # Shared fixtures, test clients, and datasets
├── test_tier1_features.py          # Tier 1: 90 Feature Coverage tests (5 * 18 points)
├── test_tier2_boundaries.py        # Tier 2: 90 Boundary & Corner Case tests (5 * 18 points)
├── test_tier3_combinations.py      # Tier 3: 18 Cross-Feature Combination tests
└── test_tier4_scenarios.py         # Tier 4: 12 Real-World Application Scenario tests
```

### Tier Breakdown

| Tier | Focus | Test Count | Description |
| :--- | :--- | :---: | :--- |
| **Tier 1** | Feature Coverage | 90 | Tests primary behavior and interface contracts for points 1–18 (5 tests each). |
| **Tier 2** | Boundary & Corner Cases | 90 | Tests type validations, boundary sizes ($N, d_{\text{feat}}, K$), concurrency conflicts, and edge conditions (5 tests each). |
| **Tier 3** | Cross-Feature Combinations | 18 | Tests interactions between shared systems (e.g. concurrency + model init, cache + memory bounds, SDK parity + JSON data). |
| **Tier 4** | Real-World Application Scenarios | 12 | Tests realistic end-to-end pipelines (active learning loop, multi-tenant benchmark concurrency, MCP session, container security). |
| **Total** | **Full Audit 18 E2E Suite** | **210** | Comprehensive requirement-driven verification across all 18 audit points. |

---

## 3. 18 Audit Points Coverage Matrix

| Point ID | Req. | Description | Target Modules | Tier 1 Tests | Tier 2 Tests | Tier 3 & 4 Tests |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **P01 (F01)** | R1 | Docker Cache & Permissions (`ENV MICROLLM_CACHE_DIR=/app/.cache`, chown `appuser`) | `Dockerfile`, `src/microllm/server.py` | 5 | 5 | 4 |
| **P02 (F02)** | R1 | Server Path Resolution Fallback (safe resolution outside git repo) | `src/microllm/server.py`, `benchmarks.py` | 5 | 5 | 3 |
| **P03 (F03)** | R1 | Secure Profile Pull Path (remove `cache_dir` from schema, enforce default) | `src/microllm/server_models.py`, `server.py` | 5 | 5 | 3 |
| **P04 (F04)** | R1 | OOM Bounds & Type Validation (reject bools, bounds $N \le 10k, d \le 1024, K \le 256$) | `src/microllm/server_models.py`, `server.py` | 5 | 5 | 4 |
| **P05 (F05)** | R1 | MCP Calibrate Validation (full validation matching HTTP `CalibrateRequest`) | `src/microllm/mcp_server.py` | 5 | 5 | 4 |
| **P06 (F06)** | R1 | HF Space Sync Race Condition (decoupled ownership or concurrency groups) | `.github/workflows/sync-hf-*.yml` | 5 | 5 | 2 |
| **P07 (F07)** | R1 | Concurrent Benchmark Safety & Atomic Writes (HTTP 409 lock + atomic JSON writes) | `src/microllm/server.py` | 5 | 5 | 4 |
| **P08 (F08)** | R1 | Non-Blocking Model Init & Concurrency (`asyncio.Lock` + `asyncio.to_thread`) | `src/microllm/server.py` | 5 | 5 | 4 |
| **P09 (F09)** | R1 | Package Name Consistency (align error messages and docs to `micro-jev`) | `src/microllm/backends/base.py`, `registry.py` | 5 | 5 | 3 |
| **P10 (F10)** | R1 | Remove Artificial RLCD Sleep (eliminate sleep in `fit` & `server.py`) | `src/microllm/rlcd.py`, `src/microllm/server.py` | 5 | 5 | 4 |
| **P11 (F11)** | R1 | MCP Neural Backend Caching (memoize `.npz` neural instances in MCP server) | `src/microllm/mcp_server.py` | 5 | 5 | 4 |
| **P12 (F12)** | R1 | Verdict-MQ Cache Exploitation (single `encode_state` call + reuse embeddings) | `src/microllm/mcp_server.py`, `backends/base.py` | 5 | 5 | 3 |
| **P13 (F13)** | R1 | Standalone Wheel CI Integration Test (build & test wheel in isolated venv) | `.github/workflows/ci.yml`, `tests/test_standalone_wheel.py` | 5 | 5 | 3 |
| **P14 (F14)** | R1 | Secure Jekyll Docker Workflow (`contents: read`, no `chmod 777`, pinned actions) | `.github/workflows/jekyll-docker.yml` | 5 | 5 | 2 |
| **P15 (F15)** | R1 | Client SDK Deduplication (`_BaseMicroJevClient` request & response handling) | `src/microllm/client.py` | 5 | 5 | 4 |
| **P16 (F16)** | R1 | Static Data Extraction to Packaged JSON (10 JSON files in `src/microllm/data/`) | `src/microllm/data/`, `papers.py`, `pyproject.toml` | 5 | 5 | 4 |
| **P17 (F17)** | R1 | Dead Code Removal & Strict Linting (remove `_get_client`, no `--exit-zero`) | `src/microllm/mcp_server.py`, `lint.yml` | 5 | 5 | 2 |
| **P18 (F18)** | R1 | CI Workflow Deduplication (clear matrix test vs packaging workflow ownership) | `.github/workflows/ci.yml`, `unit-tests.yml` | 5 | 5 | 2 |

---

## 4. Test Execution & CLI Commands

### 4.1 Running the Full Audit 18 E2E Suite
```powershell
pytest tests/test_e2e_audit18/ -v
```

### 4.2 Running by Individual Tier
```powershell
# Tier 1: Feature Coverage (90 tests)
pytest tests/test_e2e_audit18/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases (90 tests)
pytest tests/test_e2e_audit18/test_tier2_boundaries.py -v

# Tier 3: Cross-Feature Combinations (18 tests)
pytest tests/test_e2e_audit18/test_tier3_combinations.py -v

# Tier 4: Real-World Application Scenarios (12 tests)
pytest tests/test_e2e_audit18/test_tier4_scenarios.py -v
```

### 4.3 Running by Audit Point
```powershell
# Run all tests for Point 1 (Docker Cache & Permissions)
pytest tests/test_e2e_audit18/ -k "p01" -v

# Run all tests for Point 4 (OOM Bounds & Type Validation)
pytest tests/test_e2e_audit18/ -k "p04" -v

# Run all tests for Point 7 (Benchmark Concurrency & Atomic Writes)
pytest tests/test_e2e_audit18/ -k "p07" -v

# Run all tests for Point 10 (Remove Artificial Sleep)
pytest tests/test_e2e_audit18/ -k "p10" -v
```

---

## 5. Fixtures and Test Isolation

The test suite leverages `tests/test_e2e_audit18/conftest.py` with the following isolation guarantees:
1. **Opaque HTTP Status Inspection**: `Starlette TestClient` configured with `raise_server_exceptions=False` to inspect authentic status codes (409, 422, 500) rather than raising Python exceptions.
2. **Environment Isolation**: Temporary directories (`tmp_path`) injected via `MICROLLM_CACHE_DIR` environment patches to guarantee tests do not pollute user workstations or share disk state.
3. **App Instance Refresh**: Fresh `create_app()` instances created per test module/function where state mutation occurs.
4. **Deterministic Seeds**: Synthetic NumPy arrays and logits generated with fixed seeds (`np.random.RandomState(42)`) for reproducible mathematical verification.
