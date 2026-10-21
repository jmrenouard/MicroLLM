# Action Logbook — MicroLLM Project

> **Governance Rule ([`AGENTS.md`](../AGENTS.md) §2 & [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md))**:
> This Action Logbook **MUST be systematically updated on every request, every change, and every result**.
> - Every new request is logged under **"1. Ongoing Actions (`IN PROGRESS`)"** as soon as its GitHub Issue is opened.
> - Once validated by unit tests (`pytest`) and its Pull Request is created/merged, the entry is completed with the **Changes** and quantified **Result**, then archived under **"2. Completed Actions (`COMPLETED`)"**.

---

## 1. Ongoing Actions (`IN PROGRESS`)

| Action ID | Date / Time (ISO) | Issue / Branch | Status | Request, Ongoing Changes & Expected Results |
| :--- | :--- | :--- | :--- | :--- |
| *(None — all requested actions completed)* | `2026-10-21T22:52+02:00` | — | `IDLE` | All active user requests (`#1`–`#10`, `#29`, `#32`, `#33`, `#35`, `#37`, `#40`, `#43`, `#44`) have been implemented, verified with `pytest`, and logged below. |

---

## 2. Completed Actions (`COMPLETED`)

### ACT-018 — Complete Test Plan (`docs/TEST_PLAN.md`) & 750–1,000 Test Case Quality & Performance Validation Suite (Issue `#32`)
- **Date**: `2026-10-21T22:52+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#32` (`test/issue-32-quality-test-plan-750-cases`)
- **Request**:
  - Write a complete test plan to check performance and result quality, build 500 to 1,000 test cases to validate project quality, and add this to the RCA register (`docs/RCA_REGISTER.md`).
- **Changes**:
  - Created [`docs/TEST_PLAN.md`](TEST_PLAN.md) specifying the 8-track validation architecture (750 default test cases, scalable from 500 to 1,000 test cases), acceptance criteria, quality gates (`G1`–`G8`), and performance SLAs (`p50 <= 10 ms`, `p95 <= 25 ms`, `>= 100 decisions/sec`, INT8 quantization MSE `<= 1e-3`).
  - Created [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) (`build_quality_test_cases`, `run_quality_validation_suite`, `format_quality_report_markdown`, and CLI entry point `python -m microllm.quality_suite`).
  - Created TDD test suite [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py) validating `docs/TEST_PLAN.md`, 750-case and 1,000-case generation, full 8-track execution (`all_gates_passed == True`), and governance register entries.
  - Updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) with `RCA-012`, `APP-007`, `IDEA-021`, and Section 5 (*Complete Performance & Result Quality Test Plan*).
- **Result**:
  - `pytest tests/test_quality_validation_suite.py -v`: **4 / 4 tests passed (100%)**.
  - **750 / 750 test cases** (over 1,250 individual engine decisions) and **1,000 / 1,000 scaled test cases** pass **8 / 8 automated Quality & Performance SLA Gates** (`98.1%` JevBench accuracy, `100%` DecisionPivot pair accuracy, `0.000000` AnyJev $L_0$ order-flip rate, `100%` Guardrail F1, `3.67%` calibrated ECE, `100%` `jevcompat` contract/boundary pass rate with `0` HTTP `5xx`, `100%` EN/FR/ES/DE + XAI pass rate, and CPU `p50 < 2.0 ms`).

---

### ACT-017 — Merge `docs/ROADMAP_20_IDEAS.md` into `docs/RCA_REGISTER.md` (Issue `#43`)
- **Date**: `2026-10-21T22:29+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#43` (`docs/issue-43-merge-roadmap-rca-register`)
- **Request**:
  - Merge `docs/ROADMAP_20_IDEAS.md` with the RCA Register (`docs/RCA_REGISTER.md`) so that all 20 high-impact architectural & scientific improvements (`IDEA-001` through `IDEA-020`) reside in a single canonical governance register.
- **Changes**:
  - Merged the Executive Summary Matrix (`§4.1`) and Detailed Specifications across all 6 architectural pillars (`§4.2`, `IDEA-001` through `IDEA-020`) from `docs/ROADMAP_20_IDEAS.md` into Section 4 (`4. Unimplemented Ideas & Future Roadmap (IDEA)`) of [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
  - Removed the standalone `docs/ROADMAP_20_IDEAS.md` file and logged `IMP-005` (why maintaining separate roadmap files outside `docs/RCA_REGISTER.md` is a dead-end) and `RCA-011` (fixing Windows CP1252 `0x97` byte corruption in governance Markdown files).
  - Consolidated the 20 high-impact improvements implementation (`src/microllm/*`, [`tests/test_20_high_impact_ideas.py`](../tests/test_20_high_impact_ideas.py)) and updated [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) to assert that `IDEA-001`..`IDEA-020` and all 6 pillars are in `docs/RCA_REGISTER.md` and that `docs/ROADMAP_20_IDEAS.md` no longer exists as a separate file.
- **Result**:
  - `pytest tests/test_project_governance_rules.py tests/test_20_high_impact_ideas.py -v`: **15 / 15 tests passed (100%)**.
  - Full `pytest --cov=microllm` suite passing at **100%** with **94% coverage**.

---

### ACT-016 — Add "Why MicroLLM?" Section & Top-of-README 1-Command Benchmark Table (Issue `#44` / PR `#45`)
- **Date**: `2026-10-21T22:24+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#44` · PR `#45` (`docs/issue-44-readme-why-and-benchmark-table`, merged into `main`, commit `8af1bde`)
- **Request**:
  - Add a clear "Why MicroLLM?" section and surface the 1-command reproducible benchmark table (`microllm benchmark --all`) at the top of `README.md`.
- **Changes**:
  - Updated [`README.md`](../README.md) with the top-level value proposition and the 5-track benchmark results table (`540` items).
- **Result**:
  - PR `#45` merged into `main`; Issue `#44` closed automatically.

---

### ACT-015 — Add Obsidian Craft SVG Logo & Hero Header Banner Across Docs and Decision Studio (Issue `#40` / PR `#41`)
- **Date**: `2026-10-21T22:22+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#40` · PR `#41` (`feat/issue-40-brand-logo-header-banner`, merged into `main`, commit `bf68e24`)
- **Request**:
  - Design and integrate a high-craft SVG brand logo and header banner across `README.md`, `docs/README.md`, `docs/index.html`, and the interactive Decision Studio (`src/microllm/static/index.html`).
- **Changes**:
  - Created [`docs/assets/logo.svg`](assets/logo.svg) and [`docs/assets/header-banner.svg`](assets/header-banner.svg) and integrated them into [`src/microllm/static/index.html`](../src/microllm/static/index.html), [`docs/index.html`](index.html), and [`README.md`](../README.md).
- **Result**:
  - PR `#41` merged into `main`; Issue `#40` closed automatically.

---

### ACT-014 — 540-Item 5-Track Reproducible Evaluation Corpus & 1-Command Benchmark Runner (Issue `#37` / PR `#38`)
- **Date**: `2026-10-21T22:21+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#37` · PR `#38` (`feat/issue-37-benchmark-corpus-540`, merged into `main`, commit `1e772fd`)
- **Request**:
  - Build a 540-item 5-track reproducible evaluation corpus (`JevBench-Core`, `DecisionPivot`, `AnyJev-Order`, `Calibration-ECE`, `HighCard-50`) and a 1-command benchmark runner (`microllm benchmark`).
- **Changes**:
  - Added [`src/microllm/benchmarks.py`](../src/microllm/benchmarks.py), [`benchmarks/microllm_eval_corpus.jsonl`](../benchmarks/microllm_eval_corpus.jsonl), [`benchmarks/latest_results.json`](../benchmarks/latest_results.json), and [`tests/test_benchmark_corpus.py`](../tests/test_benchmark_corpus.py).
- **Result**:
  - `JevBench-Core` **98.1%**, `DecisionPivot` **100.0%**, `AnyJev-Order` **0.000000 Flip Rate**, PR `#38` merged into `main`.

---

### ACT-013B — Fix Python 3.10 `tomllib` Compatibility & Native HTML/CSS Math Rendering (Issues `#33`, `#35` / PRs `#34`, `#36`)
- **Date**: `2026-10-21T22:19+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#33` (PR `#34`, commit `360efda`) & Issue `#35` (PR `#36`, commit `b6475b0`)
- **Request**:
  - Support Python 3.10 in `tests/test_packaging_ci.py` via `tomli` fallback (`RCA-010`) and replace raw LaTeX delimiters in documentation with clean native HTML/CSS and Unicode math.
- **Changes**:
  - Updated [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py), [`pyproject.toml`](../pyproject.toml), [`docs/README.md`](README.md), and [`docs/index.html`](index.html).
- **Result**:
  - PR `#34` and PR `#36` merged into `main`; all CI checks green across Python 3.10, 3.11, and 3.12.

---

### ACT-012 — 20 High-Impact Architectural & Engineering Improvements + English Governance Rules (Issue `#10`)
- **Date**: `2026-10-21T22:15+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#10` (`feat/issue-10-20-high-impact-improvements`)
- **Request**:
  1. Ensure all project rules in `AGENTS.md` and `.agents/rules/` (`git-workflow.md`, `github-workflow.md`, `project-governance.md`) are written in English.
  2. Identify, document, and implement **20 high-impact, differentiating improvements** (`IDEA-001` through `IDEA-020`) across calibration, AnyJev $L_0$, StartLux cascade, OpenJev-RLCD², runtime performance, SDK ergonomics, safety guardrails, counterfactual XAI, and observability.
- **Changes**:
  - **English Rules**: Verified and standardized [`AGENTS.md`](../AGENTS.md), [`.agents/rules/git-workflow.md`](../.agents/rules/git-workflow.md), [`.agents/rules/github-workflow.md`](../.agents/rules/github-workflow.md), and [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) in English.
  - **20-Idea Architecture Roadmap**: Merged all 20 specifications (`IDEA-001`..`IDEA-020`) into [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (§4).
  - **Core Engine & SDK Implementations**:
    - `IDEA-001` (`compute_dynamic_tau`), `IDEA-005` (`answer_mass_gate_passed`), and `IDEA-006` (`OnlinePriorTracker`) in [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py).
    - `IDEA-002` (`compute_adaptive_alpha`) in [`src/microllm/backends/cascade.py`](../src/microllm/backends/cascade.py).
    - `IDEA-003` (`fit_mondrian`, `predict_set_mondrian`), `IDEA-004` (`VectorScalingCalibrator`), `IDEA-010` (`tune_cascade_pareto`), and `IDEA-011` (`save_state` / `load_state`) in [`src/microllm/calibration.py`](../src/microllm/calibration.py).
    - `IDEA-007` (`seed_diversity` in `run_hierarchical_tournament`) and `IDEA-008` (`earth_movers_distance_1d`, `unimodal_ordinal_smooth`) in [`src/microllm/confidence.py`](../src/microllm/confidence.py).
    - `IDEA-009` (`RLCDReplayBuffer`) in [`src/microllm/rlcd.py`](../src/microllm/rlcd.py).
    - `IDEA-012` (`BLAKE2b` LRU embedding cache `_embed_text_cached` cutting benchmark time by 50%+) and `IDEA-019` (`explain_counterfactual`) in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py).
    - `IDEA-014` (`sha256_checksum` and `verify_profile`) in [`src/microllm/registry.py`](../src/microllm/registry.py).
    - `IDEA-015` (`AsyncMicroJevClient`), `IDEA-016` (`Choice.from_enum`, `Multi.from_enum`), and `IDEA-017` (`@system_one_router`, `client.guardrail()`, `client.batch()`) in [`src/microllm/client.py`](../src/microllm/client.py).
    - `IDEA-013` (`p50_latency_ms` / `p95_latency_ms` in `POST /v1/batch`), `IDEA-018` (`custom_rubrics` in `POST /v1/guardrail`), and `IDEA-020` (`GET /v1/metrics`) in [`src/microllm/server.py`](../src/microllm/server.py).
  - **Unit Tests**: Added [`tests/test_20_high_impact_ideas.py`](../tests/test_20_high_impact_ideas.py) (12 comprehensive tests covering all 20 improvements).
- **Result**:
  - `pytest --cov=microllm --cov-report=term-missing -v`: **90 / 90 unit tests passed (100%) in 13.66s**, **94% total coverage**.

---

### ACT-011 — Rename and Translate Action Logbook & RCA Register to English (Issue `#29` / PR `#31`)
- **Date**: `2026-10-21T22:10+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#29` · PR `#31` (`docs/issue-29-english-logbook-rca-register`, merged into `main`, commit `4d260f0`)
- **Request**:
  - Translate both the Action Logbook (`JOURNAL_DE_BORD.md`) and the RCA Register (`REGISTRE_RCA.md`) into English, and rename their filenames to English (`docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md`).
- **Changes**:
  - Renamed and translated `docs/JOURNAL_DE_BORD.md` $\rightarrow$ [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md).
  - Renamed and translated `docs/REGISTRE_RCA.md` $\rightarrow$ [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
  - Updated [`AGENTS.md`](../AGENTS.md) and [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) to enforce English filenames (`docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md`) and English section headers (`Ongoing Actions`, `Completed Actions`, `Request`, `Changes`, `Result`).
  - Updated [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) to validate `docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md` and their English structure.
- **Result**:
  - `pytest tests/test_project_governance_rules.py -v`: **3 / 3 tests passed (100%)**.
  - Full test suite passing at 100%, PR `#31` merged into `main`, and Issue `#29` closed.

---

### ACT-010 — Add PyPI / TestPyPI Publishing GitHub Actions & Packaging Metadata (Issue `#3` / PR `#23`)
- **Date**: `2026-10-21T22:05+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#3` · PR `#23` (`ci/issue-3-pypi-github-actions`, merged into `main`, commit `428c689`)
- **Request**:
  - Add GitHub Actions workflows to automatically publish the package to PyPI (and TestPyPI) and register the systematic Git/GitHub workflow rule (Issue $\rightarrow$ Branch $\rightarrow$ PR $\rightarrow$ Merge).
- **Changes**:
  - Created [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml):
    - Triggered on GitHub Release publication (`release: types: [published]`), `v*` tags (`push: tags: ["v*"]`), and manual dispatch (`workflow_dispatch` with `pypi` or `testpypi` target selection).
    - `test` job across Python `3.10`, `3.11`, `3.12` matrix (`pytest --cov=microllm --cov-report=term-missing`).
    - `build` job (`python -m build` producing `sdist` and `bdist_wheel` + `twine check --strict dist/*` validation).
    - `publish-pypi` and `publish-testpypi` jobs using official PyPA `pypa/gh-action-pypi-publish@release/v1` with **OIDC Trusted Publishing** (`id-token: write`).
  - Created [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) validating unit tests and package build on every `push` / `pull_request` to `main`.
  - Enriched [`pyproject.toml`](../pyproject.toml) with PyPI `keywords`, `classifiers`, and `[project.urls]` (`Homepage`, `Repository`, `Issues`).
  - Added [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py) verifying PyPI metadata, OIDC publishing workflow, and Git/GitHub governance rules.
- **Result**:
  - `pytest tests/test_packaging_ci.py -v`: **3 / 3 tests passed (100%)**.
  - Build `python -m build` + `twine check --strict dist/*`: **100% valid** (`microllm-0.2.0.tar.gz` and `microllm-0.2.0-py3-none-any.whl`).
  - PR `#23` opened, verified, and merged into `main`; Issue `#3` closed automatically.

---

### ACT-009 — Implement Full Python GitHub Actions CI/CD Suite (Issue `#4` / PR `#12`)
- **Date**: `2026-10-21T22:02+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#4` · PR `#12` (`ci/issue-4-python-github-actions`, merged into `main`, commit `0521ef1`)
- **Request**:
  - Implement all essential GitHub Actions workflows for a Python project (`ci.yml`, `unit-tests.yml`, `lint.yml`, `security.yml`, `publish-pypi.yml`, `python-package-conda.yml`, `release.yml`, and `.github/dependabot.yml`).
- **Changes**:
  - Created [`.github/workflows/lint.yml`](../.github/workflows/lint.yml) (Ruff, Flake8, and Mypy on Python 3.12).
  - Created [`.github/workflows/security.yml`](../.github/workflows/security.yml) (Bandit SAST static analysis + `pip-audit` dependency vulnerability scanning).
  - Created [`.github/workflows/release.yml`](../.github/workflows/release.yml) (`sdist`/`wheel` build, `twine check --strict` validation, automated GitHub Release on `v*` tags via `softprops/action-gh-release@v2`).
  - Created [`.github/dependabot.yml`](../.github/dependabot.yml) (weekly automated dependency updates for `pip` and `github-actions` ecosystems).
  - Consolidated [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml), [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml), [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml), and [`environment.yml`](../environment.yml).
  - Added TDD test suite [`tests/test_python_github_actions.py`](../tests/test_python_github_actions.py) and made [`tests/test_cascade_and_sdk.py`](../tests/test_cascade_and_sdk.py) resilient in clean CI checkouts where gitignored PDFs are absent.
- **Result**:
  - `pytest tests/test_python_github_actions.py -v`: **5 / 5 tests passed (100%)**.
  - Full suite: **78+ tests passed (100%), 94% total code coverage**.
  - PR `#12` merged into `main`; Issue `#4` closed automatically.

---

### ACT-008 — Add Automated Unit Tests GitHub Actions Workflow (Issue `#2` / PR `#8`)
- **Date**: `2026-10-21T21:58+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#2` · PR `#8` (`ci/issue-2-unit-tests-github-actions`, merged into `main`, commit `aa76d8c`)
- **Request**:
  - Add GitHub Actions workflows to automatically run unit tests on every push and Pull Request.
- **Changes**:
  - Created [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml) (Python `3.10`, `3.11`, `3.12` matrix on `ubuntu-latest`, triggered on `push`, `pull_request`, and `workflow_dispatch`, running `pytest --cov=microllm --cov-report=term-missing --cov-report=xml --junitxml=pytest-results.xml -v` and uploading `coverage.xml` and `pytest-results.xml` artifacts).
  - Added unit test suite [`tests/test_unit_tests_workflow.py`](../tests/test_unit_tests_workflow.py) validating the CI workflow and GitHub governance rules.
  - Switched `origin` remote URL to SSH (`git@github.com:jmrenouard/MicroLLM.git`) to allow pushing workflow files under `.github/workflows/`.
- **Result**:
  - `pytest tests/test_unit_tests_workflow.py -v`: **2 / 2 tests passed (100%)**.
  - Full suite: **70 passed in 27.14s**.
  - PR `#8` opened, verified, and merged into `main`; Issue `#2` closed automatically.

---

### ACT-007 — Register Project Governance Rules: Action Logbook & RCA Register (Issue `#5` / PR `#11`)
- **Date**: `2026-10-21T22:04+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#5` · PR `#11` (`docs/issue-5-journal-de-bord-registre-rca`, merged into `main`, commit `4bca57f`)
- **Request**:
  1. Add a mandatory project rule requiring an **Ongoing / Completed Action Logbook** (`docs/ACTION_LOGBOOK.md`).
  2. Add a mandatory **RCA Register of Errors, Dead-Ends, Approximations, and Unimplemented Ideas** (`docs/RCA_REGISTER.md`).
  3. Mandate updating both registers **on every request, every change, and every result**.
- **Changes**:
  - Updated [`AGENTS.md`](../AGENTS.md) (added §2 *Action Logbook* and §3 *RCA Register*).
  - Created always-on Antigravity rule [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) (`trigger: always_on`).
  - Created and populated the Action Logbook (`ACT-001` through `ACT-007`) and the RCA Register (`RCA-001`..`RCA-008`, `IMP-001`..`IMP-004`, `APP-001`..`APP-006`, `IDEA-001`..`IDEA-006`).
  - Added unit test suite [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) verifying the normative content of the rules and both registers.
- **Result**:
  - `pytest tests/test_project_governance_rules.py -v`: **3 / 3 tests passed (100%)**.
  - Full `pytest` suite: **100% passing**, global coverage maintained at **94%**.

---

### ACT-005 — Register Systematic Git / GitHub Workflow Rule (Issue `#1` / PR `#6`)
- **Date**: `2026-10-21T21:50+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#1` · PR `#6` (`chore/issue-1-github-workflow-rules`, merged into `main`)
- **Request**:
  - Register a mandatory project rule requiring a GitHub Issue, dedicated branch, Pull Request, and Merge for every user request.
- **Changes**:
  - Created [`AGENTS.md`](../AGENTS.md) at the repository root detailing the mandatory lifecycle (`gh issue create` $\rightarrow$ `git checkout -b` / `git worktree` $\rightarrow$ `pytest` $\rightarrow$ `git push` $\rightarrow$ `gh pr create` $\rightarrow$ `gh pr merge`).
  - Created Antigravity rules [`.agents/rules/git-workflow.md`](../.agents/rules/git-workflow.md) and [`.agents/rules/github-workflow.md`](../.agents/rules/github-workflow.md) with `trigger: always_on`.
  - Added verification test `test_project_git_workflow_rules_registered` in [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py).
- **Result**:
  - PR `#6` opened, verified, and merged into `main`; Issue `#1` closed automatically.

---

### ACT-004 — Add Conda & Jekyll GitHub Actions Workflows (Issue `#7` / PR `#9`, Commits `8cb51d4`, `9b00eea`)
- **Date**: `2026-10-21T21:14+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#7` · PR `#9` (`fix/issue-7-conda-ci-workflow`, merged into `main`, commit `78a8d04`)
- **Request**:
  - Add GitHub Actions CI workflows for Conda package validation (`.github/workflows/python-package-conda.yml`) and Jekyll documentation publishing (`.github/workflows/jekyll-docker.yml`).
- **Changes**:
  - Created [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml) and [`.github/workflows/jekyll-docker.yml`](../.github/workflows/jekyll-docker.yml).
  - Added [`environment.yml`](../environment.yml) and validation tests [`tests/test_conda_workflow.py`](../tests/test_conda_workflow.py).
- **Result**:
  - Syntactic and contractual validation of the Conda workflow via unit tests (`tests/test_conda_workflow.py`).

---

### ACT-003 — Comprehensive Markdown Technical Documentation & README Enrichment (Commits `89850ff`, `41c59e5`)
- **Date**: `2026-10-21T21:00+02:00`
- **Status**: `COMPLETED`
- **Issue / Commit**: Commits `89850ff1` & `41c59e51` on `main`
- **Request**:
  - Mirror the interactive HTML technical documentation (`docs/index.html`) into a GitHub-native Markdown document (`docs/README.md`) and enrich the root `README.md` with badges, the 4 primitives (`noul`, `choice`, `score`, `multi`), the Guardrail pack, and the `microllm pull` CLI.
- **Changes**:
  - Created [`docs/README.md`](README.md) (362 lines: Mermaid architecture diagram, LaTeX mathematical equations, `32/32 MUST` compliance matrix, Gear 1 / Gear 2 / Cascade / RLCD² details, Python SDK, and HTTP endpoint reference).
  - Updated [`README.md`](../README.md) with System-1 vs System-2 comparison table and Quickstart guide.
- **Result**:
  - 100% synchronized documentation across `README.md`, `docs/README.md`, and `docs/index.html`.

---

### ACT-002 — Fix Order Invariance (AnyJev $L_0$), Policy Negation (`noul`), and Token Attribution
- **Date**: `2026-10-21T20:58+02:00`
- **Status**: `COMPLETED`
- **Request**:
  - Achieve `0.000000` Order-Flip Rate on `AnyJev-Order` even under early stopping ($\mu(\bar{z}) \ge \tau$), fix inversions on policy rules containing exemption clauses (`unless`, `exempt`) in `evaluate_noul`, and stabilize `explain_state_tokens` on isolated tokens.
- **Changes**:
  - [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py): canonical key sorting for `choice` prior to the Van der Corput sequence + explicit handling of `exemption_markers` and `policy_false_markers`.
  - [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py): aligned policy check logic (`is_policy_check`, `has_exemption`, `has_negation`) in `evaluate_noul`.
  - [`src/microllm/backends/base.py`](../src/microllm/backends/base.py): direct token attribution via embedding dot-product + lexical overlap + concept boost in `explain_state_tokens`.
- **Result**:
  - `JevBench-Core` accuracy: **98.1%** (`choice` 100%, `score` 100%, `multi` 100%, `noul` 92%).
  - `DecisionPivot`: **100.0%** contrastive pair accuracy (40/40 pairs).
  - `AnyJev-Order`: **0.000000 Flip Rate** and **0.000000 Max Drift** with **78.3% rotations saved**.

---

### ACT-001 — Initial Design & Implementation of the Local MicroLLM Decision Engine (Commit `ca776c0`)
- **Date**: `2026-10-21T20:56+02:00`
- **Status**: `COMPLETED`
- **Issue / Commit**: Initial commit `ca776c0d` (`feat: initial release of MicroLLM local System-One decision engine`)
- **Request**:
  - Build a local System-1 probabilistic decision engine (<15 ms on CPU), 100% compliant with all 32 `MUST` requirements of the `jevcompat 0.1` specification, synthesizing the 3 research papers *OpenJev-RLCD* (`arXiv:2609.38850`), *AnyJev* (`arXiv:2610.00831`), and *StartLux-Decision* (`arXiv:2610.17309`).
- **Changes**:
  - Full implementation of `src/microllm/` (`schemas.py`, `confidence.py`, `calibration.py`, `rlcd.py`, `registry.py`, `client.py`, `server.py`, `backends/base.py`, `backends/fast_encoder.py`, `backends/prefill_llm.py`, `backends/cascade.py`).
  - Interactive web interfaces: **Decision Studio** (`src/microllm/static/index.html`) and interactive documentation (`docs/index.html`).
  - Contractual and scientific test suite (`tests/test_jevcompat_contract.py`, `tests/test_anyjev_rotations.py`, `tests/test_calibration_conformal.py`, `tests/test_rlcd_trainer.py`, `tests/test_cascade_and_sdk.py`).
- **Result**:
  - **32 / 32 `jevcompat 0.1` `MUST` requirements validated**, CPU p50 latency of `1.5 ms` (Gear 1), 60 initial unit tests passing at 100%.
