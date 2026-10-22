---
trigger: always_on
description: Mandatory MicroLLM language rule — all project documentation, governance registers, rules, source code, comments, docstrings, and tests must be written exclusively in English.
---

# Project Rule: Mandatory English-Only Policy for Documentation and Code

All artifacts in the **MicroLLM** repository **MUST be written exclusively in English** without exception:

1. **Documentation & Governance Registers**:
   - Root and module documentation (`README.md`, `BENCHMARKS.md`, `docs/README.md`, `docs/index.html`, `docs/PRODUCT_DOCUMENTATION.md`, `docs/TECHNICAL_SPECIFICATIONS.md`, `docs/TEST_PLAN.md`, `docs/LITERATURE_REVIEW.md`, `docs/EXTERNAL_DATASETS_BENCHMARKS.md`, `docs/papers/README.md`).
   - Living governance registers (`docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md`).
   - Agent instructions and rules (`AGENTS.md` and `.agents/rules/*.md`).

2. **Source Code, Schemas, CLI & Web UI**:
   - All Python modules (`src/microllm/**/*.py`), identifiers, type annotations, docstrings, inline comments, error messages, CLI help strings, and Decision Studio UI labels (`src/microllm/static/index.html`).

3. **Tests, Benchmarks & CI/CD**:
   - All test suites (`tests/**/*.py`), GitHub Actions workflows (`.github/workflows/*.yml`), and commit/PR/Issue descriptions.
   - *Permitted Exception*: Explicit multilingual test input strings (`EN` / `FR` / `ES` / `DE` sample sentences inside benchmark/test fixtures that specifically evaluate `tokenize_multilingual` or cross-lingual NLI routing) and verbatim quotes of user prompts in `docs/ACTION_LOGBOOK.md` accompanied by an English translation.
