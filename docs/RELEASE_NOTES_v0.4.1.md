# MicroLLM v0.4.1 — Portable Vault Configuration, Release Hygiene & Reproducible Packaging

**MicroLLM v0.4.1** (`micro-jev`) is a maintenance, privacy-hygiene, and packaging reproducibility patch release following `v0.4.0`. It aligns published PyPI (`sdist` / `wheel`), GitHub Release, and Docker Hub artifacts with the scrubbed Git tree, generalizes Hugging Face credential resolution for portable environments, and reinforces CI standalone-wheel and repository hygiene checks.

---

## Key Highlights

- **Portable KeePass Vault Resolution (`microllm.hf_publishing`)**:
  - Generalized `get_vault_paths()` and `resolve_hf_token()` to use environment variables (`MICROLLM_KEEPASS_DB`, `KEEPASS_DB`, `KEEPASS_DB_PATH` and `MICROLLM_KEEPASS_KEY`, `KEEPASS_KEY`, `KEEPASS_KEY_PATH`) with a clean cross-platform default fallback at `~/.config/microllm/vault.kdbx` (`#269`).
  - Removed legacy workstation-specific path references from packaged distributions (`wheel` and `sdist`).
- **Repository Privacy & Packaging Hygiene Suite**:
  - Added automated CI regression suite (`tests/test_repo_privacy_hygiene.py`) enforcing `.gitignore` coverage on local vault files (`*.kdbx`, `*.keyx`) and private outreach artifacts, zero hardcoded OS user paths, and zero leaked personal contact metadata across all tracked files (`#269`).
  - Re-aligned cryptographic `SHA256SUMS.txt`, PyPI distributions, and Git tag provenance (`RCA-093`, `RCA-095`).
- **CI Standalone Wheel & Version Synchronization Guardrails**:
  - Updated isolated out-of-tree wheel integration assertions in `.github/workflows/ci.yml` (`#267`, `RCA-092`).
  - Added `test_all_version_surfaces_synchronized` in `tests/test_packaging_ci.py` to guarantee lockstep alignment across `pyproject.toml`, `microllm.__version__`, FastAPI `create_app().version`, MCP `serverInfo.version`, and `.github/workflows/ci.yml`.
  - Hardened `packaged_wheel` fixture in `tests/test_audit18_m4_cicd_packaging.py` so standalone packaging checks build the wheel on demand if `dist/` is empty.
- **Documentation & Governance Traceability**:
  - Updated `docs/ACTION_LOGBOOK.md` (`ACT-121`–`ACT-124`) and `docs/RCA_REGISTER.md` (`RCA-092`–`RCA-095`).

---

## Quality & Performance Summary

- **Test Suite**: `1,371` unit, integration, adversarial, and privacy-hygiene tests passing (`0` failed, `1` skipped; `92%` code coverage across `9,449` statements), `0` `flake8` / `mypy` issues, `16/16` LaTeX math syntax checks passing.
- **Core v0.4.x Engine Capabilities Preserved**:
  - Logit-space `S1-SHAP` counterfactual attribution (`original_log_odds`, `logit_drop`).
  - Inter-clause evidence-competition hijack defense (`x_hijack_suspected`, `x_abstain=True`).
  - Coding-agent `policy_boundaries` (`allowed` / `forbidden`) with remainder-based destructive hazard inspection and NFKC Unicode normalization.
  - Verified CPU throughput: `0.013 ms` p50 (`38,917 decisions/s` on `test_cases_11000`) across the 12-suite / 28-dataset benchmark campaign.

---

## Installation & Quickstart

### 1. Package Upgrade

```bash
pip install --upgrade micro-jev==0.4.1
```

Or run via multi-arch Docker (`linux/amd64`, `linux/arm64`):

```bash
docker pull jmrenouard/microllm:v0.4.1
docker run --rm -p 8000:8000 jmrenouard/microllm:v0.4.1
```

### 2. Python SDK Quickstart

```python
from microllm import MicroJevClient

client = MicroJevClient()

decision = client.choice(
    state="Payment gateway reports timeout on transaction #TX-98421.",
    question_id="recovery_action",
    question_text="Select the recommended recovery action.",
    options=[
        ("retry_idempotent", "Retry with idempotent key"),
        ("reverse", "Reverse transaction and refund"),
        ("alert_oncall", "Alert engineering oncall"),
    ],
)

print(decision.best_option, round(decision.confidence, 4), decision.x_abstain)
```

### 3. CLI Server

```bash
microllm serve --host 0.0.0.0 --port 8000 --strategy auto --neural-model minilm
```
