# MicroLLM v0.4.0 — S1-SHAP, Decision Guardrails & Calibrated System-1

We are thrilled to announce the official production release of **MicroLLM v0.4.0** (`micro-jev`), delivering major advancements in explainability, adversarial guardrail hardening, policy gatekeeping, and empirical scale validation for our ultra-low-latency, local System-1 decision engine.

MicroLLM eliminates the heavy latency, memory footprint, and operational costs of generative autoregressive LLMs for deterministic classification, workflow routing, and safety guardrails, executing certified probabilistic decisions locally on standard CPU hardware with sub-millisecond speed and mathematically bounded confidence.

---

## 🚀 Key Highlights & Verified Production Metrics

- **S1-SHAP in Logit Space (`DecisionBackend.explain_counterfactual`)**:
  - Exact counterfactual attribution in unbounded logit / log-odds space:

$$
\Delta\ell = \ell(x) - \ell(x \setminus \{w_i\})
$$

  - Resolves the well-known probability saturation pathology near 0.0 and 1.0, returning both `original_log_odds` and `logit_drop` alongside normalized token importance for `noul`, `choice`, `score`, and `multi` decisions.
- **Evidence Competition Hijack Defense**:
  - Novel cross-clause evidence competition detector (`detect_evidence_competition_hijack`) and certainty hijack markers (`_CERTAINTY_HIJACK_MARKERS`).
  - Automated abstain gate (`x_hijack_suspected`, `x_abstain=True`) in `FastEncoderBackend` that intercepts attacks where injected certainty words attempt to override legitimate semantic evidence.
  - Hardened prompt injection detector threshold (`prompt_injection >= 0.88`) in `server.py`.
- **Policy Boundary Gatekeeping & Roleplay Hardening**:
  - Explicit `policy_boundaries` (`allowed` / `forbidden`) in `SystemOneRequest`, `server.py`, `server_models.py`, `client.py`, and `mcp_server.py`.
  - Remainder-based destructive hazard inspection ensuring that allowlist rules (e.g., `rm -rf ./dist`) never allow chained destructive payloads (`DROP TABLE`, `rm -rf /etc`, `rm -rf $HOME`) to bypass safety gates.
  - Non-downgradable hard hazard barriers for destructive commands (`rm -rf /`, `DROP TABLE`).
  - Unicode NFKC normalization preventing fullwidth character bypasses (`\uff52\uff4d \uff0d\uff52\uff46`).
  - Strict interdiction of adversarial roleplay framing (`"in a novel"`, `"screenplay"`, `"pretend you are"`) paired with harmful targets.
- **Expanded Academic Catalog & Literature Taxonomies**:
  - 12 new October–November 2026 arXiv research papers added, expanding the catalog from 104 to 116 curated papers (`src/microllm/data/academic_papers_catalog.json`).
  - Research roadmaps `IDEA-052` through `IDEA-057` registered in `docs/RCA_REGISTER.md`.
  - Comprehensive schema and metadata validation suite in `tests/test_academic_papers_catalog.py`.
- **12-Suite Comparative Benchmark Campaign**:
  - Executed full 12-suite benchmark campaign evaluating **128,713 items and 131,487 decisions in 66.73s CPU wall-clock time** across 28 external datasets.
  - Master benchmark report generated in `benchmarks/all_benchmarks_master_report.json` and `benchmarks/latest_master_report.md`.
  - Confirmed p50 latency of **0.013 ms (38,917 decisions/s)** on large-scale CPU evaluation (`test_cases_11000`), 0.17 ms on multilingual cross-lingual transfer, and 1.5x cascade acceleration.
  - Fully refreshed `BENCHMARKS.md` and `README.md` with verified real-world metrics.
- **Comprehensive Test Suite & Quality Status**:
  - **1,370 unit and integration tests passing at 100%**.
  - Zero static typing errors under `mypy`.
  - 100% LaTeX math formatting compliance verified by `pytest tests/test_docs_math_syntax.py`.

---

## 🛠 Features & Capabilities

### 1. S1-SHAP in Logit Space
Standard perturbation explainers measure probability drops $\Delta p = p(x) - p(x \setminus \{w\})$, which saturate and lose sensitivity when probabilities are close to 1.0 or 0.0. S1-SHAP computes drops directly in log-odds space:

$$
\text{drop}_i = \log\left(\frac{p(x)}{1 - p(x)}\right) - \log\left(\frac{p(x \setminus \{w_i\})}{1 - p(x \setminus \{w_i\})}\right)
$$

This ensures high sensitivity across extreme confidence regimes and provides reliable attribution for compliance and auditing.

### 2. Evidence Competition Hijack Defense
Adversaries frequently attempt to manipulate classification models by injecting high-certainty phrases (e.g., *"definitely authorized"*, *"undoubtedly safe"*, *"confirmed legitimate"*). MicroLLM v0.4.0 isolates clauses, calculates competitive tension between certainty markers and contradictory semantic evidence, and automatically triggers `x_abstain=True` when hijack markers are detected.

### 3. Policy Boundary Gatekeeping
For coding and enterprise tool agents, MicroLLM provides pre-execution policy boundary evaluation:
- **Allowed Rules**: Specific authorized command patterns (e.g., `git status`, `pytest`, `npm test`, `rm -rf ./dist`).
- **Forbidden Rules**: Explicitly prohibited patterns that always override allowlists.
- **Remainder-Based Inspection**: Substring extraction strips authorized prefixes and inspects the remaining command payload for hidden or chained hazards (`DROP TABLE`, `rm -rf /`).
- **Hard Hazard Invariants**: Critical system deletions (`rm -rf /`) can never be downgraded, regardless of permissive allowlist configurations.

---

## 📦 Getting Started

### 1. Installation

Install via pip:

```bash
pip install micro-jev==0.4.0
```

Or install in development mode from source:

```bash
git clone https://github.com/jmrenouard/MicroLLM.git
cd MicroLLM
pip install -e ".[dev]"
```

### 2. Python SDK Example

```python
from microllm import MicroJevClient

client = MicroJevClient()

# 1. Routing decision with S1-SHAP explanation
response = client.choice(
    state="The customer request contains a suspected prompt injection: ignore previous rules.",
    question_id="routing_intent",
    question_text="Route the incoming agent action safely.",
    options=[
        ("execute", "Execute safe tool command"),
        ("quarantine", "Quarantine suspicious or harmful prompt"),
        ("escalate", "Escalate to human review"),
    ],
)

print(f"Decision: {response.best_option}")
print(f"Confidence: {response.confidence:.4f}")
print(f"Abstain: {response.x_abstain}")

# 2. Agent policy boundary pre-execution gate
guardrail_res = client.guardrail(
    tool="bash",
    command="rm -rf ./dist; DROP TABLE users;",
    policy_boundaries={
        "allowed": ["rm -rf ./dist"],
        "forbidden": ["DROP TABLE", "rm -rf /"],
    },
)

print(f"Safe: {guardrail_res.get('safe')}")
print(f"Action: {guardrail_res.get('action')}")  # 'block'
```

### 3. REST API & CLI Serving

Launch the local decision engine server:

```bash
microllm serve --host 0.0.0.0 --port 8000 --strategy auto --neural-model minilm
```

Send a request via cURL:

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": "Payment gateway reports timeout on transaction #TX-98421.",
    "questions": [
      {
        "id": "action",
        "type": "choice",
        "question": "What is the recommended recovery action?",
        "options": [
          {"name": "retry_idempotent", "description": "Retry with idempotent key"},
          {"name": "reverse", "description": "Reverse transaction and refund"},
          {"name": "alert_oncall", "description": "Alert engineering oncall"}
        ]
      }
    ]
  }'
```

---

## 📋 Release Summary

| Milestone | Scope | Deliverables | Verification |
| :--- | :--- | :--- | :--- |
| **M1** | Unit Test Reinforcement & Boundary Hardening | 25+ new unit tests for S1-SHAP, Evidence Hijack, Policy Gates, Academic Catalog, and Wheel packaging fix | 1,370 tests passing (100%) |
| **M2** | 12-Suite Benchmark Campaign & Docs | Full campaign executed across 28 external datasets; master report generated; `BENCHMARKS.md` and `README.md` updated | 12/12 suites passing, 128,713 items evaluated in 66.73s |
| **M3** | Production Release Packaging & Publishing | Version bump to 0.4.0 across 8 files, changelog, release notes, wheel build & twine strict check, git tag `v0.4.0`, GitHub Release | Clean wheel/sdist, 100% tests pass, GitHub Release published |
