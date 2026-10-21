# MicroLLM — Local Hybrid System-One Decision Engine

[![JevCompat 0.1](https://img.shields.io/badge/JevCompat_0.1-32%2F32_MUST-10b981)](docs/index.html)
[![Tests](https://img.shields.io/badge/tests-60_passed-3b82f6)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-94%25-10b981)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-8b5cf6)](LICENSE)

A fast, local **System One** probabilistic decision engine (<15 ms on CPU) that is **100% compliant with the 32 `MUST` requirements** of the `jevcompat` 0.1 specification (`POST /v1/systemone`, `GET /v1/models`) plus first-class extensions (`multi`, `POST /v1/guardrail`, `POST /v1/batch`, and `microllm pull` ONNX INT8 profiles).

Engineered by synthesizing three foundational research papers (see [`docs/papers/README.md`](docs/papers/README.md)):
1. **OpenJev-RLCD** ([*arXiv:2609.38850*](https://arxiv.org/abs/2609.38850), Gao & Wang, Sept. 2026): Two-stage training recipe (*Stage 1 Pathwise Brier* + *Stage 2 Leave-One-Out REINFORCE with KL Anchor*) and the group-mean variance identity proof (Proposition 1).
2. **AnyJev Technical Report** ([*arXiv:2610.00831*](https://arxiv.org/abs/2610.00831), Nokia Applied Research, Sept. 2026): Training-free first-token prefill readout, log-space circular rotation averaging ($L_0$) with the binary-reflected van der Corput schedule and early stopping ($\mu(\bar{z}) \ge \tau$), *Answer Mass Gate*, and empirical prior debiasing.
3. **StartLux-Decision** ([*arXiv:2610.17309*](https://arxiv.org/abs/2610.17309), StartLux & HKU, Oct. 2026): *Read-or-Think* cascade with weighted fusion $p = (1-\alpha)p_0 + \alpha p_1$ ($\alpha=0.75$), hierarchical tournament for up to 255 options, and $\text{RLCD}^2$ $\text{Beta}(8,2)$-weighted scoring rule.

---

## Key Features

- **4 Decision Primitives**:
  - `noul` — Binary validation (`value: bool`, `probability: float`)
  - `choice` — Mutually exclusive classification (`value`, `probabilities`, normalized entropy margin `confidence`, up to 255 options via 8-way tournament)
  - `score` — Ordinal rubric evaluation ($K=2..5$, normalized expectation $s \in [0,1]$, peak concentration `confidence`)
  - `multi` — Independent multi-label tagging (`values: list[str]`, per-tag probabilities, custom `threshold`)
- **Zero-Shot AI Safety Guardrail Pack (`POST /v1/guardrail`)**: Evaluates 4 calibrated safety rubrics in a single pass (<5 ms on CPU): `prompt_injection`, `destructive_operation`, `pii_or_secret_leak`, and `hallucination_risk` (with `groundedness_score`).
- **1-Command ONNX INT8 Profile Manager (`microllm pull`)**: Ollama-style profile registry (`fast-cpu-int8`, `multilingual`, `guardrail`) with symmetric per-row INT8 quantization into `~/.cache/microllm/`.
- **Interactive Decision Studio Web UI**: Visual No-Code Question Builder, 1-click Export-to-Code (Python / TypeScript / cURL), Token Attribution Highlighter, Gear Comparator, Guardrail Shield, and Batch CSV/JSONL Evaluator.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/jmrenouard/MicroLLM.git
cd MicroLLM
python -m venv .venv
# Windows: .\.venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Pull a Quantized ONNX INT8 Profile & Start the Server

```bash
# List and pull a pre-configured ONNX INT8 hardware profile (~60 MB in ~/.cache/microllm)
microllm list
microllm pull fast-cpu-int8

# Launch the FastAPI server & Decision Studio Web UI
microllm serve
```

- **Decision Studio Web UI**: **http://127.0.0.1:8000/**
- **Complete Technical Documentation**: **http://127.0.0.1:8000/documentation** (or open [`docs/index.html`](docs/index.html))
- **OpenAPI / Swagger UI**: **http://127.0.0.1:8000/docs**

### 3. Use the Typed Python SDK (`MicroJevClient`)

```python
from microllm import MicroJevClient, Noul, Choice, Score, Multi

client = MicroJevClient(base_url="http://127.0.0.1:8000")

res = client.ask(
    state="I was billed twice for invoice #4092! Please refund the duplicate charge today.",
    questions={
        "is_refund": Noul("Is the customer requesting an immediate refund?"),
        "department": Choice({
            "billing": "Invoices, double charges, refunds",
            "technical": "Bugs, crashes, API outages",
            "shipping": "Delayed package, lost parcel",
        }),
        "urgency": Score(["Low", "Medium", "High / Immediate"]),
        "tags": Multi({
            "billing": "Payment or invoice issue",
            "urgent": "Immediate customer escalation",
            "bug": "Software crash or error",
        }, threshold=0.5),
    },
)

print(res["is_refund"].value, res["is_refund"].probability)
print(res["department"].value, res["department"].confidence)
print(res["urgency"].value, res["urgency"].score)
print(res["tags"].values, res["tags"].probabilities)
```

### 4. Run the Automated Test Suite (60 tests, 94% coverage)

```bash
pytest --cov=microllm --cov-report=term-missing -v
```
