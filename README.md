# MicroLLM — Local Hybrid System-One Decision Engine

A fast, local **System One** probabilistic decision engine (<15 ms on CPU) that is **100% compliant with the 32 `MUST` requirements** of the `jevcompat` 0.1 specification (`POST /v1/systemone`).

Engineered by synthesizing three foundational research papers (archived locally in [`docs/papers/`](docs/papers/)):
1. **OpenJev-RLCD** (*arXiv:2609.38850*, Gao & Wang, Sept. 2026) — [`docs/papers/2609.38850_OpenJev_RLCD.pdf`](docs/papers/2609.38850_OpenJev_RLCD.pdf): Two-stage training recipe (*Stage 1 Pathwise Brier* + *Stage 2 Leave-One-Out REINFORCE with KL Anchor*) and the group-mean variance identity proof (Proposition 1).
2. **AnyJev Technical Report** (*arXiv:2610.00831*, Nokia Applied Research, Sept. 2026) — [`docs/papers/2610.00831_AnyJev.pdf`](docs/papers/2610.00831_AnyJev.pdf): Training-free first-token prefill readout, log-space circular rotation averaging ($L_0$) with the binary-reflected van der Corput schedule and early stopping ($\mu(\bar{z}) \ge \tau$), *Answer Mass Gate*, and empirical prior debiasing.
3. **StartLux-Decision** (*arXiv:2610.17309*, StartLux & HKU, Oct. 2026) — [`docs/papers/2610.17309_StartLux_Decision.pdf`](docs/papers/2610.17309_StartLux_Decision.pdf): *Read-or-Think* cascade with weighted fusion $p = (1-\alpha)p_0 + \alpha p_1$ ($\alpha=0.75$), hierarchical tournament for up to 255 options, and $\text{RLCD}^2$ $\text{Beta}(8,2)$-weighted scoring rule.

---

## Quickstart

### 1. Start the FastAPI Server & Decision Studio Web UI

```powershell
.\.venv\Scripts\python -m microllm.server
```

- **Decision Studio Web UI**: **http://127.0.0.1:8000/** (Interactive `noul` / `choice` / `score` playground, Gear 1 vs AnyJev $L_0$ vs Cascade comparator, option-order invariance audit, and local RLCD² calibration lab).
- **Complete Technical Documentation**: **http://127.0.0.1:8000/documentation** (or open [`docs/index.html`](docs/index.html) directly in your browser).
- **OpenAPI / Swagger UI**: **http://127.0.0.1:8000/docs**

### 2. Use the Typed Python SDK (`MicroJevClient`)

```python
from microllm import MicroJevClient, Noul, Choice, Score

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
    },
)

print(res["is_refund"].value, res["is_refund"].probability)
print(res["department"].value, res["department"].confidence)
print(res["urgency"].value, res["urgency"].score)
```

### 3. Run the Automated Test Suite (58 tests, 96% coverage)

```powershell
.\.venv\Scripts\pytest --cov=microllm --cov-report=term-missing -v
```
