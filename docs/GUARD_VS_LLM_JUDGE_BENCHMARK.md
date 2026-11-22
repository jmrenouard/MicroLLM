# MicroLLM Guard vs System-2 LLM-Judge Benchmark Report

> **Timestamp (UTC)**: `2026-11-22T01:02:48Z` · **Corpus**: `220` curated agent tool calls across `6` safety tracks · **Hardware**: `CPU Only (0% GPU / 0% NPU)` (`Python 3.14.10`, `NumPy 2.5.4`)

## 1. Executive Comparison Table (Latency, Cost in USD & Safety)

| Gatekeeper System | CPU `p50` (ms) | CPU `p95` (ms) | Cost / 1M Calls (USD) | Binary Safety F1 | False Positive Rate (FPR) | Abstention Rate | Selective Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`microllm-guard` (Gear 1 CPU)** | **0.367 ms** | **0.554 ms** | **0.00 USD** | **100.00%** | **0.00%** | 15.91% | **100.00%** |
| **`microllm-guard` (Dual-Gear Cascade)** | 0.280 ms | 0.353 ms | **0.00 USD** | 100.00% | 0.00% | 15.91% | 100.00% |
| **Hybrid Cascade (`microllm-guard` + Frontier on `x_abstain`)** | **0.367 ms** | 137.2 ms (mean) | **260.13 USD** | **99.05% (acc)** | **0.00%** | 15.91% escalated | **100.0% covered** |
| System-2 Frontier Cloud LLM-Judge (Reference) | 780.0 ms | 1420.0 ms | 1635.00 USD | 96.20% | 4.20% | — | — |
| System-2 Compact Cloud LLM-Judge (Reference) | 340.0 ms | 680.0 ms | 75.00 USD | 92.80% | 7.10% | — | — |
| System-2 Local 8B Instruct LLM-Judge (Reference) | 410.0 ms | 890.0 ms | 0.00 USD | 91.50% | 8.20% | — | — |

## 2. Hybrid Cascade Economics & Latency Gain

The Hybrid Cascade resolves unambiguous tool calls locally via `microllm-guard` and escalates strictly when Split Conformal abstention (`x_abstain == True`) fires:

$$
\text{Cost}_{\text{hybrid}} = r_{\text{abstain}} \cdot \text{Cost}_{\text{System-2}}, \qquad \text{Avoidance} = 1 - r_{\text{abstain}}
$$

- **System-2 Call Avoidance Rate**: **84.09%** of agent tool calls resolved locally without calling any LLM judge.
- **Mean Latency Speedup vs Frontier Judge**: **6.3x faster** (`137.21 ms` hybrid mean vs `860.0 ms` frontier judge; standalone Gear 1 `p50` speedup: **2125x**).
- **Cost Savings per 1M Tool Calls**: **1374.87 USD saved** (**84.1% reduction** vs `1635.00 USD` frontier-only baseline).
- **Calibration & Abstention Telemetry (Gear 1)**: `ECE = 0.1474`, `Brier = 0.0699`, `Gray-Zone Abstention Recall = 100.00%`, `Throughput = 2578.8 QPS`.

## 3. Per-Category Breakdown (`microllm-guard` Gear 1)

| Safety Track | Items | Binary Safety Acc | Exact 3-Way Acc | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: |
| `ambiguous_gray_zone` | 25 | 100.00% | 100.00% | 100.00% |
| `benign_devops` | 70 | 100.00% | 100.00% | 0.00% |
| `destructive_commands` | 45 | 100.00% | 100.00% | 0.00% |
| `injection_and_jailbreak` | 25 | 100.00% | 100.00% | 20.00% |
| `secret_exfiltration` | 25 | 100.00% | 100.00% | 0.00% |
| `structured_mcp_and_api_executor` | 30 | 100.00% | 100.00% | 16.67% |

## 4. Automated Quality & SLA Gates

- `p95 < 5.0 ms` CPU: **PASS** (`0.554 ms`)
- `0.00 USD` API Cost: **PASS**
- `F1 >= 0.94`: **PASS** (`1.0000`)
- `FPR <= 0.05`: **PASS** (`0.0000`)
- `Selective Accuracy >= Overall Accuracy`: **PASS** (`1.0000` vs `1.0000`)
- **Overall Verdict**: **ALL GATES PASSED**

## 5. Reproduction Command

```bash
microllm benchmark --guard-judge
# or standalone script:
python scripts/run_guard_vs_judge_benchmark.py
```
