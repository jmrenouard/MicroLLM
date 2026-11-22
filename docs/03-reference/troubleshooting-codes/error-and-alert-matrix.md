# Reference: Error Codes, Threat Matrix, and SLA Alerts

Diagnostic lookup table mapping HTTP errors, `microllm.guard.v1` threats, JSON-RPC faults, and SLA breaches to runbooks.

## 1. HTTP Status Codes and FastAPI Error Shapes (`src/microllm/server.py`)

| HTTP Status | Error Condition / Shape | Probable Cause | Remediation Runbook |
| :---: | :--- | :--- | :--- |
| `401` | `{"detail": {"error_type": "authentication_error", "message": "Missing Authorization: Bearer <key> header."}}` | Request to auth-protected `/v1/*` route omitted `Authorization: Bearer` while `MICROLLM_API_KEY` is active. | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| `401` | `{"detail": {"error_type": "authentication_error", "message": "Invalid API key provided."}}` | Bearer token failed constant-time `hmac.compare_digest` check against `MICROLLM_API_KEY`. | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| `404` | `{"detail": "Model '<name>' not found."}` | Unknown `model` on `POST /v1/systemone` or `POST /v1/batch`. | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| `404` | `{"detail": "Unknown profile: <profile>"}` | Unknown `profile` name sent to `POST /v1/profiles/pull`. | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| `404` | `{"detail": "SVG asset not found."}` or `"PDF paper not found."` or `"arXiv paper '<id>' not found in catalog."` | Missing static SVG asset, uncached arXiv PDF, or unknown `arxiv_id` query on `/v1/papers`. | [Deploy MicroLLM with Docker](../../02-how-to-guides/deployment/deploy-microllm-docker.md) |
| `409` | `{"detail": "A benchmark run is already in progress."}` | Concurrent `POST` to `/v1/benchmark`, `/v1/benchmark/run`, `/v1/benchmark/external`, or `/v1/benchmark/hf-jev-models` while `_benchmark_lock` is held. | [Run Reproducible Benchmarks](../../02-how-to-guides/operations/run-reproducible-benchmarks.md) |
| `422` | `{"detail": [{"loc": ["body", ...], "msg": ..., "type": ...}]}` | Pydantic v2 validation error, `NaN`/`Infinity` float rejected by `_ensure_finite_json`, boolean in numeric array, `items > 500` in `/v1/batch`, or `criteria` bounds exceeded (`2..255` `choice`/`multi`, `2..10` `score`). | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `422` | `HTTPException(status_code=422, detail=...)` | Invalid RLCD dataset dimensions/labels on `/v1/rlcd` or `/v1/studio/rlcd-demo`, or invalid profile pull parameters. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| `5xx` / Probe Fail | Container `HEALTHCHECK` failure on `GET /healthz` (`jevcompat` `errors.no-5xx` violation) | Uvicorn worker crash, port mismatch (`HOST`/`PORT`), or unwritable `MICROLLM_CACHE_DIR`. | [Deploy MicroLLM with Docker](../../02-how-to-guides/deployment/deploy-microllm-docker.md) |

## 2. Gatekeeper Decisions, `primary_threat` Values, and CLI Exit Codes

| Condition / Threat Code | Decision | Exit Code | Trigger Logic (`guard.py`, `guard_cli.py`, `claude_code_hook.py`) | Remediation Runbook |
| :--- | :---: | :---: | :--- | :--- |
| `prompt_injection` | `block` | `2` | Explicit jailbreak/override marker or roleplay-harm combination with score `>= eff_block_threshold` (`0.75`). | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `destructive_operation` | `block` | `2` | Matches `DANGEROUS_COMMAND_REGEX` (pipe-to-shell, fork bomb, raw `dd`/`mkfs`, `/etc/shadow` read) or hard destructive markers (`rm -rf`, `DROP TABLE`) at `>= 0.75`. | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| `pii_or_secret_leak` | `block` | `2` | Matches secret token patterns (`sk-`, `ghp_`, `github_pat_`, `AKIA`, PEM private keys, `password=`, `bearer `) at `>= 0.75`. | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| `hallucination_risk` | `block` | `2` | Ungrounded numeric values, polarity conflict against `context`, or explicit fabrication markers at `>= 0.75`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `policy_boundary_violation` | `block` | `2` | Payload matches `policy_boundaries["forbidden"]` (`0.94`), or Claude `Write`/`Edit` targets `/etc/shadow`, `/etc/sudoers`, `/etc/passwd`, or `.ssh/authorized_keys`. | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| `evidence_hijack` | `escalate` (`abstain=true`) | `3` (or `2` if `--fail-closed`) | `hijack_suspected == true` (certainty-claim or evidence-competition markers) without hard block. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| `ambiguous_tool_action` | `escalate` (`abstain=true`) | `3` (or `2` if `--fail-closed`) | Privileged infrastructure mutation (`kubectl apply/delete`, `terraform apply/destroy`, `ALTER TABLE`, `git push --force`, `chmod -R`) or Split Conformal set cardinality `> 1`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| Invalid CLI / Policy Input | `error` | `1` | Malformed JSON on `stdin` or `--args-json`, missing `--policy-file`, invalid `--threshold` outside `[0.1, 0.99]`, or missing command after `--` in `mcp-proxy`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `claude-hook` Parse / Empty Stdin | `deny` (fail-closed) | `2` | Empty `stdin`, non-object JSON, or `ValidationError` inside `run_claude_hook_stdin`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |

## 3. MCP JSON-RPC 2.0 Error Codes (`src/microllm/middleware/mcp_guard.py`)

| JSON-RPC Code | Message Prefix | Source Condition | Remediation Runbook |
| :---: | :--- | :--- | :--- |
| `-32700` | `Parse error` | Malformed JSON or non-finite constant (`NaN`/`Infinity`) received on `mcp-proxy` `stdin`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `-32600` | `Invalid Request` | Non-object JSON frame or empty JSON-RPC batch array (`[]`). | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `-32602` | `Invalid params` | `tools/call` frame missing object `params`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `-32603` | `Upstream MCP server...` | Upstream stdio pipe closed, write `OSError`, or malformed upstream JSON response. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |

## 4. Runtime Telemetry Alerts and Profile Integrity Faults

| Alert / Symptom | Telemetry Signal / Function | Probable Cause | Remediation Runbook |
| :--- | :--- | :--- | :--- |
| **Abstention Spike** | `abstained_decisions` growth in `GET /v1/metrics` or `open_set_abstain` regime | Domain drift, low softmax margin, `gate_threshold` (`0.75`) or `conformal_alpha` (`0.10`) miscalibration, or missing candidate descriptions. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| **CPU Latency Spike** | `latency_p50_ms` / `latency_p95_ms` breach in `GET /v1/metrics` | Frequent `X-Cascade-Triggered: true` escalation to `neural` `mdeberta`, cold ONNX model load, or external `MICROLLM_SLM_BASE_URL` network delay. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| **Profile Verification Failure** | `verify_profile()` returns `verified: false` (`"Profile not installed."` / `"Quantized bundle missing."` / `sha256_match: false`) | Unpulled profile, corrupted `model_quantized_int8.npz`, or wrong `MICROLLM_CACHE_DIR` mount. | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| **Container Startup / Cache Error** | `PermissionError` or non-200 `GET /healthz` in container | Host volume mounted over `/app/.cache` without UID `10001` (`appuser`) write permissions. | [Deploy MicroLLM with Docker](../../02-how-to-guides/deployment/deploy-microllm-docker.md) |

## 5. Quality & Performance SLA Gates (`G1`–`G8` in `src/microllm/quality_suite.py`)

| Gate Identifier | Pass Threshold | Failure Condition | Remediation Runbook |
| :--- | :--- | :--- | :--- |
| `G1_jevbench_core_accuracy_ge_95pct` | `cascade_accuracy >= 0.95` | Core 5-track accuracy drops below `95%`. | [Run Reproducible Benchmarks](../../02-how-to-guides/operations/run-reproducible-benchmarks.md) |
| `G2_decision_pivot_pair_accuracy_ge_90pct` | `pair_accuracy >= 0.90` | Contrastive minimal-pair accuracy drops below `90%`. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| `G3_anyjev_l0_zero_order_flip` | `gear2_l0_flip_rate == 0.0` | Non-zero permutation flip drift in `AnyJevPrefillBackend`. | [Run Reproducible Benchmarks](../../02-how-to-guides/operations/run-reproducible-benchmarks.md) |
| `G4_guardrail_f1_ge_94pct` | `f1_score >= 0.94` | Safety guardrail F1 drops below `94%`. | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| `G5_calibration_ece_and_brier_improved` | `calibrated_ece < raw_ece` and `rlcd2_brier < raw_brier` | Per-$K$ temperature scaling or RLCD² fails to improve ECE or Brier. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |
| `G6_jevcompat_contract_100pct_and_zero_5xx` | `pass_rate == 1.0` and `http_5xx_count == 0` | Any `jevcompat 0.1` contract failure or HTTP `5xx` emitted. | [Run Reproducible Benchmarks](../../02-how-to-guides/operations/run-reproducible-benchmarks.md) |
| `G7_multilingual_and_xai_ge_95pct` | `pass_rate >= 0.95` | Cross-lingual (`en`, `fr`, `es`, `pt`) or token attribution pass rate `< 95%`. | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| `G8_performance_sla_p50_le_10ms_and_int8_mse` | `pass_rate == 1.0`, `p50_ms <= 10.0`, `int8_mse <= 1e-3` | CPU median latency exceeds `10.0` ms, batch SLA fails, or INT8 quantization MSE `> 0.001`. | [Remediate Abstention or Latency Spike](../../02-how-to-guides/incident-response/remediate-abstention-or-latency-spike.md) |

## Related Documentation

| Topic | Document |
| :--- | :--- |
| Configuration & Endpoints | [Environment Variables and Model Profiles](../configuration/environment-and-model-profiles.md) |
| CLI Flags & Exit Codes | [CLI Commands and Exit Codes](../cli-and-scripts/cli-commands-and-exit-codes.md) |
| System Topology | [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md) |
| Calibration & Order Invariance | [Calibration and Order Invariance](../../04-explanation/deep-dives/calibration-and-order-invariance.md) |
