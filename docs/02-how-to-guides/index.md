# How-To Guides and Operational Runbooks

This section gives goal-oriented procedures and operational runbooks for **MicroLLM** (`micro-jev`) in
**ASD-STE100 Issue 9**.

## Deployment

| Guide | Purpose |
| :--- | :--- |
| [Deploy MicroLLM with Docker](deployment/deploy-microllm-docker.md) | Make the container image and start the HTTP server on port `8000`. |

## Operations

| Guide | Purpose |
| :--- | :--- |
| [Pull and Verify ONNX Profiles](operations/pull-and-verify-onnx-profiles.md) | Pull INT8 profiles (`fast-cpu-int8`, `multilingual`, `guardrail`, `rlcd2-decision-head`) and do checksum checks. |
| [Integrate MicroLLM Guard Middleware](operations/integrate-microllm-guard.md) | Set up `microllm-guard` for CLI evaluation, MCP stdio proxy, Claude Code hooks, and LangGraph. |
| [Run Reproducible Benchmarks](operations/run-reproducible-benchmarks.md) | Start fast and full CPU benchmark suites and examine JSON output reports. |

## Incident Response

| Guide | Purpose |
| :--- | :--- |
| [Remediate Abstention or Latency Spike](incident-response/remediate-abstention-or-latency-spike.md) | Isolate and correct high `x_abstain` rates or CPU `p95` latency spikes on `/v1/systemone`. |

## Security

| Guide | Purpose |
| :--- | :--- |
| [Rotate API Key and Policy Boundaries](security/rotate-api-key-and-policy-boundaries.md) | Replace `MICROLLM_API_KEY` and change `.microllm-guard.json` rules without downtime. |

## Other Documentation Quadrants

- [Documentation Portal](../index.md)
- [Tutorials](../01-tutorials/index.md)
- [Reference Specifications](../03-reference/index.md)
- [Architecture Explanation](../04-explanation/index.md)
