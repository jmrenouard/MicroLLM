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
| [Batch Classify Database Tables](operations/batch-classify-database-tables.md) | Export [jmrenouard/test_db](https://github.com/jmrenouard/test_db) `sakila.film` and `employees.titles` rows, classify via `POST /v1/batch`, and verify parity. |
| [Enrich Sirene PostgreSQL Records](operations/enrich-sirene-postgresql-records.md) | Enrich [jmrenouard/test_db](https://github.com/jmrenouard/test_db) Sirene `unites_legales` rows via `POST /v1/systemone` and quarantine `x_abstain` rows. |

## Incident Response

| Guide | Purpose |
| :--- | :--- |
| [Remediate Abstention or Latency Spike](incident-response/remediate-abstention-or-latency-spike.md) | Isolate and correct high `x_abstain` rates or CPU `p95` latency spikes on `/v1/systemone`. |

## Security

| Guide | Purpose |
| :--- | :--- |
| [Rotate API Key and Policy Boundaries](security/rotate-api-key-and-policy-boundaries.md) | Replace `MICROLLM_API_KEY` and change `.microllm-guard.json` rules without downtime. |
| [Enforce SQL Agent Guardrails on test_db](security/enforce-sql-agent-guardrails-on-test-db.md) | Configure `.microllm-guard.json` boundaries for [jmrenouard/test_db](https://github.com/jmrenouard/test_db) and enforce exit codes `0`, `2`, and `3`. |

## Other Documentation Quadrants

- [Documentation Portal](../index.md)
- [Tutorials](../01-tutorials/index.md)
- [Reference Specifications](../03-reference/index.md)
- [Architecture Explanation](../04-explanation/index.md)
