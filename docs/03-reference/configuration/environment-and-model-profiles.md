# Reference: Environment Variables, Model Profiles, Ports, and HTTP Endpoints

Normative configuration matrix for **MicroLLM** (`micro-jev` `0.4.1`) across runtime, profiles, and HTTP routes.

## 1. Environment Variables

| Variable | Type | Default | Allowed Values / Format | Consumer Module(s) | Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HOST` | string | `127.0.0.1` (`0.0.0.0` in Docker) | Valid IPv4/IPv6 or hostname | `cli.py`, `Dockerfile` | Bind address for `microllm serve`. |
| `PORT` | integer | `8000` (`7860` in HF Space fallback) | `1`–`65535` | `cli.py`, `hf_space_sync.py`, `Dockerfile` | TCP port for `microllm serve` or fallback Gradio Space port. |
| `GRADIO_SERVER_PORT` | integer | Fallback to `PORT` or `7860` | `1`–`65535` | `hf_space_sync.py` | TCP port override for Hugging Face Space `app.py`. |
| `MICROLLM_API_KEY` | string | Unset (`None`) | Non-empty secret string | `server.py`, `cli.py` | Enables constant-time `Authorization: Bearer <key>` validation. |
| `MICROLLM_MULTILINGUAL_STRATEGY` | string | `auto` | `fast`, `neural`, `auto` | `server.py`, `cli.py`, `gradio_app.py` | Default multilingual routing strategy when omitted in requests. |
| `MICROLLM_NEURAL_MODEL` | string | `minilm` | `minilm`, `mdeberta` | `server.py`, `cli.py`, `gradio_app.py` | Default ONNX neural model for `neural` and `auto` strategies. |
| `MICROLLM_HF_MODEL` | string | `""` (empty) | Hugging Face model repo ID | `server.py` | Overrides `semantic_match_logit` with `create_hf_zero_shot_scorer`. |
| `MICROLLM_CACHE_DIR` | path | `~/.cache/microllm` (`/app/.cache` in Docker) | Writable directory path | `registry.py`, `server.py`, `Dockerfile` | Cache root for pulled ONNX/NPZ profiles, staging, and fallback data. |
| `MICROLLM_SLM_BASE_URL` | URL | `""` (empty) | HTTP/HTTPS base URL | `backends/prefill_llm.py` | Enables Gear 2 OpenAI-compatible (`/v1`, `:11434`, `:8000`) or `llama.cpp` slot scorer. |
| `MICROLLM_SLM_MODEL` | string | `qwen2.5:0.5b` | Model identifier string | `backends/prefill_llm.py` | Model name sent to OpenAI-compatible SLM slot scorer. |
| `HF_TOKEN` | string | Unset | Hugging Face API token | `registry.py`, `hf_publishing.py` | Primary environment token for `publish`, `sync-space`, and `sync-hf-readme`. |
| `HUGGING_FACE_HUB_TOKEN` | string | Unset | Hugging Face API token | `hf_publishing.py` | Secondary Hugging Face Hub token fallback after `HF_TOKEN`. |
| `HUGGINGFACE_TOKEN` | string | Unset | Hugging Face API token | `registry.py`, `hf_publishing.py` | Tertiary Hugging Face Hub token fallback. |
| `MICROLLM_KEEPASS_DB` | path | `~/.config/microllm/vault.kdbx` | `.kdbx` file path | `hf_publishing.py` | KeePass database path for Hub token lookup (`KEEPASS_DB`, `KEEPASS_DB_PATH` aliases). |
| `MICROLLM_KEEPASS_KEY` | path | `~/.config/microllm/vault.keyx` | `.keyx` keyfile path | `hf_publishing.py` | KeePass keyfile path (`KEEPASS_KEY`, `KEEPASS_KEY_PATH` aliases). |
| `OPENAI_API_KEY` | string | Unset | OpenAI-compatible Bearer token | `cli.py`, `run_guard_vs_judge_benchmark.py` | Bearer token forwarded to `--live-judge-url` during `--guard-judge`. |
| `MICROLLM_DOCS_REF` | string | `main` | Git ref (`WORKTREE`/`HEAD`/`origin/main` -> `main`) | `mkdocs_hooks.py`, `build_versioned_docs.py` | Git ref used when rewriting relative source links in MkDocs builds. |
| `MICROLLM_DOCS_SITE_URL` | URL | `""` (empty) | HTTP/HTTPS site URL | `mkdocs_hooks.py`, `build_versioned_docs.py` | Overrides `config.site_url` per built documentation version slug. |
| `GITHUB_SHA` | string | `local-dev` | Git commit SHA | `hf_space_sync.py` | Recorded in generated Space cards and `staging_manifest.json`. |
| `GITHUB_REF` | string | `refs/heads/main` | Git ref string | `hf_space_sync.py` | Recorded in generated Space `staging_manifest.json`. |
| `PYTHONUNBUFFERED` | string | `1` (Docker) | `0` or `1` | `Dockerfile` | Forces unbuffered container `stdout`/`stderr` streams. |
| `PYTHONDONTWRITEBYTECODE` | string | `1` (Docker) | `0` or `1` | `Dockerfile` | Suppresses `.pyc` bytecode writes inside container image. |

## 2. Registered ONNX and Safetensors Model Profiles

All 8 entries defined in `MODEL_PROFILES` (`src/microllm/registry.py`).

| Profile Key | Upstream Hugging Face Repo | MicroLLM HF Repo | Pipeline Tag | Format / Subpath | Size (MB) | Target Latency (ms) | Rec. Temp |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `fast-cpu-int8` | `MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary` | `jmrenouard/microllm-fast-cpu-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `58.4` | `8.5` | `0.75` |
| `guardrail` | `protectai/deberta-v3-small-prompt-injection-v2` | `jmrenouard/microllm-guardrail-int8` | `text-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `64.2` | `5.0` | `0.70` |
| `multilingual` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `jmrenouard/microllm-multilingual-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `134.0` | `18.0` | `0.82` |
| `multilingual-minilm-int8` | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | `jmrenouard/microllm-multilingual-minilm-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `68.5` | `7.5` | `0.80` |
| `multilingual-mdeberta-int8` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `jmrenouard/microllm-multilingual-mdeberta-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `134.0` | `18.0` | `0.82` |
| `rlcd2-decision-head` | `jmrenouard/microllm-rlcd2-decision-head` | `jmrenouard/microllm-rlcd2-decision-head` | `tabular-classification` | `RLCD2_SAFETENSORS` (`model.safetensors`) | `12.0` | `0.15` | `0.78` |
| `deberta-large-int8` | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | `jmrenouard/microllm-deberta-large-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `435.0` | `32.0` | `0.85` |
| `bert-large-int8` | `yoshitomo-matsubara/bert-large-uncased-mnli` | `jmrenouard/microllm-bert-large-int8` | `zero-shot-classification` | `INT8_DYNAMIC` (`onnx/model_quantized.onnx`) | `340.0` | `26.0` | `0.85` |

### Profile Notes and `entailment_idx` Mapping

- Profiles `multilingual`, `multilingual-minilm-int8`, `multilingual-mdeberta-int8`, and `deberta-large-int8`
  set `entailment_idx = 0`.
- Profile `bert-large-int8` specifies `entailment_idx = 2` with tokenizer `bert-large-uncased`.
- `pull_profile` writes `model_quantized_int8.npz` (`256x64` INT8 weights, scale, `[0.75, 0.80, 0.85, 0.88, 0.95]`
  temperatures) and `profile_manifest.json` (`sha256_checksum`).
- `load_profile_scorer` defaults: `profile_name = "fast-cpu-int8"`, `margin_threshold = 0.65`, `revision = "main"`.

### Profile Aliases (`PROFILE_ALIASES`)

| Canonical Profile | Accepted Aliases (Case-Insensitive, Optional `jmrenouard/` Prefix) |
| :--- | :--- |
| `fast-cpu-int8` | `microllm-fast-cpu-int8`, `microllm-gear1-fast` |
| `guardrail` | `guardrail-int8`, `microllm-guardrail-int8`, `jmrenouard/microllm-guardrail-int8` |
| `multilingual` | `multilingual-int8`, `microllm-multilingual-int8` |
| `multilingual-minilm-int8` | `minilm`, `minilm-int8`, `multilingual-minilm`, `microllm-multilingual-minilm-int8`, `jmrenouard/microllm-multilingual-minilm-int8` |
| `multilingual-mdeberta-int8` | `mdeberta`, `mdeberta-int8`, `multilingual-mdeberta`, `microllm-multilingual-mdeberta-int8`, `jmrenouard/microllm-multilingual-mdeberta-int8` |
| `rlcd2-decision-head` | `microllm-rlcd2-decision-head` |
| `deberta-large-int8` | `deberta-large`, `deberta-v3-large`, `deberta-v3-large-int8`, `microllm-deberta-large-int8`, `jmrenouard/microllm-deberta-large-int8` |
| `bert-large-int8` | `bert-large`, `bert-large-uncased`, `bert-large-mnli`, `bert-large-nli`, `bert-large-classifier`, `microllm-bert-large-*`, `jmrenouard/microllm-bert-large-*` |

## 3. Engine, Calibration, Schema, and Guardrail Defaults

| Component / Schema | Parameter | Type | Default | Unit / Range | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `FastEncoderBackend` | `default_temperature` | float | `0.85` | scalar (`> 0`) | `backends/fast_encoder.py` |
| `FastEncoderBackend` | `temperatures_by_k` | map | `{2: 0.75, 3: 0.80, 4: 0.85, 5: 0.88, 10: 0.95}` | per-$K$ temperature | `backends/fast_encoder.py` |
| `FastEncoderBackend` | `conformal_alpha` | float | `0.10` | `(0, 1)` (`90%` coverage) | `backends/fast_encoder.py` |
| `FastEncoderBackend` | `max_tokens` | integer | `512` | tokens (context guard) | `backends/fast_encoder.py` |
| `DualGearCascadeBackend` | `gate_threshold` | float | `0.75` | `[0.0, 1.0]` | `backends/cascade.py` |
| `DualGearCascadeBackend` | `alpha` | float | `0.75` | `[0.0, 1.0]` (`p = 0.25 p0 + 0.75 p1`) | `backends/cascade.py` |
| `MultilingualCascadeBackend` | `margin_threshold` | float | `0.12` | `[0.0, 1.0]` | `server.py` |
| `MultilingualCascadeBackend` | `entropy_threshold` | float | `0.88` | `[0.0, 1.0]` | `server.py` |
| `MultilingualCascadeBackend` | `fusion_alpha` | float | `0.75` | `[0.0, 1.0]` | `server.py` |
| `VerdictMQMultiQuestionCache` | `max_entries` | integer | `1024` | cached encoded states | `server.py` |
| `SystemOneRequest` | `model` | string | `jev-latest` | see `/v1/models` + aliases | `schemas.py` |
| `ChoiceQuestion` / `MultiQuestion` | `criteria` cardinality | integer | - | `2`–`255` options (`>25` triggers tournament) | `schemas.py` |
| `MultiQuestion` | `threshold` | float | `0.5` | `[0.0, 1.0]` | `schemas.py` |
| `ScoreQuestion` | `criteria` cardinality | integer | - | `2`–`10` ordered levels | `schemas.py` |
| `BatchRequest` | `items` max length | integer | - | `<= 500` states (`model`: `microllm-gear1-fast`) | `server_models.py` |
| `CalibrateRequest` | `alpha` / `target_risk` / `delta` | float | `0.10` / `0.10` / `0.05` | `(0.0, 1.0)`; `<= 10,000` samples; `K <= 256` | `server_models.py` |
| `StudioRLCDDemoRequest` | `stage1_steps` / `stage2_steps` | integer | `35` / `25` | `[5, 100]`; `<= 10,000` samples; `d_feat <= 1,024` | `server_models.py` |
| `GuardrailRequest` | `threshold` | float | `0.55` | `[0.1, 0.99]` (hard block at `>= 0.75`) | `server_models.py`, `server.py` |
| `GuardConfig` | `environment` | string | `production` | `production` (`0.50`), `staging` (`0.60`), `development` (`0.70`) | `guard.py` |
| `GuardConfig` | `block_threshold` | float | `0.75` | `[0.5, 0.99]` | `guard.py` |
| `GuardConfig` | `abstain_policy` | string | `escalate` | `escalate`, `block` | `guard.py` |
| `GuardConfig` | `conformal_alpha` / `conformal_q_hat` | float | `0.10` / `0.72` | `(0, 1)` / `[0.5, 1.0)` | `guard.py` |
| `telemetry_state` | `latencies_ms` window | integer | `10000` | rolling samples (`deque(maxlen=10000)`) | `server.py` |

## 4. Network Ports, Transports, and Served Model Identifiers

| Interface | Protocol / Transport | Default Port | Exposed By | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| System-One FastAPI Server | HTTP/1.1 (`TCP`) | `8000` | `microllm serve`, `Dockerfile` (`EXPOSE 8000`) | REST inference, guardrails, telemetry, Decision Studio HTML (`/`). |
| Gradio Decision Studio UI | HTTP/1.1 (`TCP`) | `7860` | `microllm studio`, HF Space `app.py` | Interactive 7-tab Gradio UI. |
| Native MCP Server | JSON-RPC 2.0 (`stdio`) | `N/A` | `microllm mcp` | Exposes 5 MCP tools (`systemone_*`). |
| MCP Guard Stdio Proxy | JSON-RPC 2.0 (`stdio`) | `N/A` | `microllm-guard mcp-proxy` | Screens upstream MCP `tools/call` frames (`shell=False`). |
| Claude Code Hook | JSON (`stdin`/`stdout`) | `N/A` | `microllm-guard claude-hook` | Evaluates `PreToolUse` payloads locally. |

### `/v1/models` Identifiers

| Model ID | Type | Description |
| :--- | :--- | :--- |
| `jev-latest` | Canonical | Dual-Gear Hybrid Cascade (default for `/v1/systemone`). |
| `microllm-gear1-fast` | Canonical | Gear 1 single-pass encoder (default for `/v1/batch`). |
| `microllm-gear2-anyjev-l0` | Canonical | Gear 2 AnyJev $L_0$ circular rotation prefill reader. |
| `micro-jev`, `microllm` | Alias | Equivalent to `jev-latest` on `/v1/systemone` and `/v1/batch`. |
| `minilm`, `mdeberta`, `multilingual-minilm-int8`, `multilingual-mdeberta-int8` | Neural Alias | Direct neural ONNX encoder routing on `/v1/systemone` and `/v1/batch`. |

## 5. Complete HTTP REST Endpoint Matrix (`src/microllm/server.py`)

| Method | Path | Bearer Auth | Request / Query Schema | Response / Headers / Lock Behavior |
| :--- | :--- | :---: | :--- | :--- |
| `GET` | `/healthz` | No | None | `{"status": "ok", "service": "microllm"}` (`HEALTHCHECK` target). |
| `GET` | `/v1/models` | No | None | List of served model descriptors (`jevcompat` `http.models`). |
| `POST` | `/v1/systemone` | Yes | `SystemOneRequest` | `SystemOneResponse` + `X-Strategy-Used`, `X-Cascade-Triggered`, `X-Neural-Model`. |
| `POST` | `/v1/calibrate` | Yes | `CalibrateRequest` | Temperature, raw/calibrated Brier & ECE, conformal `q_hat`, Clopper-Pearson, bins. |
| `POST` | `/v1/studio/analyze` | Yes | `StudioAnalyzeRequest` | Gear 1 / Gear 2 / Cascade outputs, flip audit, XAI, coherence, VOC diagnostics. |
| `POST` | `/v1/studio/rlcd-demo` | Yes | `StudioRLCDDemoRequest` | RLCD² training metrics, reliability bins, risk-coverage curve, Proposition 1 check. |
| `POST` | `/v1/rlcd` | Yes | `StudioRLCDDemoRequest` | Identical payload handler to `/v1/studio/rlcd-demo`. |
| `POST` | `/v1/guardrail` | Yes | `GuardrailRequest` | `safe`, `action` (`allow`/`review`/`block`), `risk_score`, `checks`, `token_attribution`. |
| `GET` | `/v1/guard/contract` | No | None | Packaged `microllm.guard.v1` Draft 2020-12 JSON Schema. |
| `POST` | `/v1/guard/evaluate` | Yes | `GuardEvaluateRequest` (`GatekeeperRequest`) | `GatekeeperDecision` (`microllm.guard.v1` JSON object). |
| `GET` | `/v1/profiles` | No | None | List of all 8 `MODEL_PROFILES` entries with `installed` and `local_dir`. |
| `POST` | `/v1/profiles/pull` | Yes | `ProfilePullRequest` | Pulls/quantizes profile; returns `404` on unknown profile. |
| `POST` | `/v1/mcp` | Yes | JSON-RPC 2.0 object or batch array | Dispatches MCP JSON-RPC 2.0 tools over HTTP. |
| `POST` | `/v1/batch` | Yes | `BatchRequest` (`<= 500` items) | Summary latencies (`avg`, `p50`, `p95`), cache stats, and per-item answers. |
| `GET` | `/v1/metrics` | Yes | None | Request counters, block/abstain totals, rolling `p50`/`p95` ms, `verdict_mq_cache`. |
| `GET` | `/v1/benchmark` | Yes | None | Cached `benchmarks/latest_results.json` or executes 540-item suite. |
| `POST` | `/v1/benchmark` | Yes | Optional `{"save_artifacts": bool}` | Runs 540-item suite under `_benchmark_lock` (`409` if busy). |
| `POST` | `/v1/benchmark/run` | Yes | Optional `{"save_artifacts": bool}` | Route alias for `POST /v1/benchmark`. |
| `GET` | `/v1/benchmark/external` | Yes | Query `full_splits: bool = false` | Cached or computed 16-dataset external benchmark report. |
| `POST` | `/v1/benchmark/external` | Yes | `save_artifacts`, `full_splits`, `entries_per_dataset`, `datasets`, `fetch_live` | Runs 16-dataset external suite under `_benchmark_lock` (`409` if busy). |
| `GET` | `/v1/benchmark/hf-jev-models` | Yes | None | Cached or computed 12-dataset Hugging Face JEV models report. |
| `POST` | `/v1/benchmark/hf-jev-models` | Yes | `save_artifacts`, `entries_per_dataset`, `datasets`, `fetch_live` | Runs 12-dataset HF JEV models suite under `_benchmark_lock` (`409` if busy). |
| `GET` | `/v1/jev-models` | No | Query `q`, `primitive`, `dataset` | Lists and filters the 25 Hugging Face JEV ecosystem models and 12 datasets. |
| `GET` | `/v1/papers` | No | Query `pillar`, `q`, `arxiv_id` | Searches 116-paper catalog; returns `404` when `arxiv_id` is unknown. |
| `GET` | `/` | No | None | Decision Studio HTML (`src/microllm/static/index.html`). |
| `GET` | `/documentation` | No | None | Static documentation portal HTML (`docs/index.html`). |
| `GET` | `/slides` | No | None | Interactive LightPath presentation deck (`docs/slides.html`). |
| `GET` | `/assets/{asset_name}` | No | Path parameter `asset_name` | Serves `.svg` asset (`image/svg+xml`); `404` otherwise. |
| `GET` | `/papers/{paper_name}` | No | Path parameter `paper_name` | Serves `.pdf` paper (`application/pdf`); `404` otherwise. |

## Related Documentation

| Topic | Document |
| :--- | :--- |
| CLI Flags & Exit Codes | [CLI Commands and Exit Codes](../cli-and-scripts/cli-commands-and-exit-codes.md) |
| Operational Error Mapping | [Error and Alert Matrix](../troubleshooting-codes/error-and-alert-matrix.md) |
| Protocol Contract | [Technical Specifications](../../TECHNICAL_SPECIFICATIONS.md) |
| Docker Deployment SOP | [Deploy MicroLLM Docker](../../02-how-to-guides/deployment/deploy-microllm-docker.md) |
| ONNX Profile Operations | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| Security & Policy Boundaries | [Rotate API Key and Policy Boundaries](../../02-how-to-guides/security/rotate-api-key-and-policy-boundaries.md) |
| Architecture Overview | [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md) |
| Design Decision (Cascade) | [ADR-001 Dual-Gear Conformal Cascade](../../04-explanation/design-decisions/adr-001-dual-gear-conformal-cascade.md) |
