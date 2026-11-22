# Reference: CLI Commands, Script Flags, and Process Exit Codes

Normative CLI specification for `microllm`, `micro-jev`, `microllm-guard`, and `scripts/build_versioned_docs.py`.

## 1. Console Entry Points (`pyproject.toml`)

| Command | Target Function | Default When No Subcommand Supplied |
| :--- | :--- | :--- |
| `microllm` | `microllm.server:main` (`microllm.cli:main`) | Executes `microllm serve`. |
| `micro-jev` | `microllm.server:main` (`microllm.cli:main`) | Identical CLI entrypoint alias for `microllm`. |
| `microllm-guard` | `microllm.guard_cli:main` | Evaluates CLI flags (`--command`/`--args-json`/`--tool`) or JSON on `stdin`. |

## 2. `microllm` / `micro-jev` Subcommands (`src/microllm/cli.py`)

### 2.1 Server, Profiles, Studio, and MCP Subcommands

| Subcommand | Flag / Positional | Type / Choices | Default | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `serve` | `--host` | string | `HOST` env or `127.0.0.1` | Host interface passed to Uvicorn (`microllm.server:create_app`). |
| `serve` | `--port` | integer | `PORT` env or `8000` | TCP listening port. |
| `serve` | `--strategy`, `--multilingual-strategy` | `fast`, `neural`, `auto` | `MICROLLM_MULTILINGUAL_STRATEGY` or `auto` | Sets server default multilingual strategy. |
| `serve` | `--neural-model` | `minilm`, `mdeberta` | `MICROLLM_NEURAL_MODEL` or `minilm` | Sets default neural ONNX encoder. |
| `serve` | `--api-key` | string | `MICROLLM_API_KEY` (`None`) | Sets Bearer authentication secret. |
| `serve` | `--reload` | flag (`bool`) | `False` | Enables Uvicorn development auto-reload. |
| `pull` | `profile` (positional) | `MODEL_PROFILES` keys | Required | Target profile name (`fast-cpu-int8`, `multilingual`, `guardrail`, etc.). |
| `pull` | `--cache-dir` | path | `None` (`~/.cache/microllm`) | Overrides `MICROLLM_CACHE_DIR`. |
| `pull` | `--download` | flag (`bool`) | `False` | Fetches remote Hugging Face `config.json`. |
| `pull` | `--from-hub` | flag (`bool`) | `False` | Downloads quantized ONNX artifact via `huggingface_hub`. |
| `list` | `--cache-dir` | path | `None` (`~/.cache/microllm`) | Prints JSON array of all profiles and local install state. |
| `mcp` | *(none)* | - | - | Starts JSON-RPC 2.0 MCP server over `stdio`. |
| `guard` | `guard_args...` | remainder | - | Forwards all remaining arguments to `microllm.guard_cli:run_guard_cli`. |
| `studio`, `gradio` | `--port` | integer | `7860` | Gradio server port on `127.0.0.1`. |
| `studio`, `gradio` | `--share` | flag (`bool`) | `False` | Requests public Gradio share tunnel URL. |
| `studio`, `gradio` | `--strategy`, `--multilingual-strategy` | `fast`, `neural`, `auto` | `None` | Overrides `MICROLLM_MULTILINGUAL_STRATEGY` for Studio session. |
| `studio`, `gradio` | `--neural-model` | `minilm`, `mdeberta` | `None` | Overrides `MICROLLM_NEURAL_MODEL` for Studio session. |

### 2.2 Benchmark, Research, RCA, and Publishing Subcommands

| Subcommand | Flag | Type / Choices | Default | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `benchmark` | `--output` | path | `None` | Custom JSON output file path. |
| `benchmark` | `--all-suites`, `--all` | flag (`bool`) | `False` | Runs unified orchestrator across all benchmark suites. |
| `benchmark` | `--suites` | comma-separated IDs | `None` | Filters suite IDs when `--all-suites` is active. |
| `benchmark` | `--skip-full-splits`, `--fast`, `--quick` | flag (`bool`) | `False` | Excludes `114,018`-row `web_external_full_splits` under `--all-suites`. |
| `benchmark` | `--external`, `--web` | flag (`bool`) | `False` | Runs 16-dataset external benchmark suite. |
| `benchmark` | `--hf-jev-models` | flag (`bool`) | `False` | Runs 12-dataset Hugging Face JEV models benchmark suite. |
| `benchmark` | `--full-splits`, `--no-sampling` | flag (`bool`) | `False` | Evaluates unsampled official splits (`114,018` rows) under `--external`. |
| `benchmark` | `--entries-per-dataset`, `--scale-per-dataset` | integer | `None` | Target entry count per external dataset (e.g., `40`). |
| `benchmark` | `--fetch-live` | flag (`bool`) | `False` | Queries Hugging Face Datasets Server API for sampled external runs. |
| `benchmark` | `--multilingual` | flag (`bool`) | `False` | Runs quadrilingual (`en`, `fr`, `es`, `pt`) evaluation suite. |
| `benchmark` | `--multilingual-strategy` | `fast`, `neural`, `auto`, `all` | `all` | Strategy filter for `--multilingual`. |
| `benchmark` | `--neural-model` | `minilm`, `mdeberta`, `all` | `all` | Neural model filter for `--multilingual`. |
| `benchmark` | `--language` | `en`, `fr`, `es`, `pt`, `all` | `all` | Language filter for `--multilingual`. |
| `benchmark` | `--pareto` | flag (`bool`) | `False` | Runs comparative Pareto scaling suite (`IDEA-048`). |
| `benchmark` | `--guard-judge` | flag (`bool`) | `False` | Runs `microllm-guard` vs System-2 LLM-Judge benchmark (`220` cases). |
| `benchmark` | `--live-judge-url` | URL | `None` | Optional OpenAI-compatible endpoint URL for `--guard-judge`. |
| `benchmark` | `--live-judge-model` | string | `None` | Optional model ID paired with `--live-judge-url`. |
| `benchmark` | `--json` | flag (`bool`) | `False` | Emits JSON output to `stdout` instead of Markdown summary. |
| `papers` | `--pillar` | string | `None` | Filters 116-paper catalog by pillar ID or title substring. |
| `papers` | `--query`, `-q` | string | `None` | Searches title, authors, summary, or relevance. |
| `papers` | `--format` | `json`, `markdown` | `json` | Output format for catalog queries. |
| `papers` | `--download-pdfs` | flag (`bool`) | `False` | Downloads matching arXiv PDFs into `docs/papers/`. |
| `papers` | `--verify-pdfs` | flag (`bool`) | `False` | Verifies local PDF presence in `docs/papers/`. |
| `papers` | `--output-dir` | path | `None` | Custom PDF output directory. |
| `rca` | `--rca` | string | `None` | Targets one RCA ID (e.g., `RCA-001`). |
| `rca` | `--all` | flag (`bool`) | `False` | Runs all 15 registered RCAs sequentially (`RCA-001`–`RCA-015`). |
| `rca` | `--list` | flag (`bool`) | `False` | Lists registered RCAs and SLA targets. |
| `rca` | `--iterations` | integer | `2` | Profiling iterations per RCA. |
| `rca` | `--format` | `markdown`, `json` | `markdown` | Output format for RCA listing and report. |
| `sync-hf-readme` | `--dry-run` | flag (`bool`) | `False` | Builds all 6 Hugging Face cards locally without uploading. |
| `sync-hf-readme` | `--token` | string | `None` | Explicit HF API token (fallback: `HF_TOKEN`, `HUGGINGFACE_TOKEN`). |
| `publish` | `--dataset` / `--model` / `--all` / `--space` | flag / string | `False` / `None` | Selects dataset, model alias, Space, or combined publish (`--all` default). |
| `publish` | `--dry-run` | flag (`bool`) | `False` | Stages locally with zero network calls. |
| `publish` | `--staging-dir` / `--token` / `--commit-message` | path / string | `None` | Staging directory override, HF token, and Hub commit message. |
| `publish` | `--space-repo-id` | string | `jmrenouard/microllm-decision-studio` | Target Space repo when `--space` is set. |
| `publish` | `--format` | `json`, `text` | `json` | Output format. |
| `sync-space` | `--repo-id` | string | `jmrenouard/microllm-decision-studio` | Target Hugging Face Space repository ID. |
| `sync-space` | `--dry-run` / `--staging-dir` / `--token` / `--commit-message` / `--format` | mixed | `False` / `None` / `json` | Stages and optionally pushes Gradio Space bundle. |

## 3. `microllm-guard` CLI Modes and Flags (`src/microllm/guard_cli.py`)

### 3.1 Evaluation Mode (`microllm-guard [flags]`)

| Flag | Type / Choices | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `--tool` | string | `None` | Tool name (`bash`, `sql`, `file_write`, etc.); overrides `tool` key from `stdin`. |
| `--command` | string | `None` | Command, SQL statement, or payload text to evaluate. |
| `--args-json` | JSON string | `None` | JSON-encoded `arguments` field (strict finite JSON; rejects `NaN`/`Infinity`). |
| `--context` | string | `None` | Optional reference context for groundedness / hallucination verification. |
| `--env`, `--environment` | `production`, `staging`, `development` (`prod`, `stage`, `dev`) | `None` (`production`) | Environment threshold preset (`0.50` / `0.60` / `0.70`). |
| `--policy-file` | path | `None` | Path to JSON file defining `policy_boundaries` (`allowed`, `forbidden`). |
| `--threshold` | float | `None` | Overrides risk threshold in `[0.1, 0.99]`. |
| `--fail-closed` | flag (`bool`) | `False` | Maps conformal abstention (`abstain=true`) to `block` (exit `2`) instead of `escalate` (exit `3`). |
| `--abstain-policy` | `escalate`, `block` | `None` (`escalate`) | Action taken on conformal abstention without a hard block. |
| `--json` | flag (`bool`) | `False` | Emits single-object `microllm.guard.v1` JSON to `stdout`. |
| `--stdin` | flag (`bool`) | `False` | Forces reading base JSON payload from `stdin` even when `--command` or `--args-json` is passed. |
| `--contract` | flag (`bool`) | `False` | Prints packaged `microllm.guard.v1` JSON Schema (`gatekeeper_contract_v1.json`) and exits `0`. |

### 3.2 Guard Integration Subcommands

| Subcommand | Flag | Type / Choices | Default | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `mcp-proxy` | `--env`, `--environment` | string | `production` | Sets `GuardConfig.environment` for MCP `tools/call` screening. |
| `mcp-proxy` | `--policy-file` | path | `None` | Loads `policy_boundaries` JSON file. |
| `mcp-proxy` | `--threshold` | float | `None` | Overrides risk threshold in `[0.1, 0.99]`. |
| `mcp-proxy` | `--fail-closed` | flag (`bool`) | `False` | Forces `abstain_policy = "block"`. |
| `mcp-proxy` | `--abstain-policy` | `escalate`, `block` | `escalate` | Abstention disposition for non-hard-block calls. |
| `mcp-proxy` | `-- <upstream_cmd...>` | argv list | Required | Upstream stdio MCP server command executed with `shell=False`. |
| `claude-hook` | `--env`, `--environment` | string | `None` | Optional environment override (otherwise uses `.microllm-guard.json` or `production`). |
| `claude-hook` | `--policy-file` | path | `None` | Explicit policy JSON path (overrides `<cwd>/.microllm-guard.json`). |
| `claude-hook` | `--fail-closed` | flag (`bool`) | `False` | Maps abstentions to `permissionDecision: "deny"`. |
| `claude-hook` | `--exit-on-block` | flag (`bool`) | `False` | Exits `2` on `block` instead of `0` with `hookSpecificOutput`. |
| `init-claude-hooks` | `--print` | flag (`bool`) | `False` (default behavior) | Prints `.claude/settings.json` hook JSON to `stdout`. |
| `init-claude-hooks` | `--write` | path | `None` | Idempotently merges `PreToolUse` hook into target settings file. |
| `init-claude-hooks` | `--command` | string | `microllm-guard claude-hook` | Hook command string placed in `hooks.PreToolUse`. |
| `init-claude-hooks` | `--matcher` | regex string | <code>Bash&#124;Write&#124;Edit&#124;MultiEdit&#124;WebFetch&#124;mcp__.*</code> | Claude Code tool matcher regex. |

## 4. Documentation Build Script (`scripts/build_versioned_docs.py`)

| Flag | Type | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `--output` | path | `site_dist` | Target multi-version build output directory (rejects repo root/protected directories). |
| `--base-url` | URL | `https://microllm.lightpath.fr` | Base URL used to build `MICROLLM_DOCS_SITE_URL` per version directory. |
| `--main-ref` | Git ref | `WORKTREE` | Git ref built into `/main/` (`WORKTREE` reads current working tree; `origin/main` uses archive). |
| `--tags` | list (`nargs="*"`) | `None` | Space- or comma-separated release tags; defaults to all semver Git tags sorted descending. |
| `--skip-tags` | flag (`bool`) | `False` | Builds only `/main/` without historical Git release tags. |
| `--no-strict` | flag (`bool`) | `False` | Disables `mkdocs build --strict` link and nav validation. |
| `--repo-root` | path | Repository root (`parents[1]`) | Overrides repository root path for tag discovery and overlays. |

## 5. Deterministic Process Exit Codes (`0`, `1`, `2`, `3`)

| Exit Code | Constant / Symbol | `microllm` / `micro-jev` Meaning | `microllm-guard` Meaning | `build_versioned_docs.py` Meaning |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | `EXIT_ALLOW` (guard) | Subcommand succeeded; all SLA gates (`--all-suites`, `--guard-judge`) or RCA checks passed. | `decision == "allow"` (`abstain == False`), `--contract` output succeeded, `init-claude-hooks` succeeded, or `claude-hook` completed without `--exit-on-block`. | Multi-version MkDocs build and `assert_no_private_files` passed cleanly. |
| **`1`** | `EXIT_INVALID_INPUT` (guard) | Unhandled subcommand fallback (prints help), any suite failed SLA gates (`--all-suites` / `--guard-judge`), or any RCA non-regression check failed (`rca`). | Usage/flag `ValueError`, malformed JSON on `stdin` or `--args-json`, missing/invalid `--policy-file`, missing `mcp-proxy` upstream command, or Pydantic `ValidationError`. | Unhandled `ValueError` (unsafe `--output` / invalid Git ref), `RuntimeError` (privacy leak), or `mkdocs --strict` error. |
| **`2`** | `EXIT_BLOCK` (guard) | Invalid `--suites` selection in `benchmark --all-suites`, invalid `--live-judge-url` in `--guard-judge`, or `argparse` syntax error. | `decision == "block"` (hard safety hazard, forbidden `policy_boundaries`, or abstention with `--fail-closed` / `--abstain-policy block`); `claude-hook` empty/invalid `stdin` or `block` with `--exit-on-block`. | Standard `argparse` syntax error. |
| **`3`** | `EXIT_ESCALATE` (guard) | Not emitted by `microllm` top-level commands (unless routed through `microllm guard ...`). | `decision == "escalate"` (`abstain == True` under `abstain_policy == "escalate"`, certainty hijack, ambiguous privileged mutation, or multi-class conformal set). | Not emitted. |

## Related Documentation

| Topic | Document |
| :--- | :--- |
| Configuration & Routes | [Environment Variables and Model Profiles](../configuration/environment-and-model-profiles.md) |
| Error & Alert Resolution | [Error and Alert Matrix](../troubleshooting-codes/error-and-alert-matrix.md) |
| ONNX Profile Operations | [Pull and Verify ONNX Profiles](../../02-how-to-guides/operations/pull-and-verify-onnx-profiles.md) |
| Guard Integration Guide | [Integrate MicroLLM Guard](../../02-how-to-guides/operations/integrate-microllm-guard.md) |
| Benchmark Operations | [Run Reproducible Benchmarks](../../02-how-to-guides/operations/run-reproducible-benchmarks.md) |
| Docs Architecture Decision | [ADR-002 Diataxis Versioned MkDocs Publishing](../../04-explanation/design-decisions/adr-002-diataxis-versioned-mkdocs-publishing.md) |
