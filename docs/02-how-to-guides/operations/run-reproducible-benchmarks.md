# Run Reproducible Benchmarks

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that package `micro-jev` is installed in your active Python environment.
2. Stop heavy CPU processes before you start latency measurements.
3. Make sure that your terminal is in the MicroLLM repository root directory.

## Safety Instructions

**CAUTION:** Set `--output` to a temporary path before you start a benchmark on a tracked workspace.
Default commands replace JSON files in `benchmarks/`.

## Procedure

1. Start the fast 12-suite offline benchmark (`--skip-full-splits`) and write output to a temporary file:

   ```bash
   microllm benchmark --all-suites --skip-full-splits --output /tmp/microllm_bench_fast.json
   ```

2. If you want the full 13-suite check with 114,018 external rows, do not include `--skip-full-splits`:

   ```bash
   microllm benchmark --all-suites --output /tmp/microllm_bench_full.json
   ```

3. If you want to compare `microllm-guard` against System-2 LLM-Judge baselines, start `--guard-judge`:

   ```bash
   microllm benchmark --guard-judge --output /tmp/guard_judge.json --json
   ```

4. If you want to test a live OpenAI-compatible judge endpoint, set `OPENAI_API_KEY` and add `--live-judge-url`:

   ```bash
   microllm benchmark --guard-judge \
     --live-judge-url http://127.0.0.1:11434 \
     --live-judge-model llama3.1:8b \
     --output /tmp/guard_live_judge.json
   ```

5. If you want a selected subset of suites, add `--suites` with comma-separated identifiers:

   ```bash
   microllm benchmark --all-suites --suites eval_540,guard_vs_llm_judge --output /tmp/microllm_subset.json
   ```

**NOTE:** Scripts `scripts/run_all_benchmarks.ps1 --fast` and `scripts/run_all_benchmarks.sh --fast` write
standard artifacts in `benchmarks/`.

## Verification Check

1. Examine the CLI exit code and make sure that it is `0`.
2. Open your JSON file and make sure that `"all_suites_passed"` is `true`.
   For `--guard-judge`, make sure that `"all_gates_passed"` is `true`.
3. Record `p50_latency_ms` and `p95_latency_ms` for each suite in the report.

## Rollback Procedure

If a benchmark command replaced tracked files in `benchmarks/`, do these steps:

1. Put tracked files in `benchmarks/` back to the previous Git state:

   ```bash
   git checkout -- benchmarks/
   ```

2. Remove temporary JSON files made out of `benchmarks/`.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
