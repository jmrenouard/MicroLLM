# Remediate Abstention or Latency Spike

Refer to the
[Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
before you do this procedure.

## Preconditions

1. Make sure that you have access to the host or container that serves `microllm serve`.
2. Make sure that you have the active `MICROLLM_API_KEY` token if authentication is active.
3. Record current SLA targets for CPU `p95` latency and `abstained_decisions` rate.

## Safety Instructions

**CAUTION:** Do not increase `threshold` above `0.70` in production to mask abstention spikes.
High thresholds permit dangerous tool calls or low-confidence decisions without review.

## Procedure

1. Query `/v1/metrics` on port `8000` and record `abstained_decisions` and `latency_p95_ms`:

   ```bash
   curl -fsS http://127.0.0.1:8000/v1/metrics \
     -H "Authorization: Bearer ${MICROLLM_API_KEY}"
   ```

2. Send a sample request to `/v1/systemone` and examine headers `X-Strategy-Used` and `X-Neural-Model`:

   ```bash
   curl -fsSI -X POST http://127.0.0.1:8000/v1/systemone \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MICROLLM_API_KEY}" \
     -d '{"state":"Order delayed at customs","questions":{"q1":{"type":"choice","instructions":"Route ticket","criteria":{"shipping":"Delivery delay","billing":"Invoice charge"}}}}'
   ```

3. If `X-Neural-Model` shows `mdeberta` during a latency spike, change to `minilm`:

   ```bash
   export MICROLLM_NEURAL_MODEL=minilm
   ```

4. If `latency_p95_ms` stays above SLA, set `MICROLLM_MULTILINGUAL_STRATEGY=fast` and remove `MICROLLM_SLM_BASE_URL`:

   ```bash
   export MICROLLM_MULTILINGUAL_STRATEGY=fast
   unset MICROLLM_SLM_BASE_URL
   ```

5. Restart `microllm serve` to load the new strategy and neural model:

   ```bash
   microllm serve --host 0.0.0.0 --port 8000 --strategy fast
   ```

6. If `abstained_decisions` stays high, start all 15 RCA checks to isolate calibration drift:

   ```bash
   microllm rca --all
   ```

7. If `microllm rca --all` succeeds, calibrate temperature and `conformal_q_hat` with `/v1/calibrate`:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/calibrate \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MICROLLM_API_KEY}" \
     -d '{"logits":[[2.4,0.3],[0.2,2.1],[1.9,0.4],[0.4,2.3]],"labels":[0,1,0,1],"alpha":0.10}'
   ```

8. If `guard_escalated` spikes on safe commands, add exact command prefixes to `policy_boundaries.allowed`
   in `.microllm-guard.json`.

**NOTE:** Strategy `auto` escalates when top-two margin is `0.12` or less.
It also escalates when entropy is `0.88` or more.

## Verification Check

1. Query `/healthz` and make sure that status is `ok`:

   ```bash
   curl -fsS http://127.0.0.1:8000/healthz
   ```

2. Send 20 sample requests and query `/v1/metrics` to examine updated `latency_p95_ms` and `abstained_decisions`.
3. Make sure that `microllm rca --all` exits with code `0`.

## Rollback Procedure

If latency or accuracy decreases after correction, do these steps:

1. Set `MICROLLM_MULTILINGUAL_STRATEGY` and `MICROLLM_NEURAL_MODEL` back to the previous values.
2. Put the previous `.microllm-guard.json` policy file back if you changed `policy_boundaries`.
3. Restart `microllm serve` and query `/healthz` to make sure that the service is active.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
