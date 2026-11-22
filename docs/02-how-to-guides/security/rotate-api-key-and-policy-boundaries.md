# Rotate API Key and Policy Boundaries

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that you have access to the secret store that holds `MICROLLM_API_KEY`.
2. Make sure that you can restart `microllm serve` or replace the container on port `8000`.
3. Make a backup copy of `.microllm-guard.json` before you change a rule.

## Safety Instructions

**CAUTION:** Keep the previous `MICROLLM_API_KEY` in a safe vault before rotation.
Immediate removal without client updates causes HTTP `401` errors across active clients.

**CAUTION:** Do not write plaintext API keys in tracked files or `.env` templates committed to Git.
Key leaks give unpermitted access to protected endpoints.

## Procedure

1. Make a random 32-byte token in your secret manager for the new `MICROLLM_API_KEY`.
2. Keep the new token in your runtime secret environment:

   ```bash
   export MICROLLM_API_KEY="replace-with-new-32-byte-token"
   ```

3. Restart `microllm serve` to load the new `MICROLLM_API_KEY` value in `create_app()`:

   ```bash
   microllm serve --host 0.0.0.0 --port 8000
   ```

4. Update all client orchestrators to send the new token in the `Authorization: Bearer` header.
5. Change `.microllm-guard.json` to update `environment`, `abstain_policy`, and `policy_boundaries`:

   ```json
   {
     "environment": "production",
     "abstain_policy": "escalate",
     "policy_boundaries": {
       "allowed": [
         "git status",
         "pytest -q"
       ],
       "forbidden": [
         "rm -rf /",
         "DROP TABLE",
         "/etc/shadow"
       ]
     }
   }
   ```

6. Evaluate a forbidden command locally against `.microllm-guard.json` before deployment:

   ```bash
   microllm-guard --tool bash --command "DROP TABLE users" --policy-file .microllm-guard.json --json
   ```

**NOTE:** Hard hazards stay blocked when `policy_boundaries.allowed` matches part of the input.

## Verification Check

1. Send a request to `/v1/systemone` with the expired key and make sure that HTTP status is `401`:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/systemone \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer old-key" \
     -d '{"state":"ok","questions":{"q1":{"type":"noul","instructions":"Safe?"}}}'
   ```

2. Send a request to `/v1/systemone` with the new key and make sure that HTTP status is `200`:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/v1/systemone \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer replace-with-new-32-byte-token" \
     -d '{"state":"ok","questions":{"q1":{"type":"noul","instructions":"Safe?"}}}'
   ```

3. Evaluate an allowed command with `microllm-guard` and make sure that exit code is `0`.
4. Evaluate a forbidden command with `microllm-guard` and make sure that exit code is `2`.

## Rollback Procedure

If permitted clients receive HTTP `401` errors or `microllm-guard` rejects safe tools, do these steps:

1. Set `MICROLLM_API_KEY` back to the previous key from your secret vault.
2. Replace `.microllm-guard.json` with the backup copy.
3. Restart `microllm serve` and make sure that `/v1/systemone` outputs HTTP `200` with the previous key.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
