# Enforce SQL Agent Guardrails on test_db

Refer to the
[Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
before you do this procedure.

## Preconditions

1. Make sure that `microllm-guard` is on your system `PATH`.
2. Make sure that you have write access to the SQL agent project root directory.
3. If `microllm serve` checks HTTP calls, make sure that port `8000` is active.
4. Make a backup copy of `.microllm-guard.json` if the file is on disk.

## Safety Instructions

**CAUTION:** Make a backup copy of `.microllm-guard.json` before you change `policy_boundaries`.
Incorrect rules can accept `DROP TABLE` queries or stop valid `SELECT` queries.

**CAUTION:** Keep sensitive `employees.salaries` tables inside `policy_boundaries.forbidden` in production.
Read access to `salaries` without boundaries gives unpermitted access to payroll records.

## Procedure

1. Write a backup copy of `.microllm-guard.json` to `/tmp/microllm-guard.json.bak` if the file is on disk:

   ```bash
   if [ -f .microllm-guard.json ]; then
     cp .microllm-guard.json /tmp/microllm-guard.json.bak
   fi
   ```

2. Write `.microllm-guard.json` with `policy_boundaries` for `employees`, `sakila`, and `sirene` in
   [test_db](https://github.com/jmrenouard/test_db):

   ```json
   {
     "environment": "production",
     "abstain_policy": "escalate",
     "policy_boundaries": {
       "allowed": [
         "SELECT emp_no, title, from_date, to_date FROM employees.titles",
         "SELECT film_id, title, rating FROM sakila.film",
         "SELECT siren, denomination, code_naf FROM unites_legales"
       ],
       "forbidden": [
         "DROP TABLE",
         "TRUNCATE",
         "DELETE FROM employees",
         "DELETE FROM unites_legales",
         "salaries"
       ]
     }
   }
   ```

3. Start `microllm-guard` in CLI mode before your SQL agent sends a query:

   ```bash
   microllm-guard \
     --tool sql \
     --command "SELECT film_id, title, rating FROM sakila.film LIMIT 10;" \
     --policy-file .microllm-guard.json \
     --json
   ```

4. If your agent sends HTTP requests, send `POST` calls to `/v1/guard/evaluate` with `policy_boundaries`:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/guard/evaluate \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MICROLLM_API_KEY:-local-jev-key}" \
     -d '{
       "tool_name": "sql",
       "command": "SELECT emp_no, salary FROM salaries LIMIT 5;",
       "environment": "production",
       "policy_boundaries": {
         "allowed": ["SELECT emp_no, title, from_date, to_date FROM employees.titles"],
         "forbidden": ["DROP TABLE", "salaries"]
       }
     }'
   ```

5. If your SQL agent uses an MCP server, start it behind `microllm-guard mcp-proxy`:

   ```bash
   microllm-guard mcp-proxy \
     --env production \
     --policy-file .microllm-guard.json \
     -- python -m sql_mcp_server
   ```

**NOTE:** Hard hazards stay blocked when `policy_boundaries.allowed` matches part of the SQL string.

## Verification Check

1. Evaluate a permitted read-only `employees.titles` query and make sure that exit code is `0`:

   ```bash
   microllm-guard \
     --tool sql \
     --command "SELECT emp_no, title, from_date, to_date FROM employees.titles LIMIT 10;" \
     --policy-file .microllm-guard.json \
     --json
   echo "Exit code: $?"
   ```

2. Evaluate a forbidden `DROP TABLE salaries;` query and make sure that exit code is `2`:

   ```bash
   microllm-guard \
     --tool sql \
     --command "DROP TABLE salaries;" \
     --policy-file .microllm-guard.json \
     --json
   echo "Exit code: $?"
   ```

3. Evaluate a protected `salaries` read query and make sure that exit code is `2`:

   ```bash
   microllm-guard \
     --tool sql \
     --command "SELECT emp_no, salary FROM salaries LIMIT 5;" \
     --policy-file .microllm-guard.json \
     --json
   echo "Exit code: $?"
   ```

4. Evaluate a schema mutation on Sirene `unites_legales` and make sure that exit code is `3`:

   ```bash
   microllm-guard \
     --tool sql \
     --command "ALTER TABLE unites_legales ADD COLUMN tier VARCHAR(16);" \
     --policy-file .microllm-guard.json \
     --json
   echo "Exit code: $?"
   ```

## Rollback Procedure

If `.microllm-guard.json` stops valid SQL agent queries, do these steps:

1. Stop `microllm-guard mcp-proxy` if the proxy process is active.
2. Put the backup `.microllm-guard.json` file back in the project root:

   ```bash
   if [ -f /tmp/microllm-guard.json.bak ]; then
     cp /tmp/microllm-guard.json.bak .microllm-guard.json
   else
     rm -f .microllm-guard.json
   fi
   ```

3. Evaluate a safe `SELECT` query against `sakila.film` and make sure that exit code is `0`:

   ```bash
   microllm-guard --tool sql --command "SELECT film_id, title, rating FROM sakila.film LIMIT 5;" --json
   ```

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
