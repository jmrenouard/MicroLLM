# Enrich Sirene PostgreSQL Records

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that `micro-jev` and `psycopg` are installed in your active Python environment.
2. Make sure that Docker Engine and `psql` are available on the host.
3. Make sure that ports `15432` and `8000` are free on `127.0.0.1`.
4. Clone [test_db](https://github.com/jmrenouard/test_db) into `./sandbox/test_db` if the directory is not on disk.

## Safety Instructions

**CAUTION:** Write predictions only to the isolated `microllm_sirene_predictions` table.
Direct updates to `unites_legales` can cause damage to reference Sirene rows.

**CAUTION:** Quarantine rows with `x_abstain = true` before downstream business use.
Sentinel `00.00Z` rows do not have sector data and give unreliable labels.

## Procedure

1. Pull the `multilingual` profile and show installed profiles:

   ```bash
   microllm pull multilingual
   microllm list
   ```

2. Start `microllm serve` with strategy `auto` on port `8000`:

   ```bash
   microllm serve --host 127.0.0.1 --port 8000 --strategy auto
   ```

3. Start the `test_db` Sirene PostgreSQL 18 container and inject synthetic data:

   ```bash
   make -C ./sandbox/test_db sirene-up
   make -C ./sandbox/test_db sirene-inject
   ```

4. Make the isolated `microllm_sirene_predictions` table on `localhost:15432`:

   ```bash
   psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
     CREATE TABLE IF NOT EXISTS microllm_sirene_predictions (
         siren VARCHAR(9) PRIMARY KEY REFERENCES unites_legales(siren) ON DELETE CASCADE,
         code_naf VARCHAR(6) NOT NULL,
         predicted_vertical VARCHAR(64) NOT NULL,
         vertical_confidence DOUBLE PRECISION NOT NULL,
         b2b_eligible_prob DOUBLE PRECISION NOT NULL,
         lead_tier DOUBLE PRECISION NOT NULL,
         x_abstain BOOLEAN NOT NULL,
         quarantined BOOLEAN NOT NULL DEFAULT FALSE,
         evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
     );
   "
   ```

5. Send one test Sirene record to `POST /v1/systemone` with `curl`:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/systemone \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MICROLLM_API_KEY:-local-jev-key}" \
     -d '{
       "model": "jev-latest",
       "multilingual_strategy": "fast",
       "state": "Programmation informatique - Information et communication - 20 a 49 salaries",
       "questions": {
         "vertical": {
           "type": "choice",
           "instructions": "Classify French business vertical.",
           "criteria": {
             "software": "Programmation informatique, information et communication, logiciels, conseil informatique",
             "construction": "Construction de batiments, travaux installation electrique",
             "commerce": "Commerce de detail, supermarches, vente a distance",
             "public_admin": "Administration publique generale et services publics"
           }
         }
       }
     }'
   ```

6. Start a Python script to get rows from `unites_legales`, `ref_naf`, and `ref_tranches_effectifs`:

   ```python
   import os
   import httpx
   import psycopg
   from microllm import SplitConformalCalibrator

   dsn = "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene"
   api_key = os.environ.get("MICROLLM_API_KEY", "local-jev-key")
   calibrator = SplitConformalCalibrator(alpha=0.10)
   questions = {
       "vertical": {
           "type": "choice",
           "instructions": "Classify French business vertical.",
           "criteria": {
               "software": "Programmation informatique, information et communication, logiciels, conseil informatique",
               "construction": "Construction de batiments, travaux installation electrique",
               "commerce": "Commerce de detail, supermarches, vente a distance",
               "public_admin": "Administration publique generale et services publics",
           },
       },
       "eligible": {
           "type": "noul",
           "instructions": "Is this company an active commercial enterprise?",
           "criteria": {
               "true": "Active company with known NAF economic activity",
               "false": "Ceased unit or unknown 00.00Z sentinel activity",
           },
       },
       "priority": {
           "type": "score",
           "instructions": "Rate B2B commercial priority.",
           "criteria": [
               "Low priority micro unit",
               "Standard SME prospect",
               "High priority mid-market lead",
           ],
       },
   }

   query_sql = """
       SELECT
           ul.siren,
           ul.code_naf,
           COALESCE(n.libelle_sous_classe, 'Activite inconnue') AS libelle_sous_classe,
           COALESCE(n.libelle_section, 'unknown') AS libelle_section,
           COALESCE(te.libelle, 'unknown') AS tranche_libelle
       FROM unites_legales ul
       LEFT JOIN ref_naf n ON n.code_naf = ul.code_naf
       LEFT JOIN ref_tranches_effectifs te ON te.code_tranche = ul.code_tranche_effectifs
       ORDER BY ul.siren
       LIMIT 250;
   """

   with psycopg.connect(dsn) as conn, httpx.Client(
       base_url="http://127.0.0.1:8000",
       headers={"Authorization": f"Bearer {api_key}"},
       timeout=30.0,
   ) as http:
       rows = conn.execute(query_sql).fetchall()
       for siren, code_naf, sous_classe, section, tranche in rows:
           state_text = f"{sous_classe} - {section} - {tranche}"
           resp = http.post(
               "/v1/systemone",
               json={
                   "model": "jev-latest",
                   "state": state_text,
                   "multilingual_strategy": "fast",
                   "questions": questions,
               },
           )
           resp.raise_for_status()
           answers = resp.json()["answers"]
           vert = answers["vertical"]
           conformal = calibrator.predict_set(vert["probabilities"])
           is_abstain = bool(conformal["abstain"])
           conn.execute(
               """
               INSERT INTO microllm_sirene_predictions (
                   siren, code_naf, predicted_vertical, vertical_confidence,
                   b2b_eligible_prob, lead_tier, x_abstain, quarantined
               )
               VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
               ON CONFLICT (siren) DO UPDATE SET
                   predicted_vertical = EXCLUDED.predicted_vertical,
                   vertical_confidence = EXCLUDED.vertical_confidence,
                   b2b_eligible_prob = EXCLUDED.b2b_eligible_prob,
                   lead_tier = EXCLUDED.lead_tier,
                   x_abstain = EXCLUDED.x_abstain,
                   evaluated_at = NOW();
               """,
               (
                   siren,
                   code_naf,
                   vert["choice"],
                   float(vert["confidence"]),
                   float(answers["eligible"]["noul"]),
                   float(answers["priority"]["score"]),
                   is_abstain,
               ),
           )
       conn.commit()
   ```

7. Quarantine all `x_abstain = true` and `00.00Z` rows in `microllm_sirene_predictions`:

   ```bash
   psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
     UPDATE microllm_sirene_predictions
     SET quarantined = TRUE
     WHERE x_abstain = TRUE OR code_naf = '00.00Z';
   "
   ```

**NOTE:** `SplitConformalCalibrator` sets `abstain` to `true` when more than one candidate class is above threshold.

## Verification Check

1. Compare row counts between the `unites_legales` source slice and `microllm_sirene_predictions`:

   ```bash
   psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
     SELECT
         (SELECT COUNT(*) FROM (SELECT siren FROM unites_legales ORDER BY siren LIMIT 250) src) AS source_rows,
         (SELECT COUNT(*) FROM microllm_sirene_predictions) AS prediction_rows;
   "
   ```

2. Query quarantined rows and make sure that `00.00Z` rows show `x_abstain = t` and `quarantined = t`:

   ```bash
   psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
     SELECT siren, code_naf, predicted_vertical, vertical_confidence, x_abstain, quarantined
     FROM microllm_sirene_predictions
     WHERE x_abstain = TRUE
     ORDER BY siren
     LIMIT 20;
   "
   ```

3. Query `/v1/metrics` on port `8000` and record `systemone_requests` and `abstained_decisions`.

## Rollback Procedure

If row parity is not equal or prediction results show drift, do these steps:

1. Drop `microllm_sirene_predictions` from Sirene PostgreSQL 18 on `localhost:15432`:

   ```bash
   psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
     DROP TABLE IF EXISTS microllm_sirene_predictions;
   "
   ```

2. Make sure that `\dt microllm_sirene_predictions` shows zero rows.
3. Stop the ephemeral Sirene PostgreSQL container when maintenance ends:

   ```bash
   make -C ./sandbox/test_db sirene-down
   ```

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
- Explanation: [Calibration and Order Invariance](../../04-explanation/deep-dives/calibration-and-order-invariance.md)
