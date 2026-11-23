# Batch Classify Database Tables

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that package `micro-jev` is installed in your active Python environment.
2. Make sure that `microllm serve` is active on port `8000`.
3. Make sure that you can read `sakila.film` and `employees.titles` in
   [test_db](https://github.com/jmrenouard/test_db).
4. Make a temporary work folder at `/tmp/microllm-batch` before you export table rows.

## Safety Instructions

**CAUTION:** Write batch output only to temporary files in `/tmp/microllm-batch` before parity checks.
Direct writes to source database tables can cause damage to valid records.

**CAUTION:** Send a maximum of `500` items in each `POST /v1/batch` request.
Requests with more than `500` items cause HTTP status `422`.

## Procedure

1. Export `sakila.film` rows from `test_db` to JSONL in `/tmp/microllm-batch/sakila_film_export.jsonl`:

   ```bash
   mkdir -p /tmp/microllm-batch
   mariadb -h 127.0.0.1 -u root -N -B sakila -e "
     SELECT JSON_OBJECT(
       'film_id', film_id,
       'title', title,
       'description', description,
       'release_year', release_year,
       'rating', rating,
       'special_features', special_features
     )
     FROM sakila.film
     ORDER BY film_id
     LIMIT 100;
   " > /tmp/microllm-batch/sakila_film_export.jsonl
   ```

2. If you classify `employees.titles` rows from `test_db`, export CSV to
   `/tmp/microllm-batch/employees_titles_export.csv`:

   ```bash
   mariadb -h 127.0.0.1 -u root -B employees -e "
     SELECT emp_no, title, from_date, to_date
     FROM employees.titles
     ORDER BY emp_no, from_date
     LIMIT 100;
   " | tr '\t' ',' > /tmp/microllm-batch/employees_titles_export.csv
   ```

3. If no SQL database server is active, write sample `sakila.film` rows to
   `/tmp/microllm-batch/sakila_film_export.jsonl`:

   ```bash
   mkdir -p /tmp/microllm-batch
   cat <<'EOF' > /tmp/microllm-batch/sakila_film_export.jsonl
   {"film_id": 1, "title": "ACADEMY DINOSAUR", "description": "A Epic Drama of a Feminist And a Mad Scientist who must Battle a Teacher in The Canadian Rockies", "release_year": 2006, "rating": "PG", "special_features": "Deleted Scenes,Behind the Scenes"}
   {"film_id": 2, "title": "ACE GOLDFINGER", "description": "A Astounding Epistle of a Database Administrator And a Explorer who must Find a Car in Ancient China", "release_year": 2006, "rating": "G", "special_features": "Trailers,Deleted Scenes"}
   {"film_id": 3, "title": "ADAPTATION HOLES", "description": "A Astounding Reflection of a Lumberjack And a Car who must Sink a Lumberjack in A Baloon Factory", "release_year": 2006, "rating": "NC-17", "special_features": "Trailers,Deleted Scenes"}
   EOF
   ```

4. Send one test batch directly to `POST /v1/batch` with `curl` to examine the response schema:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/batch \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MICROLLM_API_KEY:-local-jev-key}" \
     -d '{
       "model": "microllm-gear1-fast",
       "items": [
         {"film_id": 1, "title": "ACADEMY DINOSAUR", "description": "A Epic Drama of a Feminist And a Mad Scientist who must Battle a Teacher in The Canadian Rockies", "rating": "PG"}
       ],
       "questions": {
         "genre": {
           "type": "choice",
           "instructions": "Classify film genre.",
           "criteria": {
             "Drama": "Drama, emotional conflict, feminist, teacher",
             "Action": "Action, battle, explorer, car chase",
             "Documentary": "Documentary, history, reflection, factual"
           }
         }
       }
     }'
   ```

5. Start a Python script with `MicroJevClient.batch` to classify `sakila.film` rows and write temporary output:

   ```python
   import json
   from pathlib import Path
   from microllm import Choice, MicroJevClient, Score

   export_path = Path("/tmp/microllm-batch/sakila_film_export.jsonl")
   output_path = Path("/tmp/microllm-batch/sakila_film_predictions.jsonl")

   rows = [
       json.loads(line)
       for line in export_path.read_text(encoding="utf-8").splitlines()
       if line.strip()
   ]
   questions = {
       "genre": Choice(
           {
               "Drama": "Drama, emotional conflict, feminist, teacher",
               "Action": "Action, battle, explorer, car chase",
               "Documentary": "Documentary, history, reflection, factual",
           },
           instructions="Classify film genre.",
       ),
       "intensity": Score(
           ["Family safe", "Mild drama", "Mature theme"],
           instructions="Rate audience intensity.",
       ),
   }

   client = MicroJevClient(base_url="http://127.0.0.1:8000")
   output_rows = []
   for offset in range(0, len(rows), 500):
       chunk = rows[offset : offset + 500]
       response = client.batch(
           items=chunk,
           questions=questions,
           model="microllm-gear1-fast",
       )
       for item in response["results"]:
           source_row = chunk[item["index"]]
           output_rows.append(
               {
                   "film_id": source_row["film_id"],
                   "title": source_row["title"],
                   "answers": item["answers"],
                   "x_abstain": bool(item["abstain"]),
                   "latency_ms": item["latency_ms"],
               }
           )

   output_path.write_text(
       "\n".join(json.dumps(rec) for rec in output_rows) + "\n",
       encoding="utf-8",
   )
   ```

6. If you classify `employees.titles` CSV rows, parse CSV rows and send chunks with `MicroJevClient.batch`:

   ```python
   import csv
   import json
   from pathlib import Path
   from microllm import Choice, MicroJevClient

   csv_path = Path("/tmp/microllm-batch/employees_titles_export.csv")
   output_path = Path("/tmp/microllm-batch/employees_titles_predictions.jsonl")

   with csv_path.open("r", encoding="utf-8", newline="") as handle:
       rows = list(csv.DictReader(handle))

   questions = {
       "track": Choice(
           {
               "engineering": "Engineer, Senior Engineer, Assistant Engineer, technical systems",
               "operations": "Staff, Senior Staff, operations administration",
           },
           instructions="Classify job track.",
       ),
   }

   client = MicroJevClient(base_url="http://127.0.0.1:8000")
   output_rows = []
   for offset in range(0, len(rows), 500):
       chunk = rows[offset : offset + 500]
       response = client.batch(
           items=chunk,
           questions=questions,
           model="microllm-gear1-fast",
       )
       for item in response["results"]:
           source_row = chunk[item["index"]]
           output_rows.append(
               {
                   "emp_no": source_row["emp_no"],
                   "title": source_row["title"],
                   "answers": item["answers"],
                   "x_abstain": bool(item["abstain"]),
               }
           )

   output_path.write_text(
       "\n".join(json.dumps(rec) for rec in output_rows) + "\n",
       encoding="utf-8",
   )
   ```

**NOTE:** Each `POST /v1/batch` response puts `abstain` on each item and `abstained_count` in `summary`.

## Verification Check

1. Compare line counts between the source JSONL file and the output JSONL file:

   ```bash
   wc -l /tmp/microllm-batch/sakila_film_export.jsonl /tmp/microllm-batch/sakila_film_predictions.jsonl
   ```

2. Do a check on record parity and `x_abstain` flags with Python:

   ```python
   import json
   from pathlib import Path

   src = Path("/tmp/microllm-batch/sakila_film_export.jsonl").read_text(encoding="utf-8").splitlines()
   dst = Path("/tmp/microllm-batch/sakila_film_predictions.jsonl").read_text(encoding="utf-8").splitlines()
   assert len(src) == len(dst) and len(dst) > 0
   for line in dst:
       rec = json.loads(line)
       assert "answers" in rec and isinstance(rec["x_abstain"], bool)
   print(f"Parity OK: {len(dst)} records verified.")
   ```

3. Make sure that the output shows `Parity OK` and equal line counts.

## Rollback Procedure

If record counts are not equal or predictions show errors, do these steps:

1. Delete all temporary export and prediction files in `/tmp/microllm-batch`:

   ```bash
   rm -rf /tmp/microllm-batch
   ```

2. Make sure that `/tmp/microllm-batch` is not on disk:

   ```bash
   test ! -d /tmp/microllm-batch && echo "Rollback complete"
   ```

3. Correct the export query or question criteria before you start a new batch test.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
