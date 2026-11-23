# Tutorial 5: Sirene French Business Enrichment Lab

> **Learning Goal:** Enrich French `unites_legales` and `ref_naf` company records from `jmrenouard/test_db` using
> MicroLLM's multilingual strategies (`fast` and `auto`), combine `Choice`, `Noul`, and `Score`, and detect `00.00Z`
> sentinel rows with Split Conformal `x_abstain`.

> [!WARNING]
> **Isolated Local Sandbox Only:** Run every step inside `./sandbox/sirene-lab` against `127.0.0.1` and the
> ephemeral Sirene PostgreSQL 18 container (`localhost:15432`). Never expose lab credentials outside localhost.

---

## Prerequisites

Prepare your local workstation before starting:

- Python `3.10+` with `micro-jev` and `httpx` installed
- Free local TCP port `8000` on `127.0.0.1`
- **Optional (Option A):** Docker Compose, `make`, and `psycopg` (or `psql`) to run the live
  [`jmrenouard/test_db`](https://github.com/jmrenouard/test_db) Sirene PostgreSQL 18 3NF stack on port `15432`.
  Every reader can run **Option B** immediately on CPU with zero Docker containers.

---

## What You Will Build

By the end of this lesson, you will have:

1. Extracted French company rows from `unites_legales`, `etablissements`, and `ref_naf` (via live Docker PostgreSQL 18
   or the offline JSON fixture).
2. Started `microllm serve` with `--strategy auto`.
3. Evaluated each legal unit across `multilingual_strategy="fast"` and `"auto"` using `Choice`, `Noul`, and `Score`.
4. Flagged `00.00Z` / closed (`'C'`) sentinel rows via `x_abstain = True` and torn down the Sirene stack cleanly.

---

## Step 1: Load Sirene Records (Live Docker or Zero-Docker Fixture)

Create an isolated local workspace:

=== "bash"

    ```bash
    mkdir -p ./sandbox/sirene-lab
    cd ./sandbox/sirene-lab
    ```

=== "PowerShell"

    ```powershell
    New-Item -ItemType Directory -Force -Path ./sandbox/sirene-lab | Out-Null
    Set-Location ./sandbox/sirene-lab
    ```

### Option A: Live Docker Sirene PostgreSQL 18 3NF Stack

Clone `jmrenouard/test_db` inside `./sandbox/test_db` and launch the synthetic Sirene dataset (`15,000` legal units
and `30,000` establishments):

```bash
git clone --depth 1 https://github.com/jmrenouard/test_db.git ../test_db
make -C ../test_db sirene-up
make -C ../test_db sirene-inject
make -C ../test_db sirene-verify
```

Query active legal units alongside one sentinel `00.00Z` row using the ephemeral connection URI
`postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene`:

```bash
psql "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene" -c "
SELECT
    ul.siren,
    ul.etat_administratif,
    ul.denomination,
    ul.categorie_entreprise,
    ul.code_cj,
    ul.code_naf,
    ul.code_tranche_effectifs,
    ul.economie_sociale_solidaire,
    ul.societe_mission,
    n.libelle_section,
    n.libelle_sous_classe,
    COUNT(e.siret) AS establishment_count
FROM unites_legales ul
JOIN ref_naf n ON n.code_naf = ul.code_naf
LEFT JOIN etablissements e ON e.siren = ul.siren
WHERE ul.siren IN ('900000001', '900000002', '900000003', '900014991')
GROUP BY
    ul.siren, ul.etat_administratif, ul.denomination, ul.categorie_entreprise,
    ul.code_cj, ul.code_naf, ul.code_tranche_effectifs,
    ul.economie_sociale_solidaire, ul.societe_mission,
    n.libelle_section, n.libelle_sous_classe
ORDER BY ul.siren;"
```

### Option B: Immediate Zero-Docker Offline JSON Fixture

If Docker is not running, save these four verbatim synthetic rows from `sirene/data/sql/02_load_unites_legales.sql` and
`sirene/data/sql/01_load_referentials.sql` into `./sandbox/sirene-lab/sirene_sample.json`:

```json
[
  {
    "siren": "900000001",
    "etat_administratif": "A",
    "denomination": "ACME TECHNOLOGIE",
    "categorie_entreprise": "PME",
    "code_cj": "5710",
    "code_naf": "62.01Z",
    "code_tranche_effectifs": "21",
    "economie_sociale_solidaire": "N",
    "societe_mission": "O",
    "libelle_section": "Information et communication",
    "libelle_sous_classe": "Programmation informatique"
  },
  {
    "siren": "900000002",
    "etat_administratif": "A",
    "denomination": "BOULANGERIE DUPONT",
    "categorie_entreprise": "PME",
    "code_cj": "5499",
    "code_naf": "10.71C",
    "code_tranche_effectifs": "02",
    "economie_sociale_solidaire": "N",
    "societe_mission": "N",
    "libelle_section": "Industrie manufacturière",
    "libelle_sous_classe": "Boulangerie et boulangerie-pâtisserie"
  },
  {
    "siren": "900000003",
    "etat_administratif": "A",
    "denomination": "AURA DATA SERVICES",
    "categorie_entreprise": "PME",
    "code_cj": "5710",
    "code_naf": "62.02A",
    "code_tranche_effectifs": "12",
    "economie_sociale_solidaire": "O",
    "societe_mission": "O",
    "libelle_section": "Information et communication",
    "libelle_sous_classe": "Conseil en systèmes et logiciels informatiques"
  },
  {
    "siren": "900014991",
    "etat_administratif": "C",
    "denomination": null,
    "categorie_entreprise": null,
    "code_cj": "0000",
    "code_naf": "00.00Z",
    "code_tranche_effectifs": "UN",
    "economie_sociale_solidaire": null,
    "societe_mission": null,
    "libelle_section": null,
    "libelle_sous_classe": "Activité inconnue ou non précisée"
  }
]
```

---

## Step 2: Start `microllm serve` for Multilingual Evaluation

In Terminal 1, launch `microllm serve` with `--strategy auto` (routes confident rows via Gear 1 `< 1 ms` and escalates
low-margin records cleanly):

```bash
microllm serve --host 127.0.0.1 --port 8000 --strategy auto
```

Verify readiness from Terminal 2:

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected output:

```json
{"status":"ok","service":"microllm"}
```

---

## Step 3: Enrich French Records Across `fast` and `auto` Strategies

Write `./sandbox/sirene-lab/enrich_sirene.py`. If `psycopg` is installed and `USE_PG=1` is set, the script queries
`localhost:15432`; otherwise it reads `sirene_sample.json`:

```python
import json
import os
from pathlib import Path
import httpx
from microllm import Choice, MicroJevClient, Noul, Score


def load_records() -> list[dict]:
    if os.environ.get("USE_PG") == "1":
        import psycopg

        dsn = "postgresql://sirene:pg_sirene_ephemeral@localhost:15432/sirene"
        sql = """
        SELECT
            ul.siren, ul.etat_administratif, ul.denomination,
            ul.categorie_entreprise, ul.code_cj, ul.code_naf,
            ul.code_tranche_effectifs, ul.economie_sociale_solidaire,
            ul.societe_mission, n.libelle_section, n.libelle_sous_classe
        FROM unites_legales ul
        JOIN ref_naf n ON n.code_naf = ul.code_naf
        WHERE ul.siren IN ('900000001', '900000002', '900000003', '900014991')
        ORDER BY ul.siren
        """
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [desc.name for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    return json.loads(Path("sirene_sample.json").read_text(encoding="utf-8"))


client = MicroJevClient(base_url="http://127.0.0.1:8000")
records = load_records()

questions = {
    "b2b_vertical": Choice(
        options={
            "tech_software_it": (
                "Programmation informatique, logiciels, conseil systèmes, "
                "information et communication"
            ),
            "food_bakery_craft": (
                "Boulangerie, pâtisserie, alimentation, fabrication artisanale manufacturière"
            ),
            "logistics_energy": "Transport routier, entreposage logistique, énergie et immobilier",
        },
        instructions="Classify the French enterprise into one commercial sector.",
    ),
    "b2b_eligible": Noul(
        instructions="Is this enterprise an active unit (etat_administratif A) suitable for B2B outreach?",
        true_description="Active company with known NAF activity and valid denomination",
        false_description="Closed company etat_administratif C, missing denomination, or unknown sentinel 00.00Z",
    ),
    "lead_priority": Score(
        levels=[
            "Tier 0: Closed or sentinel unknown record (do not contact)",
            "Tier 1: Micro-craft or low-fit local shop",
            "Tier 2: Growing SME in digital or B2B services",
            "Tier 3: High-priority software, data, or mission-driven tech enterprise",
        ],
        instructions="Assign B2B outbound lead priority tier.",
    ),
}

for strategy in ("fast", "auto"):
    print(f"=== strategy={strategy} ===")
    for row in records:
        res = client.ask(state=row, questions=questions, multilingual_strategy=strategy)
        diag = httpx.post(
            "http://127.0.0.1:8000/v1/studio/analyze",
            json={
                "state": row,
                "questions": {qid: q.to_payload() for qid, q in questions.items()},
            },
            timeout=30.0,
        ).json()
        meta = diag["cascade"]["b2b_vertical"]["meta"]
        print(
            f"SIREN {row['siren']} ({row['code_naf']}): "
            f"vertical={res['b2b_vertical'].value} (conf={res['b2b_vertical'].confidence:.4f}) "
            f"eligible={res['b2b_eligible'].value} (p={res['b2b_eligible'].probability:.4f}) "
            f"tier={res['lead_priority'].score:.3f} "
            f"x_abstain={meta['x_abstain']} conformal_set={meta['x_conformal_set']}"
        )
```

Run the script from Terminal 2:

```bash
python enrich_sirene.py
```

Expected output:

```text
=== strategy=fast ===
SIREN 900000001 (62.01Z): vertical=tech_software_it (conf=0.9250) eligible=True (p=0.7600) tier=1.670 x_abstain=False conformal_set=['tech_software_it']
SIREN 900000002 (10.71C): vertical=food_bakery_craft (conf=0.8570) eligible=True (p=0.7600) tier=1.537 x_abstain=False conformal_set=['food_bakery_craft']
SIREN 900000003 (62.02A): vertical=tech_software_it (conf=0.9429) eligible=True (p=0.7600) tier=1.754 x_abstain=False conformal_set=['tech_software_it']
SIREN 900014991 (00.00Z): vertical=food_bakery_craft (conf=0.0709) eligible=False (p=0.0294) tier=1.336 x_abstain=True conformal_set=['food_bakery_craft', 'tech_software_it', 'logistics_energy']
=== strategy=auto ===
SIREN 900000001 (62.01Z): vertical=tech_software_it (conf=0.9250) eligible=True (p=0.7600) tier=1.660 x_abstain=False conformal_set=['tech_software_it']
SIREN 900000002 (10.71C): vertical=food_bakery_craft (conf=0.8570) eligible=True (p=0.7600) tier=1.536 x_abstain=False conformal_set=['food_bakery_craft']
SIREN 900000003 (62.02A): vertical=tech_software_it (conf=0.9429) eligible=True (p=0.7600) tier=1.738 x_abstain=False conformal_set=['tech_software_it']
SIREN 900014991 (00.00Z): vertical=food_bakery_craft (conf=0.0755) eligible=False (p=0.0294) tier=1.347 x_abstain=True conformal_set=['food_bakery_craft', 'tech_software_it', 'logistics_energy']
```

---

## Step 4: Quarantine Sentinel `00.00Z` Rows Using `x_abstain`

Examine how Split Conformal abstention protects downstream CRM pipelines:

- **Active IT and craft companies (`900000001`, `900000002`, `900000003`)**: Each unit resolves to a singleton
  `conformal_set` with `x_abstain = False`, confidence between `0.8570` and `0.9429`, and `b2b_eligible = True`
  (`p = 0.7600`). Notice `900000003` (`societe_mission = 'O'`, `economie_sociale_solidaire = 'O'`) also receives the
  highest `lead_priority` (`1.754`).
- **Sentinel `00.00Z` row (`900014991`)**: Because `etat_administratif = 'C'`, `denomination = null`, and
  `code_naf = '00.00Z'`, `b2b_eligible` drops cleanly to `False` (`p = 0.0294`), vertical confidence collapses to
  `0.0709`, and `x_abstain = True` returns all three labels in `conformal_set` for quarantine.

---

## Step 5: Stop the Sirene Stack and Clean Up

If you launched Docker Sirene in Step 1 (Option A), shut down the PostgreSQL 18 stack cleanly:

```bash
make -C ../test_db sirene-down
```

Stop `microllm serve` in Terminal 1 (`Ctrl+C`), then remove your temporary `./sandbox/sirene-lab` folder:

=== "bash"

    ```bash
    curl -sf http://127.0.0.1:8000/healthz || echo "Local server stopped cleanly."
    cd ../..
    rm -rf ./sandbox/sirene-lab ./sandbox/test_db
    ```

=== "PowerShell"

    ```powershell
    try { Invoke-RestMethod http://127.0.0.1:8000/healthz | Out-Null } catch { "Local server stopped cleanly." }
    Set-Location ../..
    Remove-Item -Recurse -Force ./sandbox/sirene-lab, ./sandbox/test_db -ErrorAction SilentlyContinue
    ```

---

## Next Steps

- Continue to [Tutorial 6: Guard SQL Agents on Employees DB](06-guard-sql-agents-on-employees-db.md)
- Automate PostgreSQL staging enrichment and `00.00Z` quarantine:
  [Enrich Sirene PostgreSQL Records](../02-how-to-guides/operations/enrich-sirene-postgresql-records.md)
  ([All How-To Guides](../02-how-to-guides/index.md))
- Look up `MICROLLM_MULTILINGUAL_STRATEGY` and model profiles: [Reference](../03-reference/index.md)
- Read about StartLux cascades and Split Conformal calibration:
  [Calibration and Order Invariance](../04-explanation/deep-dives/calibration-and-order-invariance.md)
  ([All Explanations](../04-explanation/index.md))
