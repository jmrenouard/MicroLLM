"""TDD unit tests for the 104-paper Jev, RLCD, System-1 & Calibrated LLM academic catalog (Issue #42)."""

from __future__ import annotations

import asyncio
import json
import pathlib
from fastapi.testclient import TestClient

from microllm import (
    ACADEMIC_PAPERS_CATALOG,
    LITERATURE_PILLARS,
    AsyncMicroJevClient,
    MicroJevClient,
    create_app,
    format_papers_markdown,
    get_academic_paper,
    get_literature_taxonomy,
    list_academic_papers,
)
from microllm.server import run_cli

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_academic_papers_catalog_completeness_and_six_pillars() -> None:
    """Verify the catalog contains 98+ Jev/System-1 papers + foundational LLM papers across all 6 pillars."""
    assert len(LITERATURE_PILLARS) == 6
    assert len(ACADEMIC_PAPERS_CATALOG) >= 100

    taxonomy = get_literature_taxonomy()
    assert taxonomy["total_papers"] == len(ACADEMIC_PAPERS_CATALOG)
    assert taxonomy["total_pillars"] == 6
    assert taxonomy["foundational_core_ids"] == ["2609.38850", "2610.00831", "2610.17309"]

    pillar_counts = {p["id"]: p["count"] for p in taxonomy["pillars"]}
    for pid, count in pillar_counts.items():
        assert count >= 14, f"Pillar {pid} should contain at least 14 papers, got {count}"

    # Verify all required fields and valid arXiv URLs on every entry
    seen_ids: set[str] = set()
    for paper in ACADEMIC_PAPERS_CATALOG:
        aid = paper["arxiv_id"]
        assert aid not in seen_ids, f"Duplicate arXiv ID {aid} in catalog"
        seen_ids.add(aid)
        assert paper["title"]
        assert isinstance(paper["authors"], list) and len(paper["authors"]) >= 1
        assert paper["abs_url"].startswith("https://arxiv.org/abs/")
        assert paper["pdf_url"].startswith("https://arxiv.org/pdf/")
        assert paper["summary"]
        assert paper["microllm_relevance"]


def test_list_and_get_academic_paper_filtering() -> None:
    """Verify filtering by pillar, keyword query, and exact/versioned arXiv ID lookup."""
    openjev = get_academic_paper("2609.38850")
    assert openjev is not None
    assert "OpenJev-RLCD" in openjev["title"]

    anyjev = get_academic_paper("arXiv:2610.00831v1")
    assert anyjev is not None
    assert "AnyJev" in anyjev["title"]

    assert get_academic_paper("9999.99999") is None

    safety_papers = list_academic_papers(pillar="pillar_5_safety_adversarial")
    assert len(safety_papers) >= 15
    assert any("2609.29429" == p["arxiv_id"] for p in safety_papers)

    conformal_papers = list_academic_papers(query="conformal")
    assert len(conformal_papers) >= 2

    md = format_papers_markdown(pillar="pillar_1_core_architectures", query="Verdict")
    assert "2610.20778" in md
    assert "Verdict: Encode Once, Decide Many" in md


def test_http_endpoint_sdk_and_cli_for_papers(capsys: object) -> None:
    """Verify GET /v1/papers, MicroJevClient.papers(), AsyncMicroJevClient.papers(), and CLI `microllm papers`."""
    app = create_app()
    tc = TestClient(app)

    # 1. Direct HTTP GET /v1/papers
    resp_all = tc.get("/v1/papers")
    assert resp_all.status_code == 200
    body_all = resp_all.json()
    assert body_all["total"] == len(ACADEMIC_PAPERS_CATALOG)
    assert len(body_all["taxonomy"]["pillars"]) == 6

    # Filter by arxiv_id
    resp_single = tc.get("/v1/papers", params={"arxiv_id": "2610.17309"})
    assert resp_single.status_code == 200
    assert resp_single.json()["total"] == 1
    assert "StartLux-Decision" in resp_single.json()["papers"][0]["title"]

    # 404 on unknown arxiv_id
    resp_404 = tc.get("/v1/papers", params={"arxiv_id": "0000.00000"})
    assert resp_404.status_code == 404

    # 2. Sync SDK client.papers()
    client = MicroJevClient(http_client=tc)
    sdk_res = client.papers(pillar="calibration", query="Sys1Cal")
    assert sdk_res["total"] >= 1
    assert sdk_res["papers"][0]["arxiv_id"] == "2609.35342"

    # 3. Async SDK async_client.papers()
    async def _check_async() -> None:
        ac = AsyncMicroJevClient(http_client=tc)
        ares = await ac.papers(query="DecisionPivot")
        assert ares["total"] >= 2
        ids = {p["arxiv_id"] for p in ares["papers"]}
        assert "2610.28522" in ids

    asyncio.run(_check_async())

    # 4. CLI `microllm papers` JSON and Markdown modes
    rc_json = run_cli(["papers", "--pillar", "pillar_2_calibration_coherence", "--query", "Probability Contracts"])
    assert rc_json == 0
    captured_json = capsys.readouterr().out  # type: ignore[attr-defined]
    parsed_cli = json.loads(captured_json)
    assert parsed_cli["total"] >= 1
    assert parsed_cli["papers"][0]["arxiv_id"] == "2609.37470"

    rc_md = run_cli(["papers", "--query", "NumericJev", "--format", "markdown"])
    assert rc_md == 0
    captured_md = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "2609.28587" in captured_md


def test_documentation_files_updated_with_comprehensive_literature_review() -> None:
    """Verify docs/papers/README.md and docs/LITERATURE_REVIEW.md exist, cover all 6 pillars, and have zero raw '$' math."""
    papers_readme = ROOT / "docs" / "papers" / "README.md"
    lit_review = ROOT / "docs" / "LITERATURE_REVIEW.md"

    assert papers_readme.is_file(), "docs/papers/README.md must exist"
    assert lit_review.is_file(), "docs/LITERATURE_REVIEW.md must exist"

    readme_text = papers_readme.read_text(encoding="utf-8")
    lit_text = lit_review.read_text(encoding="utf-8")

    assert "$" not in readme_text, "docs/papers/README.md must not contain raw '$' characters"
    assert "$" not in lit_text, "docs/LITERATURE_REVIEW.md must not contain raw '$' characters"

    for p in LITERATURE_PILLARS:
        assert p["title"] in readme_text
        assert p["title"].split("—")[1].strip() in lit_text

    # Verify key discovered arXiv papers are documented in both files
    key_ids = [
        "2609.38850",
        "2610.00831",
        "2610.17309",
        "2610.18773",
        "2610.20778",
        "2609.33209",
        "2609.35342",
        "2609.37470",
        "2610.28522",
        "2609.26758",
        "2609.26550",
        "2609.29429",
        "2609.31142",
        "2609.28587",
        "2610.27677",
    ]
    for kid in key_ids:
        assert kid in readme_text, f"Missing {kid} in docs/papers/README.md"
        assert kid in lit_text, f"Missing {kid} in docs/LITERATURE_REVIEW.md"
