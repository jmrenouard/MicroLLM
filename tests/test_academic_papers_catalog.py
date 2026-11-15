"""TDD unit tests for the 104-paper Jev, RLCD, System-1 & Calibrated LLM academic catalog (Issue #42)."""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
from fastapi.testclient import TestClient

from microllm import (
    ACADEMIC_PAPERS_CATALOG,
    LITERATURE_PILLARS,
    AsyncMicroJevClient,
    MicroJevClient,
    create_app,
    download_academic_papers,
    format_papers_markdown,
    get_academic_paper,
    get_literature_taxonomy,
    list_academic_papers,
    verify_downloaded_pdfs,
)
from microllm.papers import slugify_paper_title
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


def test_oct_nov_2026_wave_papers_registered_and_documented() -> None:
    """Verify the 12 curated Oct-Nov 2026 papers are registered in the catalog and docs (Issue #247)."""
    new_ids = [
        "2610.31521",
        "2610.29217",
        "2610.34329",
        "2610.38623",
        "2611.12521",
        "2611.01370",
        "2611.13843",
        "2611.15777",
        "2611.16961",
        "2611.02155",
        "2611.02522",
        "2611.16937",
    ]
    assert len(ACADEMIC_PAPERS_CATALOG) == 116

    readme_text = (ROOT / "docs" / "papers" / "README.md").read_text(encoding="utf-8")
    lit_text = (ROOT / "docs" / "LITERATURE_REVIEW.md").read_text(encoding="utf-8")
    assert "$" not in readme_text
    assert "$" not in lit_text

    for aid in new_ids:
        paper = get_academic_paper(aid)
        assert paper is not None, f"Missing Oct-Nov 2026 paper {aid} in academic_papers_catalog.json"
        assert "$" not in paper["summary"]
        assert "$" not in paper["microllm_relevance"]
        assert aid in readme_text, f"Missing {aid} in docs/papers/README.md"
        assert aid in lit_text, f"Missing {aid} in docs/LITERATURE_REVIEW.md"


def test_download_and_verify_academic_pdfs(tmp_path: pathlib.Path, capsys: object) -> None:
    """Verify download_academic_papers, verify_downloaded_pdfs, and CLI --download-pdfs / --verify-pdfs (Issue #67)."""
    fake_pdf = b"%PDF-1.4\n%Test Academic Paper Stub\n1 0 obj\n<<>>\nendobj\n%%EOF\n"

    dl_res = download_academic_papers(
        output_dir=str(tmp_path),
        arxiv_ids=["2609.38850", "2610.00831"],
        rate_limit_seconds=0.0,
        fetcher=lambda url, aid: fake_pdf,
    )
    assert dl_res["total_selected"] == 2
    assert dl_res["downloaded_count"] == 2
    assert dl_res["skipped_existing_count"] == 0
    assert dl_res["failed_count"] == 0

    # Calling a second time should skip already-valid PDFs
    dl_skip = download_academic_papers(
        output_dir=str(tmp_path),
        arxiv_ids=["2609.38850", "2610.00831"],
        rate_limit_seconds=0.0,
        fetcher=lambda url, aid: fake_pdf,
    )
    assert dl_skip["downloaded_count"] == 0
    assert dl_skip["skipped_existing_count"] == 2

    ver = verify_downloaded_pdfs(output_dir=str(tmp_path), query="2609.38850")
    assert ver["all_present"] is True
    assert ver["present_count"] == 1

    # Test CLI --verify-pdfs
    rc_ver = run_cli(["papers", "--verify-pdfs", "--output-dir", str(tmp_path), "--query", "2609.38850"])
    assert rc_ver == 0
    out_ver = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert out_ver["all_present"] is True


def test_all_116_academic_papers_schema_and_url_integrity() -> None:
    """Verify all 116 academic papers satisfy strict schema, regex, URL, and math formatting invariants."""
    assert len(ACADEMIC_PAPERS_CATALOG) == 116
    valid_pillar_ids = {p["id"] for p in LITERATURE_PILLARS}

    for paper in ACADEMIC_PAPERS_CATALOG:
        aid = paper["arxiv_id"]
        assert re.match(r"^\d{4}\.\d{4,5}$", aid), f"Invalid arxiv_id format: {aid}"
        assert re.match(r"^\d{4}\.\d{4,5}v\d+$", paper["versioned_id"]), (
            f"Invalid versioned_id format: {paper['versioned_id']}"
        )
        assert len(paper["title"]) >= 8, f"Title too short for {aid}: {paper['title']}"
        assert isinstance(paper["authors"], list) and len(paper["authors"]) >= 1
        for author in paper["authors"]:
            assert isinstance(author, str) and len(author.strip()) >= 3
        assert 2017 <= int(paper["published"][:4]) <= 2026, f"Unexpected year {paper['published']} for {aid}"
        assert paper["pillar_id"] in valid_pillar_ids, f"Invalid pillar_id for {aid}: {paper['pillar_id']}"
        assert paper["abs_url"] == f"https://arxiv.org/abs/{aid}"
        assert paper["pdf_url"] == f"https://arxiv.org/pdf/{aid}.pdf"
        assert paper["summary"], f"Empty summary for {aid}"
        assert "$" not in paper["summary"], f"Raw $ in summary for {aid}"
        assert paper["microllm_relevance"], f"Empty microllm_relevance for {aid}"
        assert "$" not in paper["microllm_relevance"], f"Raw $ in microllm_relevance for {aid}"


def test_get_academic_paper_case_insensitivity_and_prefix_handling() -> None:
    """Verify get_academic_paper handles case variations, arXiv prefixes, versions, whitespace, and unknown IDs."""
    # S1-SHAP paper 2611.12521
    p1 = get_academic_paper("2611.12521")
    assert p1 is not None
    assert p1["arxiv_id"] == "2611.12521"

    p2 = get_academic_paper("ARXIV:2611.12521")
    assert p2 is not None
    assert p2["arxiv_id"] == "2611.12521"

    p3 = get_academic_paper("arxiv:2611.12521")
    assert p3 is not None
    assert p3["arxiv_id"] == "2611.12521"

    p4 = get_academic_paper("arXiv:2611.12521v1")
    assert p4 is not None
    assert p4["arxiv_id"] == "2611.12521"

    p5 = get_academic_paper("  2611.12521  ")
    assert p5 is not None
    assert p5["arxiv_id"] == "2611.12521"

    assert get_academic_paper("0000.00000") is None
    assert get_academic_paper("") is None


def test_list_academic_papers_query_and_pillar_combinations() -> None:
    """Verify list_academic_papers query and pillar filtering combinations."""
    assert list_academic_papers(pillar="invalid_pillar_id") == []
    assert list_academic_papers(query="nonexistent_keyword_xyz123") == []

    # Case-insensitive query results
    res_upper = list_academic_papers(query="S1-SHAP")
    res_lower = list_academic_papers(query="s1-shap")
    assert len(res_upper) >= 1
    assert [p["arxiv_id"] for p in res_upper] == [p["arxiv_id"] for p in res_lower]

    # Specific pillar and query combination
    res_combo = list_academic_papers(pillar="pillar_5_safety_adversarial", query="policy boundaries")
    assert len(res_combo) >= 1
    assert any("2611.02155" == p["arxiv_id"] for p in res_combo)


def test_verify_downloaded_pdfs_missing_handling(tmp_path: pathlib.Path) -> None:
    """Verify verify_downloaded_pdfs correctly reports missing PDFs when directory is empty."""
    ver = verify_downloaded_pdfs(output_dir=str(tmp_path), query="2611.12521")
    assert ver["all_present"] is False
    assert ver["missing_count"] == 1
    assert "2611.12521" in ver["missing_arxiv_ids"]
    assert ver["present_count"] == 0


def test_slugify_paper_title_truncation_and_special_chars() -> None:
    """Verify slugify_paper_title cleans special characters, truncates to max_len, and strips trailing underscores."""
    long_title = "OpenJev-RLCD: Ultra-Fast Micro-Decision Engine for LLM Cascades and Real-Time Routing"
    slug = slugify_paper_title(long_title, max_len=42)
    assert len(slug) <= 42
    assert not slug.endswith("_")
    assert not slug.startswith("_")

    special_title = "Verdict: Encode Once, Decide Many! [Fast / Low-Cost]"
    slug_spec = slugify_paper_title(special_title)
    assert re.match(r"^[A-Za-z0-9_]+$", slug_spec)
