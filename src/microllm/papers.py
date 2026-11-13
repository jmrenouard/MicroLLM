"""Structured Academic Literature Catalog & Search Engine for Jev, RLCD, System-1 & Calibrated LLMs (Issue #42)."""

from __future__ import annotations

import functools
import importlib.resources
import json
from typing import Any


@functools.lru_cache(maxsize=1)
def _load_literature_pillars() -> list[dict[str, str]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("literature_pillars.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


@functools.lru_cache(maxsize=1)
def _load_academic_papers_catalog() -> list[dict[str, Any]]:
    raw = (
        importlib.resources.files("microllm.data")
        .joinpath("academic_papers_catalog.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)


LITERATURE_PILLARS: list[dict[str, str]] = _load_literature_pillars()
ACADEMIC_PAPERS_CATALOG: list[dict[str, Any]] = _load_academic_papers_catalog()

# Backward-compatibility dictionary for E2E tests (assert "paper-openjev-rlcd" in PAPERS, etc.)
PAPERS: dict[str, Any] = {
    "paper-openjev-rlcd": next(
        (p for p in ACADEMIC_PAPERS_CATALOG if p.get("arxiv_id") == "2609.38850"),
        ACADEMIC_PAPERS_CATALOG[0],
    ),
    "paper-anyjev": next(
        (p for p in ACADEMIC_PAPERS_CATALOG if p.get("arxiv_id") == "2610.00831"),
        ACADEMIC_PAPERS_CATALOG[1] if len(ACADEMIC_PAPERS_CATALOG) > 1 else ACADEMIC_PAPERS_CATALOG[0],
    ),
    "paper-startlux-decision": next(
        (p for p in ACADEMIC_PAPERS_CATALOG if p.get("arxiv_id") == "2610.17309"),
        ACADEMIC_PAPERS_CATALOG[2] if len(ACADEMIC_PAPERS_CATALOG) > 2 else ACADEMIC_PAPERS_CATALOG[0],
    ),
}
for _p in ACADEMIC_PAPERS_CATALOG:
    _aid = _p.get("arxiv_id", "")
    if _aid and _aid not in PAPERS:
        PAPERS[_aid] = _p


def list_academic_papers(
    pillar: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return academic papers filtered by thematic pillar ID/substring and/or keyword query."""
    results: list[dict[str, Any]] = []
    pillar_norm = pillar.strip().lower() if pillar else None
    query_norm = query.strip().lower() if query else None

    for entry in ACADEMIC_PAPERS_CATALOG:
        if pillar_norm:
            if (
                pillar_norm not in entry["pillar_id"].lower()
                and pillar_norm not in entry["pillar_title"].lower()
            ):
                continue
        if query_norm:
            haystack = " ".join(
                [
                    entry["arxiv_id"],
                    entry["title"],
                    " ".join(entry["authors"]),
                    entry["summary"],
                    entry["microllm_relevance"],
                    entry["pillar_title"],
                ]
            ).lower()
            if query_norm not in haystack:
                continue
        results.append(dict(entry))
    return results


def get_academic_paper(arxiv_id: str) -> dict[str, Any] | None:
    """Look up a specific academic paper by its arXiv identifier (e.g. '2609.38850' or '2609.38850v1')."""
    clean_id = arxiv_id.strip().replace("arXiv:", "").split("v")[0]
    for entry in ACADEMIC_PAPERS_CATALOG:
        if entry["arxiv_id"] == clean_id or entry["versioned_id"] == arxiv_id.strip():
            return dict(entry)
    return None


def get_literature_taxonomy() -> dict[str, Any]:
    """Return the 6-pillar academic taxonomy with paper counts and foundational highlights."""
    pillars_summary: list[dict[str, Any]] = []
    for p in LITERATURE_PILLARS:
        papers_in_pillar = [
            e for e in ACADEMIC_PAPERS_CATALOG if e["pillar_id"] == p["id"]
        ]
        pillars_summary.append(
            {
                "id": p["id"],
                "title": p["title"],
                "description": p["description"],
                "count": len(papers_in_pillar),
                "arxiv_ids": [e["arxiv_id"] for e in papers_in_pillar],
            }
        )
    return {
        "total_papers": len(ACADEMIC_PAPERS_CATALOG),
        "total_pillars": len(LITERATURE_PILLARS),
        "foundational_core_ids": ["2609.38850", "2610.00831", "2610.17309"],
        "pillars": pillars_summary,
    }


def format_papers_markdown(
    pillar: str | None = None,
    query: str | None = None,
) -> str:
    """Format the filtered academic paper catalog as a GitHub-Flavored Markdown table."""
    papers = list_academic_papers(pillar=pillar, query=query)
    lines = [
        f"# MicroLLM Academic Literature Catalog ({len(papers)} papers)",
        "",
        "| arXiv ID | Title | Authors | Published | Pillar | MicroLLM Architectural Mapping |",
        "| :--- | :--- | :--- | :---: | :--- | :--- |",
    ]
    for p in papers:
        authors_short = ", ".join(p["authors"][:2]) + (" et al." if len(p["authors"]) > 2 else "")
        lines.append(
            f"| [`{p['arxiv_id']}`]({p['abs_url']}) ([PDF]({p['pdf_url']})) "
            f"| **{p['title']}** | {authors_short} | `{p['published']}` "
            f"| {p['pillar_title'].split('—')[0].strip()} | {p['microllm_relevance']} |"
        )
    return "\n".join(lines)


def slugify_paper_title(title: str, max_len: int = 42) -> str:
    """Convert a paper title into a filesystem-safe slug for local PDF filenames."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned


def verify_downloaded_pdfs(
    output_dir: str | None = None,
    pillar: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Verify local PDF presence and `%PDF-` header integrity in `docs/papers/` across the catalog."""
    from pathlib import Path

    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "docs" / "papers"
    )
    papers = list_academic_papers(pillar=pillar, query=query)
    present: list[dict[str, Any]] = []
    missing: list[str] = []

    for entry in papers:
        aid = entry["arxiv_id"]
        found_path: Path | None = None
        if base_dir.is_dir():
            for candidate in base_dir.glob(f"{aid}*.pdf"):
                if candidate.is_file() and candidate.stat().st_size > 8:
                    with candidate.open("rb") as f:
                        if f.read(5) == b"%PDF-":
                            found_path = candidate
                            break
        if found_path is not None:
            present.append(
                {
                    "arxiv_id": aid,
                    "filename": found_path.name,
                    "size_bytes": found_path.stat().st_size,
                }
            )
        else:
            missing.append(aid)

    return {
        "output_dir": str(base_dir),
        "total_expected": len(papers),
        "present_count": len(present),
        "missing_count": len(missing),
        "all_present": len(missing) == 0,
        "present": present,
        "missing_arxiv_ids": missing,
    }


def download_academic_papers(
    output_dir: str | None = None,
    pillar: str | None = None,
    query: str | None = None,
    arxiv_ids: list[str] | None = None,
    rate_limit_seconds: float = 3.0,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    """Download academic PDFs into `docs/papers/` while respecting arXiv's 1 req / 3s rate limit (Issue #67)."""
    from pathlib import Path
    import time
    import httpx

    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "docs" / "papers"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    papers = list_academic_papers(pillar=pillar, query=query)
    if arxiv_ids is not None:
        allowed = {x.strip().replace("arXiv:", "").split("v")[0] for x in arxiv_ids}
        papers = [p for p in papers if p["arxiv_id"] in allowed]

    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for idx, entry in enumerate(papers):
        aid = entry["arxiv_id"]
        existing: Path | None = None
        for candidate in base_dir.glob(f"{aid}*.pdf"):
            if candidate.is_file() and candidate.stat().st_size > 8:
                with candidate.open("rb") as f:
                    if f.read(5) == b"%PDF-":
                        existing = candidate
                        break
        if existing is not None:
            skipped.append(existing.name)
            continue

        slug = slugify_paper_title(entry["title"])
        target_path = base_dir / f"{aid}_{slug}.pdf"
        url = entry["pdf_url"]

        try:
            if fetcher is not None:
                pdf_bytes = fetcher(url, aid)
            else:
                if idx > 0 and rate_limit_seconds > 0:
                    time.sleep(rate_limit_seconds)
                with httpx.Client(
                    follow_redirects=True,
                    timeout=60.0,
                    headers={"User-Agent": "MicroLLM-ArXiv-Client/0.2 (https://github.com/jmrenouard/MicroLLM)"},
                ) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    pdf_bytes = resp.content

            if pdf_bytes and pdf_bytes[:5] == b"%PDF-":
                target_path.write_bytes(pdf_bytes)
                downloaded.append(target_path.name)
            else:
                failed.append(aid)
        except Exception:  # noqa: BLE001
            failed.append(aid)

    return {
        "output_dir": str(base_dir),
        "total_selected": len(papers),
        "downloaded_count": len(downloaded),
        "skipped_existing_count": len(skipped),
        "failed_count": len(failed),
        "downloaded_files": downloaded,
        "skipped_files": skipped,
        "failed_arxiv_ids": failed,
    }
