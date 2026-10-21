"""TDD test suite for the fun, high-converting System-2 vs System-1 visual header banner and logo (Issue #50)."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER_SVG = ROOT / "docs" / "assets" / "header-banner.svg"
LOGO_SVG = ROOT / "docs" / "assets" / "logo.svg"


def test_header_banner_svg_is_valid_xml_and_illustrative() -> None:
    """Verify docs/assets/header-banner.svg is valid SVG XML and tells the System-2 vs System-1 visual story."""
    assert HEADER_SVG.is_file(), "docs/assets/header-banner.svg must exist"
    raw = HEADER_SVG.read_text(encoding="utf-8")

    # 1. Must parse as valid XML / SVG
    root = ET.fromstring(raw)
    assert root.tag.endswith("svg"), "Root element must be <svg>"
    assert "viewBox" in root.attrib

    # 2. Must include the fun, memorable visual contrast (System-2 Giants vs System-1 Colibri Reflex Engine)
    lower = raw.lower()
    assert "system-2" in lower or "system 2" in lower
    assert "system-1" in lower or "system 1" in lower
    assert "70b" in lower
    assert "1.03 ms" in lower
    assert "anthropic" in lower and "claude" in lower
    assert "openai" in lower and "chatgpt" in lower
    assert "gemini argon" in lower

    # 3. Must highlight the concrete pain points of System-2 vs MicroLLM's guarantees
    assert "json" in lower
    assert "0.000000" in raw
    assert "noul" in lower
    assert "choice" in lower
    assert "score" in lower
    assert "multi" in lower
    assert "abstain" in lower or "conformal" in lower
    assert "anyjev" in lower


def test_logo_svg_is_valid_xml_and_crisp() -> None:
    """Verify docs/assets/logo.svg is valid SVG XML with dual-gear and conformal elements."""
    assert LOGO_SVG.is_file(), "docs/assets/logo.svg must exist"
    raw = LOGO_SVG.read_text(encoding="utf-8")
    root = ET.fromstring(raw)
    assert root.tag.endswith("svg")
    assert "viewBox" in root.attrib
    assert "#10b981" in raw
    assert "#0ea5e9" in raw
    assert "#f59e0b" in raw


def test_header_banner_integrated_across_readme_docs_and_studio() -> None:
    """Verify the visual header banner is embedded in README.md, docs/README.md, docs/index.html, and Studio UI."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    docs_html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    studio_html = (ROOT / "src" / "microllm" / "static" / "index.html").read_text(encoding="utf-8")

    assert "docs/assets/header-banner.svg" in readme
    assert "assets/header-banner.svg" in docs_readme
    assert "header-banner.svg" in docs_html
    assert "header-banner.svg" in studio_html
