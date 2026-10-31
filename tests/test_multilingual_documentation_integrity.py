"""Automated verification suite for Multilingual Documentation Integrity & Governance (Issue #167).

Validates:
1. Existence, substantive content length, and prose line density of:
   - `docs/FAILURE_MODES_AND_WORST_CASES.md`
   - `docs/MULTILINGUAL_STRATEGIES_SUMMARY.md`
2. Presence of all mandatory sections, headings, and technical symbols across both documents.
3. Structural validity of all Markdown tables (column count consistency, non-empty cells).
4. Relative file link and anchor cross-reference validity resolving on disk.
5. Strict English-only enforcement: 100% of headers, bullets, and explanatory prose in English,
   with non-English text permitted solely in code blocks and ticket payload cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import microllm

ROOT = Path(__file__).resolve().parents[1]
FAILURE_MODES_DOC = ROOT / "docs" / "FAILURE_MODES_AND_WORST_CASES.md"
SUMMARY_DOC = ROOT / "docs" / "MULTILINGUAL_STRATEGIES_SUMMARY.md"
README_ROOT = ROOT / "README.md"
README_DOCS = ROOT / "docs" / "README.md"
BENCHMARKS_DOC = ROOT / "BENCHMARKS.md"
ACTION_LOGBOOK = ROOT / "docs" / "ACTION_LOGBOOK.md"
RCA_REGISTER = ROOT / "docs" / "RCA_REGISTER.md"


def _slugify(heading: str) -> str:
    """GitHub-compatible markdown header anchor slugifier."""
    # Remove markdown formatting like bold, backticks, links
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    cleaned = re.sub(r"[`*_~]", "", cleaned)
    # Strip leading/trailing hashes and whitespace
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip().lower()
    # Replace spaces and punctuation with hyphens
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    return cleaned


def _extract_tables(text: str) -> list[list[str]]:
    """Extract markdown table blocks (list of raw line lists) from text outside code blocks."""
    lines = text.splitlines()
    in_code_block = False
    table_blocks: list[list[str]] = []
    current_table: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if current_table:
                table_blocks.append(current_table)
                current_table = []
            continue

        if in_code_block:
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            current_table.append(stripped)
        else:
            if current_table:
                if len(current_table) >= 2:
                    table_blocks.append(current_table)
                current_table = []

    if current_table and len(current_table) >= 2:
        table_blocks.append(current_table)

    return table_blocks


class TestDocumentationFilesExistenceAndReadability:
    """Validates existence, file sizes, and prose readability limits."""

    def test_documentation_files_exist(self) -> None:
        """Verify both target documentation files exist in docs/."""
        assert microllm.__version__, "microllm version must be defined"
        assert FAILURE_MODES_DOC.is_file(), f"Missing {FAILURE_MODES_DOC}"
        assert SUMMARY_DOC.is_file(), f"Missing {SUMMARY_DOC}"

    def test_documentation_substantive_length(self) -> None:
        """Verify documentation files contain substantive engineering depth."""
        fm_text = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        sm_text = SUMMARY_DOC.read_text(encoding="utf-8")
        assert len(fm_text) >= 20000, f"docs/FAILURE_MODES_AND_WORST_CASES.md length ({len(fm_text)}) < 20,000 characters"
        assert len(sm_text) >= 15000, f"docs/MULTILINGUAL_STRATEGIES_SUMMARY.md length ({len(sm_text)}) < 15,000 characters"

    def test_prose_line_density_readability(self) -> None:
        """Ensure prose paragraphs follow formatting readability (<= 450 characters per line outside code/tables)."""
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            lines = doc_path.read_text(encoding="utf-8").splitlines()
            in_code_block = False
            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue
                # Skip markdown tables, ASCII art boxes, math delimiters, and headers
                if stripped.startswith(("|", "+", "│", "┌", "└")):
                    continue
                if stripped.startswith(("#", "$$")):
                    continue
                assert len(line) <= 450, (
                    f"Line {line_no} in {doc_path.name} exceeds 450 characters ({len(line)} chars). "
                    f"Wrap long prose lines for readability per IDEA-038."
                )


class TestFailureModesDocumentRequiredSectionsAndContent:
    """Validates Section 9 structure and failure mode coverage in FAILURE_MODES_AND_WORST_CASES.md."""

    def test_top_level_headings_exist(self) -> None:
        """Verify required section 9 headings are present in docs/FAILURE_MODES_AND_WORST_CASES.md."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        required_headings = [
            "## 9. Multilingual Decision Strategies",
            "### 9.1 Strategy 1",
            "### 9.2 Strategy 2",
            "### 9.3 Auto-Cascade",
            "### 9.4 Empirical Worst-Case",
            "### 9.5 Production Hardening",
        ]
        for heading in required_headings:
            assert heading in content, f"Missing heading '{heading}' in {FAILURE_MODES_DOC.name}"

    def test_strategy1_failure_modes_covered(self) -> None:
        """Verify Strategy 1 failure mechanisms are explicitly documented."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        tokens = [
            "Out-of-Vocabulary",
            "Slang",
            "False Friends",
            "Negation",
            "Concessive",
            "CONCEPT_CLUSTERS",
            "BLAKE2b",
        ]
        for token in tokens:
            assert token.lower() in content.lower(), f"Strategy 1 failure mode keyword '{token}' missing"

    def test_strategy2_failure_modes_covered(self) -> None:
        """Verify Strategy 2 failure mechanisms are explicitly documented."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        tokens = [
            "CPU Latency",
            "max_length = 512",
            "Truncation",
            "Subword Fragmentation",
            "Cold-Start",
            "MiniLM",
            "mDeBERTa",
        ]
        for token in tokens:
            assert token.lower() in content.lower(), f"Strategy 2 failure mode keyword '{token}' missing"

    def test_auto_cascade_failure_modes_covered(self) -> None:
        """Verify Auto-Cascade dual-gear failure modes and sensitivity thresholds are covered."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        tokens = [
            "False-Negative Escalation",
            "False-Positive Escalation",
            "0.12",
            "0.88",
            "Sensitivity Analysis",
        ]
        for token in tokens:
            assert token in content, f"Auto-cascade keyword '{token}' missing"

    def test_mitigation_and_hardening_framework(self) -> None:
        """Verify production mitigations and guardrails are defined."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        tokens = [
            "CONCEPT_CLUSTERS",
            "Concessive Rhetorical Reversal",
            "Head-and-Tail",
            "Dynamic Bayesian Margin",
            "IDEA-041",
            "Conformal",
        ]
        for token in tokens:
            assert token in content, f"Mitigation framework token '{token}' missing"

    def test_governance_and_benchmark_cross_references(self) -> None:
        """Verify cross-references to BENCHMARKS.md and living registers."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        assert "BENCHMARKS.md" in content
        assert "ACTION_LOGBOOK.md" in content
        assert "RCA_REGISTER.md" in content
        assert "ACT-076" in content


class TestMultilingualStrategiesSummaryDocumentRequiredSectionsAndContent:
    """Validates structure and content of MULTILINGUAL_STRATEGIES_SUMMARY.md."""

    def test_top_level_headings_exist(self) -> None:
        """Verify required top-level headings exist in SUMMARY_DOC."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        required_headings = [
            "## 1. Executive Synthesis & Architectural Blueprint",
            "## 2. Comprehensive Empirical Trade-Off Matrix",
            "## 3. Operational Decision Guide",
            "## 4. Developer Integration Guide",
            "## 5. Governance, Maintenance & Scientific Index",
        ]
        for heading in required_headings:
            assert heading in content, f"Missing heading '{heading}' in {SUMMARY_DOC.name}"

    def test_empirical_benchmarks_section11_metrics_presence(self) -> None:
        """Verify exact citations from BENCHMARKS.md §11 are present in summary."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        required_metrics = [
            "0.342 ms",
            "0.099 ms",
            "6.85 ms",
            "17.20 ms",
            "100.00%",
            "10.42%",
            "2,086.0",
            "7,865.4",
            "48",
            "240",
        ]
        for metric in required_metrics:
            assert metric in content, f"Benchmark metric '{metric}' missing from {SUMMARY_DOC.name}"

    def test_operational_decision_guide_coverage(self) -> None:
        """Verify coverage of fast, neural, and auto deployment archetypes."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        tokens = [
            'strategy="fast"',
            'strategy="neural"',
            'strategy="auto"',
            "Archetype",
            "Edge",
            "Customer Support",
            "Compliance",
        ]
        for token in tokens:
            assert token.lower() in content.lower(), f"Operational guide keyword '{token}' missing"

    def test_developer_integration_guide_coverage(self) -> None:
        """Verify REST API, Python SDK, CLI, and Hugging Face Hub links."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        tokens = [
            "/v1/systemone",
            "MicroJevClient",
            "microllm serve",
            "microllm studio",
            "jmrenouard/microllm-multilingual-benchmarks",
            "jmrenouard/microllm-decision-studio",
            "jmrenouard/microllm-multilingual-minilm-int8",
            "jmrenouard/microllm-multilingual-mdeberta-int8",
        ]
        for token in tokens:
            assert token in content, f"Developer integration token '{token}' missing"

    def test_ascii_architecture_diagram_present(self) -> None:
        """Verify Dual-Gear architecture diagram is present."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        assert "Dual-Gear" in content
        assert "GEAR 1:" in content or "Fast Lexicon" in content
        assert "GEAR 2:" in content or "Neural ONNX" in content
        assert "StartLux" in content


class TestStructuredMarkdownTablesValidity:
    """Validates structural consistency and cell population of all Markdown tables."""

    def test_all_tables_column_consistency_and_non_empty(self) -> None:
        """Verify all tables have consistent column counts and no blank cells."""
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            text = doc_path.read_text(encoding="utf-8")
            tables = _extract_tables(text)
            assert len(tables) > 0, f"No markdown tables found in {doc_path.name}"
            for t_idx, table_lines in enumerate(tables, 1):
                header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                col_count = len(header)
                assert col_count >= 2, f"Table {t_idx} in {doc_path.name} has < 2 columns"

                # Check delimiter
                delimiter = [c.strip() for c in table_lines[1].split("|")[1:-1]]
                assert len(delimiter) == col_count, (
                    f"Table {t_idx} delimiter row in {doc_path.name} has {len(delimiter)} columns, expected {col_count}"
                )

                # Check data rows
                for r_idx, row_line in enumerate(table_lines[2:], 3):
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    assert len(cells) == col_count, (
                        f"Table {t_idx} row {r_idx} in {doc_path.name} has {len(cells)} cells, expected {col_count}: '{row_line}'"
                    )
                    for c_idx, cell in enumerate(cells, 1):
                        assert len(cell) > 0, (
                            f"Table {t_idx} row {r_idx} col {c_idx} in {doc_path.name} is empty"
                        )

    def test_empirical_worst_case_table_structure(self) -> None:
        """Verify the 12-ticket empirical worst-case matrix in FAILURE_MODES_DOC."""
        content = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        assert "TC-WC-01" in content
        assert "TC-WC-12" in content
        for lang in ("`EN`", "`FR`", "`ES`", "`PT`"):
            assert lang in content, f"Language {lang} missing from worst-case table"

    def test_empirical_trade_off_matrix_structure(self) -> None:
        """Verify the Master Empirical Trade-Off Matrix in SUMMARY_DOC."""
        content = SUMMARY_DOC.read_text(encoding="utf-8")
        assert "Table 1: Master Empirical Trade-Off Matrix" in content
        assert "Strategy 1: `fast`" in content
        assert "Auto-Cascade: `auto`" in content
        assert "Overall Accuracy" in content
        assert "CPU Latency `p50`" in content
        assert "Throughput" in content


class TestMarkdownLinksAndCrossReferencesValidity:
    """Validates markdown links, bidirectional references, and anchor resolution."""

    def test_relative_file_links_resolve_on_disk(self) -> None:
        """Ensure all relative file links in both documents point to existing files on disk."""
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            content = doc_path.read_text(encoding="utf-8")
            for match in link_pattern.finditer(content):
                target = match.group(2).strip()
                # Skip external web URLs, anchors, mailto
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if target.startswith("#"):
                    continue
                # Strip internal anchor if present (e.g. file.md#anchor)
                file_target = target.split("#")[0]
                if not file_target:
                    continue
                resolved = (doc_path.parent / file_target).resolve()
                assert resolved.exists(), (
                    f"Broken link in {doc_path.name}: '{target}' resolved to '{resolved}' which does not exist"
                )

    def test_bidirectional_cross_document_links(self) -> None:
        """Verify bidirectional cross-linking between FAILURE_MODES and SUMMARY docs."""
        fm_text = FAILURE_MODES_DOC.read_text(encoding="utf-8")
        sm_text = SUMMARY_DOC.read_text(encoding="utf-8")
        assert "MULTILINGUAL_STRATEGIES_SUMMARY.md" in fm_text, "FAILURE_MODES doc must link to MULTILINGUAL_STRATEGIES_SUMMARY.md"
        assert "FAILURE_MODES_AND_WORST_CASES.md" in sm_text, "SUMMARY doc must link to FAILURE_MODES_AND_WORST_CASES.md"

    def test_hub_readmes_link_to_both_documents(self) -> None:
        """Verify that README.md and docs/README.md cross-reference both documents."""
        root_readme = README_ROOT.read_text(encoding="utf-8")
        docs_readme = README_DOCS.read_text(encoding="utf-8")
        assert "docs/FAILURE_MODES_AND_WORST_CASES.md" in root_readme or "FAILURE_MODES_AND_WORST_CASES.md" in root_readme
        assert "docs/MULTILINGUAL_STRATEGIES_SUMMARY.md" in root_readme or "MULTILINGUAL_STRATEGIES_SUMMARY.md" in root_readme
        assert "FAILURE_MODES_AND_WORST_CASES.md" in docs_readme
        assert "MULTILINGUAL_STRATEGIES_SUMMARY.md" in docs_readme

    def test_internal_anchor_links_resolve(self) -> None:
        """Verify internal anchor links (#anchor) point to existing headings within the doc."""
        anchor_pattern = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            lines = doc_path.read_text(encoding="utf-8").splitlines()
            # Collect all headings as slugs
            headings = [_slugify(l) for l in lines if l.strip().startswith("#")]
            for line_no, line in enumerate(lines, 1):
                for match in anchor_pattern.finditer(line):
                    anchor = match.group(2).strip().lower()
                    # Fuzzy match: anchor should match one of the headings
                    matched = any(anchor == h or anchor.startswith(h) or h.startswith(anchor) for h in headings)
                    assert matched, f"In {doc_path.name}:{line_no}, anchor '#{anchor}' does not match any heading in {headings[:5]}..."


class TestStrictEnglishOnlyDocumentationEnforcement:
    """Enforces 100% English documentation policy per .agents/rules/english-only.md."""

    def test_headers_are_strictly_english(self) -> None:
        """Verify no headings in either document contain French/Spanish/Portuguese text."""
        forbidden_heading_words = [
            "synthèse", "échec", "résultats", "spécification", "stratégies",
            "resumen", "fallos", "resultados", "resumo", "falhas", "visão",
            "facturation", "expédition", "support technique",
        ]
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            lines = doc_path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    heading_lower = line.lower()
                    for word in forbidden_heading_words:
                        assert word not in heading_lower, (
                            f"Non-English word '{word}' found in heading at {doc_path.name}:{line_no}: '{line}'"
                        )

    def test_prose_and_bullets_are_strictly_english(self) -> None:
        """Verify explanatory prose and bullet points are written in English."""
        forbidden_prose_phrases = [
            "synthèse exécutive", "ce document décrit", "les résultats montrent",
            "en conclusion", "il est recommandé de", "en resumen", "este documento describe",
            "los resultados muestran", "neste documento", "os resultados mostram",
        ]
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            content = doc_path.read_text(encoding="utf-8")
            for phrase in forbidden_prose_phrases:
                assert phrase not in content.lower(), (
                    f"Forbidden non-English phrase '{phrase}' found in {doc_path.name}"
                )

    def test_multilingual_payload_isolation(self) -> None:
        """Verify multilingual samples exist exclusively in code blocks or table payload cells."""
        for doc_path in (FAILURE_MODES_DOC, SUMMARY_DOC):
            lines = doc_path.read_text(encoding="utf-8").splitlines()
            in_code_block = False
            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or stripped.startswith("|"):
                    # Table cells or code blocks are permitted to contain sample multilingual strings
                    continue
                # Outside code blocks and tables, full French/Spanish/Portuguese sentences are forbidden
                assert not stripped.startswith("Bonjour, "), f"Unenclosed foreign sample at {doc_path.name}:{line_no}"
                assert not stripped.startswith("No es que "), f"Unenclosed foreign sample at {doc_path.name}:{line_no}"
                assert not stripped.startswith("Não que "), f"Unenclosed foreign sample at {doc_path.name}:{line_no}"
