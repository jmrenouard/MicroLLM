"""Empirical challenger test suite for Milestone 3: BENCHMARKS.md refactoring.

Adversarially validates:
1. Line count within target ceiling (380-450 lines).
2. Zero \\operatorname macros in BENCHMARKS.md.
3. Display math $$ delimiters on column-0 dedicated lines with surrounding blank lines.
4. Tightly bound inline math without leading/trailing space padding, with paired delimiters.
5. Currency escaping in prose (\\$0.00 or backticked `$0.00`).
6. Table of Contents anchor integrity: all 12 primary and 5 subsection TOC anchors match actual markdown headings.
7. All relative file links in BENCHMARKS.md resolve to existing files on disk.
8. Metric fidelity and LaTeX math integrity across core documentation.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_MD = ROOT / "BENCHMARKS.md"


def strip_inline_code(line: str) -> str:
    """Strip inline code spans (including double-backtick code escapes)."""
    line = re.sub(r"``.*?``", "", line)
    return re.sub(r"`[^`\n]+`", "", line)


def github_slugify(heading_text: str) -> str:
    """Generate GitHub-compatible anchor slug from a Markdown heading string."""
    text = re.sub(r"^#+\s*", "", heading_text.strip())
    # Remove inline markdown formatting like bold, italic, backticks, links
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.lower()
    # Remove all punctuation except whitespace and hyphens (matching GitHub cmark-gfm)
    text = re.sub(r"[^\w\s-]", "", text)
    # Convert every whitespace character to a hyphen (without collapsing multi-spaces)
    text = re.sub(r"\s", "-", text)
    return text


def get_all_headings(lines: list[str]) -> list[tuple[int, int, str, str]]:
    """Extract (line_idx, heading_level, raw_title, slug) for all headings outside code blocks."""
    headings: list[tuple[int, int, str, str]] = []
    in_code_block = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            raw_title = match.group(2).strip()
            slug = github_slugify(raw_title)
            headings.append((idx, level, raw_title, slug))

    return headings


def test_benchmarks_md_exists_and_line_count() -> None:
    """Verify BENCHMARKS.md exists and satisfies target line count (380-450 lines)."""
    assert BENCHMARKS_MD.is_file(), "BENCHMARKS.md must exist at repository root"
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    line_count = len(lines)
    assert 380 <= line_count <= 450, f"BENCHMARKS.md line count {line_count} must be within [380, 450]"


def test_benchmarks_md_zero_operatorname() -> None:
    """Adversarially assert zero \\operatorname macros in BENCHMARKS.md."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    violations: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        clean_line = strip_inline_code(line)
        if r"\operatorname" in clean_line:
            violations.append(f"L{idx}: {line.strip()}")

    assert not violations, f"Found \\operatorname in BENCHMARKS.md:\n" + "\n".join(violations)


def test_benchmarks_md_display_math_delimiters_column_0_and_blank_lines() -> None:
    """Assert all display math $$ delimiters are column 0, dedicated lines, and surrounded by blank lines."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    in_display_math = False
    delimiter_count = 0
    violations: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        no_inline = strip_inline_code(line)
        if "$$" in no_inline:
            delimiter_count += 1
            # Must be strictly '$$' with zero leading/trailing spaces
            if line != "$$":
                violations.append(f"L{idx}: display math delimiter not isolated on column 0 dedicated line: {repr(line)}")

            if not in_display_math:
                # Opening delimiter: preceding line must be blank
                if idx > 1 and lines[idx - 2].strip() != "":
                    violations.append(f"L{idx}: opening $$ missing preceding blank line (prev: {repr(lines[idx - 2])})")
                in_display_math = True
            else:
                # Closing delimiter: following line must be blank
                if idx < len(lines) and lines[idx].strip() != "":
                    violations.append(f"L{idx}: closing $$ missing following blank line (next: {repr(lines[idx])})")
                in_display_math = False

    assert delimiter_count > 0, "Expected display math blocks in BENCHMARKS.md"
    assert delimiter_count % 2 == 0, f"Unpaired display math delimiters: count = {delimiter_count}"
    assert not violations, f"Display math delimiter violations in BENCHMARKS.md:\n" + "\n".join(violations)


def test_benchmarks_md_inline_math_tightly_bound_and_paired() -> None:
    """Assert all inline math spans $...$ are paired and tightly bound without leading/trailing spaces."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    in_display = False
    violations: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        clean_line = strip_inline_code(line)
        if clean_line.strip() == "$$":
            in_display = not in_display
            continue
        if in_display:
            continue

        # Strip any display math if present
        no_display = re.sub(r"\$\$", "", clean_line)

        # Split line by unescaped single $
        parts = re.split(r"(?<!\\)\$", no_display)
        # Even number of delimiters means odd number of split pieces
        if len(parts) % 2 == 0:
            violations.append(f"L{idx}: odd unescaped '$' count ({len(parts)-1}): {repr(line)}")
            continue

        # Parts at odd indices (1, 3, 5...) are inside inline math spans $...$
        for math_span in parts[1::2]:
            if not math_span:
                violations.append(f"L{idx}: empty inline math span '$$' on line: {repr(line)}")
            elif math_span.startswith(" "):
                violations.append(f"L{idx}: inline math has leading space: '${math_span}$' in: {repr(line)}")
            elif math_span.endswith(" "):
                violations.append(f"L{idx}: inline math has trailing space: '${math_span}$' in: {repr(line)}")

    assert not violations, f"Inline math violations in BENCHMARKS.md:\n" + "\n".join(violations)


def test_benchmarks_md_currency_escaping() -> None:
    """Assert zero unescaped currency dollar signs in prose ($0.00 -> \\$0.00 or `$0.00`)."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    in_display = False
    violations: list[str] = []

    currency_pattern = re.compile(
        r"(?<!\\)\$\d+(?:\.\d+)?(?:\s*(?:API\s+cost|cloud\s+compute|cost|USD|EUR)\b|[\*\"\'\.,\)]+\s+[a-zA-Z])"
    )

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        clean_line = strip_inline_code(line)
        if clean_line.strip() == "$$":
            in_display = not in_display
            continue
        if in_display:
            continue

        for m in currency_pattern.finditer(clean_line):
            violations.append(f"L{idx}: unescaped currency {repr(m.group(0))} in: {repr(line)}")

    assert not violations, f"Unescaped currency symbols in BENCHMARKS.md:\n" + "\n".join(violations)


def test_benchmarks_md_table_of_contents_anchors_match_headings() -> None:
    """Verify that all Table of Contents anchors correspond to real headings in BENCHMARKS.md."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    headings = get_all_headings(lines)
    heading_slugs = {h[3] for h in headings}

    # Find Table of Contents block
    toc_lines: list[tuple[int, str]] = []
    in_toc = False

    for idx, line in enumerate(lines, start=1):
        if "**Table of Contents**" in line:
            in_toc = True
            continue
        if in_toc:
            if line.strip().startswith(">") and "[" in line and "](#" in line:
                toc_lines.append((idx, line.strip()))
            elif line.strip() == "---" or (not line.startswith(">") and line.strip() != ""):
                # End of TOC block
                break

    assert len(toc_lines) >= 12, f"Expected at least 12 TOC lines, found {len(toc_lines)}"

    anchor_pattern = re.compile(r"\[([^\]]+)\]\(#([^\)]+)\)")
    toc_links: list[tuple[int, str, str]] = []

    for idx, line in toc_lines:
        for m in anchor_pattern.finditer(line):
            title = m.group(1).strip()
            anchor = m.group(2).strip()
            toc_links.append((idx, title, anchor))

    assert len(toc_links) >= 12, f"Expected at least 12 anchor links in TOC, found {len(toc_links)}"

    missing_anchors: list[str] = []
    for idx, title, anchor in toc_links:
        if anchor not in heading_slugs:
            missing_anchors.append(
                f"TOC L{idx}: '{title}' -> anchor '#{anchor}' has no matching heading slug!\n"
            )

    assert not missing_anchors, f"Found broken TOC anchors in BENCHMARKS.md:\n" + "\n".join(missing_anchors)


def test_benchmarks_md_all_internal_anchors_valid() -> None:
    """Assert all internal '#...' anchor links in BENCHMARKS.md map to valid headings."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    headings = get_all_headings(lines)
    heading_slugs = {h[3] for h in headings}

    in_code = False
    violations: list[str] = []
    anchor_pattern = re.compile(r"\[([^\]]+)\]\(#([^\)]+)\)")

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        for m in anchor_pattern.finditer(line):
            title = m.group(1).strip()
            anchor = m.group(2).strip()
            if anchor not in heading_slugs:
                violations.append(f"L{idx}: '[{title}](#{anchor})' has no matching heading in BENCHMARKS.md")

    assert not violations, f"Found broken internal anchor links in BENCHMARKS.md:\n" + "\n".join(violations)


def test_benchmarks_md_relative_file_links_exist() -> None:
    """Assert that every relative file link in BENCHMARKS.md points to an existing file."""
    lines = BENCHMARKS_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    violations: list[str] = []
    file_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)#\s]+)\)")

    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        for m in file_link_pattern.finditer(line):
            title = m.group(1).strip()
            target_path = m.group(2).strip()
            if target_path.startswith("http://") or target_path.startswith("https://"):
                continue
            if target_path.startswith("mailto:"):
                continue

            clean_path = target_path.split("?")[0].split("#")[0]
            if not clean_path:
                continue

            resolved_path = (ROOT / clean_path).resolve()
            if not resolved_path.exists():
                violations.append(f"L{idx}: '[{title}]({target_path})' -> file does not exist: {clean_path}")

    assert not violations, f"Found broken relative file links in BENCHMARKS.md:\n" + "\n".join(violations)
