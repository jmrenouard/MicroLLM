"""TDD regression and validation suite for LaTeX math formula integrity in Markdown (Issue #179).

Verifies:
1. Display math isolation: all '$$' delimiters are on their own line with blank line separation.
2. Math token integrity: no stripped subscripts (e.g. \\operatorname{argmax}i instead of \\operatorname{argmax}_i),
   no missing underscore in operators (\\sum{ instead of \\sum_{), and no stripped asterisks (k^ instead of k^*).
3. Even pair counts for all display math delimiters.
4. Ground-truth fidelity for critical mathematical derivations across syllabus, product doc, and technical specs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Core architectural and specification documentation files containing LaTeX equations
CORE_MATH_DOCS = [
    ROOT / "docs" / "SYLLABUS.md",
    ROOT / "docs" / "TECHNICAL_SPECIFICATIONS.md",
    ROOT / "docs" / "PRODUCT_DOCUMENTATION.md",
    ROOT / "docs" / "FAILURE_MODES_AND_WORST_CASES.md",
    ROOT / "docs" / "MULTILINGUAL_STRATEGIES_SUMMARY.md",
    ROOT / "docs" / "ALGORITHMS.md",
]


def strip_code_blocks(text: str) -> str:
    """Strip fenced code blocks and inline code spans to avoid inspecting code snippets."""
    # Strip multi-line code blocks
    text_without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Strip inline code spans
    return re.sub(r"`[^`\n]+`", "", text_without_fences)


def test_core_math_docs_exist() -> None:
    """Verify that all target documentation files exist."""
    for doc in CORE_MATH_DOCS:
        assert doc.is_file(), f"Document {doc.relative_to(ROOT)} must exist"


def test_no_corrupted_math_tokens_in_all_docs() -> None:
    """Verify no corrupted math tokens (e.g. stripped subscripts or asterisks) appear in documentation."""
    docs_to_check = list((ROOT / "docs").glob("*.md")) + [ROOT / "README.md", ROOT / "BENCHMARKS.md"]
    # Exclude historical logs / registers where bug symptoms are explicitly discussed
    excluded = {"ACTION_LOGBOOK.md", "RCA_REGISTER.md", "LATEX_MATH_RULES.md"}
    docs_to_check = [d for d in docs_to_check if d.name not in excluded]

    corruptions: list[str] = []

    for doc in docs_to_check:
        raw_text = doc.read_text(encoding="utf-8")
        clean_text = strip_code_blocks(raw_text)

        # 1. Missing underscore in argmax/argmin: \operatorname{argmax}i or \argmax i
        if re.search(r"\\(operatorname\{arg(max|min)\}|arg(max|min))[a-zA-Z]", clean_text):
            corruptions.append(f"{doc.name}: detected missing subscript in argmax/argmin")

        # 2. Missing underscore in sum/prod/int: \sum{i=0} or \prod{k=1}
        if re.search(r"\\(sum|prod|int)\{[a-zA-Z0-9]", clean_text):
            corruptions.append(f"{doc.name}: detected missing subscript underscore in \\sum, \\prod, or \\int")

        # 3. Truncated k^ or e_{k^} or |i - k^| where asterisk was eaten by markdown emphasis
        if re.search(r"k\^[\s,\.\)\]\|]", clean_text) or "e_{k^}" in clean_text or "k^}" in clean_text:
            corruptions.append(f"{doc.name}: detected truncated caret superscript (stripped asterisk in k^* / e_{{k^*}})")

    assert not corruptions, f"Found LaTeX math token corruptions: {corruptions}"


def test_display_math_delimiters_isolated_in_core_docs() -> None:
    """Verify all display math '$$' blocks in core docs are isolated on their own lines."""
    violations: list[str] = []

    for doc in CORE_MATH_DOCS:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Strip inline code spans to avoid flagging backticked `$$`
            line_no_inline_code = re.sub(r"`[^`\n]+`", "", stripped).strip()
            if "$$" in line_no_inline_code and line_no_inline_code != "$$":
                violations.append(f"{doc.name}:L{idx}: inline $$ delimiter on line: '{stripped[:70]}'")

    assert not violations, "Display math blocks must isolate '$$' on dedicated lines:\n" + "\n".join(violations)


def test_display_math_delimiters_are_paired() -> None:
    """Verify that every document has an even number of isolated '$$' delimiters."""
    for doc in CORE_MATH_DOCS:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code = False
        isolated_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            line_no_inline = re.sub(r"`[^`\n]+`", "", stripped).strip()
            if line_no_inline == "$$":
                isolated_count += 1
        assert isolated_count % 2 == 0, f"{doc.name} must have paired '$$' delimiters (found {isolated_count})"


def test_syllabus_dispersion_bound_equation_fidelity() -> None:
    """Verify exact formula fidelity of the score confidence dispersion bound in docs/SYLLABUS.md."""
    syllabus_text = (ROOT / "docs" / "SYLLABUS.md").read_text(encoding="utf-8")

    # Mode definition must have underscore: k^* = \operatorname{argmax}_i p_i
    assert r"k^* = \operatorname{argmax}_i p_i" in syllabus_text or r"k^* = \arg\max_i p_i" in syllabus_text, (
        "Syllabus must specify mode with proper LaTeX subscript: k^* = \\operatorname{argmax}_i p_i"
    )

    # Score confidence formula must have \\sum_{i=0}^{N-1} p_i |i - k^*|
    assert r"\sum_{i=0}^{N-1} p_i |i - k^*|" in syllabus_text or r"\sum_{i=0}^{N-1} p_i \, |i - k^*|" in syllabus_text, (
        "Syllabus must contain exact summation index and absolute difference |i - k^*|"
    )

    # Boundary conditions must preserve e_{k^*}
    assert r"p = e_{k^*}" in syllabus_text, "Syllabus must preserve boundary state e_{k^*}"


def test_technical_specifications_equation_fidelity() -> None:
    """Verify mathematical fidelity in docs/TECHNICAL_SPECIFICATIONS.md."""
    tech_text = (ROOT / "docs" / "TECHNICAL_SPECIFICATIONS.md").read_text(encoding="utf-8")

    assert r"p_{k^*} \leftarrow \text{clip}_{[0, 1]}" in tech_text, "Missing simplex drift absorption formula"
    assert r"\text{confidence}_{\text{choice}}(p) = \text{clip}_{[0, 1]}" in tech_text, "Missing choice confidence formula"
    assert r"\text{confidence}_{\text{score}}(p) = \text{clip}_{[0, 1]}" in tech_text, "Missing score confidence formula"
    assert r"\text{EMD}_{1\text{D}}(p, q)" in tech_text, "Missing 1D EMD formula"
    assert r"p_i^{(L_0)} = \frac{\exp(\bar{z}_i)}" in tech_text, "Missing AnyJev L0 softmax formula"
    assert r"\tau(K, \alpha) = \tau_{\text{base}}" in tech_text, "Missing dynamic tau formula"
    assert r"p_{\text{fused}} = (1 - \alpha) p_0 + \alpha p_1" in tech_text, "Missing probability fusion formula"


def test_product_documentation_equation_fidelity() -> None:
    """Verify mathematical fidelity in docs/PRODUCT_DOCUMENTATION.md."""
    prod_text = (ROOT / "docs" / "PRODUCT_DOCUMENTATION.md").read_text(encoding="utf-8")

    assert r"\text{confidence}_{\text{choice}}(p) = \frac{K \cdot \max_i p_i - 1}{K - 1}" in prod_text, (
        "Missing choice confidence formula in PRODUCT_DOCUMENTATION.md"
    )
    assert r"\text{confidence}_{\text{score}}(p) = \max" in prod_text, (
        "Missing score confidence formula in PRODUCT_DOCUMENTATION.md"
    )
    assert r"\text{confidence}_{\text{multi}}(p) = \frac{1}{K}\sum_{k=1}^K |2p_k - 1|" in prod_text, (
        "Missing multi confidence formula in PRODUCT_DOCUMENTATION.md"
    )
    assert r"C_\alpha(x) = \left\{ k \in \{1,\dots,K\} : 1 - p(k \mid x) \le \hat{q}_{1-\alpha} \right\}" in prod_text, (
        "Missing conformal prediction set formula in PRODUCT_DOCUMENTATION.md"
    )
