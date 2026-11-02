"""TDD regression and validation suite for LaTeX math formula integrity in Markdown (Issues #179 & #181).

Verifies:
1. Ban on disallowed LaTeX macro \\operatorname: zero occurrences in math or prose across all markdown files.
2. Display math isolation & column-0 unindent: all '$$' delimiters are unindented at column 0 and on dedicated lines.
3. Math token integrity: no stripped subscripts (e.g. \\arg\\max i instead of \\arg\\max_i),
   no missing underscore in operators (\\sum{ instead of \\sum_{), and no stripped asterisks (k^ instead of k^*).
4. Even pair counts for all display math delimiters.
5. Ground-truth fidelity for critical mathematical derivations across syllabus, product doc, and technical specs,
   including Computational Complexity Divergence subscripts.
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


def strip_inline_code(line: str) -> str:
    """Strip inline code spans, supporting double-backtick code escapes."""
    line = re.sub(r"``.*?``", "", line)
    return re.sub(r"`[^`\n]+`", "", line)


def strip_code_blocks(text: str) -> str:
    """Strip fenced code blocks and inline code spans to avoid inspecting code snippets."""
    # Strip multi-line code blocks
    text_without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Strip inline code spans (including double backticks)
    return strip_inline_code(text_without_fences)


def get_all_repository_markdown_files() -> list[Path]:
    """Retrieve all tracked repository markdown files, excluding git internals, cache, and teamwork scratchpads."""
    return sorted([
        p for p in ROOT.rglob("*.md")
        if ".git" not in p.parts
        and ".pytest_cache" not in p.parts
        and not (len(p.relative_to(ROOT).parts) >= 2 and p.relative_to(ROOT).parts[:2] == (".agents", "teamwork"))
    ])


def test_core_math_docs_exist() -> None:
    """Verify that all target documentation files exist."""
    for doc in CORE_MATH_DOCS:
        assert doc.is_file(), f"Document {doc.relative_to(ROOT)} must exist"


def test_no_disallowed_operatorname_macro_in_any_md_file() -> None:
    """Assert zero occurrences of \\operatorname in math formulas or prose across all markdown files (R1.1, R3.1).

    GitHub's MathJax/KaTeX configuration does not permit \\operatorname{...}.
    All occurrences must be replaced by \\arg\\max, \\arg\\min, or \\text{...}.
    """
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            no_inline_code = strip_inline_code(line)
            if r"\operatorname" in no_inline_code:
                violations.append(f"{doc.relative_to(ROOT)}:L{idx}: '{line.strip()[:70]}'")

    assert not violations, (
        f"Found disallowed \\operatorname macros (must use \\arg\\max, \\arg\\min, or \\text{{...}}):\n"
        + "\n".join(violations)
    )


def test_zero_indented_display_math_delimiters_in_all_md_files() -> None:
    """Assert zero indented '$$' delimiter lines across all repository markdown files (R1.2, R3.2).

    All display math '$$' delimiters must start at column 0 (unindented) and be surrounded
    by blank lines, preventing GitHub CommonMark list-item inline collapse and underscore stripping.
    """
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # Strip inline code spans to avoid inspecting backticked code
            line_no_inline_code = strip_inline_code(line)
            # Check if line contains $$ and is indented with spaces or tabs
            if "$$" in line_no_inline_code and (line.startswith(" ") or line.startswith("\t")):
                violations.append(f"{doc.relative_to(ROOT)}:L{idx}: indented '$$' delimiter: '{line}'")

    assert not violations, (
        f"Found indented '$$' delimiters (must be column 0 unindented):\n"
        + "\n".join(violations)
    )


def test_no_corrupted_math_tokens_in_all_docs() -> None:
    """Verify no corrupted math tokens (e.g. stripped subscripts or asterisks) appear in documentation (R3.3)."""
    all_files = get_all_repository_markdown_files()
    # Exclude historical logs / registers where bug symptoms are explicitly discussed
    excluded = {"ACTION_LOGBOOK.md", "RCA_REGISTER.md", "LATEX_MATH_RULES.md"}
    docs_to_check = [d for d in all_files if d.name not in excluded]

    corruptions: list[str] = []

    for doc in docs_to_check:
        raw_text = doc.read_text(encoding="utf-8")
        clean_text = strip_code_blocks(raw_text)

        # 1. Missing underscore in argmax/argmin: \argmax i or \arg\max i or \arg\max \theta
        if re.search(r"\\arg\\?(?:max|min)(?:\s+[a-zA-Z\\]|(?<!\\)[a-zA-Z])", clean_text):
            corruptions.append(f"{doc.relative_to(ROOT)}: detected missing subscript underscore in argmax/argmin")

        # 2. Missing underscore in sum/prod/int: \sum{i=0} or \prod{k=1}
        if re.search(r"\\(sum|prod|int)\{[a-zA-Z0-9]", clean_text):
            corruptions.append(f"{doc.relative_to(ROOT)}: detected missing subscript underscore in \\sum, \\prod, or \\int")

        # 3. Missing subscript index on simplex sum (\sum p_i or \sum w_i without index)
        if re.search(r"\\sum\s+[pw]_i", clean_text):
            corruptions.append(f"{doc.relative_to(ROOT)}: detected missing subscript index on simplex sum (\\sum_i required)")

        # 4. Truncated caret superscript (stripped asterisk in k^* / e_{k^*})
        if re.search(r"[a-zA-Z]\^[\s,\.\)\]\|]", clean_text) or "e_{k^}" in clean_text or "k^}" in clean_text:
            corruptions.append(f"{doc.relative_to(ROOT)}: detected truncated caret superscript (stripped asterisk)")

    assert not corruptions, f"Found LaTeX math token corruptions:\n" + "\n".join(corruptions)


def test_inline_math_delimiters_are_paired_in_all_md_files() -> None:
    """Assert all lines outside code blocks have paired unescaped single '$' delimiters (R1.1, R3.3).

    Unpaired '$' in markdown causes GitHub/KaTeX to capture all subsequent prose as math mode,
    corrupting rendering across paragraphs and inverting subsequent math expressions.
    """
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        in_display_math = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            no_inline_code = strip_inline_code(line)
            if no_inline_code.strip() == "$$":
                in_display_math = not in_display_math
                continue
            if in_display_math:
                continue

            no_display = re.sub(r"\$\$", "", no_inline_code)
            dollars = re.findall(r"(?<!\\)\$", no_display)
            if len(dollars) % 2 != 0:
                violations.append(
                    f"{doc.relative_to(ROOT)}:L{idx}: odd count of unescaped '$' ({len(dollars)}) in line: '{line[:70]}'"
                )

    assert not violations, (
        "Found lines with unpaired unescaped single '$' delimiters (must escape with \\$ or pair math delimiters):\n"
        + "\n".join(violations)
    )


def test_no_unescaped_currency_in_prose_in_all_md_files() -> None:
    """Assert zero unescaped currency dollar signs in prose across all repository documentation (R1.1, R2.1).

    Section 4 Rule 3 of .agents/rules/latex-math-in-markdown.md strictly forbids unescaped currency
    symbols like '$0.00' or '$50' in prose because KaTeX treats them as math mode opening delimiters.
    They must be escaped ('\\$0.00', '\\$50') or wrapped in inline code (`$50`).
    """
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    # Currency pattern: unescaped $ followed by digits followed by currency terms (cost, API, cloud, USD) or sentence punctuation/words
    currency_pattern = re.compile(
        r"(?<!\\)\$\d+(?:\.\d+)?(?:\s*(?:API\s+cost|cloud\s+compute|cost|USD|EUR)\b|[\*\"'\.,\)]+\s+[a-zA-Z])"
    )

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        in_display_math = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            no_inline_code = strip_inline_code(line)
            if no_inline_code.strip() == "$$":
                in_display_math = not in_display_math
                continue
            if in_display_math:
                continue

            for m in currency_pattern.finditer(no_inline_code):
                violations.append(
                    f"{doc.relative_to(ROOT)}:L{idx}: unescaped currency in prose: '{m.group(0)}' (must be '\\${m.group(0)[1:]}' or backticked): '{line[:70]}'"
                )

    assert not violations, (
        "Found unescaped currency symbols in prose:\n"
        + "\n".join(violations)
    )


def test_display_math_blocks_have_balanced_braces_and_environments() -> None:
    """Assert all display math blocks across all md files have balanced braces, \\left/\\right, and \\begin/\\end."""
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        in_math_block = False
        current_block: list[str] = []
        block_start_idx = 0

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            line_no_inline = strip_inline_code(line).strip()
            if line_no_inline == "$$":
                if not in_math_block:
                    in_math_block = True
                    block_start_idx = idx
                    current_block = []
                else:
                    in_math_block = False
                    block_str = "\n".join(current_block)

                    # Balanced braces
                    if block_str.count("{") != block_str.count("}"):
                        violations.append(
                            f"{doc.relative_to(ROOT)}:L{block_start_idx}-{idx}: unbalanced braces ({block_str.count('{')} open vs {block_str.count('}')} close)"
                        )

                    # Balanced \left and \right
                    lefts = len(re.findall(r"\\left(?![a-zA-Z])", block_str))
                    rights = len(re.findall(r"\\right(?![a-zA-Z])", block_str))
                    if lefts != rights:
                        violations.append(
                            f"{doc.relative_to(ROOT)}:L{block_start_idx}-{idx}: unbalanced \\left ({lefts}) and \\right ({rights})"
                        )

                    # Balanced \begin{...} and \end{...}
                    begins = re.findall(r"\\begin\{([^}]+)\}", block_str)
                    ends = re.findall(r"\\end\{([^}]+)\}", block_str)
                    if begins != ends:
                        violations.append(
                            f"{doc.relative_to(ROOT)}:L{block_start_idx}-{idx}: mismatched environments ({begins} vs {ends})"
                        )
            elif in_math_block:
                current_block.append(line)

    assert not violations, (
        "Found display math blocks with unbalanced braces or environments:\n"
        + "\n".join(violations)
    )


def test_display_math_delimiters_isolated_in_all_md_files() -> None:
    """Verify all display math '$$' blocks in all repository md files are isolated on column-0 dedicated lines."""
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
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
            line_no_inline_code = strip_inline_code(line).strip()
            if "$$" in line_no_inline_code and line != "$$":
                violations.append(f"{doc.relative_to(ROOT)}:L{idx}: unisolated or indented $$ delimiter on line: '{line[:70]}'")

    assert not violations, "Display math blocks must isolate '$$' on column-0 dedicated lines:\n" + "\n".join(violations)


def test_display_math_delimiters_are_paired_in_all_md_files() -> None:
    """Verify that every repository markdown document has an even number of isolated '$$' delimiters."""
    all_md_files = get_all_repository_markdown_files()
    for doc in all_md_files:
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
            line_no_inline = strip_inline_code(line).strip()
            if line_no_inline == "$$":
                isolated_count += 1
        assert isolated_count % 2 == 0, f"{doc.relative_to(ROOT)} must have paired '$$' delimiters (found {isolated_count})"


def test_display_math_blocks_have_surrounding_blank_lines_in_all_md_files() -> None:
    """Assert all display math '$$' blocks across all repository md files are surrounded by blank lines (R1.2, R2.2).

    Every opening '$$' must have a preceding blank line, and every closing '$$' must have
    a following blank line, ensuring GitHub CommonMark isolates block equations from list items and text.
    """
    all_md_files = get_all_repository_markdown_files()
    violations: list[str] = []

    for doc in all_md_files:
        lines = doc.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        in_math_block = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            line_no_inline = strip_inline_code(line).strip()
            if line_no_inline == "$$":
                if not in_math_block:
                    # Opening $$: preceding line must be blank
                    if idx > 1:
                        prev_line = lines[idx - 2].strip()
                        if prev_line != "":
                            violations.append(
                                f"{doc.relative_to(ROOT)}:L{idx}: opening $$ missing preceding blank line (prev: '{prev_line[:50]}')"
                            )
                    in_math_block = True
                else:
                    # Closing $$: following line must be blank
                    if idx < len(lines):
                        next_line = lines[idx].strip()
                        if next_line != "":
                            violations.append(
                                f"{doc.relative_to(ROOT)}:L{idx}: closing $$ missing following blank line (next: '{next_line[:50]}')"
                            )
                    in_math_block = False

    assert not violations, (
        f"Found display math blocks without surrounding blank lines:\n"
        + "\n".join(violations)
    )


def test_computational_complexity_divergence_equation_fidelity() -> None:
    """Verify exact formula and subscript fidelity of Computational Complexity Divergence in docs/SYLLABUS.md (R1.3)."""
    syllabus_text = (ROOT / "docs" / "SYLLABUS.md").read_text(encoding="utf-8")

    assert r"\text{Complexity}_{\text{System-2}} = \mathcal{O}\left(N_{\text{layers}} \cdot L_{\text{gen}} \cdot d_{\text{model}}^2 + L_{\text{ctx}} \cdot L_{\text{gen}} \cdot d_{\text{model}}\right)" in syllabus_text, (
        "Syllabus must contain intact System-2 complexity formula with preserved subscripts"
    )
    assert r"\text{Complexity}_{\text{System-1}} = \mathcal{O}\left(L_{\text{prompt}} \cdot D + K \cdot D\right)" in syllabus_text, (
        "Syllabus must contain intact System-1 complexity formula with preserved subscripts"
    )


def test_syllabus_dispersion_bound_equation_fidelity() -> None:
    """Verify exact formula fidelity of the score confidence dispersion bound in docs/SYLLABUS.md."""
    syllabus_text = (ROOT / "docs" / "SYLLABUS.md").read_text(encoding="utf-8")

    # Mode definition must use \\arg\\max with underscore: k^* = \\arg\\max_i p_i
    assert r"k^* = \arg\max_i p_i" in syllabus_text, (
        "Syllabus must specify mode with proper LaTeX subscript using \\arg\\max: k^* = \\arg\\max_i p_i"
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


def test_algorithms_documentation_equation_fidelity() -> None:
    """Verify mathematical fidelity in docs/ALGORITHMS.md."""
    algo_text = (ROOT / "docs" / "ALGORITHMS.md").read_text(encoding="utf-8")

    assert r"\mathcal{T}(x) = \text{tokenize}(x) = [t_1, t_2, \dots, t_M]" in algo_text, (
        "Missing multilingual tokenization equation in docs/ALGORITHMS.md"
    )
    assert r"\vec{v} = [\vec{v}_{\text{concepts}} \in \mathbb{R}^{16} \;\Vert\; \vec{v}_{\text{hashes}} \in \mathbb{R}^{D - 16}]" in algo_text, (
        "Missing feature space partitioning equation in docs/ALGORITHMS.md"
    )
    assert r"\vec{v}_{\text{hashes}}[idx_{\text{word}}] \mathrel{+}= 1.6 \cdot \rho_{\text{word}}" in algo_text, (
        "Missing word hash accumulation equation in docs/ALGORITHMS.md"
    )
