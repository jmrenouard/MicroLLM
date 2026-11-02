# Universal Project Rule: LaTeX Math Integrity & Markdown Delimiter Standards

This rule applies **without exception** to every documentation file, README, specification, translation pipeline, and AI agent operating in this repository and across all engineering workspaces.

---

## 1. Problem Statement & Root Cause

Markdown parsers (CommonMark, GitHub Flavored Markdown / GFM) and automated text translation/refactoring engines process punctuation before KaTeX/MathJax evaluates mathematical spans. This introduces severe silent corruptions:

1. **Underscore / Subscript Stripping (`_`)**:
   - In Markdown, `_text_` denotes italics.
   - When a LaTeX formula contains multiple subscripts (e.g., `$\arg\max_i p_i$`), Markdown pairs the first and second underscores, converting `_i p_` into `<i>i p</i>` or stripping them during plain-text translation, yielding corrupted expressions like `\arg\max i p_i` or `\sum{i=0}^{N-1}`.
2. **Asterisk / Superscript Stripping (`*`)**:
   - In Markdown, `*text*` denotes italics.
   - Mathematical optimality markers, adjoints, and conjugates (e.g., `k^*`, `|i - k^*|`, `e_{k^*}`) contain asterisks. Markdown pairs asterisks across prose lines, stripping them and leaving truncated symbols: `k^`, `|i - k^|`, and `e_{k^}`.
3. **Display Math Degradation via Indentation (`$$`)**:
   - When `$$...$$` is indented inside lists (e.g., `  $$`) or placed inline without blank lines, CommonMark parses the block as a list continuation paragraph or inline text rather than an isolated block environment. This causes markdown emphasis parsers to evaluate underscores inside equations and triggers KaTeX syntax errors.
4. **Disallowed LaTeX Macros on GitHub (`\operatorname`)**:
   - GitHub's MathJax/KaTeX configuration does not permit `\operatorname{...}` (e.g., `\operatorname{argmax}`, `\operatorname{tokenize}`). Formulas using this macro fail to render or trigger rendering errors on GitHub.

---

## 2. Mandatory Prohibition of `\operatorname` Macro

GitHub's math rendering engine does **NOT** support `\operatorname{...}`.
- **Strictly Prohibited**: Never use `\operatorname{...}` in any Markdown math formulas.
- **GitHub-Supported Alternatives**:
  - For argmax / argmin: Use standard LaTeX built-ins `\arg\max` and `\arg\min` (e.g. `\arg\max_i p_i`, `\arg\min_k \mathcal{L}(k)`).
  - For named functions and tokenizers: Use `\text{...}` (e.g. `\text{tokenize}(x)`, `\text{Var}(X)`, `\text{Softmax}(z)`).
  - Standard built-in math operators (`\max`, `\min`, `\log`, `\exp`, `\cos`, `\sin`, `\det`, `\dim`, `\ker`, etc.) are natively supported and should be written directly.

---

## 3. Mandatory Column-0 Display Math Standards (`$$...$$`)

Display math equations must **ALWAYS** follow the column-0 unindented block standard:
1. Every `$$` delimiter line must start at column 0 (no leading spaces or tabs).
2. Every `$$` delimiter line must be preceded and followed by an empty blank line.
3. Indenting display math inside bullet points or numbered lists is **STRICTLY FORBIDDEN**.

### Correct (Standard Compliant — Standalone Block)
```markdown
Here is the formal definition of score confidence:

$$
\text{conf}(p) = \max\left(0, 1 - \frac{\sum_{i=0}^{N-1} p_i |i - k^*|}{D_{\text{uniform}}}\right)
$$

where $k^* = \arg\max_i p_i$ is the modal option.
```

### Correct (Standard Compliant — Inside List Items)
Even when an equation belongs logically to a bullet point or numbered item, unindent `$$` and its formula to column 0:
```markdown
- **Score Confidence**:
  Let $p \in \Delta^{N-1}$ be the probability distribution:

$$
\text{conf}(p) = \max\left(0, 1 - \frac{\sum_{i=0}^{N-1} p_i |i - k^*|}{D_{\text{uniform}}}\right)
$$

  satisfying boundary conditions $\text{conf}(p) = 0$ for uniform distribution and $1$ for $p = e_{k^*}$.
```

### Strictly Forbidden (Violations)
```markdown
<!-- WRONG: Display math indented inside list item -->
- **Item**:
  $$
  E = mc^2
  $$

<!-- WRONG: Single-line display math without dedicated delimiter lines -->
$$\text{conf}(p) = \max\left(0, 1 - \frac{\sum_{i=0}^{N-1} p_i |i - k^*|}{D_{\text{uniform}}}\right)$$

<!-- WRONG: Display math inline with text -->
The confidence is $$\text{conf}(p) = ...$$ which proves the theorem.

<!-- WRONG: Missing blank lines before or after $$ -->
Paragraph text:
$$
\text{formula}
$$
Following text without empty line.
```

---

## 4. Mandatory Inline Math Standards (`$...$`)

1. **Tight Delimiters (No Internal Edge Whitespace)**:
   - Correct: `$p \in \Delta^{N-1}$`
   - Prohibited: `$ p \in \Delta^{N-1} $` (many Markdown renderers fail to detect math when spaces touch the delimiters).
2. **Emphasis Collision Prevention**:
   - When an inline formula contains multiple underscores or asterisks, prefer `\( ... \)` or ensure subscript braces are explicit:
     - Correct: `$k^* = \arg\max_i p_i$`
     - Prohibited: stripping underscores `\arg\max i` or dropping asterisks `k^`.
3. **Escaping Literal Dollar Signs in Prose**:
   - Literal currency or shell variables must **NEVER** use unescaped `$`:
     - Correct: `\$100` or `` `$100` `` or `` `$VAR` ``
     - Prohibited: `$100` or `$VAR` in prose (causes KaTeX to capture all text up to the next dollar sign as math mode).

---

## 5. Multi-Row LaTeX Environments (`cases`, `aligned`, `matrix`)

Inside display math blocks, row breaks require double backslashes `\\`:
```markdown
$$
\rho_{\text{word}}(x, c) = \begin{cases}
1.0 & \text{if } c \in \mathcal{T}(x) \\
0.0 & \text{otherwise}
\end{cases}
$$
```
Never compress multi-row environments onto single lines without verifying that double backslashes `\\` are preserved.

---

## 6. Agent Instructions for Translation, Formatting & Refactoring

Every agent (human or AI) performing translation, editing, or refactoring on Markdown files must obey these rules:

1. **Math Spans Are Immutable Tokens**:
   - Content between `$ ... $`, `$$ ... $$`, `\( ... \)`, and `\[ ... \]` is strictly immutable.
   - Never translate, paraphrase, "correct", or strip symbols within math delimiters.
2. **Masking Before Processing**:
   - When passing Markdown documents to translation pipelines or LLM prompts, replace all `$ ... $` and `$$ ... $$` blocks with opaque placeholder tokens (e.g., `MATH_SPAN_001`, `MATH_SPAN_002`), run the text translation, and re-inject the exact original math spans into the translated document.
3. **Automated Verification**:
   - After modifying any `.md` file containing math, run the regression test suite:
     ```bash
     pytest tests/test_docs_math_syntax.py -v
     ```
   - No Pull Request may be opened or merged if math tokens are corrupted, display math blocks are indented, or disallowed macros like `\operatorname` are present.
