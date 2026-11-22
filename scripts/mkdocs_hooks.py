"""MkDocs Material build hooks for MicroLLM documentation publishing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import posixpath
import re
import shutil
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from mkdocs.structure.files import Files

log = logging.getLogger("mkdocs.hooks.microllm")

DEFAULT_REPO_URL = "https://github.com/jmrenouard/MicroLLM"
ROOT_VIRTUAL_DOCS: tuple[str, ...] = ("BENCHMARKS.md", "CHANGELOG.md")
QUADRANT_HISTORICAL_FALLBACKS: dict[str, str] = {
    "01-tutorials": "SYLLABUS.md",
    "02-how-to-guides": "PRODUCT_DOCUMENTATION.md",
    "03-reference": "TECHNICAL_SPECIFICATIONS.md",
    "04-explanation": "README.md",
}

GITHUB_ALERT_MAP: dict[str, str] = {
    "NOTE": "note",
    "TIP": "tip",
    "IMPORTANT": "info",
    "WARNING": "warning",
    "CAUTION": "danger",
}

_FENCE_OPEN_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
_ALERT_HEADER_RE = re.compile(
    r"^([ \t]*)>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*(.*)$",
    re.IGNORECASE,
)
_BLOCKQUOTE_LINE_RE = re.compile(r"^[ \t]*>[ \t]?(.*)$")
_RELEASE_NOTES_SEMVER_RE = re.compile(r"^RELEASE_NOTES_v?(\d+)\.(\d+)\.(\d+)\.md$")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_EXCLUDED_TARGET_RE = re.compile(r"(?i)(outreach|\.(?:kdbx|keyx|pdf)$)")


def _effective_repo_ref(raw_ref: str | None = None) -> str:
    """Resolve the Git ref used for GitHub source links."""
    ref = (raw_ref if raw_ref is not None else os.environ.get("MICROLLM_DOCS_REF", "main")).strip()
    if not ref or ref in {"WORKTREE", "HEAD", "origin/main"}:
        return "main"
    return ref


def _is_fence_close(line: str, fence_char: str, fence_len: int) -> bool:
    """Return True when a line closes the active fenced code block."""
    stripped = line.lstrip(" \t")
    if not stripped.startswith(fence_char):
        return False
    count = 0
    while count < len(stripped) and stripped[count] == fence_char:
        count += 1
    return count >= fence_len and stripped[count:].strip() == ""


def _split_fenced_blocks(markdown: str) -> list[tuple[bool, str]]:
    """Split Markdown into `(is_code_fence, text)` chunks preserving exact newlines."""
    chunks: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in markdown.splitlines(keepends=True):
        if not in_fence:
            match = _FENCE_OPEN_RE.match(line.rstrip("\r\n"))
            if match:
                if buffer:
                    chunks.append((False, "".join(buffer)))
                    buffer.clear()
                in_fence = True
                fence_token = match.group(2)
                fence_char = fence_token[0]
                fence_len = len(fence_token)
                buffer.append(line)
                continue
            buffer.append(line)
        else:
            buffer.append(line)
            if _is_fence_close(line.rstrip("\r\n"), fence_char, fence_len):
                chunks.append((True, "".join(buffer)))
                buffer.clear()
                in_fence = False
                fence_char = ""
                fence_len = 0

    if buffer:
        chunks.append((in_fence, "".join(buffer)))
    return chunks


def convert_github_alerts(markdown: str) -> str:
    """Convert GitHub-style blockquote alerts into Material admonitions outside code fences."""
    result: list[str] = []
    for is_fence, chunk in _split_fenced_blocks(markdown):
        if is_fence:
            result.append(chunk)
            continue

        lines = chunk.splitlines(keepends=True)
        out_lines: list[str] = []
        idx = 0
        while idx < len(lines):
            raw_line = lines[idx]
            line_no_nl = raw_line.rstrip("\r\n")
            newline = raw_line[len(line_no_nl) :]
            header_match = _ALERT_HEADER_RE.match(line_no_nl)
            if not header_match:
                out_lines.append(raw_line)
                idx += 1
                continue

            lead_ws = header_match.group(1)
            alert_key = header_match.group(2).upper()
            custom_title = header_match.group(3).strip()
            admon_type = GITHUB_ALERT_MAP[alert_key]
            header = (
                f'{lead_ws}!!! {admon_type} "{custom_title}"'
                if custom_title
                else f"{lead_ws}!!! {admon_type}"
            )
            out_lines.append(f"{header}{newline or chr(10)}")
            idx += 1

            while idx < len(lines):
                body_raw = lines[idx]
                body_no_nl = body_raw.rstrip("\r\n")
                body_nl = body_raw[len(body_no_nl) :]
                bq_match = _BLOCKQUOTE_LINE_RE.match(body_no_nl)
                if not bq_match:
                    break
                inner = bq_match.group(1)
                if inner.strip() == "":
                    out_lines.append(f"{body_nl or chr(10)}")
                else:
                    out_lines.append(f"{lead_ws}    {inner}{body_nl}")
                idx += 1

        result.append("".join(out_lines))
    return "".join(result)


def _is_escaped(text: str, pos: int) -> bool:
    """Return True if the character at `pos` is escaped by an odd number of backslashes."""
    backslashes = 0
    cursor = pos - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return (backslashes % 2) == 1


def _find_matching_bracket(text: str, open_idx: int) -> int | None:
    """Find matching `]` for `[` at `open_idx`, skipping inline code and escaped chars."""
    depth = 1
    idx = open_idx + 1
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch == "`":
            tick_len = 1
            while idx + tick_len < n and text[idx + tick_len] == "`":
                tick_len += 1
            close_pos = text.find("`" * tick_len, idx + tick_len)
            if close_pos == -1:
                return None
            idx = close_pos + tick_len
            continue
        if ch == "[" and not _is_escaped(text, idx):
            depth += 1
        elif ch == "]" and not _is_escaped(text, idx):
            depth -= 1
            if depth == 0:
                return idx
        elif ch == "\n" and idx + 1 < n and text[idx + 1] == "\n":
            return None
        idx += 1
    return None


def _find_matching_paren(text: str, open_idx: int) -> int | None:
    """Find matching `)` for `(` at `open_idx`, balancing nested parentheses."""
    depth = 1
    idx = open_idx + 1
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch == "\n":
            return None
        if ch == "(" and not _is_escaped(text, idx):
            depth += 1
        elif ch == ")" and not _is_escaped(text, idx):
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return None


def _split_destination_and_title(raw_dest: str) -> tuple[str, str, bool]:
    """Split `(url \"title\")` into `(url, title_suffix, had_angle_brackets)`."""
    stripped = raw_dest.strip()
    if not stripped:
        return "", "", False

    had_angles = False
    if stripped.startswith("<"):
        end_angle = stripped.find(">")
        if end_angle != -1:
            had_angles = True
            url_part = stripped[1:end_angle].strip()
            title_suffix = stripped[end_angle + 1 :]
            return url_part, title_suffix, had_angles

    # Split optional title separated by whitespace
    for idx, ch in enumerate(stripped):
        if ch.isspace():
            url_part = stripped[:idx]
            title_suffix = stripped[idx:]
            return url_part, title_suffix, had_angles
    return stripped, "", had_angles


def _resolve_link_target(
    raw_url: str,
    page_src_path: str,
    ref: str,
    repo_url: str,
    available_root_docs: frozenset[str],
    docs_dir: Path | None = None,
) -> tuple[str | None, bool]:
    """Compute rewritten target URL or `(None, True)` when excluded/neutralized."""
    if not raw_url or raw_url.startswith("#") or _SCHEME_RE.match(raw_url):
        return raw_url, False
    if raw_url.startswith(("/", "\\")):
        return raw_url, False

    if "#" in raw_url:
        path_part, anchor_part = raw_url.split("#", 1)
        anchor = f"#{anchor_part}"
    else:
        path_part = raw_url
        anchor = ""

    if not path_part:
        return raw_url, False

    norm_src = page_src_path.replace("\\", "/").lstrip("/")
    is_root_virtual = norm_src in ROOT_VIRTUAL_DOCS
    source_repo_dir = "" if is_root_virtual else posixpath.dirname(f"docs/{norm_src}")

    had_trailing_slash = path_part.endswith("/")
    repo_rel = (
        posixpath.normpath(posixpath.join(source_repo_dir, path_part))
        if source_repo_dir
        else posixpath.normpath(path_part)
    )

    if _EXCLUDED_TARGET_RE.search(repo_rel):
        return None, True

    depth = norm_src.count("/")
    prefix = "../" * depth
    clean_repo_url = repo_url.rstrip("/")

    if repo_rel == "README.md":
        return f"{prefix}index.md{anchor}", False

    if repo_rel in ROOT_VIRTUAL_DOCS:
        if repo_rel in available_root_docs:
            return f"{prefix}{repo_rel}{anchor}", False
        return f"{clean_repo_url}/blob/{ref}/{repo_rel}{anchor}", False

    if repo_rel == "docs/index.html":
        return f"{prefix}simulator.html{anchor}", False

    if repo_rel == "docs/slides.html":
        return f"{prefix}slides.html{anchor}", False

    if repo_rel.startswith("docs/"):
        doc_subpath = repo_rel[len("docs/") :]
        if (
            doc_subpath in ROOT_VIRTUAL_DOCS
            and (docs_dir is None or not (docs_dir / doc_subpath).is_file())
        ):
            if doc_subpath in available_root_docs:
                return f"{prefix}{doc_subpath}{anchor}", False
            return f"{clean_repo_url}/blob/{ref}/{doc_subpath}{anchor}", False

        if docs_dir is not None:
            top_segment = doc_subpath.split("/", 1)[0]
            if (
                top_segment in QUADRANT_HISTORICAL_FALLBACKS
                and not (docs_dir / top_segment).is_dir()
            ):
                fallback_doc = QUADRANT_HISTORICAL_FALLBACKS[top_segment]
                if (docs_dir / fallback_doc).is_file():
                    return f"{prefix}{fallback_doc}", False

            candidate = docs_dir / doc_subpath
            if candidate.is_dir() and (candidate / "index.md").is_file():
                index_subpath = posixpath.join(doc_subpath.rstrip("/"), "index.md")
                return f"{prefix}{index_subpath}{anchor}", False

        trailing = "/" if had_trailing_slash and doc_subpath and not doc_subpath.endswith("/") else ""
        return f"{prefix}{doc_subpath}{trailing}{anchor}", False

    if repo_rel == "." or repo_rel == ".." or repo_rel.startswith("../"):
        return clean_repo_url, False

    trailing = "/" if had_trailing_slash and not repo_rel.endswith("/") else ""
    return f"{clean_repo_url}/blob/{ref}/{repo_rel}{trailing}{anchor}", False


def _rewrite_prose_segment(
    text: str,
    page_src_path: str,
    ref: str,
    repo_url: str,
    available_root_docs: frozenset[str],
    docs_dir: Path | None = None,
) -> str:
    """Rewrite Markdown links in a non-fenced text segment while preserving inline code."""
    out: list[str] = []
    idx = 0
    n = len(text)

    while idx < n:
        ch = text[idx]
        if ch == "`":
            tick_len = 1
            while idx + tick_len < n and text[idx + tick_len] == "`":
                tick_len += 1
            close_pos = text.find("`" * tick_len, idx + tick_len)
            if close_pos == -1:
                out.append(text[idx:])
                break
            out.append(text[idx : close_pos + tick_len])
            idx = close_pos + tick_len
            continue

        is_image = ch == "!" and idx + 1 < n and text[idx + 1] == "[" and not _is_escaped(text, idx)
        bracket_start = idx + 1 if is_image else (idx if ch == "[" and not _is_escaped(text, idx) else -1)
        if bracket_start != -1:
            bracket_end = _find_matching_bracket(text, bracket_start)
            if bracket_end is not None and bracket_end + 1 < n and text[bracket_end + 1] == "(":
                paren_end = _find_matching_paren(text, bracket_end + 1)
                if paren_end is not None:
                    raw_label = text[bracket_start + 1 : bracket_end]
                    rewritten_label = _rewrite_prose_segment(
                        raw_label,
                        page_src_path,
                        ref,
                        repo_url,
                        available_root_docs,
                        docs_dir=docs_dir,
                    )
                    raw_dest = text[bracket_end + 2 : paren_end]
                    url_part, title_suffix, had_angles = _split_destination_and_title(raw_dest)
                    new_url, neutralized = _resolve_link_target(
                        url_part,
                        page_src_path=page_src_path,
                        ref=ref,
                        repo_url=repo_url,
                        available_root_docs=available_root_docs,
                        docs_dir=docs_dir,
                    )
                    if neutralized:
                        out.append("" if is_image else rewritten_label)
                    else:
                        formatted_url = f"<{new_url}>" if had_angles else str(new_url)
                        bang = "!" if is_image else ""
                        out.append(f"{bang}[{rewritten_label}]({formatted_url}{title_suffix})")
                    idx = paren_end + 1
                    continue

        out.append(ch)
        idx += 1

    return "".join(out)


def rewrite_relative_links(
    markdown: str,
    page_src_path: str,
    ref: str = "main",
    repo_url: str = DEFAULT_REPO_URL,
    available_root_docs: Iterable[str] | None = None,
    docs_dir: Path | None = None,
) -> str:
    """Rewrite repository-relative links for MkDocs site publishing outside fenced code blocks."""
    root_docs = (
        frozenset(ROOT_VIRTUAL_DOCS)
        if available_root_docs is None
        else frozenset(available_root_docs)
    )
    resolved_ref = _effective_repo_ref(ref)
    resolved_docs_dir = Path(docs_dir) if docs_dir is not None else None
    result: list[str] = []
    for is_fence, chunk in _split_fenced_blocks(markdown):
        if is_fence:
            result.append(chunk)
        else:
            result.append(
                _rewrite_prose_segment(
                    chunk,
                    page_src_path=page_src_path,
                    ref=resolved_ref,
                    repo_url=repo_url,
                    available_root_docs=root_docs,
                    docs_dir=resolved_docs_dir,
                )
            )
    return "".join(result)


def _nav_doc_exists(rel_path: str, docs_dir: Path, repo_root: Path) -> bool:
    """Return True if a navigation Markdown target exists in docs_dir or as a root virtual doc."""
    if (docs_dir / rel_path).is_file():
        return True
    if rel_path in ROOT_VIRTUAL_DOCS and (repo_root / rel_path).is_file():
        return True
    return False


def _discover_release_notes_entries(docs_dir: Path) -> list[dict[str, str]]:
    """Discover `RELEASE_NOTES_*.md` files in `docs_dir` sorted by semver descending."""
    semver_entries: list[tuple[tuple[int, int, int], str, str]] = []
    other_entries: list[tuple[str, str]] = []
    if not docs_dir.is_dir():
        return []

    for path in sorted(docs_dir.glob("RELEASE_NOTES_*.md")):
        if not path.is_file():
            continue
        match = _RELEASE_NOTES_SEMVER_RE.match(path.name)
        if match:
            ver = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            label = f"Release Notes v{ver[0]}.{ver[1]}.{ver[2]}"
            semver_entries.append((ver, label, path.name))
        else:
            stem_tag = path.stem.removeprefix("RELEASE_NOTES_")
            other_entries.append((f"Release Notes {stem_tag}", path.name))

    semver_entries.sort(key=lambda item: item[0], reverse=True)
    ordered: list[dict[str, str]] = [{label: fname} for _, label, fname in semver_entries]
    for label, fname in sorted(other_entries, reverse=True):
        ordered.append({label: fname})
    return ordered


def _prune_releases_section(
    children: list[Any],
    docs_dir: Path,
    repo_root: Path,
) -> list[Any]:
    """Prune `Releases & Governance` items and inject discovered `RELEASE_NOTES_*.md` files."""
    discovered_notes = _discover_release_notes_entries(docs_dir)
    notes_inserted = False
    pruned: list[Any] = []

    for child in children:
        if isinstance(child, str):
            if Path(child).name.startswith("RELEASE_NOTES_"):
                if not notes_inserted:
                    pruned.extend(discovered_notes)
                    notes_inserted = True
                continue
            if _nav_doc_exists(child, docs_dir, repo_root):
                pruned.append(child)
            continue

        if isinstance(child, dict) and len(child) == 1:
            title, target = next(iter(child.items()))
            if isinstance(target, str):
                if Path(target).name.startswith("RELEASE_NOTES_"):
                    if not notes_inserted:
                        pruned.extend(discovered_notes)
                        notes_inserted = True
                    continue
                if _nav_doc_exists(target, docs_dir, repo_root):
                    pruned.append({title: target})
                    if target == "CHANGELOG.md" and not notes_inserted:
                        pruned.extend(discovered_notes)
                        notes_inserted = True
            elif isinstance(target, list):
                nested = _prune_nav_list(target, docs_dir, repo_root)
                if nested:
                    pruned.append({title: nested})

    if not notes_inserted and discovered_notes:
        # Insert release notes ahead of governance logs if neither Changelog nor prior notes existed
        insert_idx = 0
        for idx, item in enumerate(pruned):
            if isinstance(item, dict) and next(iter(item.values())) == "CHANGELOG.md":
                insert_idx = idx + 1
                break
        pruned[insert_idx:insert_idx] = discovered_notes

    return pruned


def _prune_nav_list(
    items: list[Any],
    docs_dir: Path,
    repo_root: Path,
) -> list[Any]:
    """Recursively prune missing pages and empty sections from a MkDocs nav list."""
    pruned: list[Any] = []
    for entry in items:
        if isinstance(entry, str):
            if _nav_doc_exists(entry, docs_dir, repo_root):
                pruned.append(entry)
            continue

        if isinstance(entry, dict):
            kept_dict: dict[str, Any] = {}
            for title, target in entry.items():
                if isinstance(target, str):
                    if _nav_doc_exists(target, docs_dir, repo_root):
                        kept_dict[title] = target
                elif isinstance(target, list):
                    if title == "Releases & Governance":
                        nested = _prune_releases_section(target, docs_dir, repo_root)
                    else:
                        nested = _prune_nav_list(target, docs_dir, repo_root)
                    if nested:
                        kept_dict[title] = nested
                elif isinstance(target, dict):
                    nested_dict = _prune_nav_list([target], docs_dir, repo_root)
                    if nested_dict:
                        kept_dict[title] = nested_dict[0]
            if kept_dict:
                pruned.append(kept_dict)
    return pruned


def prune_nav_for_existing_files(
    nav: list[Any] | None,
    docs_dir: Path,
    repo_root: Path,
) -> list[Any]:
    """Prune missing files and empty sections while auto-discovering release notes."""
    if not nav:
        return []
    return _prune_nav_list(list(nav), Path(docs_dir), Path(repo_root))


def on_config(config: Any) -> Any:
    """Configure environment-driven site_url, edit_uri, and pruned navigation."""
    site_url = os.environ.get("MICROLLM_DOCS_SITE_URL", "").strip()
    if site_url:
        config.site_url = site_url.rstrip("/") + "/"

    config.edit_uri = "edit/main/docs/" if _effective_repo_ref() == "main" else ""

    docs_dir = Path(config.docs_dir)
    repo_root = docs_dir.parent

    # Prevent MkDocs from dropping docs/README.md when docs/index.md coexists;
    # on_files() re-attaches docs/README.md at /README/ cleanly.
    if (docs_dir / "index.md").is_file() and (docs_dir / "README.md").is_file():
        from pathspec.gitignore import GitIgnoreSpec

        extra_spec = GitIgnoreSpec.from_lines(["/README.md"])
        existing_spec = getattr(config, "exclude_docs", None)
        if existing_spec is None:
            config.exclude_docs = extra_spec
        elif hasattr(existing_spec, "patterns"):
            existing_spec.patterns.extend(extra_spec.patterns)

    if getattr(config, "nav", None):
        config.nav = prune_nav_for_existing_files(config.nav, docs_dir=docs_dir, repo_root=repo_root)
    return config


def on_files(files: Files, config: Any) -> Files:
    """Inject root virtual Markdown docs (`BENCHMARKS.md`, `CHANGELOG.md`), `README.md`, and `simulator.html`."""
    from mkdocs.structure.files import File, Files as MkDocsFiles, InclusionLevel

    docs_dir = Path(config.docs_dir)
    repo_root = docs_dir.parent

    # Belt-and-suspenders privacy filter for excluded outreach/vault/pdf artifacts
    kept: list[File] = []
    for file_obj in list(files):
        if _EXCLUDED_TARGET_RE.search(file_obj.src_uri):
            file_obj.inclusion = InclusionLevel.EXCLUDED
            continue
        kept.append(file_obj)

    if len(kept) != len(files):
        files = MkDocsFiles(kept)

    docs_readme = docs_dir / "README.md"
    if docs_readme.is_file() and files.get_file_from_path("README.md") is None:
        dest_uri = "README/index.html" if getattr(config, "use_directory_urls", True) else "README.html"
        files.append(
            File(
                "README.md",
                src_dir=str(docs_dir),
                dest_dir=config.site_dir,
                use_directory_urls=getattr(config, "use_directory_urls", True),
                dest_uri=dest_uri,
                inclusion=InclusionLevel.INCLUDED,
            )
        )

    for doc_name in ROOT_VIRTUAL_DOCS:
        src_path = repo_root / doc_name
        if src_path.is_file() and files.get_file_from_path(doc_name) is None:
            files.append(File.generated(config, doc_name, abs_src_path=str(src_path)))

    simulator_src = docs_dir / "index.html"
    if simulator_src.is_file() and files.get_file_from_path("simulator.html") is None:
        files.append(File.generated(config, "simulator.html", abs_src_path=str(simulator_src)))

    return files


def on_page_markdown(markdown: str, page: Any, config: Any, files: Files) -> str:
    """Normalize GitHub alerts and rewrite repository-relative links on every Markdown page."""
    page_src_path = getattr(page.file, "src_uri", None) or str(page.file.src_path).replace("\\", "/")
    ref = _effective_repo_ref()
    repo_url = (getattr(config, "repo_url", None) or DEFAULT_REPO_URL).rstrip("/")
    raw_docs_dir = getattr(config, "docs_dir", None) or (
        config.get("docs_dir") if isinstance(config, dict) else None
    )
    docs_dir = Path(raw_docs_dir) if raw_docs_dir else None

    available_root_docs = [
        name for name in ROOT_VIRTUAL_DOCS if files.get_file_from_path(name) is not None
    ]
    converted = convert_github_alerts(markdown)
    return rewrite_relative_links(
        converted,
        page_src_path=page_src_path,
        ref=ref,
        repo_url=repo_url,
        available_root_docs=available_root_docs,
        docs_dir=docs_dir,
    )


def on_post_build(config: Any) -> None:
    """Ensure `<docs_dir>/index.html` is copied to `<site_dir>/simulator.html`."""
    docs_dir = Path(config.docs_dir if hasattr(config, "docs_dir") else config["docs_dir"])
    site_dir = Path(config.site_dir if hasattr(config, "site_dir") else config["site_dir"])
    src_html = docs_dir / "index.html"
    if src_html.is_file():
        site_dir.mkdir(parents=True, exist_ok=True)
        dest_html = site_dir / "simulator.html"
        shutil.copy2(src_html, dest_html)
