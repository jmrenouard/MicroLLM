"""Unit and integration tests for versioned MkDocs Material publishing and IONOS deployment."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docs-publish.yml"
MKDOCS_CONFIG_PATH = REPO_ROOT / "mkdocs.yml"
HOOKS_SCRIPT_PATH = REPO_ROOT / "scripts" / "mkdocs_hooks.py"
BUILD_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_versioned_docs.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"


class _MkDocsYamlLoader(yaml.SafeLoader):
    """Safe YAML loader supporting PyMdownx `!!python/name:` tag references as strings."""


def _python_name_multi_constructor(
    loader: yaml.Loader,
    tag_suffix: str,
    node: yaml.Node,
) -> str:
    del loader, node
    return f"python/name:{tag_suffix}"


_MkDocsYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    _python_name_multi_constructor,
)


class _AttrDict(dict[str, Any]):
    """Dictionary exposing keys as attributes to mimic MkDocsConfig in hook unit tests."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _load_mkdocs_config() -> dict[str, Any]:
    assert MKDOCS_CONFIG_PATH.is_file(), "mkdocs.yml must exist at repository root"
    loader = _MkDocsYamlLoader(MKDOCS_CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        parsed = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(parsed, dict), "mkdocs.yml must parse into a YAML mapping"
    return parsed


def _load_script_module(module_name: str, script_path: Path) -> ModuleType:
    assert script_path.is_file(), f"Missing script module: {script_path}"
    scripts_dir = str(script_path.parent)
    added_sys_path = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_sys_path = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None, f"Cannot load spec for {script_path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added_sys_path and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


def _flatten_nav_paths(node: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(node, str):
        paths.append(node)
    elif isinstance(node, list):
        for item in node:
            paths.extend(_flatten_nav_paths(item))
    elif isinstance(node, dict):
        for val in node.values():
            paths.extend(_flatten_nav_paths(val))
    return paths


def _find_nav_section(nav: list[Any], section_name: str) -> Any:
    for item in nav:
        if isinstance(item, dict) and section_name in item:
            return item[section_name]
    return None


def _load_pyproject_toml() -> dict[str, Any]:
    raw = PYPROJECT_PATH.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(raw)
    version_match = re.search(r'^\s*version\s*=\s*"([^"]+)"', raw, re.MULTILINE)
    docs_match = re.search(r"^\s*docs\s*=\s*\[(.*?)\]", raw, re.MULTILINE | re.DOTALL)
    docs_items = re.findall(r'"([^"]+)"', docs_match.group(1)) if docs_match else []
    return {
        "project": {
            "version": version_match.group(1) if version_match else "",
            "optional-dependencies": {"docs": docs_items} if docs_match else {},
        }
    }


def _collect_workflow_steps(job_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job_cfg.get("steps", [])
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


# ==============================================================================
# 1. mkdocs.yml and Portal Assets Verification
# ==============================================================================


def test_mkdocs_yml_configuration_and_security_exclusions() -> None:
    """Verify mkdocs.yml Material config, Mermaid, KaTeX, mike switcher, hooks, and privacy exclusions."""
    config = _load_mkdocs_config()

    assert config.get("site_name") == "MicroLLM Documentation"
    assert config.get("site_url") == "https://microllm.lightpath.fr/"
    assert config.get("repo_name") == "jmrenouard/MicroLLM"
    assert config.get("repo_url") == "https://github.com/jmrenouard/MicroLLM"
    assert config.get("docs_dir") == "docs"
    assert "scripts/mkdocs_hooks.py" in config.get("hooks", [])

    theme = config.get("theme", {})
    assert isinstance(theme, dict)
    assert theme.get("name") == "material"
    assert theme.get("language") == "en"
    assert "navigation.tabs" in theme.get("features", [])

    extra = config.get("extra", {})
    assert extra.get("version", {}).get("provider") == "mike"
    assert extra.get("version", {}).get("default") == "latest"

    extra_css = config.get("extra_css", [])
    assert any("katex" in item and item.endswith(".css") for item in extra_css)
    assert "stylesheets/extra.css" in extra_css

    extra_js = config.get("extra_javascript", [])
    assert "javascripts/katex.js" in extra_js
    assert any("katex.min.js" in item for item in extra_js)
    assert any("auto-render.min.js" in item for item in extra_js)

    extensions = config.get("markdown_extensions", [])
    ext_names: set[str] = set()
    superfences_cfg: dict[str, Any] = {}
    for ext in extensions:
        if isinstance(ext, str):
            ext_names.add(ext)
        elif isinstance(ext, dict):
            ext_names.update(ext.keys())
            if "pymdownx.superfences" in ext:
                superfences_cfg = ext["pymdownx.superfences"] or {}

    assert "admonition" in ext_names
    assert "pymdownx.arithmatex" in ext_names
    assert "pymdownx.superfences" in ext_names

    custom_fences = superfences_cfg.get("custom_fences", [])
    mermaid_fences = [
        fence for fence in custom_fences if isinstance(fence, dict) and fence.get("name") == "mermaid"
    ]
    assert len(mermaid_fences) == 1, "pymdownx.superfences must configure a mermaid custom fence"
    assert mermaid_fences[0].get("class") == "mermaid"
    assert "fence_code_format" in str(mermaid_fences[0].get("format", ""))

    raw_exclude = config.get("exclude_docs", "")
    exclude_text = "\n".join(raw_exclude) if isinstance(raw_exclude, list) else str(raw_exclude)
    for pattern in ("*OUTREACH*", "index.html", "*.kdbx", "*.keyx", "papers/*.pdf"):
        assert pattern in exclude_text, f"exclude_docs must exclude '{pattern}'"

    nav = config.get("nav", [])
    assert isinstance(nav, list) and len(nav) >= 6
    tab_names = [next(iter(entry.keys())) for entry in nav if isinstance(entry, dict)]
    for expected_tab in (
        "Home",
        "Getting Started",
        "Architecture & Reference",
        "Benchmarks & Quality",
        "Research & Kits",
        "Releases & Governance",
    ):
        assert expected_tab in tab_names

    assert (REPO_ROOT / "docs" / "index.md").is_file()
    assert (REPO_ROOT / "docs" / "stylesheets" / "extra.css").is_file()
    katex_js = REPO_ROOT / "docs" / "javascripts" / "katex.js"
    assert katex_js.is_file()
    katex_js_text = katex_js.read_text(encoding="utf-8")
    assert "document$.subscribe" in katex_js_text
    assert "renderMathInElement" in katex_js_text


# ==============================================================================
# 2. scripts/mkdocs_hooks.py Unit Tests
# ==============================================================================


def test_mkdocs_hooks_convert_github_alerts_and_ignore_code_blocks() -> None:
    """Verify all 5 GitHub alerts convert to Material admonitions while fenced blocks stay untouched."""
    hooks = _load_script_module("mkdocs_hooks", HOOKS_SCRIPT_PATH)

    sample = (
        "Intro paragraph.\n\n"
        "> [!NOTE]\n"
        "> Useful context line.\n\n"
        "> [!TIP]\n"
        "> Helpful optimization.\n\n"
        "> [!IMPORTANT]\n"
        "> Essential requirement.\n\n"
        "> [!WARNING]\n"
        "> Watch out for drift.\n\n"
        "> [!CAUTION]\n"
        "> High risk operation.\n\n"
        "```markdown\n"
        "> [!NOTE]\n"
        "> Literal example inside fenced code.\n"
        "```\n"
    )

    converted = hooks.convert_github_alerts(sample)

    assert "!!! note" in converted
    assert "!!! tip" in converted
    assert "!!! info" in converted
    assert "!!! warning" in converted
    assert "!!! danger" in converted
    assert "Useful context line." in converted
    assert "High risk operation." in converted

    fenced_part = converted.split("```markdown", 1)[1]
    assert "> [!NOTE]" in fenced_part
    outside_fenced = converted.split("```markdown", 1)[0]
    for marker in ("[!" + tag + "]" for tag in ("NOTE", "TIP", "IMPORTANT", "WARNING", "CAUTION")):
        assert marker not in outside_fenced


def test_mkdocs_hooks_rewrite_relative_links_and_neutralize_excluded_targets() -> None:
    """Verify root doc links, depth adjustments, simulator redirect, GitHub blob URLs, and exclusions."""
    hooks = _load_script_module("mkdocs_hooks", HOOKS_SCRIPT_PATH)

    top_level_md = (
        "- [Benchmarks](../BENCHMARKS.md)\n"
        "- [Changelog](../CHANGELOG.md)\n"
        "- [Root Readme](../README.md)\n"
        "- [Simulator](index.html)\n"
        "- [Simulator Dot](./index.html)\n"
        "- [RLCD Source](../src/microllm/rlcd.py)\n"
        "- [Contract Tests](../tests/test_jevcompat_contract.py)\n"
        "- [Outreach](OUTREACH_CONTACTS.md)\n"
        "- [Local Paper](papers/2609.38850.pdf)\n"
        "```markdown\n"
        "[Preserved](../README.md)\n"
        "[Preserved Sim](index.html)\n"
        "```\n"
    )

    rewritten_top = hooks.rewrite_relative_links(top_level_md, "PRODUCT_DOCUMENTATION.md", ref="v0.4.1")
    assert "](BENCHMARKS.md)" in rewritten_top
    assert "](../BENCHMARKS.md)" not in rewritten_top
    assert "](CHANGELOG.md)" in rewritten_top
    assert "](../CHANGELOG.md)" not in rewritten_top
    assert "](index.md)" in rewritten_top
    assert "](../README.md)" not in rewritten_top.split("```markdown", 1)[0]
    assert "](simulator.html)" in rewritten_top
    assert "](index.html)" not in rewritten_top.split("```markdown", 1)[0]
    assert "](./index.html)" not in rewritten_top
    assert "](https://github.com/jmrenouard/MicroLLM/blob/v0.4.1/src/microllm/rlcd.py)" in rewritten_top
    assert "](https://github.com/jmrenouard/MicroLLM/blob/v0.4.1/tests/test_jevcompat_contract.py)" in rewritten_top
    assert "](OUTREACH_CONTACTS.md)" not in rewritten_top
    assert "Outreach" in rewritten_top
    assert "](papers/2609.38850.pdf)" not in rewritten_top
    assert "Local Paper" in rewritten_top

    fenced = rewritten_top.split("```markdown", 1)[1]
    assert "[Preserved](../README.md)" in fenced
    assert "[Preserved Sim](index.html)" in fenced

    nested_md = (
        "- [Root](../../README.md)\n"
        "- [Bench](../../BENCHMARKS.md)\n"
        "- [Literature](../LITERATURE_REVIEW.md)\n"
        "- [Core](../../src/microllm/rlcd.py)\n"
    )
    rewritten_nested = hooks.rewrite_relative_links(nested_md, "papers/README.md", ref="v0.4.1")
    assert "](../index.md)" in rewritten_nested
    assert "](../BENCHMARKS.md)" in rewritten_nested
    assert "](../LITERATURE_REVIEW.md)" in rewritten_nested
    assert "](https://github.com/jmrenouard/MicroLLM/blob/v0.4.1/src/microllm/rlcd.py)" in rewritten_nested


def test_mkdocs_hooks_prune_nav_for_existing_files_and_discover_release_notes(tmp_path: Path) -> None:
    """Verify nav pruning removes missing pages/empty sections and discovers release notes in semver order."""
    hooks = _load_script_module("mkdocs_hooks", HOOKS_SCRIPT_PATH)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "index.md").write_text("# Home\n", encoding="utf-8")
    (docs_dir / "PRODUCT_DOCUMENTATION.md").write_text("# Product\n", encoding="utf-8")
    (docs_dir / "RELEASE_NOTES_v0.3.0.md").write_text("# v0.3.0\n", encoding="utf-8")
    (docs_dir / "RELEASE_NOTES_v0.4.1.md").write_text("# v0.4.1\n", encoding="utf-8")
    (docs_dir / "RELEASE_NOTES_v0.10.0.md").write_text("# v0.10.0\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")

    nav = [
        {"Home": "index.md"},
        {"Getting Started": ["PRODUCT_DOCUMENTATION.md", "MISSING_DOC.md"]},
        {"Empty Section": ["DOES_NOT_EXIST.md"]},
        {"Releases & Governance": ["CHANGELOG.md", "RELEASE_NOTES_v9.9.9.md"]},
    ]

    pruned = hooks.prune_nav_for_existing_files(nav, docs_dir, tmp_path)
    all_paths = _flatten_nav_paths(pruned)

    assert "index.md" in all_paths
    assert "PRODUCT_DOCUMENTATION.md" in all_paths
    assert "MISSING_DOC.md" not in all_paths
    assert "DOES_NOT_EXIST.md" not in all_paths
    assert "RELEASE_NOTES_v9.9.9.md" not in all_paths
    assert _find_nav_section(pruned, "Empty Section") is None

    releases_section = _find_nav_section(pruned, "Releases & Governance")
    assert releases_section is not None
    release_paths = _flatten_nav_paths(releases_section)
    assert "CHANGELOG.md" in release_paths
    for note in ("RELEASE_NOTES_v0.10.0.md", "RELEASE_NOTES_v0.4.1.md", "RELEASE_NOTES_v0.3.0.md"):
        assert note in release_paths

    assert release_paths.index("RELEASE_NOTES_v0.10.0.md") < release_paths.index("RELEASE_NOTES_v0.4.1.md")
    assert release_paths.index("RELEASE_NOTES_v0.4.1.md") < release_paths.index("RELEASE_NOTES_v0.3.0.md")


def test_mkdocs_hooks_on_post_build_copies_simulator_html(tmp_path: Path) -> None:
    """Verify on_post_build copies docs/index.html to site/simulator.html when present."""
    hooks = _load_script_module("mkdocs_hooks", HOOKS_SCRIPT_PATH)

    docs_dir = tmp_path / "docs"
    site_dir = tmp_path / "site"
    docs_dir.mkdir()
    site_dir.mkdir()

    expected_html = "<!DOCTYPE html><html><body>MicroLLM Simulator</body></html>\n"
    (docs_dir / "index.html").write_text(expected_html, encoding="utf-8")

    cfg = _AttrDict({"docs_dir": str(docs_dir), "site_dir": str(site_dir)})
    hooks.on_post_build(cfg)

    sim_out = site_dir / "simulator.html"
    assert sim_out.is_file()
    assert sim_out.read_text(encoding="utf-8") == expected_html


def test_mkdocs_hooks_on_config_edit_uri_by_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify on_config enables edit_uri for main/WORKTREE and clears it for release tags."""
    hooks = _load_script_module("mkdocs_hooks", HOOKS_SCRIPT_PATH)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    monkeypatch.setenv("MICROLLM_DOCS_REF", "WORKTREE")
    monkeypatch.setenv("MICROLLM_DOCS_SITE_URL", "https://microllm.lightpath.fr/main")
    cfg_main = _AttrDict({"docs_dir": str(docs_dir), "site_url": "", "edit_uri": "", "nav": []})
    hooks.on_config(cfg_main)
    assert cfg_main["edit_uri"] == "edit/main/docs/"
    assert cfg_main["site_url"] == "https://microllm.lightpath.fr/main/"

    monkeypatch.setenv("MICROLLM_DOCS_REF", "v0.4.1")
    cfg_tag = _AttrDict({"docs_dir": str(docs_dir), "site_url": "", "edit_uri": "edit/main/docs/", "nav": []})
    hooks.on_config(cfg_tag)
    assert cfg_tag["edit_uri"] == ""


def test_mkdocs_hooks_importable_without_mkdocs_or_pathspec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify scripts/mkdocs_hooks.py imports and runs helper/hook functions without optional docs extras."""
    for blocked in (
        "mkdocs",
        "mkdocs.structure",
        "mkdocs.structure.files",
        "pathspec",
        "pathspec.gitignore",
    ):
        monkeypatch.setitem(sys.modules, blocked, None)

    try:
        hooks = _load_script_module("mkdocs_hooks_no_deps", HOOKS_SCRIPT_PATH)
        assert "!!! note" in hooks.convert_github_alerts("> [!NOTE]\n> Works without mkdocs.")
        assert "](index.md)" in hooks.rewrite_relative_links("[Root](../README.md)", "PRODUCT_DOCUMENTATION.md")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Home\n", encoding="utf-8")
        cfg = _AttrDict({"docs_dir": str(docs_dir), "site_url": "", "edit_uri": "", "nav": []})
        hooks.on_config(cfg)
        assert cfg["edit_uri"] == "edit/main/docs/"
    finally:
        sys.modules.pop("mkdocs_hooks_no_deps", None)


# ==============================================================================
# 3. scripts/build_versioned_docs.py Unit & Integration Tests
# ==============================================================================


def test_build_versioned_docs_parse_semver_tag_and_manifest_ordering() -> None:
    """Verify semver parsing, numeric descending sorting, alias assignment, and empty fallback."""
    builder = _load_script_module("build_versioned_docs", BUILD_SCRIPT_PATH)

    assert builder.parse_semver_tag("v0.1.0") == (0, 1, 0)
    assert builder.parse_semver_tag("v0.10.0") == (0, 10, 0)
    assert builder.parse_semver_tag("v1.2.3") == (1, 2, 3)
    assert builder.parse_semver_tag("main") is None
    assert builder.parse_semver_tag("latest") is None
    assert builder.parse_semver_tag("v0.4") is None

    manifest = builder.build_versions_manifest(
        ["v0.1.0", "v0.9.0", "v0.10.0", "v0.4.1"],
        include_main=True,
    )
    versions = [entry["version"] for entry in manifest]

    # Numeric semver descending puts v0.10.0 before v0.9.0
    assert versions.index("v0.10.0") < versions.index("v0.9.0")
    assert versions.index("v0.9.0") < versions.index("v0.4.1")
    assert versions.index("v0.4.1") < versions.index("v0.1.0")

    by_version = {entry["version"]: entry for entry in manifest}
    assert "latest" in by_version["v0.10.0"]["aliases"]
    assert "latest" not in by_version["v0.9.0"]["aliases"]
    assert "main" in by_version
    assert "dev" in by_version["main"]["aliases"]

    fallback_main = builder.build_versions_manifest([], include_main=True)
    assert len(fallback_main) == 1
    assert fallback_main[0]["version"] == "main"
    assert "dev" in fallback_main[0]["aliases"]

    assert builder.build_versions_manifest([], include_main=False) == []


def test_build_versioned_site_rejects_protected_or_ancestor_output_dirs(tmp_path: Path) -> None:
    """Verify build_versioned_site refuses to delete the repo root, ancestors, or protected subdirs."""
    builder = _load_script_module("build_versioned_docs", BUILD_SCRIPT_PATH)
    repo_dir = tmp_path / "repo"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)
    sentinel = docs_dir / "keep.md"
    sentinel.write_text("# Keep\n", encoding="utf-8")

    for forbidden_out in (repo_dir, tmp_path, docs_dir, docs_dir / "nested"):
        with pytest.raises(ValueError, match="protected"):
            builder.build_versioned_site(repo_root=repo_dir, output_dir=forbidden_out, skip_tags=True)
        assert sentinel.is_file()


def test_build_versioned_docs_render_htaccess_and_redirect_html() -> None:
    """Verify .htaccess rules (HTTPS 301, /latest/ & /dev/ 302, no Indexes) and redirect HTML."""
    builder = _load_script_module("build_versioned_docs", BUILD_SCRIPT_PATH)

    htaccess = builder.render_htaccess("v0.4.1", main_version="main")
    assert "Options -Indexes" in htaccess
    assert "DirectoryIndex index.html" in htaccess
    assert "RewriteEngine On" in htaccess
    assert "RewriteBase /" in htaccess
    assert "X-Content-Type-Options" in htaccess
    assert "R=301" in htaccess
    assert "^latest(?:/(.*))?$" in htaccess
    assert "/v0.4.1/$1" in htaccess
    assert "^dev(?:/(.*))?$" in htaccess
    assert "/main/$1" in htaccess
    assert "R=302" in htaccess
    assert "ErrorDocument 404 /v0.4.1/404.html" in htaccess

    redirect_html = builder.render_redirect_html("/v0.4.1/", "MicroLLM Documentation")
    assert "<!DOCTYPE html>" in redirect_html or "<!doctype html>" in redirect_html.lower()
    assert "MicroLLM Documentation" in redirect_html
    assert 'http-equiv="refresh"' in redirect_html.lower()
    assert "/v0.4.1/" in redirect_html


def test_build_versioned_docs_end_to_end_worktree_and_v041(tmp_path: Path) -> None:
    """End-to-end build of WORKTREE + v0.4.1 asserting manifest, redirects, assets, and zero outreach."""
    if importlib.util.find_spec("mkdocs") is None:
        pytest.skip("mkdocs optional extra not installed in current test environment")
    if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("Git repository metadata required for multi-version tag build test")

    tag_probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "tag", "-l", "v0.4.1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_probe.returncode != 0 or "v0.4.1" not in tag_probe.stdout.splitlines():
        pytest.skip("Tag v0.4.1 not available in local Git clone")

    out_dir = tmp_path / "site_dist"
    proc = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT_PATH),
            "--output",
            str(out_dir),
            "--base-url",
            "https://microllm.lightpath.fr",
            "--main-ref",
            "WORKTREE",
            "--tags",
            "v0.4.1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"build_versioned_docs.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    expected_files = [
        out_dir / "versions.json",
        out_dir / "index.html",
        out_dir / ".htaccess",
        out_dir / "latest" / "index.html",
        out_dir / "dev" / "index.html",
        out_dir / "main" / "index.html",
        out_dir / "main" / "simulator.html",
        out_dir / "main" / "slides.html",
        out_dir / "main" / "stylesheets" / "extra.css",
        out_dir / "main" / "javascripts" / "katex.js",
        out_dir / "v0.4.1" / "index.html",
    ]
    for path in expected_files:
        assert path.is_file(), f"Expected built artifact missing: {path}"

    manifest = json.loads((out_dir / "versions.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, list)
    by_version = {entry["version"]: entry for entry in manifest}
    assert "v0.4.1" in by_version
    assert "latest" in by_version["v0.4.1"]["aliases"]
    assert "main" in by_version
    assert "dev" in by_version["main"]["aliases"]

    htaccess_text = (out_dir / ".htaccess").read_text(encoding="utf-8")
    assert "/v0.4.1/$1" in htaccess_text
    assert "/main/$1" in htaccess_text

    outreach_hits = [
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if "outreach" in p.name.lower() or "outreach" in p.relative_to(out_dir).as_posix().lower()
    ]
    assert not outreach_hits, f"Excluded outreach files leaked into built output: {outreach_hits}"

    vault_or_pdf_hits = [
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and (p.suffix.lower() in {".kdbx", ".keyx"} or p.as_posix().endswith(".pdf"))
    ]
    assert not vault_or_pdf_hits, f"Excluded vault or PDF files leaked into built output: {vault_or_pdf_hits}"


# ==============================================================================
# 4. .github/workflows/docs-publish.yml Security & Deployment Tests
# ==============================================================================


def test_docs_publish_workflow_triggers_permissions_build_and_sftp_deploy() -> None:
    """Verify docs-publish.yml structure, least-privilege permissions, artifact flags, and SFTP steps."""
    assert WORKFLOW_PATH.is_file(), ".github/workflows/docs-publish.yml must exist"
    raw_yaml = WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_yaml)
    assert isinstance(data, dict)

    assert data.get("name") == "Publish Documentation (MkDocs Material)"
    assert data.get("permissions") == {"contents": "read"}

    # PyYAML parses unquoted `on:` as boolean True
    triggers = data.get("on") if "on" in data else data.get(True)
    assert isinstance(triggers, dict)
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main"]
    assert triggers["push"]["paths"] == triggers["pull_request"]["paths"]
    for required_path in (
        "docs/**",
        "mkdocs.yml",
        "README.md",
        "BENCHMARKS.md",
        "CHANGELOG.md",
        "scripts/mkdocs_hooks.py",
        "scripts/build_versioned_docs.py",
        ".github/workflows/docs-publish.yml",
    ):
        assert required_path in triggers["push"]["paths"]

    assert triggers["release"] == {"types": ["published"]}
    deploy_input = triggers["workflow_dispatch"]["inputs"]["deploy"]
    assert deploy_input["type"] == "boolean"
    assert deploy_input["default"] is True

    concurrency = data.get("concurrency", {})
    assert concurrency.get("group") == "docs-publish-${{ github.ref }}"
    assert "pull_request" in str(concurrency.get("cancel-in-progress", ""))

    jobs = data.get("jobs", {})
    assert "verify-and-build" in jobs
    assert "deploy-ionos" in jobs

    # Job 1: verify-and-build
    build_job = jobs["verify-and-build"]
    assert build_job.get("runs-on") == "ubuntu-latest"
    assert isinstance(build_job.get("timeout-minutes"), int) and build_job["timeout-minutes"] <= 15

    build_steps = _collect_workflow_steps(build_job)
    checkout_steps = [s for s in build_steps if s.get("uses") == "actions/checkout@v7"]
    assert len(checkout_steps) == 1
    assert checkout_steps[0].get("with", {}).get("fetch-depth") == 0

    python_steps = [s for s in build_steps if s.get("uses") == "actions/setup-python@v7"]
    assert len(python_steps) == 1
    assert python_steps[0].get("with", {}).get("python-version") == "3.12"

    build_run_script = "\n".join(str(s.get("run", "")) for s in build_steps)
    assert '.[dev,docs]' in build_run_script
    assert "tests/test_mkdocs_publishing.py" in build_run_script
    assert "scripts/build_versioned_docs.py" in build_run_script
    assert "origin/main" in build_run_script
    assert "WORKTREE" in build_run_script
    assert "site_dist/versions.json" in build_run_script
    assert "site_dist/.htaccess" in build_run_script
    assert "outreach" in build_run_script.lower()

    upload_steps = [s for s in build_steps if s.get("uses") == "actions/upload-artifact@v7"]
    assert len(upload_steps) == 1
    upload_with = upload_steps[0].get("with", {})
    assert upload_with.get("name") == "mkdocs-site-dist"
    assert upload_with.get("path") == "site_dist/"
    assert upload_with.get("include-hidden-files") is True
    assert upload_with.get("retention-days") == 7

    # Job 2: deploy-ionos
    deploy_job = jobs["deploy-ionos"]
    assert deploy_job.get("needs") == "verify-and-build"
    assert deploy_job.get("runs-on") == "ubuntu-latest"
    assert isinstance(deploy_job.get("timeout-minutes"), int) and deploy_job["timeout-minutes"] <= 15

    deploy_if = str(deploy_job.get("if", ""))
    assert "github.event_name == 'release'" in deploy_if
    assert "refs/heads/main" in deploy_if
    assert "inputs.deploy == true" in deploy_if

    deploy_steps = _collect_workflow_steps(deploy_job)
    assert not any("actions/checkout" in str(s.get("uses", "")) for s in deploy_steps)

    download_steps = [s for s in deploy_steps if s.get("uses") == "actions/download-artifact@v8"]
    assert len(download_steps) == 1
    assert download_steps[0].get("with", {}).get("name") == "mkdocs-site-dist"
    assert download_steps[0].get("with", {}).get("path") == "."

    sftp_steps = [s for s in deploy_steps if s.get("uses") == "wlixcc/SFTP-Deploy-Action@v1.2.6"]
    assert len(sftp_steps) == 2, "Expected site upload step plus explicit .htaccess upload step"

    local_paths = [step.get("with", {}).get("local_path") for step in sftp_steps]
    assert "./" in local_paths
    assert ".htaccess" in local_paths

    for step in sftp_steps:
        step_with = step.get("with", {})
        assert step_with.get("sftp_only") is True
        assert step_with.get("server") == "${{ secrets.IONOS_FTP_SERVER }}"
        assert step_with.get("username") == "${{ secrets.IONOS_FTP_USERNAME }}"
        assert step_with.get("password") == "${{ secrets.IONOS_FTP_PASSWORD }}"
        assert step_with.get("port") == "${{ secrets.IONOS_FTP_PORT || '22' }}"
        assert step_with.get("remote_path") == "${{ secrets.IONOS_REMOTE_PATH }}/"

    deploy_run_script = "\n".join(str(s.get("run", "")) for s in deploy_steps)
    assert ".htaccess" in deploy_run_script
    assert "versions.json" in deploy_run_script
    assert "https://microllm.lightpath.fr/versions.json" in deploy_run_script
    assert "https://microllm.lightpath.fr/latest/" in deploy_run_script
    assert "https://microllm.lightpath.fr/main/" in deploy_run_script

    for unsafe_pattern in (
        "chmod 777",
        "chmod -R 777",
        "chmod a+rwx",
        "sudo",
        "eval ",
        "curl -s | bash",
        "wget -O- | sh",
    ):
        assert unsafe_pattern not in raw_yaml, f"Unsafe shell pattern found in workflow: {unsafe_pattern}"


# ==============================================================================
# 5. Packaging Metadata, .gitignore, and Portal Link Consistency
# ==============================================================================


def test_gitignore_and_pyproject_docs_configuration() -> None:
    """Verify .gitignore excludes site outputs and pyproject.toml defines docs extras at v0.4.1."""
    gitignore_lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "site/" in gitignore_lines
    assert "site_dist/" in gitignore_lines

    pyproject = _load_pyproject_toml()
    project = pyproject.get("project", {})
    assert project.get("version") == "0.4.1"

    docs_deps = project.get("optional-dependencies", {}).get("docs", [])
    joined_deps = " ".join(docs_deps)
    assert "mkdocs>=" in joined_deps
    assert "mkdocs-material>=" in joined_deps
    assert "pymdown-extensions>=" in joined_deps

    for doc_path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "README.md"):
        assert "https://microllm.lightpath.fr" in doc_path.read_text(encoding="utf-8")
