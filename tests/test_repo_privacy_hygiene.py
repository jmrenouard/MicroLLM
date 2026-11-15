"""Regression tests enforcing repository privacy hygiene and `.gitignore` protections."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

# Intentionally allowlisted public/placeholder email addresses only
ALLOWED_PUBLIC_OR_SYNTHETIC_EMAILS = frozenset(
    {
        "contact@lightpath.fr",
        "git@github.com",
        "test@example.com",
        "user@domain.org",
    }
)
ALLOWED_EMAIL_SUFFIXES = (
    "@users.noreply.github.com",
    "@example.com",
    "@example.org",
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
PHONE_FR_RE = re.compile(r"\+33[\s.-]?[1-9](?:[\s.-]?\d{2}){4}\b")
GMAIL_ID_RE = re.compile(r"\b1a1dc[0-9a-f]{10,}\b|\br-?[0-9]{15,}\b")
WIN_USER_PATH_RE = re.compile(r"(?i)[a-z]:(?:[/\\]{1,2})users(?:[/\\]{1,2})(?![\\\[\(])[a-z0-9_.-]+")
POSIX_USER_PATH_RE = re.compile(r"(?i)/(?:users|home)/(?!\[|runner\b|\.cache\b)[a-z0-9_.-]+/")
OUTREACH_OR_VAULT_PATH_RE = re.compile(r"(?i)(outreach|\.(?:kdbx|keyx)$)")

SYNTHETIC_IGNORED_PROBES = (
    "docs/OUTREACH_CONTACTS.md",
    "docs/OUTREACH_EMAILS.md",
    "docs/INFLUENCER_OUTREACH_CONTACTS.md",
    "docs/INFLUENCER_OUTREACH_EMAILS.md",
    "docs/outreach/drafts.md",
    "sent_outreach_receipts.json",
    "vault.kdbx",
    "vault.keyx",
)


def _git_tracked_files() -> list[str] | None:
    """Return tracked file paths via `git ls-files`, or None outside a git worktree."""
    if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
        return None
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_gitignore_blocks_private_outreach_and_vault_files() -> None:
    """Verify `.gitignore` includes rules blocking outreach catalogs and KeePass vault files."""
    assert GITIGNORE_PATH.is_file(), ".gitignore must exist at repository root"
    gitignore_text = GITIGNORE_PATH.read_text(encoding="utf-8")
    lowered = gitignore_text.lower()
    assert "outreach" in lowered, ".gitignore must exclude private outreach files"
    assert "*.kdbx" in lowered, ".gitignore must exclude *.kdbx KeePass database files"
    assert "*.keyx" in lowered, ".gitignore must exclude *.keyx KeePass key files"

    if (REPO_ROOT / ".git").exists() and shutil.which("git") is not None:
        for probe in SYNTHETIC_IGNORED_PROBES:
            proc = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "--no-index", probe],
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, f"Probe path must be ignored by .gitignore: {probe}"


def test_no_outreach_or_vault_files_tracked_in_git() -> None:
    """Verify zero outreach or KeePass vault files remain tracked in Git index."""
    tracked = _git_tracked_files()
    if tracked is None:
        pytest.skip("Git repository metadata not present in standalone execution environment")

    offenders = [path for path in tracked if OUTREACH_OR_VAULT_PATH_RE.search(path)]
    assert not offenders, f"Private outreach or vault files must not be tracked in Git: {offenders}"


def test_zero_personal_emails_phones_gmail_ids_or_workstation_paths_in_tracked_files() -> None:
    """Verify tracked repository files contain no personal emails, phone numbers, Gmail IDs, or OS user paths."""
    tracked = _git_tracked_files()
    if tracked is None:
        candidate_paths = [
            p.relative_to(REPO_ROOT).as_posix()
            for p in REPO_ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts
        ]
    else:
        candidate_paths = tracked

    violations: list[str] = []
    for rel_path in candidate_paths:
        if OUTREACH_OR_VAULT_PATH_RE.search(rel_path):
            continue
        full_path = REPO_ROOT / rel_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            for match in EMAIL_RE.findall(line):
                lower_email = match.lower()
                if lower_email in ALLOWED_PUBLIC_OR_SYNTHETIC_EMAILS or any(
                    lower_email.endswith(suffix) for suffix in ALLOWED_EMAIL_SUFFIXES
                ):
                    continue
                violations.append(f"{rel_path}:{line_no}: unexpected email address redacted")

            if PHONE_FR_RE.search(line):
                violations.append(f"{rel_path}:{line_no}: personal phone number detected")

            if GMAIL_ID_RE.search(line):
                violations.append(f"{rel_path}:{line_no}: Gmail message/draft identifier detected")

            if WIN_USER_PATH_RE.search(line) or POSIX_USER_PATH_RE.search(line):
                violations.append(f"{rel_path}:{line_no}: hardcoded workstation user path detected")

    assert not violations, "Privacy hygiene violations found:\n" + "\n".join(violations)


def test_no_wrong_maintainer_first_name_in_tracked_markdown() -> None:
    """Verify the misspelled maintainer given name does not appear in any tracked Markdown document."""
    tracked = _git_tracked_files()
    if tracked is None:
        md_paths = [p for p in REPO_ROOT.rglob("*.md") if not OUTREACH_OR_VAULT_PATH_RE.search(p.as_posix())]
    else:
        md_paths = [REPO_ROOT / p for p in tracked if p.lower().endswith(".md")]

    offenders: list[str] = []
    for md_file in sorted(md_paths):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8")
        if "Jean-Marc" in content:
            offenders.append(md_file.relative_to(REPO_ROOT).as_posix())

    assert not offenders, f"Found misspelled maintainer given name in: {offenders}"
