"""Regression tests for MicroLLM outreach documentation and sent campaign integrity."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTREACH_EMAILS_PATH = DOCS_DIR / "OUTREACH_EMAILS.md"
INFLUENCER_EMAILS_PATH = DOCS_DIR / "INFLUENCER_OUTREACH_EMAILS.md"
OUTREACH_CONTACTS_PATH = DOCS_DIR / "OUTREACH_CONTACTS.md"

SENT_ROW_RE = re.compile(
    r"^\|\s*`(?P<msg_id>1a1dcf[0-9a-f]{10})`\s*\|.*\|\s*\*\*Sent via Gmail\*\*\s*\|$",
    re.MULTILINE,
)


def test_no_wrong_maintainer_first_name_in_docs() -> None:
    """Verify the typo 'Jean-Marc' does not appear in any Markdown document."""
    offenders: list[str] = []
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if "Jean-Marc" in content:
            offenders.append(str(md_file.relative_to(REPO_ROOT)))
    assert not offenders, f"Found wrong maintainer name 'Jean-Marc' in: {offenders}"


def test_canonical_maintainer_signature_present_in_outreach_docs() -> None:
    """Verify canonical maintainer signature appears in all outreach templates."""
    outreach_text = OUTREACH_EMAILS_PATH.read_text(encoding="utf-8")
    influencer_text = INFLUENCER_EMAILS_PATH.read_text(encoding="utf-8")

    assert outreach_text.count("Jean-Marie Renouard") >= 14
    assert influencer_text.count("Jean-Marie Renouard") >= 39


def test_deprecated_and_unverified_addresses_removed() -> None:
    """Verify invalid or unverified recipient addresses are absent from outreach docs."""
    checked_files = [
        OUTREACH_EMAILS_PATH,
        INFLUENCER_EMAILS_PATH,
        OUTREACH_CONTACTS_PATH,
    ]
    forbidden = ("rk2x@cmu.edu", "askell@anthropic.com", "such@openai.com")
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for addr in forbidden:
            assert addr not in text, f"Forbidden/stale address {addr} found in {path.name}"

    outreach_text = OUTREACH_EMAILS_PATH.read_text(encoding="utf-8")
    assert "rk2x@andrew.cmu.edu" in outreach_text
    assert "amanda@anthropic.com" in outreach_text


def test_all_48_outreach_emails_recorded_as_sent() -> None:
    """Verify all 14 Campaign 1 and 34 Campaign 2 emails carry Sent status and valid Gmail IDs."""
    outreach_text = OUTREACH_EMAILS_PATH.read_text(encoding="utf-8")
    influencer_text = INFLUENCER_EMAILS_PATH.read_text(encoding="utf-8")

    assert "**Created in Gmail**" not in outreach_text
    assert "**Created in Gmail**" not in influencer_text

    campaign1_matches = SENT_ROW_RE.findall(outreach_text)
    campaign2_matches = SENT_ROW_RE.findall(influencer_text)

    assert len(campaign1_matches) == 14, (
        f"Expected 14 Sent rows in OUTREACH_EMAILS.md, found {len(campaign1_matches)}"
    )
    assert len(campaign2_matches) == 34, (
        f"Expected 34 Sent rows in INFLUENCER_OUTREACH_EMAILS.md, found {len(campaign2_matches)}"
    )
    all_ids = campaign1_matches + campaign2_matches
    assert len(set(all_ids)) == 48, "Sent Gmail message IDs must be unique across all 48 emails"
