"""Unit tests verifying the project governance rules, Action Logbook, and RCA Register."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_governance_rules_inscribed_in_agents_and_rules_dir() -> None:
    """Verify AGENTS.md and .agents/rules/project-governance.md enforce Logbook and RCA updates."""
    agents_md = ROOT / "AGENTS.md"
    gov_rule_md = ROOT / ".agents" / "rules" / "project-governance.md"

    assert agents_md.is_file(), "AGENTS.md must exist at repository root"
    assert gov_rule_md.is_file(), ".agents/rules/project-governance.md must exist"

    agents_content = agents_md.read_text(encoding="utf-8")
    gov_content = gov_rule_md.read_text(encoding="utf-8")

    # Must be an always_on rule in Antigravity
    assert "trigger: always_on" in gov_content

    # Both files must reference the two mandatory living documents
    for content in (agents_content, gov_content):
        assert "docs/JOURNAL_DE_BORD.md" in content
        assert "docs/REGISTRE_RCA.md" in content
        # Must mandate updating at every request, change, and result
        assert "chaque demande" in content.lower()
        assert "changement" in content.lower()
        assert "résultat" in content.lower()


def test_journal_de_bord_exists_and_tracks_ongoing_and_completed_actions() -> None:
    """Verify docs/JOURNAL_DE_BORD.md exists and contains structured ongoing/completed sections."""
    journal_path = ROOT / "docs" / "JOURNAL_DE_BORD.md"
    assert journal_path.is_file(), "docs/JOURNAL_DE_BORD.md must exist"

    content = journal_path.read_text(encoding="utf-8")
    assert "Actions en Cours" in content
    assert "Actions Terminées" in content
    assert "Demande" in content
    assert "Changements" in content
    assert "Résultat" in content


def test_registre_rca_exists_and_covers_all_four_mandatory_categories() -> None:
    """Verify docs/REGISTRE_RCA.md covers Errors (RCA), Dead-Ends, Approximations, and Unimplemented Ideas."""
    rca_path = ROOT / "docs" / "REGISTRE_RCA.md"
    assert rca_path.is_file(), "docs/REGISTRE_RCA.md must exist"

    content = rca_path.read_text(encoding="utf-8")
    # 1. Erreurs & Root Cause Analysis (RCA)
    assert "Erreurs" in content and "RCA" in content
    # 2. Impasses
    assert "Impasses" in content
    # 3. Approximations
    assert "Approximations" in content
    # 4. Idées non implémentées
    assert "Idées Non Implémentées" in content or "Idées non implémentées" in content

    # Ensure structured entries exist (RCA-001, IMP-001, APP-001, IDEA-001)
    assert "RCA-001" in content
    assert "IMP-001" in content
    assert "APP-001" in content
    assert "IDEA-001" in content
