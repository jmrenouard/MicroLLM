"""Unit tests verifying the project governance rules, Action Logbook, and RCA Register."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_governance_rules_inscribed_in_agents_and_rules_dir() -> None:
    """Verify AGENTS.md and .agents/rules/project-governance.md enforce Logbook and RCA updates in English."""
    agents_md = ROOT / "AGENTS.md"
    gov_rule_md = ROOT / ".agents" / "rules" / "project-governance.md"

    assert agents_md.is_file(), "AGENTS.md must exist at repository root"
    assert gov_rule_md.is_file(), ".agents/rules/project-governance.md must exist"

    agents_content = agents_md.read_text(encoding="utf-8")
    gov_content = gov_rule_md.read_text(encoding="utf-8")

    # Must be an always_on rule in Antigravity
    assert "trigger: always_on" in gov_content

    # Both files must reference the two mandatory English living documents
    for content in (agents_content, gov_content):
        assert "docs/ACTION_LOGBOOK.md" in content
        assert "docs/RCA_REGISTER.md" in content
        # Must mandate updating on every request, change, and result
        assert "every request" in content.lower()
        assert "every change" in content.lower()
        assert "every result" in content.lower()


def test_post_merge_issue_comment_rule_enforced_in_agents_and_rules() -> None:
    """Verify AGENTS.md and .agents/rules/ mandate closing issues with a summary comment upon PR merge."""
    agents_md = ROOT / "AGENTS.md"
    git_wf_md = ROOT / ".agents" / "rules" / "git-workflow.md"
    gh_wf_md = ROOT / ".agents" / "rules" / "github-workflow.md"

    for path in (agents_md, git_wf_md, gh_wf_md):
        text = path.read_text(encoding="utf-8")
        assert "gh issue comment" in text, f"{path.name} must mandate 'gh issue comment'"
        assert "gh issue close" in text, f"{path.name} must mandate 'gh issue close'"


def test_action_logbook_exists_in_english_and_tracks_ongoing_and_completed_actions() -> None:
    """Verify docs/ACTION_LOGBOOK.md exists in English and contains structured ongoing/completed sections."""
    logbook_path = ROOT / "docs" / "ACTION_LOGBOOK.md"
    assert logbook_path.is_file(), "docs/ACTION_LOGBOOK.md must exist"
    assert not (ROOT / "docs" / "JOURNAL_DE_BORD.md").exists(), "Legacy French JOURNAL_DE_BORD.md must be removed"

    content = logbook_path.read_text(encoding="utf-8")
    assert "Ongoing Actions" in content
    assert "Completed Actions" in content
    assert "Request" in content
    assert "Changes" in content
    assert "Result" in content


def test_rca_register_exists_in_english_and_covers_all_four_mandatory_categories() -> None:
    """Verify docs/RCA_REGISTER.md covers Errors (RCA), Dead-Ends, Approximations, and Unimplemented Ideas in English."""
    rca_path = ROOT / "docs" / "RCA_REGISTER.md"
    assert rca_path.is_file(), "docs/RCA_REGISTER.md must exist"
    assert not (ROOT / "docs" / "REGISTRE_RCA.md").exists(), "Legacy French REGISTRE_RCA.md must be removed"

    content = rca_path.read_text(encoding="utf-8")
    # 1. Errors & Root Cause Analysis (RCA)
    assert "Errors & Bugs Encountered" in content and "RCA" in content
    # 2. Technical Dead-Ends
    assert "Technical Dead-Ends" in content
    # 3. Current Approximations
    assert "Current Approximations" in content
    # 4. Unimplemented Ideas
    assert "Unimplemented Ideas" in content

    # Ensure structured entries exist (RCA-001, IMP-001, APP-001, IDEA-001..IDEA-020)
    assert "RCA-001" in content
    assert "IMP-001" in content
    assert "APP-001" in content
    for i in range(1, 21):
        assert f"IDEA-{i:03d}" in content, f"IDEA-{i:03d} must be merged into docs/RCA_REGISTER.md"

    # Verify docs/ROADMAP_20_IDEAS.md is merged into docs/RCA_REGISTER.md and no longer a separate file
    assert not (ROOT / "docs" / "ROADMAP_20_IDEAS.md").exists(), (
        "docs/ROADMAP_20_IDEAS.md must be merged into docs/RCA_REGISTER.md and removed as a separate file"
    )
    assert "Detailed Specifications of the 20" in content or "Pillar 1" in content
    assert "Pillar 6" in content


def test_english_only_rule_registered_and_enforced_across_docs_and_code() -> None:
    """Verify the English-only rule is registered and all project documentation and code are in English."""
    agents_md = ROOT / "AGENTS.md"
    english_rule_md = ROOT / ".agents" / "rules" / "english-only.md"
    gov_rule_md = ROOT / ".agents" / "rules" / "project-governance.md"

    assert english_rule_md.is_file(), ".agents/rules/english-only.md must exist"
    rule_text = english_rule_md.read_text(encoding="utf-8")
    assert "trigger: always_on" in rule_text
    assert "english" in rule_text.lower()
    assert "documentation" in rule_text.lower()
    assert "code" in rule_text.lower()

    agents_text = agents_md.read_text(encoding="utf-8")
    assert "English-Only Policy" in agents_text

    gov_text = gov_rule_md.read_text(encoding="utf-8")
    assert "English-Only" in gov_text

    forbidden_french_doc_phrases = (
        "Répertoire Exhaustif",
        "Synthèse Exécutive",
        "Synthèse Produit (Français)",
        "Documentation Produit Complète",
        "Spécifications Techniques Complètes",
        "Matrice Comparative des Jeux de Tests",
        "Résultats Obtenus sur les Échantillons",
        "Prochaines Évolutions",
    )

    doc_files = [
        ROOT / "README.md",
        ROOT / "BENCHMARKS.md",
        ROOT / "AGENTS.md",
        *sorted((ROOT / ".agents" / "rules").glob("*.md")),
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "docs" / "papers" / "README.md",
    ]
    for doc_file in doc_files:
        assert doc_file.is_file(), f"Expected documentation file missing: {doc_file}"
        text = doc_file.read_text(encoding="utf-8")
        for phrase in forbidden_french_doc_phrases:
            assert phrase not in text, (
                f"Forbidden French documentation phrase '{phrase}' found in {doc_file.relative_to(ROOT)}"
            )

    # Verify all Python source modules under src/microllm/ have English docstrings and comments
    for py_file in sorted((ROOT / "src" / "microllm").rglob("*.py")):
        py_text = py_file.read_text(encoding="utf-8")
        for phrase in forbidden_french_doc_phrases:
            assert phrase not in py_text, (
                f"Forbidden French phrase '{phrase}' found in {py_file.relative_to(ROOT)}"
            )


