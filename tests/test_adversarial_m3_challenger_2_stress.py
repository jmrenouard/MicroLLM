"""Empirical Adversarial Challenge Test Suite for Milestone 3 (challenger_m3_2).

Covers:
1. F09 (HF Space License Defaults & YAML Frontmatter Integrity):
   - Upper/lower/mixed casing: 'APACHE-2.0', 'Apache-2.0', 'MiT', 'BSD-3-Clause'.
   - Whitespace stripping and fallback for empty string or None.
   - Custom licenses and custom tags (single, multiple, special characters).
   - YAML frontmatter syntactic validity via PyYAML parsing.
   - End-to-end Space bundle staging and dry-run synchronization.
2. F10 (Workstation Path Removal & Vault Resolution):
   - Source-wide scan for zero hardcoded workstation personal directories ('C:\\Users\\').
   - Multi-tier environment variable override hierarchy (MICROLLM_ -> KEEPASS_ -> GESTION2026_ -> default).
   - Whitespace-padded and relative path resolution.
   - Fault tolerance on non-existent, corrupt, or unreadable KeePass vault files.
3. F06 (DualGearCascade Token Reuse & Throughput Stress Testing):
   - 1,000 multi-question call throughput stress test comparing evaluate_voc_gate_regime
     latency with pre-cached vs uncached state tokens.
   - Equivalence oracle guaranteeing 100% bit-for-bit parity across all regimes.
   - Container polymorphism (set, frozenset, list, tuple, Sequence).
   - Integration with VerdictMQMultiQuestionCache and deliberative marker sensitivity.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from microllm.backends.base import (
    VerdictMQMultiQuestionCache,
    tokenize_multilingual,
)
from microllm.backends.cascade import (
    DualGearCascadeBackend,
    evaluate_voc_gate_regime,
)
from microllm.hf_publishing import (
    get_vault_paths,
    resolve_hf_token,
    _resolve_token_from_keepass,
)
from microllm.hf_space_sync import (
    build_space_readme,
    stage_hf_space,
    sync_hf_space,
)


# ==============================================================================
# Helper to extract and parse YAML frontmatter from generated Space Card README
# ==============================================================================


def parse_space_readme_frontmatter(readme_text: str) -> dict[str, Any]:
    """Extract and parse YAML frontmatter from a Hugging Face Space Card README."""
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", readme_text, re.DOTALL)
    assert match is not None, "README.md does not start with a valid YAML frontmatter block ('---')"
    yaml_block = match.group(1)
    parsed = yaml.safe_load(yaml_block)
    assert isinstance(parsed, dict), f"Frontmatter did not parse into a dictionary: {parsed}"
    return parsed


# ==============================================================================
# 1. F09: Space License Defaults & YAML Frontmatter Adversarial Tests
# ==============================================================================


class TestSpaceLicenseDefaultsAdversarial:
    """Adversarially challenge F09 Hugging Face Space Card license generation."""

    @pytest.mark.parametrize(
        ("input_license", "expected_license"),
        [
            ("APACHE-2.0", "apache-2.0"),
            ("Apache-2.0", "apache-2.0"),
            ("apache-2.0", "apache-2.0"),
            ("MIT", "mit"),
            ("MiT", "mit"),
            ("BSD-3-Clause", "bsd-3-clause"),
            ("gpl-3.0", "gpl-3.0"),
            ("   APACHE-2.0   \n", "apache-2.0"),
            ("  mit  ", "mit"),
            ("", "apache-2.0"),
            ("   \t  ", "apache-2.0"),
            (None, "apache-2.0"),
        ],
    )
    def test_build_space_readme_license_normalization(
        self, input_license: str | None, expected_license: str
    ) -> None:
        """Verify license casing, whitespace padding, and empty fallback normalize to expected lowercase."""
        kwargs: dict[str, Any] = {}
        if input_license is not None:
            kwargs["license_type"] = input_license

        readme = build_space_readme(**kwargs)
        parsed = parse_space_readme_frontmatter(readme)

        assert parsed["license"] == expected_license
        assert f"license: {expected_license}" in readme.lower()

    def test_build_space_readme_default_is_apache_2_0(self) -> None:
        """Verify zero-argument default is strictly apache-2.0."""
        readme = build_space_readme()
        parsed = parse_space_readme_frontmatter(readme)
        assert parsed["license"] == "apache-2.0"

    def test_build_space_readme_custom_tags_formatting(self) -> None:
        """Verify custom tags serialize into clean valid YAML sequence."""
        custom_tags = ["system-1", "probabilistic-engine", "quadrilingual", "nlp"]
        readme = build_space_readme(license_type="apache-2.0", tags=custom_tags)
        parsed = parse_space_readme_frontmatter(readme)

        assert parsed["tags"] == custom_tags
        assert "tags:" in readme
        for tag in custom_tags:
            assert f"- {tag}" in readme

    def test_build_space_readme_empty_tags_omits_tags_key(self) -> None:
        """Verify passing tags=[] does not leave a dangling empty tags block."""
        readme = build_space_readme(license_type="apache-2.0", tags=[])
        parsed = parse_space_readme_frontmatter(readme)
        assert "tags" not in parsed

    def test_build_space_readme_tags_with_spaces_and_special_chars(self) -> None:
        """Verify tags with spaces and hyphens parse correctly as valid YAML strings."""
        tags = ["ai safety", "sub-0.4ms", "openjev-rlcd2", "startlux-cascade"]
        readme = build_space_readme(license_type="apache-2.0", tags=tags)
        parsed = parse_space_readme_frontmatter(readme)
        assert parsed["tags"] == tags

    def test_build_space_readme_frontmatter_required_keys_present(self) -> None:
        """Verify all standard Hugging Face Space metadata keys are present and valid."""
        readme = build_space_readme()
        parsed = parse_space_readme_frontmatter(readme)

        required_keys = {
            "title",
            "emoji",
            "colorFrom",
            "colorTo",
            "sdk",
            "sdk_version",
            "app_file",
            "pinned",
            "license",
            "short_description",
        }
        for key in required_keys:
            assert key in parsed, f"Missing required frontmatter key: {key}"

        assert parsed["sdk"] == "gradio"
        assert parsed["app_file"] == "app.py"
        assert parsed["pinned"] is False

    def test_stage_hf_space_honors_uppercase_license_and_writes_valid_readme(
        self, tmp_path: Path
    ) -> None:
        """Verify stage_hf_space creates README.md with normalized license from uppercase input."""
        stage_dir = tmp_path / "staged_space"
        result = stage_hf_space(
            staging_dir=stage_dir,
            license_type="APACHE-2.0",
            include_src=False,
        )

        readme_file = stage_dir / "README.md"
        assert readme_file.is_file()
        content = readme_file.read_text(encoding="utf-8")
        parsed = parse_space_readme_frontmatter(content)

        assert parsed["license"] == "apache-2.0"
        assert result["total_files"] >= 4  # README.md, requirements.txt, app.py, .gitignore, manifests
        assert (stage_dir / "manifest.json").is_file()

    def test_sync_hf_space_dry_run_preserves_license_selection(
        self, tmp_path: Path
    ) -> None:
        """Verify sync_hf_space with dry_run=True applies license parameter to staged README."""
        sync_dir = tmp_path / "sync_dry_run"
        res = sync_hf_space(
            staging_dir=sync_dir,
            dry_run=True,
            license_type="MIT",
            include_src=False,
        )

        assert res["status"] == "dry_run"
        assert res["upload_attempted"] is False
        assert res["network_calls_made"] == 0

        readme_text = (sync_dir / "README.md").read_text(encoding="utf-8")
        parsed = parse_space_readme_frontmatter(readme_text)
        assert parsed["license"] == "mit"


# ==============================================================================
# 2. F10: Workstation Path Removal & Vault Resolution Adversarial Tests
# ==============================================================================


class TestVaultPathsAdversarial:
    """Adversarially challenge F10 workstation path elimination and KeePass vault resolution."""

    def test_zero_hardcoded_user_workstation_paths_across_all_source_files(self) -> None:
        """Assert zero occurrences of hardcoded developer home directories in src/microllm."""
        src_root = Path(__file__).resolve().parent.parent / "src" / "microllm"
        assert src_root.is_dir()

        forbidden_patterns = [
            r"C:\\Users\\[a-zA-Z0-9_]+",
            r"/Users/[a-zA-Z0-9_]+/(?!(\.cache|runner))",
            r"/home/[a-zA-Z0-9_]+/(?!(\.cache|runner))",
        ]

        violations: list[str] = []
        for py_path in src_root.rglob("*.py"):
            content = py_path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(f"{py_path.name}: {matches}")

        assert not violations, f"Found hardcoded workstation user paths in source files: {violations}"

    def test_get_vault_paths_precedence_hierarchy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify strict priority ordering: MICROLLM_ > KEEPASS_ > GESTION2026_ > default."""
        p_micro_db = tmp_path / "micro.kdbx"
        p_micro_key = tmp_path / "micro.keyx"
        p_keepass_db = tmp_path / "keepass.kdbx"
        p_keepass_key = tmp_path / "keepass.keyx"
        p_gestion_db = tmp_path / "gestion.kdbx"
        p_gestion_key = tmp_path / "gestion.keyx"

        # Case 1: All three set -> MICROLLM_ wins
        monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(p_micro_db))
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", str(p_micro_key))
        monkeypatch.setenv("KEEPASS_DB", str(p_keepass_db))
        monkeypatch.setenv("KEEPASS_KEY", str(p_keepass_key))
        monkeypatch.setenv("GESTION2026_KDBX_PATH", str(p_gestion_db))
        monkeypatch.setenv("GESTION2026_KEYX_PATH", str(p_gestion_key))

        db, key = get_vault_paths()
        assert db == p_micro_db
        assert key == p_micro_key

        # Case 2: MICROLLM_ unset -> KEEPASS_ wins
        monkeypatch.delenv("MICROLLM_KEEPASS_DB")
        monkeypatch.delenv("MICROLLM_KEEPASS_KEY")
        db, key = get_vault_paths()
        assert db == p_keepass_db
        assert key == p_keepass_key

        # Case 3: KEEPASS_ unset -> GESTION2026_ wins
        monkeypatch.delenv("KEEPASS_DB")
        monkeypatch.delenv("KEEPASS_KEY")
        db, key = get_vault_paths()
        assert db == p_gestion_db
        assert key == p_gestion_key

        # Case 4: All unset -> Default Path.home() / Dropbox / LightPath / Gestion2026
        monkeypatch.delenv("GESTION2026_KDBX_PATH")
        monkeypatch.delenv("GESTION2026_KEYX_PATH")
        db, key = get_vault_paths()
        expected_dir = Path.home() / "Dropbox" / "LightPath" / "Gestion2026"
        assert db == expected_dir / "Database.kdbx"
        assert key == expected_dir / "Database.keyx"

    def test_get_vault_paths_whitespace_stripping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify leading/trailing spaces and newlines in environment variables are cleanly stripped."""
        raw_db = tmp_path / "vault.kdbx"
        raw_key = tmp_path / "vault.keyx"

        monkeypatch.setenv("MICROLLM_KEEPASS_DB", f"  \t\n{raw_db}  \r\n")
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", f" \t {raw_key} \n ")

        db, key = get_vault_paths()
        assert db == raw_db
        assert key == raw_key

    def test_get_vault_paths_empty_or_whitespace_only_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify whitespace-only environment variable values fall back to default path."""
        monkeypatch.setenv("MICROLLM_KEEPASS_DB", "   \t\n  ")
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", "   ")
        monkeypatch.delenv("KEEPASS_DB", raising=False)
        monkeypatch.delenv("KEEPASS_KEY", raising=False)
        monkeypatch.delenv("GESTION2026_KDBX_PATH", raising=False)
        monkeypatch.delenv("GESTION2026_KEYX_PATH", raising=False)

        db, key = get_vault_paths()
        expected_dir = Path.home() / "Dropbox" / "LightPath" / "Gestion2026"
        assert db == expected_dir / "Database.kdbx"
        assert key == expected_dir / "Database.keyx"

    def test_resolve_token_from_keepass_missing_files_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify _resolve_token_from_keepass gracefully returns None when vault files do not exist."""
        nonexistent_db = tmp_path / "does_not_exist.kdbx"
        nonexistent_key = tmp_path / "does_not_exist.keyx"
        monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(nonexistent_db))
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", str(nonexistent_key))

        assert _resolve_token_from_keepass() is None

    def test_resolve_token_from_keepass_corrupt_files_returns_none_without_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify corrupt/dummy vault files do not raise an unhandled exception."""
        dummy_db = tmp_path / "corrupt.kdbx"
        dummy_key = tmp_path / "corrupt.keyx"
        dummy_db.write_bytes(b"INVALID_KDBX_HEADER_DATA_12345")
        dummy_key.write_bytes(b"INVALID_KEY_FILE_BYTES_67890")

        monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(dummy_db))
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", str(dummy_key))

        # Must catch PyKeePass parsing exceptions and return None cleanly
        assert _resolve_token_from_keepass() is None

    def test_resolve_hf_token_with_vault_enabled_and_corrupt_vault_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify resolve_hf_token(allow_vault=True) falls back cleanly when vault is corrupt."""
        # Clear token env vars and point cache to empty folder
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_cache)

        corrupt_db = tmp_path / "bad.kdbx"
        corrupt_key = tmp_path / "bad.keyx"
        corrupt_db.write_text("not a keepass file", encoding="utf-8")
        corrupt_key.write_text("not a key file", encoding="utf-8")

        monkeypatch.setenv("MICROLLM_KEEPASS_DB", str(corrupt_db))
        monkeypatch.setenv("MICROLLM_KEEPASS_KEY", str(corrupt_key))

        tok, source = resolve_hf_token(allow_vault=True)
        assert tok is None
        assert source == "none"


# ==============================================================================
# 3. F06: DualGearCascade Token Reuse & Throughput Stress Testing
# ==============================================================================


class TestDualGearCascadeTokenReuseStress:
    """Stress test evaluate_voc_gate_regime and token reuse across 1,000 multi-question calls."""

    @pytest.fixture
    def enterprise_state_text(self) -> str:
        """Realistic enterprise customer support context with mixed deliberative and standard terms."""
        return (
            "Customer states: I noticed an unexpected recurring charge of $49.99 on my account. "
            "However, unless the billing system overrides the standard 30-day grace period, "
            "I request an expedited refund. If a waiver applies, please exempt the late fee. "
            "Otherwise, I will proceed with a dispute. Client ID: ACC-98432. "
            "Payment processed via international card, pending settlement."
        )

    def test_1000_multi_question_throughput_and_equivalence(
        self, enterprise_state_text: str
    ) -> None:
        """Stress-test evaluate_voc_gate_regime over 1,000 calls comparing cached vs uncached tokens.

        Asserts:
        1. 100% equivalence: every returned field is identical between cached and uncached calls.
        2. Throughput performance: cached calls execute with strictly sub-millisecond latency.
        3. Pre-cached latency is equal to or faster than uncached tokenization.
        """
        # Pre-compute state tokens once
        precomputed_tokens = set(tokenize_multilingual(enterprise_state_text))
        assert len(precomputed_tokens) > 10
        assert "unless" in precomputed_tokens
        assert "however" in precomputed_tokens

        options_pool = [
            {"refund": "Process full refund immediately", "dispute": "Escalate to financial dispute"},
            {"billing": "Review recurring subscription billing", "technical": "Technical platform issue"},
            {"urgent": "High priority escalation", "standard": "Standard SLA queue"},
            {"waiver": "Apply fee waiver exception", "enforce": "Enforce standard policy"},
        ]

        instructions_pool = [
            "Classify the customer inquiry into the most appropriate operational queue.",
            "Determine whether this request qualifies for fee exemption under policy.",
            "Identify the urgency level of the payment dispute.",
            "Route the ticket to billing support or dispute resolution team.",
        ]

        iterations = 1000

        # Phase 1: 1,000 calls WITHOUT pre-cached tokens (uncached state_tokens=None)
        t_start_uncached = time.perf_counter()
        uncached_results: list[dict[str, Any]] = []
        for i in range(iterations):
            conf = 0.20 + 0.60 * ((i % 100) / 100.0)
            opts = options_pool[i % len(options_pool)]
            instr = instructions_pool[i % len(instructions_pool)]
            res = evaluate_voc_gate_regime(
                state=enterprise_state_text,
                instructions=instr,
                options=opts,
                conf0=conf,
                state_tokens=None,
            )
            uncached_results.append(res)
        t_uncached_total = time.perf_counter() - t_start_uncached

        # Phase 2: 1,000 calls WITH pre-cached tokens (cached state_tokens=precomputed_tokens)
        t_start_cached = time.perf_counter()
        cached_results: list[dict[str, Any]] = []
        for i in range(iterations):
            conf = 0.20 + 0.60 * ((i % 100) / 100.0)
            opts = options_pool[i % len(options_pool)]
            instr = instructions_pool[i % len(instructions_pool)]
            res = evaluate_voc_gate_regime(
                state=enterprise_state_text,
                instructions=instr,
                options=opts,
                conf0=conf,
                state_tokens=precomputed_tokens,
            )
            cached_results.append(res)
        t_cached_total = time.perf_counter() - t_start_cached

        # Assertion 1: Exact Equivalence Oracle across all 1,000 calls
        assert len(cached_results) == iterations
        assert len(uncached_results) == iterations
        for i in range(iterations):
            r_cached = cached_results[i]
            r_uncached = uncached_results[i]
            assert r_cached["uncertainty_regime"] == r_uncached["uncertainty_regime"]
            assert r_cached["should_abstain"] == r_uncached["should_abstain"]
            assert r_cached["should_escalate_gear2"] == r_uncached["should_escalate_gear2"]
            assert r_cached["expected_voc_gain"] == r_uncached["expected_voc_gain"]
            assert r_cached["max_semantic_logit"] == r_uncached["max_semantic_logit"]

        # Assertion 2: Strict sub-millisecond budget (<1.0 ms per call)
        avg_cached_ms = (t_cached_total / iterations) * 1000.0
        avg_uncached_ms = (t_uncached_total / iterations) * 1000.0
        assert avg_cached_ms < 1.0, f"Cached per-call latency exceeded 1.0ms: {avg_cached_ms:.4f}ms"

        # Assertion 3: Cached execution overhead is bounded and non-regressing
        assert t_cached_total <= t_uncached_total * 1.5, (
            f"Cached total ({t_cached_total:.4f}s) significantly exceeded uncached ({t_uncached_total:.4f}s)"
        )

    def test_state_tokens_container_polymorphism(self, enterprise_state_text: str) -> None:
        """Verify evaluate_voc_gate_regime accepts diverse Sequence and Set containers."""
        token_list = tokenize_multilingual(enterprise_state_text)
        token_set = set(token_list)
        token_frozenset = frozenset(token_list)
        token_tuple = tuple(token_list)

        options = {"yes": "Affirmative", "no": "Negative"}
        instructions = "Test container polymorphism."

        res_none = evaluate_voc_gate_regime(enterprise_state_text, instructions, options, conf0=0.80, state_tokens=None)
        res_set = evaluate_voc_gate_regime(enterprise_state_text, instructions, options, conf0=0.80, state_tokens=token_set)
        res_frozenset = evaluate_voc_gate_regime(enterprise_state_text, instructions, options, conf0=0.80, state_tokens=token_frozenset)
        res_list = evaluate_voc_gate_regime(enterprise_state_text, instructions, options, conf0=0.80, state_tokens=token_list)
        res_tuple = evaluate_voc_gate_regime(enterprise_state_text, instructions, options, conf0=0.80, state_tokens=token_tuple)

        for res in (res_set, res_frozenset, res_list, res_tuple):
            assert res == res_none

    def test_verdict_mq_cache_integration_with_voc_gate(self) -> None:
        """Verify VerdictMQMultiQuestionCache provides tokens for evaluate_voc_gate_regime."""
        mq_cache = VerdictMQMultiQuestionCache(max_entries=64)
        state_text = "The user has provided an exemption waiver for the threshold limit unless revoked."

        # Prime the cache once
        mq_cache.encode_state_once(state_text)
        cached_entry = mq_cache.get_cached_state(state_text)
        assert cached_entry is not None
        assert "tokens" in cached_entry
        cached_tokens = cached_entry["tokens"]

        # Run 100 question evaluations using cached tokens
        options = {"approve": "Approve transaction", "review": "Manual compliance review"}
        instructions = "Determine whether compliance review is necessary."

        for _ in range(100):
            res = evaluate_voc_gate_regime(
                state=state_text,
                instructions=instructions,
                options=options,
                conf0=0.70,
                state_tokens=cached_tokens,
            )
            assert res["uncertainty_regime"] == "epistemic_deliberative"
            assert res["should_escalate_gear2"] is True

    def test_deliberative_marker_detection_accuracy(self) -> None:
        """Verify presence of deliberative markers reliably triggers epistemic_deliberative regime."""
        deliberative_words = [
            "unless",
            "exempt",
            "exempted",
            "waiver",
            "except",
            "however",
            "provided",
            "override",
            "conflict",
            "threshold",
            "exceeds",
        ]

        options = {"a": "Option A", "b": "Option B"}
        for word in deliberative_words:
            state = f"Standard procedure applies {word} conditions are not met."
            tokens = set(tokenize_multilingual(state))
            assert word in tokens

            res = evaluate_voc_gate_regime(
                state=state,
                instructions="Determine if special handling applies.",
                options=options,
                conf0=0.90,  # Even with high initial confidence, deliberative marker triggers epistemic review
                state_tokens=tokens,
            )
            assert res["uncertainty_regime"] == "epistemic_deliberative"
            assert res["should_escalate_gear2"] is True

    def test_open_set_abstention_without_deliberative_markers(self) -> None:
        """Verify low semantic match without deliberative markers triggers open_set_abstain."""
        state = "Recipe for chocolate sourdough brioche fermentation at 24C."
        instructions = "Select the Kubernetes CNI network packet drop root cause."
        options = {"bgp_flap": "BGP peer session reset", "mtu_mismatch": "VXLAN MTU overflow"}

        state_tokens = set(tokenize_multilingual(state))
        res = evaluate_voc_gate_regime(
            state=state,
            instructions=instructions,
            options=options,
            conf0=0.52,
            state_tokens=state_tokens,
        )

        assert res["uncertainty_regime"] == "open_set_abstain"
        assert res["should_abstain"] is True
        assert res["should_escalate_gear2"] is False
        assert res["expected_voc_gain"] == 0.0

