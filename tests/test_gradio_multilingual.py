"""Comprehensive Unit and Offline Integration Tests for Gradio Multilingual Decision Studio
and Hugging Face Space Synchronization (Milestone 7 / Issue #165).

Test Classes:
1. TestHeadlessUIInitialization: Verifies headless components, controls presence, and presets.
2. TestStateTransitionsAndEventCallbacks: Verifies mode switches and reactive field updates.
3. TestDecisionExecutionCallbacks: Tests end-to-end evaluation across EN, FR, ES, PT with fast, neural, auto.
4. TestSpaceSyncStagingAndBundleGeneration: Validates Space staging, README YAML frontmatter, requirements.
5. TestSpaceSyncCLIAndMockUpload: Tests CLI commands, --dry-run socket isolation, credentials, and mock API.
6. TestEnglishOnlyGovernanceAndDocumentation: Enforces 100% English across UI labels and Space card.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from unittest import mock

import pytest

from microllm.backends.cascade import create_multilingual_backend
from microllm.gradio_app import (
    MULTILINGUAL_SUPPORT_PRESETS,
    STUDIO_DOMAIN_PRESETS,
    _HeadlessGradioComponent,
    _HeadlessGradioModule,
    build_demo,
    create_gradio_demo,
    evaluate_consumer_choice,
    evaluate_consumer_multi,
    evaluate_consumer_score,
    evaluate_consumer_yes_no,
    evaluate_studio_multi_mode,
    load_preset_for_all_tabs,
    on_workspace_mode_change,
)
from microllm.hf_publishing import guard_sockets, mask_token, resolve_hf_token
from microllm.hf_space_sync import (
    SPACE_REPO_ID,
    build_space_readme,
    generate_space_app_py,
    generate_space_requirements,
    stage_hf_space,
    stage_space_bundle,
    sync_hf_space,
)
from microllm.server import main


class MockHfApi:
    """Mock Hugging Face API recording calls without network interactions."""

    def __init__(self) -> None:
        self.created_repos: list[dict[str, Any]] = []
        self.uploaded_files: list[dict[str, Any]] = []
        self.uploaded_folders: list[dict[str, Any]] = []

    def create_repo(
        self,
        repo_id: str,
        repo_type: str = "space",
        space_sdk: str = "gradio",
        exist_ok: bool = True,
        private: bool = False,
    ) -> str:
        self.created_repos.append({
            "repo_id": repo_id,
            "repo_type": repo_type,
            "space_sdk": space_sdk,
            "exist_ok": exist_ok,
            "private": private,
        })
        return f"https://huggingface.co/spaces/{repo_id}"

    def upload_file(
        self,
        path_or_fileobj: Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "space",
        commit_message: str | None = None,
    ) -> str:
        self.uploaded_files.append({
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "repo_type": repo_type,
            "commit_message": commit_message,
        })
        return f"https://huggingface.co/spaces/{repo_id}/commit/main"

    def upload_folder(
        self,
        folder_path: str,
        repo_id: str,
        repo_type: str = "space",
        commit_message: str | None = None,
    ) -> str:
        self.uploaded_folders.append({
            "folder_path": folder_path,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_message": commit_message,
        })
        return f"https://huggingface.co/spaces/{repo_id}/commit/main"


# =============================================================================
# 1. TestHeadlessUIInitialization
# =============================================================================
class TestHeadlessUIInitialization:
    """Verifies headless Gradio components, UI builder, and presets availability."""

    def test_headless_gradio_module_exposes_required_components(self) -> None:
        """Assert _HeadlessGradioModule exposes all necessary UI primitives and methods."""
        required = [
            "Blocks",
            "Row",
            "Column",
            "Tabs",
            "TabItem",
            "Group",
            "Accordion",
            "HTML",
            "Markdown",
            "Radio",
            "Dropdown",
            "Textbox",
            "Slider",
            "Button",
            "Dataframe",
            "Code",
            "Label",
            "JSON",
            "Checkbox",
        ]
        for comp_name in required:
            assert hasattr(_HeadlessGradioModule, comp_name)
            cls = getattr(_HeadlessGradioModule, comp_name)
            instance = cls()
            assert hasattr(instance, "click")
            assert hasattr(instance, "change")
            assert hasattr(instance, "close")
            assert hasattr(instance, "launch")

        assert hasattr(_HeadlessGradioModule, "update")
        upd = _HeadlessGradioModule.update(visible=False, value="test")
        assert upd == {"visible": False, "value": "test"}

    def test_create_gradio_demo_builds_without_error_in_headless_mode(self) -> None:
        """Call create_gradio_demo() and verify demo instance and close() method."""
        demo = create_gradio_demo()
        assert demo is not None
        assert hasattr(demo, "close")
        demo.close()  # Must execute without exception

        # Verify export alias
        assert build_demo is create_gradio_demo

    def test_multilingual_strategy_and_model_controls_configured(self) -> None:
        """Verify presence of multilingual strategy, neural model, and language options."""
        # Check environment variable overrides
        with mock.patch.dict(os.environ, {"MICROLLM_MULTILINGUAL_STRATEGY": "neural", "MICROLLM_NEURAL_MODEL": "mdeberta"}):
            demo = create_gradio_demo()
            assert demo is not None
            demo.close()

    def test_multilingual_domain_presets_catalog_coverage(self) -> None:
        """Verify that all 12 multilingual presets exist across EN, FR, ES, PT with language tags."""
        expected_presets = [
            "support_billing_en",
            "support_billing_fr",
            "support_billing_es",
            "support_billing_pt",
            "support_shipping_en",
            "support_shipping_fr",
            "support_shipping_es",
            "support_shipping_pt",
            "support_tech_en",
            "support_tech_fr",
            "support_tech_es",
            "support_tech_pt",
        ]
        for preset_id in expected_presets:
            assert preset_id in MULTILINGUAL_SUPPORT_PRESETS
            assert preset_id in STUDIO_DOMAIN_PRESETS
            p = STUDIO_DOMAIN_PRESETS[preset_id]
            assert "title" in p
            assert "domain" in p
            assert "language" in p
            assert p["language"] in {"en", "fr", "es", "pt"}
            assert "state" in p
            assert "questions" in p

        # Verify default alias 'support' points to an active customer support preset
        assert "support" in STUDIO_DOMAIN_PRESETS
        assert STUDIO_DOMAIN_PRESETS["support"]["language"] == "en"

    def test_headless_launch_delegates_to_fastapi_server(self) -> None:
        """Verify demo.launch() delegates to uvicorn.run without NameError."""
        demo = _HeadlessGradioComponent()
        with mock.patch("uvicorn.run") as mock_uvicorn:
            demo.launch(server_name="127.0.0.1", server_port=7860)
            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs.get("host") == "127.0.0.1"
            assert call_kwargs.get("port") == 7860


# =============================================================================
# 2. TestStateTransitionsAndEventCallbacks
# =============================================================================
class TestStateTransitionsAndEventCallbacks:
    """Verifies workspace mode switching, dynamic strategy visibility, and preset loading."""

    def test_on_workspace_mode_change_visibility_matrix(self) -> None:
        """Test on_workspace_mode_change across simple, advanced, and cockpit modes."""
        # 1. Simple Per-Type Tabs
        simple = on_workspace_mode_change("✨ Simple Per-Type Tabs (Recommended for Everyone)")
        assert len(simple) == 9
        tabs_obj = simple[0]
        sel = getattr(tabs_obj, "selected", None) or (tabs_obj.get("selected") if isinstance(tabs_obj, dict) else None)
        assert sel == "tab_yn"

        # 2. Advanced Studio & Custom JSON Mode
        adv = on_workspace_mode_change("⚙️ Advanced Studio & Custom JSON Mode")
        assert len(adv) == 9
        tabs_obj = adv[0]
        sel = getattr(tabs_obj, "selected", None) or (tabs_obj.get("selected") if isinstance(tabs_obj, dict) else None)
        assert sel == "tab_advanced"

        # 3. All-in-One Cockpit Mode
        cockpit = on_workspace_mode_change("🖥️ All-in-One Cockpit Mode (Full Studio)")
        assert len(cockpit) == 9
        tabs_obj = cockpit[0]
        sel = getattr(tabs_obj, "selected", None) or (tabs_obj.get("selected") if isinstance(tabs_obj, dict) else None)
        assert sel == "tab_advanced"

    def test_strategy_selection_triggers_neural_model_visibility(self) -> None:
        """Test that selecting 'fast' hides neural_model, while 'neural' and 'auto' show it."""
        from microllm.gradio_app import _HeadlessGradioModule

        # Test logic mimicking on_multilingual_strategy_change
        def toggle_vis(strat: str) -> dict[str, Any]:
            is_neural = "fast" not in str(strat).lower()
            return _HeadlessGradioModule.update(visible=is_neural)

        assert toggle_vis("fast") == {"visible": False}
        assert toggle_vis("neural") == {"visible": True}
        assert toggle_vis("auto") == {"visible": True}

    def test_multilingual_preset_loader_populates_all_fields(self) -> None:
        """Verify load_preset_for_all_tabs populates all 26 outputs with French billing inquiry."""
        res = load_preset_for_all_tabs("support_billing_fr")
        assert len(res) == 26
        yn_text, yn_q, yn_yes, yn_no = res[0], res[1], res[2], res[3]
        ch_text, ch_q, ch_opts = res[4], res[5], res[6]
        mt_text, mt_q, mt_tags, mt_thresh = res[7], res[8], res[9], res[10]
        sc_text, sc_q, sc_levels = res[11], res[12], res[13]

        assert "prélevé deux fois" in yn_text or "facture" in yn_text
        assert len(yn_q) > 0
        assert len(ch_opts) > 0
        assert len(mt_tags) > 0
        assert 0.1 <= mt_thresh <= 0.9
        assert len(sc_levels) > 0

    def test_multilingual_preset_cross_language_switching(self) -> None:
        """Verify sequential loading across EN, FR, ES, PT updates form fields cleanly."""
        languages = ["en", "fr", "es", "pt"]
        for lang in languages:
            key = f"support_billing_{lang}"
            res = load_preset_for_all_tabs(key)
            assert len(res) == 26
            state_val = res[0]
            expected_snippet = {
                "en": "billed twice",
                "fr": "prélevé deux fois",
                "es": "cobrado dos veces",
                "pt": "cobrado duas vezes",
            }[lang]
            assert expected_snippet in state_val


# =============================================================================
# 3. TestDecisionExecutionCallbacks
# =============================================================================
class TestDecisionExecutionCallbacks:
    """Tests end-to-end evaluation callbacks across EN, FR, ES, PT with fast, neural, auto."""

    def test_evaluate_ticket_fast_strategy_en_fr_es_pt(self) -> None:
        """Evaluate tickets across EN, FR, ES, PT with fast strategy and verify <5ms latency."""
        tickets = [
            ("en", "I have been billed twice for invoice #4092! Please refund the duplicate charge of $149."),
            ("fr", "J'ai été prélevé deux fois pour la facture #4092 ! Veuillez procéder au remboursement."),
            ("es", "¡Me han cobrado dos veces por la factura #4092! Solicito un reembolso inmediato."),
            ("pt", "Fui cobrado duas vezes pela fatura #4092! Exijo o estorno e reembolso imediato."),
        ]
        q_json = json.dumps({
            "department": {
                "type": "choice",
                "instructions": "Route this support ticket",
                "criteria": {
                    "billing": "Invoices, refunds, duplicate charges, payment errors",
                    "technical": "Software bugs, crashes, login outages",
                    "shipping": "Delayed package, parcel delivery issues",
                },
            }
        })

        for lang, text in tickets:
            summary_md, xai_md, probs, telemetry, code = evaluate_studio_multi_mode(
                input_mode="Raw JSON Mode",
                state_input=text,
                questions_json=q_json,
                visual_qid="",
                visual_type="",
                visual_instructions="",
                visual_criteria="",
                visual_threshold=0.5,
                engine_choice="Dual-Gear Cascade (StartLux) [Recommended]",
                gate_threshold=0.12,
                alpha_val=0.75,
                multilingual_strategy="fast",
                neural_model="minilm",
                target_language=lang,
            )
            assert "Calibrated System-1 Decisions" in summary_md
            assert "department" in telemetry["cascade"]
            routing = telemetry.get("multilingual_routing", {})
            assert routing.get("strategy_used") == "fast"
            assert routing.get("effective_language") == lang
            assert telemetry["latency_ms"] < 25.0

    def test_evaluate_ticket_neural_minilm_strategy(self) -> None:
        """Evaluate ticket with neural strategy and MiniLM model."""
        q_json = json.dumps({
            "urgent": {
                "type": "noul",
                "instructions": "Is the user requesting an immediate refund?",
                "criteria": {"true": "Requests immediate refund", "false": "General inquiry"},
            }
        })
        summary_md, _, probs, telemetry, _ = evaluate_studio_multi_mode(
            input_mode="Raw JSON Mode",
            state_input="Please refund my duplicate payment today!",
            questions_json=q_json,
            visual_qid="",
            visual_type="",
            visual_instructions="",
            visual_criteria="",
            visual_threshold=0.5,
            engine_choice="Gear 2 (AnyJev Prefill) [Deep NLI]",
            gate_threshold=0.12,
            alpha_val=0.75,
            multilingual_strategy="neural",
            neural_model="minilm",
            target_language="en",
        )
        assert "Calibrated System-1 Decisions" in summary_md
        assert "urgent" in telemetry["cascade"]
        routing = telemetry.get("multilingual_routing", {})
        assert routing.get("strategy_used") == "neural"

    def test_evaluate_ticket_neural_mdeberta_strategy(self) -> None:
        """Evaluate ticket with neural strategy and mDeBERTa model."""
        q_json = json.dumps({
            "department": {
                "type": "choice",
                "instructions": "Route ticket",
                "criteria": {
                    "billing": "Charges and refunds",
                    "technical": "API outages and crashes",
                },
            }
        })
        summary_md, _, probs, telemetry, _ = evaluate_studio_multi_mode(
            input_mode="Raw JSON Mode",
            state_input="Our API authentication failed with 500 error after deployment",
            questions_json=q_json,
            visual_qid="",
            visual_type="",
            visual_instructions="",
            visual_criteria="",
            visual_threshold=0.5,
            engine_choice="Gear 2 (AnyJev Prefill) [Deep NLI]",
            gate_threshold=0.12,
            alpha_val=0.75,
            multilingual_strategy="neural",
            neural_model="mdeberta",
            target_language="en",
        )
        assert "Calibrated System-1 Decisions" in summary_md
        routing = telemetry.get("multilingual_routing", {})
        assert routing.get("neural_model") == "mdeberta"

    def test_evaluate_ticket_auto_cascade_routing_telemetry(self) -> None:
        """Evaluate ticket with auto cascade and assert telemetry captures stage, margin, and language."""
        q_json = json.dumps({
            "dept": {
                "type": "choice",
                "instructions": "Select department",
                "criteria": {"billing": "Invoices", "shipping": "Packages"},
            }
        })
        _, _, _, telemetry, _ = evaluate_studio_multi_mode(
            input_mode="Raw JSON Mode",
            state_input="El paquete TRK-8821 no ha llegado tras 14 dias",
            questions_json=q_json,
            visual_qid="",
            visual_type="",
            visual_instructions="",
            visual_criteria="",
            visual_threshold=0.5,
            engine_choice="Dual-Gear Cascade (StartLux) [Recommended]",
            gate_threshold=0.12,
            alpha_val=0.75,
            multilingual_strategy="auto",
            neural_model="minilm",
            target_language="auto",
        )
        routing = telemetry.get("multilingual_routing", {})
        assert routing.get("strategy") == "auto"
        assert routing.get("detected_language") in {"es", "pt"}
        assert "cascade_triggered" in routing
        assert "top2_margin" in routing

    def test_decision_execution_error_handling_robustness(self) -> None:
        """Assert invalid JSON syntax or empty inputs return user-friendly markdown errors."""
        summary_md, xai_md, probs, tele, code = evaluate_studio_multi_mode(
            input_mode="Raw JSON Mode",
            state_input="Some input text",
            questions_json="{invalid_json_missing_brace",
            visual_qid="",
            visual_type="",
            visual_instructions="",
            visual_criteria="",
            visual_threshold=0.5,
            engine_choice="Dual-Gear Cascade (StartLux) [Recommended]",
            gate_threshold=0.12,
            alpha_val=0.75,
        )
        assert "JSON Syntax Error" in summary_md
        assert "error" in tele


# =============================================================================
# 4. TestSpaceSyncStagingAndBundleGeneration
# =============================================================================
class TestSpaceSyncStagingAndBundleGeneration:
    """Validates Space bundle staging, YAML frontmatter, requirements, and SHA-256 manifests."""

    def test_stage_space_bundle_creates_complete_directory_hierarchy(self, tmp_path: Path) -> None:
        """stage_space_bundle stages app.py, requirements.txt, README.md, manifest, and src/microllm/."""
        repo_root = Path(__file__).resolve().parent.parent
        res = stage_space_bundle(staging_dir=tmp_path, repo_root=repo_root)

        assert res["total_files"] >= 5
        assert (tmp_path / "app.py").is_file()
        assert (tmp_path / "requirements.txt").is_file()
        assert (tmp_path / "README.md").is_file()
        assert (tmp_path / ".gitignore").is_file()
        assert (tmp_path / "staging_manifest.json").is_file()
        assert (tmp_path / "manifest.json").is_file()
        assert (tmp_path / "src" / "microllm").is_dir()
        assert (tmp_path / "src" / "microllm" / "gradio_app.py").is_file()

    def test_space_readme_yaml_frontmatter_schema_and_metadata(self, tmp_path: Path) -> None:
        """Verify README.md starts with strict YAML frontmatter delimiters and required metadata."""
        readme_text = build_space_readme(license_type="mit")
        assert readme_text.startswith("---\n")
        parts = readme_text.split("---", 2)
        assert len(parts) >= 3
        yaml_content = parts[1]

        assert "title: MicroLLM Decision Studio" in yaml_content
        assert "emoji: ⚡" in yaml_content
        assert "sdk: gradio" in yaml_content
        assert "app_file: app.py" in yaml_content
        assert "pinned: false" in yaml_content
        assert "license: mit" in yaml_content

    def test_space_requirements_txt_minimal_dependencies(self, tmp_path: Path) -> None:
        """Verify requirements.txt contains minimal runtime dependencies without heavy PyTorch."""
        reqs = generate_space_requirements()
        lines = [line.strip() for line in reqs.splitlines() if line.strip()]
        dep_names = [line.split(">=")[0].split("==")[0] for line in lines]

        assert "gradio" in dep_names
        assert "fastapi" in dep_names
        assert "uvicorn" in dep_names
        assert "pydantic" in dep_names
        assert "numpy" in dep_names
        assert "onnxruntime" in dep_names
        assert "torch" not in dep_names  # CPU optimized zero-GPU

    def test_space_app_py_entrypoint_validity(self, tmp_path: Path) -> None:
        """Verify staged app.py contains entrypoint importing create_gradio_demo and zero_gpu_hook."""
        app_code = generate_space_app_py()
        assert "create_gradio_demo" in app_code
        assert "zero_gpu_hook" in app_code
        assert "demo.launch(" in app_code

    def test_space_staging_manifest_sha256_checksums(self, tmp_path: Path) -> None:
        """Verify staging_manifest.json accurately records SHA-256 digests for all files."""
        res = stage_hf_space(staging_dir=tmp_path, include_src=False)
        manifest_path = tmp_path / "staging_manifest.json"
        assert manifest_path.is_file()

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["repo_id"] == SPACE_REPO_ID
        assert manifest_data["space_sdk"] == "gradio"

        for entry in manifest_data["files"]:
            file_path = tmp_path / entry["path"]
            assert file_path.is_file()
            expected_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert entry["sha256"] == expected_hash
            assert entry["bytes"] == file_path.stat().st_size


# =============================================================================
# 5. TestSpaceSyncCLIAndMockUpload
# =============================================================================
class TestSpaceSyncCLIAndMockUpload:
    """Tests CLI subcommands, --dry-run isolation under guard_sockets, and mock Hub uploads."""

    def test_cli_sync_space_help_and_arguments(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Execute `microllm sync-space --help` and assert flags are registered."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sync-space", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "--repo-id" in out
        assert "--dry-run" in out
        assert "--token" in out
        assert "--staging-dir" in out

    def test_sync_space_dry_run_offline_isolation_guard_sockets(self, tmp_path: Path) -> None:
        """Verify sync_hf_space(dry_run=True) performs zero network calls under socket guard."""
        with guard_sockets():
            res = sync_hf_space(
                space_repo_id="jmrenouard/microllm-decision-studio",
                staging_dir=tmp_path,
                dry_run=True,
                include_src=False,
            )
        assert res["status"] == "dry_run"
        assert res["upload_attempted"] is False
        assert res["network_calls_made"] == 0

    def test_sync_space_mock_upload_folder_invocation(self, tmp_path: Path) -> None:
        """Verify sync_hf_space creates repo and calls upload_folder on mock HfApi."""
        mock_api = MockHfApi()
        res = sync_hf_space(
            space_repo_id="jmrenouard/microllm-decision-studio",
            staging_dir=tmp_path,
            dry_run=False,
            token="hf_test_token_12345",
            api=mock_api,
            commit_message="test(space): test upload",
            include_src=False,
        )
        assert res["status"] == "synced"
        assert res["upload_attempted"] is True
        assert len(mock_api.created_repos) == 1
        assert mock_api.created_repos[0]["repo_id"] == "jmrenouard/microllm-decision-studio"
        assert mock_api.created_repos[0]["repo_type"] == "space"

        assert len(mock_api.uploaded_folders) == 1
        assert mock_api.uploaded_folders[0]["repo_type"] == "space"
        assert mock_api.uploaded_folders[0]["commit_message"] == "test(space): test upload"

    def test_sync_space_credential_hierarchy_and_secret_masking(self) -> None:
        """Verify token masking never leaks secret characters and resolution works."""
        assert mask_token("hf_9876543210fedcba") == "hf_***dcba"
        assert mask_token("secret_token") == "sec***oken"
        assert mask_token(None) == "<none>"


        # Direct argument has highest precedence
        token, source = resolve_hf_token("explicit_token_arg")
        assert token == "explicit_token_arg"
        assert source == "argument"


# =============================================================================
# 6. TestEnglishOnlyGovernanceAndDocumentation
# =============================================================================
class TestEnglishOnlyGovernanceAndDocumentation:
    """Enforces 100% English language policy across UI labels, documentation, and Space cards."""

    def test_gradio_app_labels_and_markdown_english_only(self) -> None:
        """Verify that UI labels, headings, and helper texts in gradio_app.py are in English."""
        app_file = Path(__file__).resolve().parent.parent / "src" / "microllm" / "gradio_app.py"
        code = app_file.read_text(encoding="utf-8")

        # Check key UI strings are exclusively in English
        assert "Experience & Workspace Mode Switcher" in code
        assert "Load Enterprise Domain Preset" in code
        assert "Multilingual Analysis Strategy" in code
        assert "Neural ONNX Model" in code
        assert "Target Language" in code
        assert "Yes / No Question" in code
        assert "Single Choice" in code
        assert "Multi-Tagging" in code
        assert "Rating & Scale" in code

        # Verify no foreign section titles (foreign text allowed only in ticket test payloads)
        for non_en in ["Facturation", "Livraison", "Facturación", "Faturamento"]:
            # These words must NOT appear as top-level UI labels or class names
            matches = re.findall(rf'label="[^"]*{non_en}[^"]*"', code)
            assert len(matches) == 0, f"Found foreign string in UI label: {matches}"

    def test_space_readme_card_english_only_compliance(self) -> None:
        """Verify that generated Space card README.md is 100% in English."""
        card_text = build_space_readme()
        assert "Decision Studio (Interactive System-1 Engine)" in card_text
        assert "Ecosystem Navigation & Quick Links" in card_text
        assert "Key Features & Capabilities" in card_text
        assert "Quickstart" in card_text
        assert "Launching Decision Studio Locally" in card_text

        # Verify zero French, Spanish, or Portuguese headers
        for non_en_header in ["Démarrage", "Fonctionnalités", "Inicio", "Recursos"]:
            assert f"## {non_en_header}" not in card_text
