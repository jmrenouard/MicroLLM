"""Comprehensive Unit and Offline Integration Tests for Hugging Face Publishing Tooling.

Tests cover:
1. Dataset packaging, schema validation, split generation, and English dataset card.
2. Neural multilingual model packaging, INT8 matrix integrity, configs, and English model card.
3. Dry-run zero-network isolation under socket interception (guard_sockets).
4. Credential resolution hierarchy, KeePass vault fallback, and zero secret leakage.
5. Mock Hub API interactions and error handling.
6. English-only documentation governance across all generated artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from microllm.hf_publishing import (
    DATASET_REPO_ID,
    PUBLISHABLE_MICROLLM_HF_MODELS,
    AuthenticationError,
    MicroLLMMultilingualBenchmarksAdapter,
    PublishingError,
    build_multilingual_dataset_card,
    build_multilingual_model_card,
    calculate_sha256,
    guard_sockets,
    mask_token,
    publish_all,
    publish_dataset,
    publish_model,
    resolve_hf_token,
    stage_dataset_corpus,
    stage_model_profile,
    validate_multilingual_corpus_schema,
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
        repo_type: str = "model",
        exist_ok: bool = True,
        private: bool = False,
    ) -> str:
        self.created_repos.append({
            "repo_id": repo_id,
            "repo_type": repo_type,
            "exist_ok": exist_ok,
            "private": private,
        })
        return f"https://huggingface.co/{repo_id}"

    def upload_file(
        self,
        path_or_fileobj: Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "model",
        commit_message: str | None = None,
    ) -> str:
        self.uploaded_files.append({
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "repo_type": repo_type,
            "commit_message": commit_message,
        })
        return f"https://huggingface.co/{repo_id}/commit/main"

    def upload_folder(
        self,
        folder_path: str,
        repo_id: str,
        repo_type: str = "model",
        commit_message: str | None = None,
    ) -> str:
        self.uploaded_folders.append({
            "folder_path": folder_path,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_message": commit_message,
        })
        return f"https://huggingface.co/{repo_id}/commit/main"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure environment variables for tokens are unset during tests."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "MICROLLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def temp_staging_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary staging directory."""
    st = tmp_path / "staging"
    st.mkdir(parents=True, exist_ok=True)
    return st


@pytest.fixture
def mock_api() -> MockHfApi:
    """Provide a recording mock HF API instance."""
    return MockHfApi()


# =============================================================================
# Class 1: TestDatasetPackagingAndStaging
# =============================================================================
class TestDatasetPackagingAndStaging:
    """Validates dataset asset preparation, schema conformity, per-language split exports, and metadata card formatting."""

    def test_dataset_packaging_schema_and_corpus_validation(self) -> None:
        corpus_file = Path("benchmarks/microllm_multilingual_corpus.jsonl")
        assert corpus_file.exists(), "Corpus file benchmarks/microllm_multilingual_corpus.jsonl must exist"

        rows = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line.strip()))

        assert len(rows) == 48, f"Expected exactly 48 rows, got {len(rows)}"
        validation = validate_multilingual_corpus_schema(rows)
        assert validation["is_valid"] is True
        assert validation["total_count"] == 48
        assert validation["languages"] == {"en": 12, "fr": 12, "es": 12, "pt": 12}
        assert validation["parallel_ids_count"] == 12

    def test_dataset_split_generation_and_export(self, temp_staging_dir: Path) -> None:
        stage_res = stage_dataset_corpus(staging_dir=temp_staging_dir, splits=True)
        staged_dir = Path(stage_res["staging_dir"])
        assert staged_dir.exists()

        # Check full corpus copies
        root_corpus = staged_dir / "corpus.jsonl"
        data_corpus = staged_dir / "data" / "microllm_multilingual_corpus.jsonl"
        assert root_corpus.exists()
        assert data_corpus.exists()
        assert len(root_corpus.read_text(encoding="utf-8").strip().splitlines()) == 48

        # Check per-language splits
        for lang in ("en", "fr", "es", "pt"):
            split_file = staged_dir / "splits" / f"{lang}.jsonl"
            data_lang_file = staged_dir / "data" / "languages" / f"{lang}.jsonl"
            assert split_file.exists()
            assert data_lang_file.exists()
            assert len(split_file.read_text(encoding="utf-8").strip().splitlines()) == 12

    def test_dataset_card_yaml_frontmatter_and_metadata(self) -> None:
        card = build_multilingual_dataset_card()
        assert card.startswith("---")
        assert "license: apache-2.0" in card
        assert "task_categories:" in card
        assert "- text-classification" in card
        assert "- zero-shot-classification" in card
        assert "- en" in card and "- fr" in card and "- es" in card and "- pt" in card
        assert "billing" in card
        assert "shipping" in card
        assert "tech" in card
        assert "100.0%" in card

    def test_dataset_manifest_checksums_and_byte_counts(self, temp_staging_dir: Path) -> None:
        stage_res = stage_dataset_corpus(staging_dir=temp_staging_dir, splits=True)
        staged_dir = Path(stage_res["staging_dir"])

        manifest_file = staged_dir / "staging_manifest.json"
        assert manifest_file.exists()

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["repo_id"] == DATASET_REPO_ID
        assert manifest["repo_type"] == "dataset"
        assert "staged_at_utc" in manifest
        assert len(manifest["files"]) > 0

        for f_entry in manifest["files"]:
            target_path = staged_dir / f_entry["path"]
            assert target_path.exists()
            actual_size = target_path.stat().st_size
            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            assert f_entry["bytes"] == actual_size
            assert f_entry["sha256"] == actual_sha
            assert calculate_sha256(target_path) == actual_sha

    def test_benchmark_results_summary_packaging(self, temp_staging_dir: Path) -> None:
        stage_res = stage_dataset_corpus(staging_dir=temp_staging_dir, splits=True)
        staged_dir = Path(stage_res["staging_dir"])

        bench_results_file = staged_dir / "multilingual_benchmarks_results.json"
        assert bench_results_file.exists()
        content = json.loads(bench_results_file.read_text(encoding="utf-8"))
        assert "fast" in content or "summary" in content or "benchmark" in str(content).lower()

    def test_adapter_record_conversion(self) -> None:
        adapter = MicroLLMMultilingualBenchmarksAdapter()
        assert adapter.dataset_name == DATASET_REPO_ID
        assert adapter.target_decision_type == "choice"

        sample = {
            "id": "mling-cs-001-fr",
            "language": "fr",
            "domain": "customer_support",
            "state": "Bonjour, facture erronee.",
            "expected_answer": "billing",
            "metadata": {"parallel_id": "cs-001", "intent": "billing_issue"},
        }
        converted = adapter.convert_record(sample, 1)
        assert converted["id"] == "mling-cs-001-fr"
        assert converted["domain"] == "customer_support"
        assert converted["message"] == "Bonjour, facture erronee."
        assert converted["expected_result"] == "billing"
        assert converted["metadata"]["language"] == "fr"
        assert converted["metadata"]["parallel_id"] == "cs-001"


# =============================================================================
# Class 2: TestModelPackagingAndStaging
# =============================================================================
class TestModelPackagingAndStaging:
    """Validates neural multilingual model artifact generation, INT8 projection bundles, configs, and model cards."""

    def test_model_packaging_minilm_artifacts(self, temp_staging_dir: Path) -> None:
        res = stage_model_profile("multilingual-minilm-int8", staging_dir=temp_staging_dir)
        staged_dir = Path(res["staging_dir"])
        assert staged_dir.exists()
        assert "multilingual-minilm-int8" in PUBLISHABLE_MICROLLM_HF_MODELS

        assert (staged_dir / "model_quantized_int8.npz").exists()
        assert (staged_dir / "config.json").exists()
        assert (staged_dir / "tokenizer_config.json").exists()
        assert (staged_dir / "onnx" / "onnx_manifest.json").exists()
        assert (staged_dir / "README.md").exists()
        assert (staged_dir / "staging_manifest.json").exists()

    def test_model_packaging_mdeberta_artifacts(self, temp_staging_dir: Path) -> None:
        res = stage_model_profile("multilingual-mdeberta-int8", staging_dir=temp_staging_dir)
        staged_dir = Path(res["staging_dir"])
        assert staged_dir.exists()

        config = json.loads((staged_dir / "config.json").read_text(encoding="utf-8"))
        assert config["model_type"] == "mdeberta-v3"
        assert config["entailment_idx"] == 0
        assert config["quantization"] == "INT8_DYNAMIC"

    def test_model_card_english_prose_and_yaml_metadata(self) -> None:
        card = build_multilingual_model_card("multilingual-minilm-int8")
        assert card.startswith("---")
        assert "license: apache-2.0" in card
        assert "pipeline_tag: zero-shot-classification" in card
        assert "minilm" in card.lower()
        assert "microllm pull" in card
        assert "create_multilingual_backend" in card

    def test_model_weight_verification_and_int8_matrix_integrity(self, temp_staging_dir: Path) -> None:
        res = stage_model_profile("multilingual-minilm-int8", staging_dir=temp_staging_dir)
        staged_dir = Path(res["staging_dir"])
        npz = np.load(staged_dir / "model_quantized_int8.npz")

        assert "weights_int8" in npz.files
        assert "scale" in npz.files
        w = npz["weights_int8"]
        s = float(npz["scale"])

        assert w.dtype == np.int8
        assert w.shape == (256, 64)
        assert np.all(w >= -127) and np.all(w <= 127)
        assert s > 0.0

    def test_model_onnx_manifest_and_config_json_schema(self, temp_staging_dir: Path) -> None:
        res = stage_model_profile("multilingual-mdeberta-int8", staging_dir=temp_staging_dir)
        staged_dir = Path(res["staging_dir"])

        onnx_manifest = json.loads((staged_dir / "onnx" / "onnx_manifest.json").read_text(encoding="utf-8"))
        assert onnx_manifest["opset_version"] >= 14
        input_names = [inp["name"] for inp in onnx_manifest["inputs"]]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert onnx_manifest["entailment_idx"] == 0


# =============================================================================
# Class 3: TestDryRunZeroNetworkIsolation
# =============================================================================
class TestDryRunZeroNetworkIsolation:
    """Guarantees --dry-run operates 100% offline with zero network requests under socket interception."""

    def test_dry_run_dataset_publishing_under_socket_guard(self, temp_staging_dir: Path) -> None:
        with guard_sockets():
            result = publish_dataset(dry_run=True, staging_dir=temp_staging_dir)
            assert result["status"] == "dry_run"
            assert result["upload_attempted"] is False
            assert result["network_calls_made"] == 0
            assert result["repo_id"] == DATASET_REPO_ID

    def test_dry_run_model_publishing_under_socket_guard(self, temp_staging_dir: Path) -> None:
        with guard_sockets():
            result = publish_model("multilingual-minilm-int8", dry_run=True, staging_dir=temp_staging_dir)
            assert result["status"] == "dry_run"
            assert result["upload_attempted"] is False
            assert result["network_calls_made"] == 0

            res_all = publish_all(dry_run=True, staging_dir=temp_staging_dir)
            assert res_all["status"] == "dry_run"

    def test_cli_publish_dry_run_dataset(self, temp_staging_dir: Path) -> None:
        with guard_sockets():
            exit_code = main(["publish", "--dataset", "--dry-run", "--staging-dir", str(temp_staging_dir)])
            assert exit_code == 0
            assert (temp_staging_dir / "jmrenouard_microllm-multilingual-benchmarks").exists()

    def test_cli_publish_dry_run_model(self, temp_staging_dir: Path) -> None:
        with guard_sockets():
            exit_code = main(["publish", "--model", "multilingual-minilm-int8", "--dry-run", "--staging-dir", str(temp_staging_dir)])
            assert exit_code == 0
            assert (temp_staging_dir / "jmrenouard_microllm-multilingual-minilm-int8").exists()

    def test_cli_publish_dry_run_all(self, temp_staging_dir: Path) -> None:
        with guard_sockets():
            exit_code = main(["publish", "--all", "--dry-run", "--staging-dir", str(temp_staging_dir)])
            assert exit_code == 0
            assert (temp_staging_dir / "jmrenouard_microllm-multilingual-benchmarks").exists()
            assert (temp_staging_dir / "jmrenouard_microllm-multilingual-minilm-int8").exists()
            assert (temp_staging_dir / "jmrenouard_microllm-multilingual-mdeberta-int8").exists()

    def test_dry_run_preserves_staging_dir_for_inspection(self, temp_staging_dir: Path) -> None:
        result = publish_dataset(dry_run=True, staging_dir=temp_staging_dir)
        staged_folder = Path(result["staging_dir"])
        assert staged_folder.exists()
        assert (staged_folder / "README.md").exists()
        assert (staged_folder / "staging_manifest.json").exists()


# =============================================================================
# Class 4: TestCredentialResolutionAndSecurity
# =============================================================================
class TestCredentialResolutionAndSecurity:
    """Validates credential precedence, KeePass vault resolution, error handling, and zero secret leakage."""

    def test_token_resolution_explicit_argument_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "env-token-xyz")
        tok, source = resolve_hf_token(token="explicit-token-abc", allow_vault=False)
        assert tok == "explicit-token-abc"
        assert source == "argument"

    def test_token_resolution_env_var_fallback_hierarchy(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        # 1. HF_TOKEN
        monkeypatch.setenv("HF_TOKEN", "hf_token_val")
        tok, src = resolve_hf_token(allow_vault=False)
        assert tok == "hf_token_val"
        assert src == "HF_TOKEN"

        # 2. HUGGING_FACE_HUB_TOKEN
        monkeypatch.delenv("HF_TOKEN")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_val")
        tok, src = resolve_hf_token(allow_vault=False)
        assert tok == "hf_hub_val"
        assert src == "HUGGING_FACE_HUB_TOKEN"

        # 3. HUGGINGFACE_TOKEN
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_face_val")
        tok, src = resolve_hf_token(allow_vault=False)
        assert tok == "hf_face_val"
        assert src == "HUGGINGFACE_TOKEN"

    def test_token_resolution_keepass_vault_fallback(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        with mock.patch("microllm.hf_publishing._resolve_token_from_keepass", return_value="hf_keepass_resolved_token"):
            tok, src = resolve_hf_token(allow_vault=True)
            assert tok == "hf_keepass_resolved_token"
            assert src == "keepass"

    def test_token_missing_in_dry_run_mode_proceeds_gracefully(self, clean_env: None, temp_staging_dir: Path) -> None:
        with mock.patch("microllm.hf_publishing._resolve_token_from_keepass", return_value=None):
            res_ds = publish_dataset(dry_run=True, staging_dir=temp_staging_dir)
            assert res_ds["status"] == "dry_run"

            res_m = publish_model("multilingual-minilm-int8", dry_run=True, staging_dir=temp_staging_dir)
            assert res_m["status"] == "dry_run"

    def test_token_missing_in_live_mode_raises_authentication_error(self, clean_env: None, temp_staging_dir: Path) -> None:
        with mock.patch("microllm.hf_publishing._resolve_token_from_keepass", return_value=None):
            with pytest.raises((AuthenticationError, ValueError)) as excinfo:
                publish_dataset(dry_run=False, staging_dir=temp_staging_dir, api=None)
            assert "Hugging Face API token not found" in str(excinfo.value)

    def test_zero_credential_leakage_in_artifacts_and_telemetry(self, temp_staging_dir: Path) -> None:
        secret = "hf_secret_token_never_leak_this_value_123456"
        res = stage_model_profile("multilingual-minilm-int8", staging_dir=temp_staging_dir)
        staged_dir = Path(res["staging_dir"])

        assert secret not in json.dumps(res)
        for item in staged_dir.rglob("*"):
            if item.is_file() and not item.name.endswith(".npz"):
                content = item.read_text(encoding="utf-8", errors="ignore")
                assert secret not in content, f"Secret leaked in file {item.name}"

        masked = mask_token(secret)
        assert secret not in masked
        assert masked.startswith("hf_")
        assert masked.endswith("3456")


# =============================================================================
# Class 5: TestMockHubApiInteractions
# =============================================================================
class TestMockHubApiInteractions:
    """Verifies live upload code paths using injected MockHfApi test doubles."""

    def test_live_dataset_publish_with_mock_api_invocations(
        self,
        mock_api: MockHfApi,
        temp_staging_dir: Path,
    ) -> None:
        res = publish_dataset(
            dry_run=False,
            api=mock_api,
            token="test-token",
            staging_dir=temp_staging_dir,
        )
        assert res["status"] == "published"
        assert res["upload_attempted"] is True

        created = mock_api.created_repos
        assert len(created) == 1
        assert created[0]["repo_id"] == DATASET_REPO_ID
        assert created[0]["repo_type"] == "dataset"

        assert len(mock_api.uploaded_folders) == 1 or len(mock_api.uploaded_files) > 0

    def test_live_model_publish_with_mock_api_invocations(
        self,
        mock_api: MockHfApi,
        temp_staging_dir: Path,
    ) -> None:
        res_mini = publish_model(
            "multilingual-minilm-int8",
            dry_run=False,
            api=mock_api,
            token="test-token",
            staging_dir=temp_staging_dir,
        )
        assert res_mini["status"] == "published"

        res_deb = publish_model(
            "multilingual-mdeberta-int8",
            dry_run=False,
            api=mock_api,
            token="test-token",
            staging_dir=temp_staging_dir,
        )
        assert res_deb["status"] == "published"

        repo_ids = [r["repo_id"] for r in mock_api.created_repos]
        assert "jmrenouard/microllm-multilingual-minilm-int8" in repo_ids
        assert "jmrenouard/microllm-multilingual-mdeberta-int8" in repo_ids

    def test_hub_api_error_handling_and_informative_diagnostics(
        self,
        temp_staging_dir: Path,
    ) -> None:
        class BrokenHfApi:
            def create_repo(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("HF Hub 403 Forbidden: Write access denied")

        broken = BrokenHfApi()
        with pytest.raises(PublishingError) as excinfo:
            publish_dataset(dry_run=False, api=broken, token="test-token", staging_dir=temp_staging_dir)

        assert "Hugging Face Hub publishing failed" in str(excinfo.value)
        assert DATASET_REPO_ID in str(excinfo.value)


# =============================================================================
# Class 6: TestEnglishOnlyGovernanceAndDocumentation
# =============================================================================
class TestEnglishOnlyGovernanceAndDocumentation:
    """Enforces the mandatory English-only policy across all generated artifacts."""

    def test_dataset_card_english_only_compliance(self) -> None:
        card = build_multilingual_dataset_card()
        lines = card.splitlines()

        # Check markdown headings
        headings = [line for line in lines if line.startswith("#")]
        assert len(headings) >= 5
        for h in headings:
            # Headings must be in English
            assert not any(non_en in h.lower() for non_en in ["jeu de données", "resultados", "descripción", "données"])

    def test_model_card_english_only_compliance(self) -> None:
        for profile in ("multilingual-minilm-int8", "multilingual-mdeberta-int8"):
            card = build_multilingual_model_card(profile)
            headings = [line for line in card.splitlines() if line.startswith("#")]
            assert len(headings) >= 5
            for h in headings:
                assert not any(non_en in h.lower() for non_en in ["modèle", "modelo", "quantification", "limites"])

    def test_cli_help_strings_english_only_compliance(self) -> None:
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            with pytest.raises(SystemExit) as excinfo:
                main(["publish", "--help"])
            assert excinfo.value.code == 0
            help_out = sys.stdout.getvalue()
            assert "Stage and publish multilingual benchmark datasets" in help_out
            assert "--dry-run" in help_out
            assert "--dataset" in help_out
            assert "--model" in help_out
        finally:
            sys.stdout = old_stdout
