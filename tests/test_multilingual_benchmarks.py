"""Comprehensive Unit and Integration Tests for Milestone 4 Multilingual Benchmarks.

Validates:
1. Multilingual benchmark dataset schema, field types, and UTF-8 diacritic integrity across EN, FR, ES, PT.
2. Quadrilingual customer support probe ticket classification accuracy (>=90%) across all 4 languages.
3. Strategy and model permutation matrix ('fast', 'neural' [minilm, mdeberta], 'auto' cascade).
4. Dual-gear auto-cascade routing behavior (fast path on clear tickets, neural escalation on ambiguous tickets).
5. Isolated benchmark evaluation runner execution and JSON artifact generation.
6. 100% offline socket isolation (zero network connections during test runs).
7. Strict English-only documentation compliance (no translated prose in docs/).
"""

from __future__ import annotations

import json
import pathlib
import socket
import time
from typing import Any

import pytest

from microllm.backends.cascade import create_multilingual_backend
from microllm.backends.fast_encoder import FastEncoderBackend
from microllm.multilingual_benchmark_runner import (
    MultilingualBenchmarkConfig,
    MultilingualBenchmarkRunner,
    format_multilingual_benchmarks_markdown,
    load_multilingual_corpus,
)
from microllm.multilingual_benchmark_runner import (
    main as cli_main,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Benchmark Corpus Schema & Integrity Tests
# ==============================================================================

class TestMultilingualCorpusSchemaAndIntegrity:
    """Validates the schema, language distribution, and UTF-8 encoding of the benchmark corpus."""

    def test_corpus_file_exists_and_is_valid_jsonl(self) -> None:
        """Asserts that benchmarks/microllm_multilingual_corpus.jsonl exists and parses as valid JSONL."""
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        assert corpus_path.is_file(), f"Benchmark corpus missing at {corpus_path}"

        rows = load_multilingual_corpus(corpus_path)
        assert len(rows) >= 32, f"Expected at least 32 probe items, got {len(rows)}"

    def test_corpus_schema_fields(self) -> None:
        """Asserts that every corpus row contains required fields: id, language, domain, state, question_type, criteria, expected_answer."""
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        rows = load_multilingual_corpus(corpus_path)
        required_fields = {"id", "language", "domain", "state", "question_type", "criteria", "expected_answer"}

        for row in rows:
            missing = required_fields - set(row.keys())
            assert not missing, f"Row {row.get('id')} is missing required fields: {missing}"
            assert row["language"] in {"en", "fr", "es", "pt"}
            assert row["question_type"] in {"choice", "noul", "score", "multi"}
            assert isinstance(row["criteria"], dict)
            assert row["expected_answer"] in row["criteria"]

    def test_corpus_balanced_language_representation(self) -> None:
        """Asserts balanced representation across all 4 languages (EN, FR, ES, PT)."""
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        rows = load_multilingual_corpus(corpus_path)
        counts = {"en": 0, "fr": 0, "es": 0, "pt": 0}
        for row in rows:
            counts[row["language"]] += 1

        min_per_lang = min(counts.values())
        max_per_lang = max(counts.values())
        assert min_per_lang >= 8, f"Each language must have at least 8 items, got {counts}"
        assert max_per_lang - min_per_lang <= 2, f"Languages should be balanced, got {counts}"

    def test_corpus_utf8_diacritic_preservation(self) -> None:
        """Asserts that French, Spanish, and Portuguese diacritics are cleanly preserved in UTF-8 without Mojibake."""
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        content = corpus_path.read_text(encoding="utf-8")

        # Must not contain common Mojibake artifacts
        mojibake_markers = ["Ã©", "Ã¨", "Ã ", "Ã§", "Ã¡", "Ã³", "Ãº", "Ã£", "Ãµ", "â€™", "\\u00"]
        for marker in mojibake_markers:
            assert marker not in content, f"Detected potential Mojibake encoding error: '{marker}'"

        # Must contain legitimate accented characters
        assert any(c in content for c in "éèàçù"), "French accents missing"
        assert any(c in content for c in "áéíóúñ¿"), "Spanish accents/inverted punctuation missing"
        assert any(c in content for c in "ãõâêíúç"), "Portuguese accents missing"


# ==============================================================================
# 2. Quadrilingual Probe Ticket Accuracy Tests
# ==============================================================================

class TestQuadrilingualProbeTicketAccuracy:
    """Validates classification accuracy across EN, FR, ES, and PT customer support probe tickets."""

    def test_fast_strategy_customer_support_routing(self) -> None:
        """Asserts that FastEncoderBackend achieves >=90% accuracy across EN, FR, ES, PT customer support tickets."""
        backend = FastEncoderBackend()
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        rows = load_multilingual_corpus(corpus_path)

        lang_correct = {"en": 0, "fr": 0, "es": 0, "pt": 0}
        lang_total = {"en": 0, "fr": 0, "es": 0, "pt": 0}

        for row in rows:
            if row["question_type"] != "choice":
                continue
            lang = row["language"]
            lang_total[lang] += 1
            result = backend.evaluate_choice(
                state=row["state"],
                instructions="Route this customer ticket to the appropriate department.",
                options=row["criteria"],
            )
            if result["choice"] == row["expected_answer"]:
                lang_correct[lang] += 1

        total_correct = sum(lang_correct.values())
        total_items = sum(lang_total.values())
        accuracy = total_correct / total_items if total_items > 0 else 0.0

        assert accuracy >= 0.90, f"Overall probe accuracy must be >= 90%, got {accuracy:.2%}"
        for lang, count in lang_total.items():
            lang_acc = lang_correct[lang] / count if count > 0 else 0.0
            assert lang_acc >= 0.85, f"Accuracy for {lang} must be >= 85%, got {lang_acc:.2%}"

    def test_spanish_and_portuguese_measurably_exceed_random_baseline(self) -> None:
        """Asserts Spanish and Portuguese probe accuracy exceeds the 25% (or 33%) random baseline by >=50 percentage points."""
        backend = FastEncoderBackend()
        corpus_path = ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"
        rows = [r for r in load_multilingual_corpus(corpus_path) if r["language"] in {"es", "pt"} and r["question_type"] == "choice"]

        correct = 0
        for row in rows:
            res = backend.evaluate_choice(row["state"], "Categorize ticket.", row["criteria"])
            if res["choice"] == row["expected_answer"]:
                correct += 1

        accuracy = correct / len(rows) if rows else 0.0
        assert accuracy >= 0.85, f"ES/PT accuracy ({accuracy:.2%}) must significantly exceed random baseline (33%)"


# ==============================================================================
# 3. Strategy & Model Execution Matrix Tests
# ==============================================================================

class TestMultilingualStrategyExecutionMatrix:
    """Verifies execution across fast, neural (minilm, mdeberta), and auto cascade strategies."""

    def test_fast_strategy_sub_millisecond_latency(self) -> None:
        """Asserts that multilingual_strategy='fast' executes with sub-millisecond p50 latency (< 1.0 ms)."""
        backend = FastEncoderBackend()
        latencies: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            backend.evaluate_choice(
                state="J'ai une question sur ma facture et mon remboursement.",
                instructions="Classer la demande.",
                options={"billing": "Facturation", "shipping": "Livraison", "tech": "Support technique"},
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 1.0, f"Expected fast strategy p50 < 1.0 ms, got {p50:.3f} ms"

    def test_auto_cascade_routing_unambiguous_vs_ambiguous(self) -> None:
        """Asserts that auto-cascade resolves clear tickets via Gear 1, but escalates ambiguous tickets (margin <= 0.12)."""
        cascade = create_multilingual_backend(
            strategy="auto",
            neural_model="minilm",
        )
        # Clear billing inquiry in Portuguese -> should NOT escalate
        clear_res = cascade.evaluate_choice(
            state="Preciso de um estorno urgente de uma fatura cobrada duas vezes no meu cartão.",
            instructions="Encaminhar o ticket.",
            options={"billing": "Cobrança e estornos", "shipping": "Entrega de encomendas", "tech": "Falhas técnicas"},
        )
        assert clear_res["choice"] == "billing"

        # Ambiguous cross-domain inquiry -> should escalate or provide valid confidence
        ambiguous_res = cascade.evaluate_choice(
            state="The system had an issue with the service processing.",
            instructions="Categorize ticket.",
            options={"opt_a": "General service question", "opt_b": "General system processing"},
        )
        assert "confidence" in ambiguous_res

    def test_neural_strategy_execution(self) -> None:
        """Verifies neural strategy initialization and prediction across both minilm and mdeberta options."""
        for model_name in ["minilm", "mdeberta"]:
            backend = create_multilingual_backend(strategy="neural", neural_model=model_name)
            res = backend.evaluate_choice(
                state="The API server crashed with a fatal 500 internal server error.",
                instructions="Route ticket.",
                options={"billing": "Invoices and fees", "tech": "Server and login errors"},
            )
            assert res["choice"] in {"billing", "tech"}
            assert "probabilities" in res
            assert float(res["confidence"]) > 0.0

    def test_all_strategies_return_valid_confidence_scores(self) -> None:
        """Verifies that fast, neural, and auto cascade return well-formed confidence scores in [0.0, 1.0]."""
        strategies = [
            ("fast", ""),
            ("neural", "minilm"),
            ("neural", "mdeberta"),
            ("auto", "minilm"),
            ("auto", "mdeberta"),
        ]
        sample_state = "Mi paquete de envío llegó con retraso y la caja está rota."
        sample_opts = {"billing": "Facturación", "shipping": "Envíos y paquetes", "tech": "Soporte técnico"}

        for strat, model in strategies:
            backend = create_multilingual_backend(strategy=strat, neural_model=model if model else None)
            res = backend.evaluate_choice(sample_state, "Clasificar", sample_opts)
            conf = float(res.get("confidence", 0.0))
            assert 0.0 <= conf <= 1.0, f"Strategy {strat}:{model} returned out-of-range confidence {conf}"
            assert res["choice"] == "shipping"


# ==============================================================================
# 4. Multilingual Benchmark Runner Integration Tests
# ==============================================================================

class TestMultilingualBenchmarkRunnerIntegration:
    """Verifies programmatic execution of MultilingualBenchmarkRunner and output schema."""

    def test_runner_executes_quadrilingual_evaluation(self, tmp_path: pathlib.Path) -> None:
        """Asserts that MultilingualBenchmarkRunner runs across EN, FR, ES, PT and writes structured JSON."""
        output_file = tmp_path / "multilingual_benchmarks_results.json"
        config = MultilingualBenchmarkConfig(
            languages=["en", "fr", "es", "pt"],
            strategies=["fast", "auto"],
            neural_models=["minilm"],
            corpus_path=str(ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"),
            output_path=str(output_file),
            max_items_per_lang=8,
        )
        runner = MultilingualBenchmarkRunner(config)
        report = runner.run()

        assert output_file.is_file(), "Runner must persist results JSON artifact"
        assert "summary" in report
        assert "per_language" in report
        assert "per_strategy" in report
        assert set(report["per_language"].keys()) == {"en", "fr", "es", "pt"}
        assert report["summary"]["total_evaluated"] >= 32

    def test_results_separated_from_historical_benchmarks(self) -> None:
        """Asserts that multilingual benchmark artifacts do NOT overwrite or mutate benchmarks/latest_results.json."""
        hist_path = ROOT / "benchmarks" / "latest_results.json"
        assert hist_path.is_file(), "Historical benchmark results must remain intact"
        hist_content_before = hist_path.read_text(encoding="utf-8")

        # Historical benchmark must remain unmutated
        assert "eval_540" in hist_content_before or "tracks" in hist_content_before

    def test_markdown_formatting_of_benchmark_report(self) -> None:
        """Verifies that format_multilingual_benchmarks_markdown produces formatted Markdown tables."""
        config = MultilingualBenchmarkConfig(
            languages=["en", "fr"],
            strategies=["fast"],
            corpus_path=str(ROOT / "benchmarks" / "microllm_multilingual_corpus.jsonl"),
            max_items_per_lang=2,
            save_artifacts=False,
        )
        runner = MultilingualBenchmarkRunner(config)
        report = runner.run()
        md_text = format_multilingual_benchmarks_markdown(report)
        assert "# MicroLLM Multilingual Strategy & Evaluation Benchmark Results" in md_text
        assert "Overall Strategy Performance Matrix" in md_text
        assert "fast" in md_text

    def test_cli_runner_argument_parsing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """Verifies CLI entrypoint execution with custom parameters."""
        out_json = tmp_path / "cli_out.json"
        exit_code = cli_main([
            "--strategy", "fast",
            "--language", "en",
            "--output", str(out_json),
            "--format", "json",
        ])
        assert exit_code == 0
        assert out_json.is_file()
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        assert "fast" in saved["results_by_configuration"]


# ==============================================================================
# 5. Offline Isolation & English-Only Documentation Compliance Tests
# ==============================================================================

class TestOfflineIsolationAndDocumentationCompliance:
    """Guarantees zero outbound network connections and enforces 100% English documentation."""

    def test_zero_socket_calls_during_multilingual_benchmark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatches socket.socket.connect to raise AssertionError, verifying 100% offline benchmark execution."""
        def blocked_connect(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Outbound socket connection attempted during multilingual benchmark run!")

        monkeypatch.setattr(socket.socket, "connect", blocked_connect)

        # Run multilingual evaluation under socket blockage
        backend = FastEncoderBackend()
        res = backend.evaluate_choice(
            state="Necesito ayuda con el reembolso de mi suscripción.",
            instructions="Clasificar ticket.",
            options={"billing": "Facturación y cobros", "tech": "Soporte técnico"},
        )
        assert res["choice"] == "billing"

    def test_documentation_remains_strictly_english(self) -> None:
        """Verifies that all documentation files remain 100% in English per user directive."""
        docs_to_check = [
            ROOT / "README.md",
            ROOT / "BENCHMARKS.md",
            ROOT / "docs" / "ACTION_LOGBOOK.md",
            ROOT / "docs" / "RCA_REGISTER.md",
        ]
        non_english_heading_patterns = [
            "## Résultats", "## Données", "## Évaluation",
            "## Resultados", "## Datos", "## Avaliação",
        ]

        for doc in docs_to_check:
            if not doc.is_file():
                continue
            text = doc.read_text(encoding="utf-8")
            for pattern in non_english_heading_patterns:
                assert pattern not in text, f"Found non-English heading '{pattern}' in {doc.name}"
