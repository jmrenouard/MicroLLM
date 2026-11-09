"""Console CLI argument parser and command runner for MicroLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import uvicorn

def build_cli_parser() -> Any:
    """Build and return the MicroLLM CLI argument parser."""
    from microllm.registry import MODEL_PROFILES

    parser = argparse.ArgumentParser(
        prog="microllm",
        description="MicroLLM Local System-One Engine CLI, Benchmark Suite & ONNX INT8 Profile Manager",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Run the FastAPI System-One server")
    serve_p.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind (default: 8000)",
    )
    serve_p.add_argument(
        "--strategy",
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto"],
        default=os.environ.get("MICROLLM_MULTILINGUAL_STRATEGY", "auto"),
        help="Multilingual analysis strategy ('fast' | 'neural' | 'auto', default: 'auto')",
    )
    serve_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta"],
        default=os.environ.get("MICROLLM_NEURAL_MODEL", "minilm"),
        help="Neural ONNX model ('minilm' | 'mdeberta', default: 'minilm')",
    )
    serve_p.add_argument(
        "--api-key",
        default=os.environ.get("MICROLLM_API_KEY"),
        help="Optional API key for Bearer authentication",
    )
    serve_p.add_argument(
        "--reload",
        action="store_true",
        help="Enable development reload",
    )

    pull_p = sub.add_parser("pull", help="Download and quantize an ONNX INT8 hardware profile")
    pull_p.add_argument(
        "profile",
        choices=sorted(MODEL_PROFILES.keys()),
        help="Profile name (fast-cpu-int8, multilingual, guardrail)",
    )
    pull_p.add_argument("--cache-dir", default=None, help="Custom cache directory")
    pull_p.add_argument("--download", action="store_true", help="Fetch remote HF config metadata")
    pull_p.add_argument("--from-hub", dest="from_hub", action="store_true", help="Download quantized ONNX model from Hugging Face Hub")

    sub.add_parser("mcp", help="Run the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio")

    studio_p = sub.add_parser(
        "studio",
        aliases=["gradio"],
        help="Launch the Gradio Decision Studio UI (Hugging Face Spaces compatible engine)",
    )
    studio_p.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860)")
    studio_p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    studio_p.add_argument(
        "--strategy",
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto"],
        default=None,
        help="Default multilingual strategy ('fast' | 'neural' | 'auto')",
    )
    studio_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta"],
        default=None,
        help="Default neural ONNX model ('minilm' | 'mdeberta')",
    )

    list_p = sub.add_parser("list", help="List available and installed hardware profiles")
    list_p.add_argument("--cache-dir", default=None, help="Custom cache directory")

    bench_p = sub.add_parser("benchmark", help="Run the 540-item, 16-dataset external, or unified all-suites evaluation")
    bench_p.add_argument("--output", default=None, help="Optional JSON output file path")
    bench_p.add_argument(
        "--all-suites",
        "--all",
        dest="all_suites",
        action="store_true",
        help="Run the unified automation orchestrator across all MicroLLM benchmark suites",
    )
    bench_p.add_argument(
        "--suites",
        default=None,
        help="Comma-separated list of suite IDs when running with --all-suites",
    )
    bench_p.add_argument(
        "--skip-full-splits",
        "--fast",
        "--quick",
        dest="skip_full_splits",
        action="store_true",
        help="Skip the 114,018-row full unsampled external split evaluation when running --all-suites",
    )
    bench_p.add_argument(
        "--external",
        "--web",
        dest="external",
        action="store_true",
        help="Run the 16-dataset external web benchmark suite (BoolQ, MultiNLI, Banking77, CLINC150, WildGuard, RouteLLM, JevBench, etc.)",
    )
    bench_p.add_argument(
        "--hf-jev-models",
        dest="hf_jev_models",
        action="store_true",
        help="Run the 12-dataset Hugging Face JEV models ecosystem benchmark suite (huggingface.co/models?search=jev)",
    )
    bench_p.add_argument(
        "--full-splits",
        "--no-sampling",
        dest="full_splits",
        action="store_true",
        help="Evaluate complete unsampled official evaluation splits across all 16 external datasets (114,018 rows)",
    )
    bench_p.add_argument(
        "--entries-per-dataset",
        "--scale-per-dataset",
        dest="entries_per_dataset",
        type=int,
        default=None,
        help="Optional target entries per external dataset (e.g. 40 for 640 total entries)",
    )
    bench_p.add_argument(
        "--fetch-live",
        dest="fetch_live",
        action="store_true",
        help="Fetch live rows from Hugging Face Datasets Server API when running sampled external benchmarks",
    )
    bench_p.add_argument(
        "--multilingual",
        dest="multilingual",
        action="store_true",
        help="Run the quadrilingual evaluation benchmark across EN, FR, ES, PT with fast, neural, and auto cascade options",
    )
    bench_p.add_argument(
        "--multilingual-strategy",
        dest="multilingual_strategy",
        choices=["fast", "neural", "auto", "all"],
        default="all",
        help="Multilingual strategy option to benchmark ('fast', 'neural', 'auto', or 'all')",
    )
    bench_p.add_argument(
        "--neural-model",
        dest="neural_model",
        choices=["minilm", "mdeberta", "all"],
        default="all",
        help="Neural model option for neural and auto strategies ('minilm', 'mdeberta', or 'all')",
    )
    bench_p.add_argument(
        "--language",
        dest="language",
        choices=["en", "fr", "es", "pt", "all"],
        default="all",
        help="Language to evaluate ('en', 'fr', 'es', 'pt', or 'all')",
    )
    bench_p.add_argument(
        "--pareto",
        dest="pareto",
        action="store_true",
        help="Run the comparative Pareto benchmark across Tier-1, Tier-2, Tier-3 (deberta-large-int8), and decision-gate-cascade (IDEA-048)",
    )
    bench_p.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Output benchmark results in JSON format to stdout",
    )

    papers_p = sub.add_parser(
        "papers",
        help="Search, verify, and download the Jev, RLCD, System-1 & Calibrated LLM academic paper catalog",
    )
    papers_p.add_argument("--pillar", default=None, help="Filter by pillar ID or title substring")
    papers_p.add_argument("--query", "-q", default=None, help="Keyword search in title, authors, summary, or relevance")
    papers_p.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format")
    papers_p.add_argument("--download-pdfs", action="store_true", help="Download matching arXiv PDFs into docs/papers/")
    papers_p.add_argument("--verify-pdfs", action="store_true", help="Verify local PDF presence in docs/papers/")
    papers_p.add_argument("--output-dir", default=None, help="Optional custom output directory for PDFs")

    rca_p = sub.add_parser("rca", help="Run sequential or targeted RCA non-regression verification")
    rca_p.add_argument("--rca", default=None, help="Target specific RCA ID (e.g., RCA-001)")
    rca_p.add_argument("--all", action="store_true", help="Run all 15 RCAs sequentially one by one")
    rca_p.add_argument("--list", action="store_true", help="List all registered RCAs in catalog")
    rca_p.add_argument("--iterations", type=int, default=2, help="Number of profiling iterations per RCA (default: 2)")
    rca_p.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")

    sync_hf_p = sub.add_parser(
        "sync-hf-readme",
        help="Synchronize root README.md to all 6 official Hugging Face repositories (IDEA-038)",
    )
    sync_hf_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build all 6 Hugging Face cards without uploading to the Hub",
    )
    sync_hf_p.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face API token (defaults to HF_TOKEN or HUGGINGFACE_TOKEN env var)",
    )

    publish_p = sub.add_parser(
        "publish",
        help="Stage and publish multilingual benchmark datasets and neural models to Hugging Face Hub",
        description="Stage and publish multilingual benchmark datasets and neural models to Hugging Face Hub",
    )
    publish_p.add_argument(
        "--dataset",
        action="store_true",
        help="Package and publish the multilingual benchmark dataset (jmrenouard/microllm-multilingual-benchmarks)",
    )
    publish_p.add_argument(
        "--model",
        default=None,
        help="Profile name or alias of the neural multilingual model to package and publish (e.g. multilingual-minilm-int8, multilingual-mdeberta-int8)",
    )
    publish_p.add_argument(
        "--all",
        action="store_true",
        help="Package and publish both the multilingual benchmark dataset and all neural multilingual models",
    )
    publish_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage files locally without publishing to Hugging Face Hub (zero network calls)",
    )
    publish_p.add_argument(
        "--staging-dir",
        default=None,
        help="Custom directory to stage repository artifacts",
    )
    publish_p.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN or KeePass vault)",
    )
    publish_p.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message for Hub uploads",
    )
    publish_p.add_argument(
        "--space",
        action="store_true",
        help="Stage and synchronize Gradio Decision Studio to Hugging Face Space",
    )
    publish_p.add_argument(
        "--space-repo-id",
        default="jmrenouard/microllm-decision-studio",
        help="Target Hugging Face Space repository ID (default: jmrenouard/microllm-decision-studio)",
    )
    publish_p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (json or text, default: json)",
    )

    sync_space_p = sub.add_parser(
        "sync-space",
        help="Stage and synchronize Gradio Decision Studio to Hugging Face Space (jmrenouard/microllm-decision-studio)",
        description="Stage and synchronize Gradio Decision Studio to Hugging Face Space",
    )
    sync_space_p.add_argument(
        "--repo-id",
        default="jmrenouard/microllm-decision-studio",
        help="Target Hugging Face Space repository ID (default: jmrenouard/microllm-decision-studio)",
    )
    sync_space_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage Space bundle locally and validate manifests without uploading (zero network calls)",
    )
    sync_space_p.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN or KeePass vault)",
    )
    sync_space_p.add_argument(
        "--staging-dir",
        default=None,
        help="Custom staging directory for Space bundle",
    )
    sync_space_p.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message for Hugging Face Space upload",
    )
    sync_space_p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (json or text, default: json)",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """CLI handler supporting `microllm pull`, `microllm list`, `microllm benchmark`, and `microllm serve`."""
    from microllm.benchmarks import run_benchmark_suite
    from microllm.registry import list_profiles, pull_profile

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["serve"]

    parser = build_cli_parser()
    parsed = parser.parse_args(args_list)

    if parsed.command == "sync-space":
        from microllm.hf_space_sync import sync_hf_space

        res = sync_hf_space(
            space_repo_id=parsed.repo_id,
            staging_dir=parsed.staging_dir,
            dry_run=bool(parsed.dry_run),
            token=parsed.token,
            commit_message=parsed.commit_message,
        )
        if parsed.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"Status: {res.get('status')}")
            print(f"Space Repo: {res.get('repo_id')}")
            print(f"Staging Directory: {res.get('staging_dir')}")
        return 0
    if parsed.command == "publish":
        from microllm.hf_publishing import publish_cli

        return publish_cli(parsed)
    if parsed.command == "serve":
        if parsed.multilingual_strategy:
            os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] = str(parsed.multilingual_strategy)
        if parsed.neural_model:
            os.environ["MICROLLM_NEURAL_MODEL"] = str(parsed.neural_model)
        if parsed.api_key:
            os.environ["MICROLLM_API_KEY"] = str(parsed.api_key)
        uvicorn.run(
            "microllm.server:create_app",
            factory=True,
            host=parsed.host,
            port=parsed.port,
            reload=bool(getattr(parsed, "reload", False)),
        )
        return 0
    if parsed.command == "sync-hf-readme":
        from microllm.registry import sync_readme_to_huggingface_hub

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sync_res = sync_readme_to_huggingface_hub(
            token=parsed.token,
            dry_run=bool(parsed.dry_run),
        )
        print(json.dumps(sync_res, indent=2))
        return 0
    if parsed.command == "pull":
        manifest = pull_profile(
            parsed.profile,
            cache_dir=parsed.cache_dir,
            download_weights=parsed.download,
            from_hub=bool(getattr(parsed, "from_hub", False)),
        )
        print(json.dumps(manifest, indent=2))
        return 0
    if parsed.command == "mcp":
        from microllm.mcp_server import run_mcp_stdio_server

        return run_mcp_stdio_server()
    if parsed.command in {"studio", "gradio"}:
        if getattr(parsed, "multilingual_strategy", None):
            os.environ["MICROLLM_MULTILINGUAL_STRATEGY"] = str(parsed.multilingual_strategy)
        if getattr(parsed, "neural_model", None):
            os.environ["MICROLLM_NEURAL_MODEL"] = str(parsed.neural_model)
        from microllm.gradio_app import create_gradio_demo

        demo = create_gradio_demo()
        demo.launch(server_name="127.0.0.1", server_port=int(parsed.port), share=bool(parsed.share))
        return 0

    if parsed.command == "list":
        print(json.dumps(list_profiles(cache_dir=parsed.cache_dir), indent=2))
        return 0
    if parsed.command == "benchmark":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if getattr(parsed, "all_suites", False):
            from microllm.benchmark_runner import (
                format_master_benchmark_markdown,
                run_all_benchmark_suites,
            )

            raw_suites = getattr(parsed, "suites", None)
            suite_list = (
                [s.strip() for s in raw_suites.split(",") if s.strip()]
                if raw_suites
                else None
            )
            try:
                master_rep = run_all_benchmark_suites(
                    suites=suite_list,
                    include_full_splits=not bool(getattr(parsed, "skip_full_splits", False)),
                    master_output_path=parsed.output,
                    save_artifacts=True,
                )
            except ValueError as exc:
                sys.stderr.write(f"Error: {exc}\n")
                return 2
            print(format_master_benchmark_markdown(master_rep))
            return 0 if master_rep["all_suites_passed"] else 1
        if getattr(parsed, "hf_jev_models", False):
            from microllm.web_benchmark_suite import (
                format_hf_jev_models_benchmarks_markdown,
                run_hf_jev_model_benchmarks,
            )

            default_jev_out = (
                Path(parsed.output)
                if parsed.output
                else Path(__file__).resolve().parents[2]
                / "benchmarks"
                / "hf_jev_models_benchmarks_results.json"
            )
            jev_rep = run_hf_jev_model_benchmarks(
                output_path=default_jev_out,
                entries_per_dataset=getattr(parsed, "entries_per_dataset", None),
                fetch_live=bool(getattr(parsed, "fetch_live", False)),
            )
            print(format_hf_jev_models_benchmarks_markdown(jev_rep))
            return 0
        if getattr(parsed, "external", False):
            from microllm.web_benchmark_suite import (
                format_web_benchmarks_markdown,
                run_all_web_benchmarks,
            )

            full_splits = bool(getattr(parsed, "full_splits", False))
            default_filename = (
                "web_benchmarks_full_results.json"
                if full_splits
                else "web_benchmarks_results.json"
            )
            default_ext_out = (
                Path(parsed.output)
                if parsed.output
                else Path(__file__).resolve().parents[2] / "benchmarks" / default_filename
            )
            ext_rep = run_all_web_benchmarks(
                output_path=default_ext_out,
                entries_per_dataset=getattr(parsed, "entries_per_dataset", None),
                fetch_live=bool(getattr(parsed, "fetch_live", False)),
                full_splits=full_splits,
            )
            print(format_web_benchmarks_markdown(ext_rep))
            return 0
        if getattr(parsed, "multilingual", False):
            from microllm.multilingual_benchmark_runner import (
                format_multilingual_benchmarks_markdown,
                run_multilingual_benchmarks,
            )

            strats = (
                ["fast", "neural", "auto"]
                if getattr(parsed, "multilingual_strategy", "all") == "all"
                else [parsed.multilingual_strategy]
            )
            models = (
                ["minilm", "mdeberta"]
                if getattr(parsed, "neural_model", "all") == "all"
                else [parsed.neural_model]
            )
            langs = (
                ["en", "fr", "es", "pt"]
                if getattr(parsed, "language", "all") == "all"
                else [parsed.language]
            )
            m_rep = run_multilingual_benchmarks(
                output_path=parsed.output,
                strategies=strats,
                neural_models=models,
                languages=langs,
                save_artifacts=True,
            )
            if getattr(parsed, "json", False):
                print(json.dumps(m_rep, indent=2))
            else:
                print(format_multilingual_benchmarks_markdown(m_rep))
            return 0
        if getattr(parsed, "pareto", False):
            from microllm.pareto_benchmark import (
                format_pareto_benchmark_markdown,
                run_pareto_benchmark,
            )

            p_rep = run_pareto_benchmark(output_path=parsed.output, save_artifacts=True)
            if getattr(parsed, "json", False):
                print(json.dumps(p_rep, indent=2))
            else:
                print(format_pareto_benchmark_markdown(p_rep["profiles"], p_rep.get("summary_findings")))
            return 0
        rep = run_benchmark_suite(save_artifacts=True, output_path=parsed.output)
        if getattr(parsed, "json", False):
            print(json.dumps(rep, indent=2))
        else:
            print(rep["markdown_table"])
        return 0
    if parsed.command == "papers":
        from microllm.papers import (
            download_academic_papers,
            format_papers_markdown,
            get_literature_taxonomy,
            list_academic_papers,
            verify_downloaded_pdfs,
        )

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if parsed.download_pdfs:
            dl_res = download_academic_papers(
                output_dir=parsed.output_dir,
                pillar=parsed.pillar,
                query=parsed.query,
            )
            print(json.dumps(dl_res, indent=2))
            return 0
        if parsed.verify_pdfs:
            ver_res = verify_downloaded_pdfs(
                output_dir=parsed.output_dir,
                pillar=parsed.pillar,
                query=parsed.query,
            )
            print(json.dumps(ver_res, indent=2))
            return 0
        if parsed.format == "markdown":
            print(format_papers_markdown(pillar=parsed.pillar, query=parsed.query))
        else:
            items = list_academic_papers(pillar=parsed.pillar, query=parsed.query)
            print(
                json.dumps(
                    {
                        "total": len(items),
                        "taxonomy": get_literature_taxonomy(),
                        "papers": items,
                    },
                    indent=2,
                )
            )
        return 0
    if parsed.command == "rca":
        from microllm.rca_runner import RCARunner

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        runner = RCARunner()
        if parsed.list:
            if parsed.format == "json":
                print(json.dumps(runner.list_rcas(), indent=2))
            else:
                print("| RCA ID | Title | Module | Target SLA | Latency SLA |")
                print("| :--- | :--- | :--- | :---: | :---: |")
                for r in runner.list_rcas():
                    print(f"| **{r['id']}** | {r['title']} | `{r['target_module']}` | {r['accuracy_sla']*100:.0f}% | {r['latency_p50_sla_ms']:.1f} ms |")
            return 0

        target_ids = [parsed.rca] if parsed.rca else None
        results = runner.run_sequential(
            rca_ids=target_ids,
            iterations_per_rca=parsed.iterations,
        )

        if parsed.format == "json":
            from dataclasses import asdict
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print(runner.format_report_markdown(results))

        all_ok = all(r.non_regression_passed for r in results)
        return 0 if all_ok else 1

    parser.print_help()
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `microllm`."""
    code = run_cli(argv)
    if argv is None:
        raise SystemExit(code)
    return code



__all__ = ["build_cli_parser", "run_cli", "main"]

if __name__ == "__main__":
    main()
