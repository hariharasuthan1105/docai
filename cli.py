"""
Comprehensive Production CLI for Document AI.

Subcommands:
- predict: Process single document (PDF, image, text)
- batch: Process a directory of documents
- evaluate: Run ground-truth evaluation & metrics report
- benchmark: Run component ablation study & robustness benchmarks
- serve: Launch FastAPI REST microservice
- demo: Launch Streamlit visual review UI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
for _p in [_parent_dir, _curr_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from docai.config import load_custom_config
    from docai.evaluation.ablation import format_ablation_markdown, run_ablation_study
    from docai.evaluation.dataset import load_evaluation_dataset
    from docai.evaluation.evaluator import DocumentAIEvaluator
    from docai.evaluation.robustness import format_robustness_markdown, run_robustness_benchmark
    from docai.models.extraction_schema import FinalDocument, ReviewDecision
    from docai.pipeline import DocumentAIPipeline
except ImportError:
    from config import load_custom_config
    from evaluation.ablation import format_ablation_markdown, run_ablation_study
    from evaluation.dataset import load_evaluation_dataset
    from evaluation.evaluator import DocumentAIEvaluator
    from evaluation.robustness import format_robustness_markdown, run_robustness_benchmark
    from models.extraction_schema import FinalDocument, ReviewDecision
    from pipeline import DocumentAIPipeline

BANNER_LINE = "=" * 60


def format_field_display(name: str, field_obj, unit: str = "") -> str:
    """Format an extracted field for terminal presentation."""
    if field_obj.is_present():
        val_str = f"{field_obj.value} {unit}".strip() if unit else str(field_obj.value)
        conf_str = f"{field_obj.confidence:.2f}"
    else:
        val_str = "(not found)"
        conf_str = "0.00"

    return f"{name:<17}: {val_str}\nConfidence       : {conf_str}\n"


def format_visual_mark(name: str, mark_obj) -> str:
    """Format a visual mark for terminal display."""
    if mark_obj.status == "not_implemented":
        status_str = "NOT IMPLEMENTED"
    elif mark_obj.present:
        status_str = f"DETECTED (conf: {mark_obj.confidence:.2f})"
    else:
        status_str = f"NOT DETECTED (conf: {mark_obj.confidence:.2f})"

    return f"{name:<17}: {status_str}"


def display_results(final_doc: FinalDocument) -> None:
    """Print clean formatted card to terminal."""
    print("\n" + BANNER_LINE)
    print("RESULT")
    print(BANNER_LINE + "\n")

    print(format_field_display("Dealer Name", final_doc.dealer_name))
    print(format_field_display("Model Name", final_doc.model_name))
    print(format_field_display("Horse Power", final_doc.horse_power, unit="HP"))
    print(format_field_display("Asset Cost", final_doc.asset_cost))

    if final_doc.invoice_number.is_present():
        print(format_field_display("Invoice Number", final_doc.invoice_number))
    if final_doc.invoice_date.is_present():
        print(format_field_display("Invoice Date", final_doc.invoice_date))
    if final_doc.customer_name.is_present():
        print(format_field_display("Customer Name", final_doc.customer_name))
    if final_doc.phone_number.is_present():
        print(format_field_display("Phone Number", final_doc.phone_number))

    print(format_visual_mark("Dealer Signature", final_doc.dealer_signature))
    print(format_visual_mark("Dealer Stamp", final_doc.dealer_stamp))

    print("\n" + BANNER_LINE)
    print(f"Overall Score    : {final_doc.overall_confidence:.4f}")
    if final_doc.decision == ReviewDecision.AUTO_APPROVE:
        print("Decision         : AUTO-APPROVED")
    elif final_doc.decision == ReviewDecision.REVIEW:
        print("Decision         : REVIEW REQUIRED")
    else:
        print("Decision         : MANUAL REVIEW REQUIRED")

    if final_doc.needs_human_review and final_doc.review_reasons:
        print("\nReview Reasons:")
        for r in final_doc.review_reasons:
            print(f"  • {r}")

    print(f"Processing Time  : {final_doc.processing_time_ms:.1f} ms")
    print(BANNER_LINE + "\n")


def cmd_predict(args: argparse.Namespace) -> int:
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: '{input_file}'", file=sys.stderr)
        sys.exit(1)

    dealer_master, model_master = load_custom_config(args.config)
    pipeline = DocumentAIPipeline(
        dealer_master=dealer_master,
        model_master=model_master,
        ocr_lang=args.language,
        enable_preprocessing=not args.no_preprocess,
    )

    print("\n" + BANNER_LINE)
    print("DOCUMENT AI FIELD EXTRACTION")
    print(BANNER_LINE)
    print(f"\nInput: {input_file}\n")

    def print_progress(msg: str, step: int, total: int):
        print(f"[{step}/{total}] {msg}")

    final_doc = pipeline.process(
        input_file,
        document_id=os.path.basename(input_file),
        on_progress=print_progress if not args.quiet else None,
    )

    display_results(final_doc)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_doc.to_legacy_dict(), f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved structured JSON output to: {args.output}\n")

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    dealer_master, model_master = load_custom_config(args.config)
    pipeline = DocumentAIPipeline(
        dealer_master=dealer_master,
        model_master=model_master,
        ocr_lang=args.language,
        enable_preprocessing=not args.no_preprocess,
    )

    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".txt"))
    ]

    if not files:
        print(f"No document files found in '{input_dir}'")
        return 0

    print(f"\nProcessing batch of {len(files)} documents from '{input_dir}'...\n")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Processing {fname}...")
        res = pipeline.process(fpath, document_id=fname)
        if args.output_dir:
            out_file = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_result.json")
            with open(out_file, "w", encoding="utf-8") as fp:
                json.dump(res.to_legacy_dict(), fp, indent=2, ensure_ascii=False)
        print(f"  -> Decision: {res.decision.value} (Score: {res.overall_confidence:.2f})")

    print("\n[OK] Batch processing complete.\n")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    print("\n" + BANNER_LINE)
    print("RUNNING DOCUMENT AI EVALUATION BENCHMARK")
    print(BANNER_LINE + "\n")

    dataset = load_evaluation_dataset(args.data_dir)
    print(f"Loaded {len(dataset)} evaluation documents with ground truth annotations.")

    evaluator = DocumentAIEvaluator()
    metrics = evaluator.evaluate_dataset(dataset)
    report_md = evaluator.generate_report_markdown(metrics)

    print("\n" + report_md + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[OK] Evaluation report saved to: {args.output}\n")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    print("\n" + BANNER_LINE)
    print("RUNNING COMPONENT ABLATION & ROBUSTNESS BENCHMARKS")
    print(BANNER_LINE + "\n")

    dataset = load_evaluation_dataset(args.data_dir)

    print("1. Running Ablation Study across system layers (Exp A - Exp E)...")
    ablation_results = run_ablation_study(dataset)
    ablation_md = format_ablation_markdown(ablation_results)
    print("\n" + ablation_md + "\n")

    print("2. Running Robustness Benchmark under degraded document conditions...")
    robust_results = run_robustness_benchmark(dataset)
    robust_md = format_robustness_markdown(robust_results)
    print("\n" + robust_md + "\n")

    if args.output:
        full_report = ablation_md + "\n\n---\n\n" + robust_md
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"[OK] Benchmark report saved to: {args.output}\n")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from docai.api.app import create_app

    print(f"\nStarting Document AI FastAPI REST Service on {args.host}:{args.port}...")
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "app.py")
    print(f"\nLaunching Streamlit Dashboard on port {args.port}...")
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", demo_path, "--server.port", str(args.port)]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docai",
        description="Deterministic Document AI CLI (PaddleOCR + Regex + Fuzzy Matching)",
    )

    parser.add_argument("-o", "--output", help="Path to write output JSON")
    parser.add_argument("-c", "--config", help="Path to custom catalog JSON")
    parser.add_argument("-l", "--language", default="en", help="OCR language code (en, hi, gu)")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable CV preprocessing")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    subparsers = parser.add_subparsers(dest="command")

    # Predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Process a single document file")
    pred_parser.add_argument("input_file", help="Path to PDF, PNG, JPG, or TXT document")
    pred_parser.add_argument("-o", "--output", help="Save structured JSON result to file")
    pred_parser.add_argument("-c", "--config", help="Custom dealer/model catalog JSON")
    pred_parser.add_argument("-l", "--language", default="en", help="OCR language code (en, hi, gu)")
    pred_parser.add_argument("--no-preprocess", action="store_true", help="Disable CV preprocessing")
    pred_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # Batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Process a directory of documents")
    batch_parser.add_argument("input_dir", help="Path to folder containing documents")
    batch_parser.add_argument("-o", "--output-dir", help="Directory to save JSON results")
    batch_parser.add_argument("-c", "--config", help="Custom catalog JSON")
    batch_parser.add_argument("-l", "--language", default="en", help="OCR language code")
    batch_parser.add_argument("--no-preprocess", action="store_true", help="Disable CV preprocessing")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Run ground-truth evaluation metrics")
    eval_parser.add_argument("data_dir", nargs="?", default="data", help="Directory containing sample documents")
    eval_parser.add_argument("-o", "--output", help="Save evaluation markdown report to file")

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run ablation study & robustness benchmarks")
    bench_parser.add_argument("data_dir", nargs="?", default="data", help="Directory containing sample documents")
    bench_parser.add_argument("-o", "--output", help="Save benchmark report to file")

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST microservice")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", help="Launch Streamlit interactive visual UI")
    demo_parser.add_argument("--port", type=int, default=8501, help="Port number")

    return parser


def main(argv: Optional[list] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    subcommands = {"predict", "batch", "evaluate", "benchmark", "serve", "demo"}
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["predict"] + argv
    elif not argv:
        argv = ["--help"]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return cmd_predict(args)
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "demo":
        return cmd_demo(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

