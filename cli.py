"""
Comprehensive Production CLI for Multi-Document AI.

Subcommands:
- predict: Process any single document (restaurant receipt, tractor invoice, generic PDF/image)
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
    from docai.schemas.schema_loader import get_schema_registry
except ImportError:
    from config import load_custom_config
    from evaluation.ablation import format_ablation_markdown, run_ablation_study
    from evaluation.dataset import load_evaluation_dataset
    from evaluation.evaluator import DocumentAIEvaluator
    from evaluation.robustness import format_robustness_markdown, run_robustness_benchmark
    from models.extraction_schema import FinalDocument, ReviewDecision
    from pipeline import DocumentAIPipeline
    from schemas.schema_loader import get_schema_registry

BANNER_LINE = "=" * 60


def format_field_display(name: str, field_obj, unit: str = "") -> str:
    """Format an extracted field for terminal presentation."""
    if field_obj is not None and getattr(field_obj, "is_present", lambda: False)():
        val_str = f"{field_obj.value} {unit}".strip() if unit else str(field_obj.value)
        conf_str = f"{field_obj.confidence:.2f}"
        return f"{name:<19}: {val_str:<32} (conf: {conf_str})"
    elif field_obj is not None and getattr(field_obj, "value", None) is not None:
        val_str = f"{field_obj.value} {unit}".strip() if unit else str(field_obj.value)
        conf_str = f"{getattr(field_obj, 'confidence', 1.0):.2f}"
        return f"{name:<19}: {val_str:<32} (conf: {conf_str})"
    return f"{name:<19}: (not found)"


def format_visual_mark(name: str, mark_obj) -> str:
    """Format a visual mark for terminal display."""
    if mark_obj.status == "not_implemented":
        status_str = "NOT IMPLEMENTED"
    elif mark_obj.present:
        status_str = f"DETECTED (conf: {mark_obj.confidence:.2f})"
    else:
        status_str = f"NOT DETECTED (conf: {mark_obj.confidence:.2f})"

    return f"{name:<19}: {status_str}"


def display_results(final_doc: FinalDocument) -> None:
    """Print clean formatted multi-document result card to terminal."""
    print("\n" + BANNER_LINE)
    print("DOCUMENT AI EXTRACTION RESULT")
    print(BANNER_LINE + "\n")

    dtype = final_doc.document_type
    print(f"Document Type      : {dtype.replace('_', ' ').title()} (conf: {final_doc.document_type_confidence:.2f})\n")

    all_fields = final_doc.get_all_fields()

    if dtype == "restaurant_receipt":
        # --- Merchant ---
        print("[ MERCHANT DETAILS ]")
        if "merchant_name" in all_fields:
            print(format_field_display("Establishment", all_fields["merchant_name"]))
        if "merchant_tagline" in all_fields and all_fields["merchant_tagline"].is_present():
            print(format_field_display("Tagline", all_fields["merchant_tagline"]))
        if "merchant_address" in all_fields and all_fields["merchant_address"].is_present():
            print(format_field_display("Address", all_fields["merchant_address"]))
        if "merchant_phone" in all_fields and all_fields["merchant_phone"].is_present():
            print(format_field_display("Phone", all_fields["merchant_phone"]))

        # --- Metadata ---
        print("\n[ ORDER METADATA ]")
        for k in ["invoice_number", "invoice_date", "invoice_time", "order_number", "table_number", "cashier"]:
            if k in all_fields and all_fields[k].is_present():
                lbl = k.replace("_", " ").title()
                print(format_field_display(lbl, all_fields[k]))

        # --- Tables (Line Items) ---
        for tab in final_doc.tables:
            print(f"\n[ {tab.name.upper()} ({len(tab.rows)} items) ]")
            print(f"  {'S.No':<5} {'Item Name':<28} {'Qty':<5} {'Unit Price':<12} {'Amount':<10}")
            print("  " + "-" * 64)
            for r in tab.rows:
                sno = str(r.get("sno", ""))
                item = str(r.get("item", ""))[:26]
                qty = str(r.get("qty", ""))
                uprice = f"₹{r.get('unit_price', ''):.2f}" if isinstance(r.get('unit_price'), (int, float)) else str(r.get('unit_price', ''))
                amt = f"₹{r.get('amount', ''):.2f}" if isinstance(r.get('amount'), (int, float)) else str(r.get('amount', ''))
                print(f"  {sno:<5} {item:<28} {qty:<5} {uprice:<12} {amt:<10}")

        # --- Totals ---
        print("\n[ FINANCIAL TOTALS & TAXES ]")
        for k in ["subtotal", "discount_percentage", "discount_amount", "taxable_amount", "cgst_rate_pct", "cgst_amount", "sgst_rate_pct", "sgst_amount", "grand_total", "amount_in_words"]:
            if k in all_fields and all_fields[k].is_present():
                lbl = k.replace("_", " ").title()
                unit = "Rs." if "amount" in k or "total" in k else ("%" if "rate" in k or "pct" in k or "percentage" in k else "")
                print(format_field_display(lbl, all_fields[k], unit=unit))

        # --- Payment ---
        print("\n[ PAYMENT INFO ]")
        for k in ["payment_mode", "upi_ref_no", "payment_status"]:
            if k in all_fields and all_fields[k].is_present():
                lbl = k.replace("_", " ").title()
                print(format_field_display(lbl, all_fields[k]))

    elif dtype == "tractor_invoice":
        # --- Tractor Invoice View ---
        print("[ EQUIPMENT & FINANCING ]")
        print(format_field_display("Dealer Name", final_doc.dealer_name))
        print(format_field_display("Model Name", final_doc.model_name))
        print(format_field_display("Horse Power", final_doc.horse_power, unit="HP"))
        print(format_field_display("Asset Cost", final_doc.asset_cost, unit="Rs."))


        print("\n[ DOCUMENT METADATA ]")
        if final_doc.invoice_number.is_present():
            print(format_field_display("Invoice Number", final_doc.invoice_number))
        if final_doc.invoice_date.is_present():
            print(format_field_display("Invoice Date", final_doc.invoice_date))
        if final_doc.customer_name.is_present():
            print(format_field_display("Customer Name", final_doc.customer_name))
        if final_doc.phone_number.is_present():
            print(format_field_display("Phone Number", final_doc.phone_number))

        print("\n[ VISUAL MARKS ]")
        print(format_visual_mark("Dealer Signature", final_doc.dealer_signature))
        print(format_visual_mark("Dealer Stamp", final_doc.dealer_stamp))

    else:
        # --- Generic Unknown Document View ---
        print("[ DISCOVERED KEY-VALUES ]")
        for k, fval in all_fields.items():
            if fval.is_present():
                lbl = k.replace("_", " ").title()
                print(format_field_display(lbl, fval))

        for tab in final_doc.tables:
            print(f"\n[ DETECTED TABLE: {tab.name.upper()} ({len(tab.rows)} rows) ]")
            for r in tab.rows:
                print("  " + str(r))

    print("\n" + BANNER_LINE)
    print(f"Overall Confidence : {final_doc.overall_confidence:.4f}")
    if final_doc.decision == ReviewDecision.AUTO_APPROVE:
        print("Review Decision    : AUTO-APPROVED")
    elif final_doc.decision == ReviewDecision.REVIEW:
        print("Review Decision    : REVIEW REQUIRED")
    else:
        print("Review Decision    : MANUAL REVIEW REQUIRED")

    if final_doc.needs_human_review and final_doc.review_reasons:
        print("\nReview Reasons / Audit Trail:")
        for r in final_doc.review_reasons:
            print(f"  • {r}")

    print(f"Processing Time    : {final_doc.processing_time_ms:.1f} ms")
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
        document_type=getattr(args, "schema", None),
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
        res = pipeline.process(fpath, document_id=fname, document_type=getattr(args, "schema", None))
        if args.output_dir:
            out_file = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_result.json")
            with open(out_file, "w", encoding="utf-8") as fp:
                json.dump(res.to_legacy_dict(), fp, indent=2, ensure_ascii=False)
        print(f"  -> Type: {res.document_type} | Decision: {res.decision.value} (Score: {res.overall_confidence:.2f})")

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
    try:
        import uvicorn
        from docai.api.app import create_app
    except ImportError:
        print("Error: FastAPI or Uvicorn not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    app = create_app()
    print(f"\nStarting Document AI REST API service on http://{args.host}:{args.port}...\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    import subprocess
    demo_file = os.path.join(os.path.dirname(__file__), "demo", "app.py")
    if not os.path.exists(demo_file):
        print(f"Error: Demo file not found at {demo_file}", file=sys.stderr)
        return 1

    print(f"\nLaunching Streamlit Visual Review UI on port {args.port}...\n")
    cmd = [sys.executable, "-m", "streamlit", "run", demo_file, "--server.port", str(args.port)]
    subprocess.run(cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docai",
        description="Document AI — Production Multi-Document Field Extraction Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. predict
    p_pred = subparsers.add_parser("predict", help="Extract fields from a document")
    p_pred.add_argument("input_file", help="Path to input PDF, image, or text file")
    p_pred.add_argument("--schema", default=None, help="Force specific schema (restaurant_receipt, tractor_invoice, generic)")
    p_pred.add_argument("--output", "-o", default=None, help="Save structured JSON output to file")
    p_pred.add_argument("--config", "-c", default=None, help="Path to custom JSON config with master lists")
    p_pred.add_argument("--language", "-l", default="en", help="PaddleOCR language code (default: en)")
    p_pred.add_argument("--no-preprocess", action="store_true", help="Disable computer vision deskew/contrast enhancement")
    p_pred.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logging")

    # 2. batch
    p_batch = subparsers.add_parser("batch", help="Batch extract fields from directory of documents")
    p_batch.add_argument("input_dir", help="Directory containing documents")
    p_batch.add_argument("--schema", default=None, help="Force specific schema")
    p_batch.add_argument("--output-dir", "-o", default=None, help="Directory to save JSON results")
    p_batch.add_argument("--config", "-c", default=None, help="Path to custom JSON config with master lists")
    p_batch.add_argument("--language", "-l", default="en", help="PaddleOCR language code")
    p_batch.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")

    # 3. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run ground truth evaluation metrics")
    p_eval.add_argument("data_dir", nargs="?", default="data", help="Directory with samples/ and ground_truth/")
    p_eval.add_argument("--output", "-o", default=None, help="Path to save evaluation markdown report")

    # 4. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run ablation study and robustness degradation benchmarks")
    p_bench.add_argument("data_dir", nargs="?", default="data", help="Directory with evaluation samples")
    p_bench.add_argument("--output", "-o", default=None, help="Path to save benchmark report")

    # 5. serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST service")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # 6. demo
    p_demo = subparsers.add_parser("demo", help="Launch Streamlit visual review UI")
    p_demo.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()

    # Backward compatibility: default to predict if positional file passed
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if raw_args and raw_args[0] not in ["predict", "batch", "evaluate", "benchmark", "serve", "demo", "-h", "--help"]:
        raw_args.insert(0, "predict")

    args = parser.parse_args(raw_args)

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "predict": cmd_predict,
        "batch": cmd_batch,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
        "demo": cmd_demo,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
