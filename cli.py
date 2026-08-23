"""
Pure Deterministic Command-Line Interface (CLI) for Document AI.

Operates completely from the local terminal without any external LLM,
web UI, or API key dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
for _p in [_parent_dir, _curr_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from docai.config import load_custom_config
    from docai.models.extraction_schema import FinalDocument
    from docai.pipeline import DocumentAIPipeline
except ImportError:
    from config import load_custom_config
    from models.extraction_schema import FinalDocument
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


def format_visual_mark_display(name: str, mark_obj) -> str:
    """Format visual mark status."""
    if mark_obj.status == "not_implemented":
        status_str = "NOT IMPLEMENTED"
    elif mark_obj.present:
        status_str = f"DETECTED ({mark_obj.confidence:.2f})"
    else:
        status_str = "NOT DETECTED"
    return f"{name:<17}: {status_str}"


def print_cli_summary(
    input_file: str,
    doc: FinalDocument,
    output_path: Optional[str] = None,
):
    """Print readable terminal output."""
    print("\n" + BANNER_LINE)
    print("RESULT")
    print(BANNER_LINE + "\n")

    hp_unit = "HP" if doc.horse_power.is_present() else ""
    print(format_field_display("Dealer Name", doc.dealer_name))
    print(format_field_display("Model Name", doc.model_name))
    print(format_field_display("Horse Power", doc.horse_power, unit=hp_unit))
    print(format_field_display("Asset Cost", doc.asset_cost))

    print(format_visual_mark_display("Dealer Signature", doc.dealer_signature))
    print(format_visual_mark_display("Dealer Stamp", doc.dealer_stamp))

    print("\n" + BANNER_LINE)
    decision = "NEEDS HUMAN REVIEW" if doc.needs_human_review else "AUTO-APPROVED"
    print(f"Overall Score    : {doc.overall_confidence:.4f}")
    print(f"Decision         : {decision}")
    if doc.review_reasons:
        print("Review Reasons   :")
        for r in doc.review_reasons:
            print(f"  - {r}")
    print(BANNER_LINE)

    if output_path:
        print(f"\n[✓] Saved structured JSON output to: {output_path}")


def export_json_result(input_file: str, doc: FinalDocument, output_path: str):
    """Export the structured extraction result matching the required schema."""
    result = {
        "document": input_file,
        "fields": {
            "dealer_name": {
                "value": doc.dealer_name.value,
                "confidence": doc.dealer_name.confidence,
                "bbox": doc.dealer_name.bbox,
            },
            "model_name": {
                "value": doc.model_name.value,
                "confidence": doc.model_name.confidence,
                "bbox": doc.model_name.bbox,
            },
            "horse_power": {
                "value": doc.horse_power.value,
                "confidence": doc.horse_power.confidence,
                "bbox": doc.horse_power.bbox,
            },
            "asset_cost": {
                "value": doc.asset_cost.value,
                "confidence": doc.asset_cost.confidence,
                "bbox": doc.asset_cost.bbox,
            },
            "dealer_signature": {
                "status": doc.dealer_signature.status,
            },
            "dealer_stamp": {
                "status": doc.dealer_stamp.status,
            },
        },
        "validation": {
            "overall_confidence": doc.overall_confidence,
            "needs_human_review": doc.needs_human_review,
            "review_reasons": doc.review_reasons,
        },
    }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docai",
        description="Deterministic Document AI CLI — Extract fields from invoices using PaddleOCR, regex, fuzzy matching, and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m docai invoice.pdf
  python -m docai invoice.png --output result.json
  python -m docai invoice.pdf --language hi
  python -m docai invoice.pdf --config custom_catalogs.json
        """,
    )
    parser.add_argument(
        "input_file",
        help="Path to input invoice or document (PDF, PNG, JPG, or TXT)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default=None,
        help="Path to save full structured extraction result in JSON format",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=None,
        help="Path to optional JSON file with custom dealer/model master catalogs",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        default="en",
        help="OCR language code: en (English), hi (Hindi), gu (Gujarati). Default: en",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable detailed logging output",
    )
    return parser


def main(argv: Optional[list] = None):
    # Ensure UTF-8 console output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    print(BANNER_LINE)
    print("DOCUMENT AI FIELD EXTRACTION")
    print(BANNER_LINE)
    print(f"\nInput: {input_path}\n")

    # Load custom catalogs if provided
    dealers, models = load_custom_config(args.config_file)

    def progress_callback(msg: str, step: int, total: int):
        print(f"[{step}/{total}] {msg}")

    try:
        pipeline = DocumentAIPipeline(
            dealer_master=dealers,
            model_master=models,
            ocr_lang=args.language,
        )

        doc = pipeline.process(input_path, on_progress=progress_callback)

        if args.output_file:
            export_json_result(input_path, doc, args.output_file)

        print_cli_summary(input_path, doc, output_path=args.output_file)

    except Exception as e:
        print(f"\nError processing document: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
