"""
End-to-End Schema-Driven Multi-Document AI Pipeline.

Coordinates:
1. Intelligent Preprocessing (Deskew, Rotation Check, CLAHE, Denoise)
2. PaddleOCR Multi-Page Text & Bounding Box Extraction
3. Table Boundary & Header Detection (separating tabular rows from scalar fields)
4. Dynamic Document Type Classification (schema-driven, no hardcoded type checks)
5. Schema-Driven Field Extraction (Key-Value, Regex, Entity Matching, Header Positions)
6. Generic & Schema-Driven Multi-Column Table Extraction (e.g. line items)
7. Computer Vision Mark Detection (configured entirely from schema.visual_marks YAML)
8. Schema-Driven Validation (table math, arithmetic balances, ranges, required fields)
9. Calibrated Confidence Scoring & Human-in-the-Loop Decisioning

NO domain-specific field names appear in this file.
Adding a new document type requires ONLY a new YAML schema file.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from docai.classification.classifier import ClassificationResult, DocumentClassifier
from docai.extraction.fuzzy_matcher import FuzzyMatcher
from docai.extraction.schema_extractor import SchemaExtractor
from docai.layout.kv_extractor import LayoutKVExtractor
from docai.layout.table_detector import TableDetector
from docai.layout.table_extractor import TableExtractor
from docai.models.extraction_schema import (
    DocumentResult,
    FieldSource,
    FieldValue,
    ReviewDecision,
    TableResult,
    ValidationSummary,
    VisualMark,
)
from docai.ocr.paddleocr_engine import OCREngine, PaddleOCREngine
from docai.schemas.schema_loader import DocumentSchema, SchemaRegistry, get_schema_registry
from docai.validation.confidence import ConfidenceCalibrator
from docai.validation.schema_validator import SchemaValidator
from docai.vision.signature_detector import CVSignatureDetector, SignatureDetector
from docai.vision.stamp_detector import CVStampDetector, StampDetector

# FinalDocument alias for backward compatibility
FinalDocument = DocumentResult

logger = logging.getLogger(__name__)


class DocumentAIPipeline:
    """
    Schema-driven, multi-document intelligence pipeline.
    No hardcoded domain fields or document-type checks.
    """

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        registry: Optional[SchemaRegistry] = None,
        signature_detector: Optional[SignatureDetector] = None,
        stamp_detector: Optional[StampDetector] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        ocr_lang: str = "en",
        enable_preprocessing: bool = True,
        # Legacy kwargs accepted but ignored (catalogs now live in YAML)
        dealer_master: Optional[List[str]] = None,
        model_master: Optional[List[str]] = None,
    ):
        self.ocr_engine = ocr_engine or PaddleOCREngine(
            lang=ocr_lang,
            enable_preprocessing=enable_preprocessing,
        )
        self.registry = registry or get_schema_registry()
        self.classifier = DocumentClassifier(registry=self.registry)
        self.table_detector = TableDetector()
        self.table_extractor = TableExtractor()
        self.signature_detector = signature_detector or CVSignatureDetector()
        self.stamp_detector = stamp_detector or CVStampDetector()
        self.calibrator = calibrator
        # Fuzzy matcher — no hardcoded catalogs; catalogs come from schema YAML
        self.fuzzy_matcher = FuzzyMatcher()

    def process(
        self,
        document_path: str,
        document_id: str = "doc_001",
        document_type: Optional[str] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> DocumentResult:
        """
        Execute end-to-end extraction pipeline on an input document.
        Returns a generic DocumentResult (alias: FinalDocument).
        """
        start_time = time.perf_counter()

        def report(msg: str, step: int, total: int = 6):
            if on_progress:
                on_progress(msg, step, total)
            logger.info("[%d/%d] %s", step, total, msg)

        # 1. OCR + Preprocessing
        report(f"Preprocessing document and running OCR on {document_path}...", 1)
        ocr_result = self.ocr_engine.extract_text(document_path)

        # 2. Document Type Classification & Schema Selection
        report("Classifying document type and selecting schema...", 2)
        if document_type and document_type != "auto":
            selected_schema = self.registry.get_schema(document_type) or self.registry.get_schema("generic")
            class_res = ClassificationResult(
                document_type=document_type,
                confidence=1.0,
                title=selected_schema.title if selected_schema else document_type,
                schema=selected_schema,
            )
        else:
            class_res = self.classifier.classify(ocr_result)
            selected_schema = class_res.schema or self.registry.get_schema("generic")

        # 3. Table Boundary & Header Detection (stop keywords from schema YAML)
        report("Analyzing layout, table boundaries, and column structures...", 3)
        self.table_detector.table_defs = selected_schema.tables if selected_schema else {}
        self.table_detector.stop_keywords = selected_schema.stop_keywords if selected_schema else []
        table_regions = self.table_detector.detect_tables(ocr_result)
        table_lines = self.table_detector.get_table_line_indices(ocr_result, table_regions)

        # 4. Extract Tables
        extracted_tables: List[TableResult] = []
        table_map = {}
        for region in table_regions:
            ext_tab = self.table_extractor.extract_table(region)
            table_map[region.name] = ext_tab
            extracted_tables.append(
                TableResult(
                    name=ext_tab.name,
                    rows=ext_tab.to_dict_list(),
                    confidence=ext_tab.confidence,
                    bbox=ext_tab.bbox.to_list() if ext_tab.bbox else None,
                )
            )

        # 5. Schema-Driven Field Extraction (on non-table lines)
        report(f"Extracting fields for schema '{class_res.document_type}'...", 4)
        if selected_schema and selected_schema.document_type != "generic":
            # Pass schema so FuzzyMatcher can use schema entity_catalogs
            schema_extractor = SchemaExtractor(selected_schema, fuzzy_matcher=self.fuzzy_matcher)
            extracted_fields = schema_extractor.extract(ocr_result, excluded_line_indices=table_lines)
        else:
            # Fallback generic unknown document: extract plain key-value pairs
            kv_ext = LayoutKVExtractor()
            generic_kvs = kv_ext.extract_generic_key_values(ocr_result.lines, excluded_line_indices=table_lines)
            extracted_fields = {}
            for g in generic_kvs:
                key_slug = g["label"].lower().replace(" ", "_")
                extracted_fields[key_slug] = FieldValue(
                    value=g["value"],
                    confidence=round(g["confidence"], 4),
                    source=FieldSource.LAYOUT_KV,
                    source_text=f"{g['label']}: {g['value']}",
                    evidence=str(g["value"]),
                    bbox=g["bbox"],
                    page=g["page"],
                    method="generic_kv_split",
                )

        # 6. Visual Marks Detection — driven by schema.visual_marks YAML, not hardcoded type checks
        visual_marks: Dict[str, VisualMark] = {}
        if selected_schema and selected_schema.visual_marks:
            report("Detecting schema-defined visual marks (signatures, stamps)...", 5)
            for mark_def in selected_schema.visual_marks:
                if mark_def.mark_type == "signature":
                    vm = self.signature_detector.detect(document_path)
                    vm.mark_type = "signature"
                    visual_marks[mark_def.name] = vm
                elif mark_def.mark_type == "stamp":
                    vm = self.stamp_detector.detect(document_path)
                    vm.mark_type = "stamp"
                    visual_marks[mark_def.name] = vm

        # 7. Schema-Driven Validation
        report("Running schema validation rules and consistency checks...", 6)
        validation_errors: List[str] = []
        validation_warnings: List[str] = []
        passed_rules: List[str] = []

        if selected_schema:
            validator = SchemaValidator(selected_schema)
            val_report = validator.validate(extracted_fields, table_map)
            validation_errors = val_report.errors
            validation_warnings = val_report.warnings
            passed_rules = val_report.passed_rules

        # Group fields into sections (schema-driven section names)
        sections_dict: Dict[str, Dict[str, Any]] = {}
        if selected_schema:
            for s in selected_schema.sections:
                sections_dict[s] = {}
            for fname, fval in extracted_fields.items():
                fdef = selected_schema.get_field(fname)
                sec = fdef.section if fdef else "metadata"
                if sec not in sections_dict:
                    sections_dict[sec] = {}
                sections_dict[sec][fname] = fval.value

        # 8. Calculate Overall Confidence (field-weights from schema)
        field_weights = None
        if selected_schema:
            field_weights = {
                fname: fdef.confidence_weight
                for fname, fdef in selected_schema.fields.items()
            }

        field_confs = [f.confidence for f in extracted_fields.values() if f.is_present()]
        for t in extracted_tables:
            field_confs.append(t.confidence)

        if field_confs:
            base_score = sum(field_confs) / len(field_confs)
        else:
            base_score = 0.50

        if validation_errors:
            base_score = max(0.20, base_score - len(validation_errors) * 0.15)
        if validation_warnings:
            base_score = max(0.30, base_score - len(validation_warnings) * 0.05)

        overall_conf = round(min(0.99, max(0.10, base_score)), 4)
        calibrated_conf = self.calibrator.calibrate(overall_conf) if self.calibrator else None

        # 9. Human Review Decision
        auto_thresh = selected_schema.auto_approve_threshold if selected_schema else 0.88
        rev_thresh = selected_schema.review_threshold if selected_schema else 0.70

        review_reasons = list(validation_errors) + list(validation_warnings)
        if overall_conf < auto_thresh:
            review_reasons.append(f"Overall confidence ({overall_conf:.2f}) below auto-approval threshold ({auto_thresh:.2f}).")

        if overall_conf >= auto_thresh and not validation_errors:
            decision = ReviewDecision.AUTO_APPROVE
            needs_review = False
        elif overall_conf >= rev_thresh:
            decision = ReviewDecision.REVIEW
            needs_review = True
        else:
            decision = ReviewDecision.MANUAL_REVIEW
            needs_review = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DocumentResult(
            document_id=document_id,
            document=document_path,
            document_type=class_res.document_type,
            document_type_confidence=class_res.confidence,
            fields=extracted_fields,
            tables=extracted_tables,
            sections=sections_dict,
            visual_marks=visual_marks,
            validation=ValidationSummary(
                is_valid=not bool(validation_errors),
                passed_rules=passed_rules,
                errors=validation_errors,
                warnings=validation_warnings,
            ),
            overall_confidence=overall_conf,
            calibrated_confidence=calibrated_conf,
            decision=decision,
            review_required=needs_review,
            review_reasons=review_reasons,
            processing_time_ms=elapsed_ms,
            metadata={
                "classification": {
                    "document_type": class_res.document_type,
                    "confidence": class_res.confidence,
                    "matched_keywords": class_res.matched_keywords,
                    "scores": class_res.scores,
                },
                "validation": {
                    "passed_rules": passed_rules,
                    "errors": validation_errors,
                    "warnings": validation_warnings,
                },
                "ocr_lines_count": len(ocr_result.lines),
                "page_count": ocr_result.page_count,
            },
        )


def run_pipeline(
    document_path: str,
    ocr_engine: Optional[OCREngine] = None,
    registry: Optional[SchemaRegistry] = None,
    dealer_master: Optional[List[str]] = None,   # Legacy arg: accepted but ignored
    model_master: Optional[List[str]] = None,    # Legacy arg: accepted but ignored
    ocr_lang: str = "en",
    document_type: Optional[str] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> DocumentResult:
    """Convenience helper to run the pipeline with defaults."""
    pipeline = DocumentAIPipeline(
        ocr_engine=ocr_engine,
        registry=registry,
        ocr_lang=ocr_lang,
    )
    return pipeline.process(document_path, document_type=document_type, on_progress=on_progress)
