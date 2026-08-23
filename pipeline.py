"""
End-to-End Production Document AI Pipeline.

Coordinates:
1. Intelligent Preprocessing (Deskew, Rotation Check, CLAHE, Denoise)
2. PaddleOCR Multi-Page Text & Bounding Box Extraction
3. 2D Document Layout Intelligence & Spatial Key-Value Parsing
4. Layered Field Extraction (Layout -> Pattern -> Fuzzy Matching -> Semantic)
5. Genuine Computer Vision Signature & Stamp Detection
6. Extensible Business Rules & Domain Validation
7. Multi-signal Confidence Scoring & Statistical Calibration
8. Human-in-the-Loop Review Decisioning
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from docai.config import DEFAULT_DEALER_MASTER, DEFAULT_MODEL_MASTER
from docai.extraction.fuzzy_matcher import EntityMatcher
from docai.extraction.layout_extractor import LayeredFieldExtractor
from docai.extraction.regex_engine import RegexExtractionEngine
from docai.layout.kv_extractor import LayoutKVExtractor
from docai.models.extraction_schema import (
    FieldSource,
    FieldValue,
    FinalDocument,
    ReviewDecision,
    VisualMark,
)
from docai.ocr.paddleocr_engine import OCREngine, PaddleOCREngine
from docai.validation.business_rules import RuleEngine
from docai.validation.confidence import (
    ConfidenceCalibrator,
    compute_field_confidence,
    compute_overall_confidence,
    make_review_decision,
)
from docai.vision.signature_detector import CVSignatureDetector, SignatureDetector
from docai.vision.stamp_detector import CVStampDetector, StampDetector

logger = logging.getLogger(__name__)


class DocumentAIPipeline:
    """
    Production-ready Document AI Pipeline.
    """

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        signature_detector: Optional[SignatureDetector] = None,
        stamp_detector: Optional[StampDetector] = None,
        rule_engine: Optional[RuleEngine] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        dealer_master: Optional[List[str]] = None,
        model_master: Optional[List[str]] = None,
        ocr_lang: str = "en",
        enable_preprocessing: bool = True,
    ):
        self.ocr_engine = ocr_engine or PaddleOCREngine(
            lang=ocr_lang,
            enable_preprocessing=enable_preprocessing,
        )
        self.signature_detector = signature_detector or CVSignatureDetector()
        self.stamp_detector = stamp_detector or CVStampDetector()
        self.rule_engine = rule_engine or RuleEngine()
        self.calibrator = calibrator

        self.entity_matcher = EntityMatcher(
            dealer_master=dealer_master or DEFAULT_DEALER_MASTER,
            model_master=model_master or DEFAULT_MODEL_MASTER,
        )
        self.regex_engine = RegexExtractionEngine()
        self.layout_extractor = LayoutKVExtractor()

        self.layered_extractor = LayeredFieldExtractor(
            entity_matcher=self.entity_matcher,
            regex_engine=self.regex_engine,
            layout_extractor=self.layout_extractor,
        )

    def process(
        self,
        document_path: str,
        document_id: str = "doc_001",
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> FinalDocument:
        """
        Execute end-to-end extraction pipeline on an input document.
        """
        start_time = time.perf_counter()

        def report(msg: str, step: int, total: int = 6):
            if on_progress:
                on_progress(msg, step, total)
            logger.info("[%d/%d] %s", step, total, msg)

        # 1. OCR + Preprocessing
        report(f"Preprocessing document and running OCR on {document_path}...", 1)
        ocr_result = self.ocr_engine.extract_text(document_path)

        # 2. Layered Field Extraction (Layout -> Pattern -> Fuzzy Matching)
        report("Extracting fields with 2D spatial layout and pattern rules...", 2)
        extracted_fields = self.layered_extractor.extract(ocr_result)

        # 3. Vision Marks Detection (CV Signature & Stamp)
        report("Detecting handwritten signatures and official rubber stamps...", 3)
        sig_result = self.signature_detector.detect(document_path)
        stamp_result = self.stamp_detector.detect(document_path)

        # 4. Domain & Business Rule Validation
        report("Executing business rules and domain validations...", 4)
        validation_reasons, _ = self.rule_engine.validate(extracted_fields)

        # 5. Multi-Signal Confidence Scoring & Calibration
        report("Calibrating multi-signal confidence scores...", 5)
        # Recalculate field-level confidences with validation signals
        for fname, fval in extracted_fields.items():
            if fval.is_present():
                fval.confidence = compute_field_confidence(fval)

        overall_conf = compute_overall_confidence(extracted_fields)
        calibrated_conf = self.calibrator.calibrate(overall_conf) if self.calibrator else None

        # 6. Human-in-the-Loop Decision
        report("Computing review decision and audit trail...", 6)
        decision, review_needed, review_reasons = make_review_decision(
            overall_confidence=overall_conf,
            validation_reasons=validation_reasons,
            fields=extracted_fields,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        final_doc = FinalDocument(
            document_id=document_id,
            document=document_path,
            dealer_name=extracted_fields.get("dealer_name", FieldValue()),
            model_name=extracted_fields.get("model_name", FieldValue()),
            horse_power=extracted_fields.get("horse_power", FieldValue()),
            asset_cost=extracted_fields.get("asset_cost", FieldValue()),
            invoice_number=extracted_fields.get("invoice_number", FieldValue()),
            invoice_date=extracted_fields.get("invoice_date", FieldValue()),
            customer_name=extracted_fields.get("customer_name", FieldValue()),
            customer_address=extracted_fields.get("customer_address", FieldValue()),
            phone_number=extracted_fields.get("phone_number", FieldValue()),
            registration_number=extracted_fields.get("registration_number", FieldValue()),
            serial_number=extracted_fields.get("serial_number", FieldValue()),
            dealer_signature=sig_result,
            dealer_stamp=stamp_result,
            overall_confidence=overall_conf,
            calibrated_confidence=calibrated_conf,
            decision=decision,
            needs_human_review=review_needed,
            review_reasons=review_reasons,
            processing_time_ms=elapsed_ms,
            metadata={
                "ocr_lines_count": len(ocr_result.lines),
                "page_count": ocr_result.page_count,
                "ocr_metadata": ocr_result.metadata,
            },
        )

        return final_doc


def run_pipeline(
    document_path: str,
    ocr_engine: Optional[OCREngine] = None,
    dealer_master: Optional[List[str]] = None,
    model_master: Optional[List[str]] = None,
    ocr_lang: str = "en",
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> FinalDocument:
    """Convenience helper to run the pipeline with defaults."""
    pipeline = DocumentAIPipeline(
        ocr_engine=ocr_engine,
        dealer_master=dealer_master,
        model_master=model_master,
        ocr_lang=ocr_lang,
    )
    return pipeline.process(document_path, on_progress=on_progress)
