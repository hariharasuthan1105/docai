"""
End-to-end deterministic Document AI extraction pipeline.

Architecture:
  PDF / Image / Text
        │
        ▼
    PaddleOCR (OCR text + bounding boxes)
        │
        ▼
    Regex Extraction (HP, Asset Cost, candidates with OCR line bboxes)
        │
        ▼
    Fuzzy Dealer Matching (against dealer catalog)
        │
        ▼
    Exact Model Matching (against model catalog with normalized OCR text)
        │
        ▼
    Vision Mark Check (stub reporting status="not_implemented")
        │
        ▼
    Business Rules & Range Validation
        │
        ▼
    Confidence Scoring & Review Flagging
        │
        ▼
    FinalDocument (Structured JSON Output)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from docai.config import DEFAULT_DEALER_MASTER, DEFAULT_MODEL_MASTER
from docai.extraction.fuzzy_matcher import EntityMatcher
from docai.extraction.regex_engine import RegexExtractionEngine
from docai.models.extraction_schema import FieldValue, FinalDocument
from docai.ocr.paddleocr_engine import OCREngine, PaddleOCREngine
from docai.validation.business_rules import apply_business_rules
from docai.validation.confidence import compute_overall_confidence, needs_human_review
from docai.vision.signature_detector import SignatureDetector, StubSignatureDetector
from docai.vision.stamp_detector import StampDetector, StubStampDetector

logger = logging.getLogger(__name__)


class DocumentAIPipeline:
    """
    100% deterministic Document AI pipeline.
    Does not require any generative AI, LLMs, or external APIs.
    """

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        signature_detector: Optional[SignatureDetector] = None,
        stamp_detector: Optional[StampDetector] = None,
        dealer_master: Optional[List[str]] = None,
        model_master: Optional[List[str]] = None,
        ocr_lang: str = "en",
    ):
        self.ocr_engine = ocr_engine or PaddleOCREngine(lang=ocr_lang)
        self.signature_detector = signature_detector or StubSignatureDetector()
        self.stamp_detector = stamp_detector or StubStampDetector()

        self.regex_engine = RegexExtractionEngine()
        self.entity_matcher = EntityMatcher(
            dealer_master=dealer_master or DEFAULT_DEALER_MASTER,
            model_master=model_master or DEFAULT_MODEL_MASTER,
        )

    def process(
        self,
        image_path: str,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> FinalDocument:
        """
        Run the complete deterministic pipeline on an input document file.
        """

        def report(msg: str, step: int, total: int = 5):
            if on_progress:
                on_progress(msg, step, total)
            logger.info("[%d/%d] %s", step, total, msg)

        # 1. OCR
        report(f"Running PaddleOCR on {image_path}...", 1)
        ocr_result = self.ocr_engine.extract_text(image_path)

        # 2. Regex Rule-Based Extraction with Bounding Boxes
        report("Extracting regex fields (Horse Power, Asset Cost, Candidates)...", 2)
        regex_fields = self.regex_engine.extract_from_ocr_result(ocr_result)

        # 3. Entity Matching (Fuzzy Dealer & Exact Model)
        report("Matching dealer (fuzzy) and model (exact)...", 3)
        dealer_field = self.entity_matcher.match_dealer_from_ocr(
            ocr_result,
            candidate_dealer=regex_fields.get("dealer_name"),
        ) or regex_fields.get("dealer_name", FieldValue())

        model_field = self.entity_matcher.match_model_from_ocr(
            ocr_result,
            candidate_model=regex_fields.get("model_name"),
        ) or regex_fields.get("model_name", FieldValue())

        fields: Dict[str, FieldValue] = {
            "dealer_name": dealer_field,
            "model_name": model_field,
            "horse_power": regex_fields.get("horse_power", FieldValue()),
            "asset_cost": regex_fields.get("asset_cost", FieldValue()),
        }

        # 4. Vision Marks (Stub Detectors)
        report("Checking signature and stamp marks...", 4)
        sig_result = self.signature_detector.detect(image_path)
        stamp_result = self.stamp_detector.detect(image_path)

        # 5. Validation & Confidence Scoring
        report("Validating business rules & computing confidence...", 5)
        rule_reasons, _ = apply_business_rules(fields)
        overall_conf = compute_overall_confidence(fields)
        review_required, all_reasons = needs_human_review(overall_conf, rule_reasons)

        final_doc = FinalDocument(
            dealer_name=fields["dealer_name"],
            model_name=fields["model_name"],
            horse_power=fields["horse_power"],
            asset_cost=fields["asset_cost"],
            dealer_signature=sig_result,
            dealer_stamp=stamp_result,
            overall_confidence=overall_conf,
            needs_human_review=review_required,
            review_reasons=all_reasons,
        )

        return final_doc


def run_pipeline(
    image_path: str,
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
    return pipeline.process(image_path, on_progress=on_progress)
