"""
Layered Multi-Strategy Field Extractor.

Coordinates:
- Level 1: 2D Spatial Layout & Geometric Key-Value Parsing
- Level 2: Regex & Pattern Extraction
- Level 3: Fuzzy & Exact Entity Matching
- Level 4: Reconciliation and Multi-pass Consensus
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from docai.extraction.fuzzy_matcher import EntityMatcher
from docai.extraction.regex_engine import (
    RegexExtractionEngine,
    extract_asset_cost_from_line,
    extract_horse_power_from_line,
    normalize_numeric_string,
)
from docai.layout.kv_extractor import LayoutKVExtractor
from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRResult

logger = logging.getLogger(__name__)


class LayeredFieldExtractor:
    """
    Combines Spatial Layout Intelligence with Pattern/Regex and Entity Matching.
    """

    def __init__(
        self,
        entity_matcher: Optional[EntityMatcher] = None,
        regex_engine: Optional[RegexExtractionEngine] = None,
        layout_extractor: Optional[LayoutKVExtractor] = None,
    ):
        self.entity_matcher = entity_matcher
        self.regex_engine = regex_engine or RegexExtractionEngine()
        self.layout_extractor = layout_extractor or LayoutKVExtractor()

    def extract(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """
        Extract all target document fields using the layered strategy.
        """
        # Step 1: 2D Spatial Layout Key-Value Extraction
        layout_fields: Dict[str, FieldValue] = {}
        for page_idx in range(1, ocr_result.page_count + 1):
            page_lines = ocr_result.get_page_lines(page_idx) or ocr_result.lines
            p_fields = self.layout_extractor.extract_fields(page_lines, page=page_idx)
            for k, v in p_fields.items():
                if k not in layout_fields or layout_fields[k].confidence < v.confidence:
                    layout_fields[k] = v

        # Step 2: Global Line-by-Line Regex Extraction
        regex_fields = self.regex_engine.extract_from_ocr_result(ocr_result)

        # Step 3: Entity Matching for Dealer and Model
        dealer_candidate = layout_fields.get("dealer_name") or regex_fields.get("dealer_name")
        model_candidate = layout_fields.get("model_name") or regex_fields.get("model_name")

        matched_dealer: Optional[FieldValue] = None
        matched_model: Optional[FieldValue] = None

        if self.entity_matcher:
            matched_dealer = self.entity_matcher.match_dealer_from_ocr(
                ocr_result,
                candidate_dealer=dealer_candidate,
            )
            matched_model = self.entity_matcher.match_model_from_ocr(
                ocr_result,
                candidate_model=model_candidate,
            )

        # Step 4: Layered Field Reconciliation
        reconciled: Dict[str, FieldValue] = {}

        # 4a. Dealer Name
        if matched_dealer and matched_dealer.is_present():
            reconciled["dealer_name"] = matched_dealer
        elif dealer_candidate and dealer_candidate.is_present():
            reconciled["dealer_name"] = dealer_candidate
        else:
            reconciled["dealer_name"] = regex_fields.get("dealer_name", FieldValue())

        # 4b. Model Name
        if matched_model and matched_model.is_present():
            reconciled["model_name"] = matched_model
        elif model_candidate and model_candidate.is_present():
            reconciled["model_name"] = model_candidate
        else:
            reconciled["model_name"] = regex_fields.get("model_name", FieldValue())

        # 4c. Horse Power (Layout priority, normalized numeric parsing)
        hp_val: Optional[FieldValue] = None
        if "horse_power" in layout_fields and layout_fields["horse_power"].is_present():
            raw_hp = str(layout_fields["horse_power"].value)
            parsed_hp = extract_horse_power_from_line(raw_hp, page=layout_fields["horse_power"].page)
            if parsed_hp:
                hp_val = FieldValue(
                    value=parsed_hp.value,
                    confidence=max(parsed_hp.confidence, layout_fields["horse_power"].confidence),
                    source=FieldSource.REGEX,
                    source_text=raw_hp,
                    evidence=raw_hp,
                    bbox=layout_fields["horse_power"].bbox,
                    page=layout_fields["horse_power"].page,
                    method="layout_hp",
                )
        if not hp_val and regex_fields.get("horse_power", FieldValue()).is_present():
            hp_val = regex_fields["horse_power"]
        reconciled["horse_power"] = hp_val or FieldValue()

        # 4d. Asset Cost (Layout priority, numeric normalization)
        cost_val: Optional[FieldValue] = None
        if "asset_cost" in layout_fields and layout_fields["asset_cost"].is_present():
            raw_cost = str(layout_fields["asset_cost"].value)
            parsed_cost = extract_asset_cost_from_line(raw_cost, page=layout_fields["asset_cost"].page)
            if parsed_cost:
                cost_val = FieldValue(
                    value=parsed_cost.value,
                    confidence=max(parsed_cost.confidence, layout_fields["asset_cost"].confidence),
                    source=FieldSource.REGEX,
                    source_text=raw_cost,
                    evidence=raw_cost,
                    bbox=layout_fields["asset_cost"].bbox,
                    page=layout_fields["asset_cost"].page,
                    method="layout_cost",
                )
            else:
                num = normalize_numeric_string(raw_cost)
                if num and num >= 1000.0:
                    cost_val = FieldValue(
                        value=num,
                        confidence=layout_fields["asset_cost"].confidence,
                        source=FieldSource.LAYOUT_KV,
                        source_text=raw_cost,
                        evidence=raw_cost,
                        bbox=layout_fields["asset_cost"].bbox,
                        page=layout_fields["asset_cost"].page,
                        method="layout_cost_normalized",
                    )

        if not cost_val and regex_fields.get("asset_cost", FieldValue()).is_present():
            cost_val = regex_fields["asset_cost"]
        reconciled["asset_cost"] = cost_val or FieldValue()

        # 4e. Metadata and Customer Fields
        for field_name in [
            "invoice_number",
            "invoice_date",
            "customer_name",
            "customer_address",
            "phone_number",
            "registration_number",
            "serial_number",
        ]:
            if field_name in layout_fields and layout_fields[field_name].is_present():
                reconciled[field_name] = layout_fields[field_name]
            elif field_name in regex_fields and regex_fields[field_name].is_present():
                reconciled[field_name] = regex_fields[field_name]
            else:
                reconciled[field_name] = FieldValue()

        return reconciled
