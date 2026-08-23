"""
Schema-Driven Generic Field Extractor.

Interprets any DocumentSchema dynamically without hardcoded domain knowledge.
Executes declared extraction strategies (key_value, regex, entity_match, position_header)
and type normalizations across non-table OCR regions.

Extraction strategies are configured entirely in YAML schema fields:
  - key_value:        label-anchor spatial matching
  - regex:            schema-defined regex patterns
  - position_header:  positional heuristics using field.position metadata
  - entity_match:     fuzzy/exact catalog matching using field.catalog_ref or field.catalog_master
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from docai.extraction.fuzzy_matcher import FuzzyMatcher
from docai.layout.kv_extractor import LayoutKVExtractor, clean_extracted_value
from docai.models.extraction_schema import BoundingBox, FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult
from docai.schemas.schema_loader import DocumentSchema, FieldDefinition

logger = logging.getLogger(__name__)


def parse_numeric(val: Any) -> Optional[float]:
    """Extract float value from string, ignoring currency symbols and percentage prefixes."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    cleaned = val.replace(",", "").replace("\u20b9", "").replace("Rs.", "").replace("INR", "").strip()
    # Remove parenthesized percentage expressions e.g. '(10%)'
    cleaned = re.sub(r"\(\s*[-+]?\d+(?:\.\d+)?\s*%\s*\)", "", cleaned).strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if match:
        try:
            return abs(float(match.group(0)))
        except ValueError:
            pass
    return None


def _bbox_to_list(bbox: Any) -> Optional[List[float]]:
    if bbox is None:
        return None
    if hasattr(bbox, "to_list"):
        return bbox.to_list()
    if isinstance(bbox, (list, tuple)):
        return [float(x) for x in bbox]
    return [0.0, 0.0, 0.0, 0.0]


class SchemaExtractor:
    """
    Dynamically extracts fields defined by any DocumentSchema.
    No knowledge of specific field names (dealer_name, horse_power, merchant_name, etc.)
    is present in this class. All extraction behavior is driven by YAML schema metadata.
    """

    def __init__(self, schema: DocumentSchema, fuzzy_matcher: Optional[FuzzyMatcher] = None):
        self.schema = schema
        self.kv_extractor = LayoutKVExtractor(schema=schema)
        self.fuzzy_matcher = fuzzy_matcher or FuzzyMatcher()

    def extract(
        self,
        ocr_result: OCRResult,
        excluded_line_indices: Optional[Set[int]] = None,
    ) -> Dict[str, FieldValue]:
        """
        Extract all schema fields from OCR text and bounding boxes.
        """
        extracted_fields: Dict[str, FieldValue] = {}
        excluded = excluded_line_indices or set()
        active_lines = [line for idx, line in enumerate(ocr_result.lines) if idx not in excluded]

        # 1. Run Layout Key-Value Extraction on active lines
        kv_fields = self.kv_extractor.extract_fields(
            ocr_result.lines,
            page=1,
            excluded_line_indices=excluded,
        )

        # 2. Iterate through all schema fields and apply configured extraction strategies
        for fname, fdef in self.schema.fields.items():
            best_candidate: Optional[FieldValue] = None

            # Strategy A: Check if Key-Value extractor captured this field
            if fname in kv_fields:
                raw_val = kv_fields[fname].value
                norm_val, is_valid = self._normalize_type(raw_val, fdef.type)
                if is_valid:
                    best_candidate = FieldValue(
                        value=norm_val,
                        confidence=kv_fields[fname].confidence,
                        source=FieldSource.LAYOUT_KV,
                        source_text=kv_fields[fname].source_text,
                        evidence=str(raw_val),
                        bbox=kv_fields[fname].bbox,
                        page=kv_fields[fname].page,
                        method=kv_fields[fname].method,
                    )

            # Strategy B: Regex Pattern Matching across active lines
            if (best_candidate is None or best_candidate.confidence < 0.90) and "regex" in fdef.extraction_strategies:
                regex_cand = self._extract_regex(active_lines, fdef)
                if regex_cand and (best_candidate is None or regex_cand.confidence > best_candidate.confidence):
                    best_candidate = regex_cand

            # Strategy C: Position Header (uses fdef.position metadata, NOT field name)
            if best_candidate is None and "position_header" in fdef.extraction_strategies:
                hdr_cand = self._extract_position_header(active_lines, fdef)
                if hdr_cand:
                    best_candidate = hdr_cand

            # Strategy D: Catalog / Entity Matching (uses fdef.catalog_ref or fdef.catalog_master)
            if (best_candidate is None or best_candidate.confidence < 0.85) and "entity_match" in fdef.extraction_strategies:
                entity_cand = self._extract_entity_match(active_lines, fdef)
                if entity_cand and (best_candidate is None or entity_cand.confidence > best_candidate.confidence):
                    best_candidate = entity_cand

            if best_candidate:
                extracted_fields[fname] = best_candidate

        return extracted_fields

    def _extract_regex(self, lines: List[OCRLine], fdef: FieldDefinition) -> Optional[FieldValue]:
        """Extract field value using schema-defined regex patterns."""
        compiled_regexes = fdef.get_compiled_regexes()
        if not compiled_regexes:
            return None

        for line in lines:
            text = line.text.strip()
            for pattern in compiled_regexes:
                match = pattern.search(text)
                if match:
                    raw_val = match.group(1) if match.groups() else match.group(0)
                    cleaned = clean_extracted_value(raw_val)
                    norm_val, is_valid = self._normalize_type(cleaned, fdef.type)
                    if is_valid:
                        return FieldValue(
                            value=norm_val,
                            confidence=round(line.confidence * 0.95, 4),
                            source=FieldSource.REGEX,
                            source_text=text,
                            evidence=cleaned,
                            bbox=_bbox_to_list(line.bbox),
                            page=1,
                            method="regex_pattern",
                        )
        return None

    def _extract_position_header(
        self, lines: List[OCRLine], fdef: FieldDefinition
    ) -> Optional[FieldValue]:
        """
        Extract positional header fields using schema metadata on fdef.
        Uses fdef.position ("first_header", "second_header", "header_block"),
        fdef.tagline_markers, and fdef.address_keywords — NOT field name strings.
        """
        if not lines or not fdef.position:
            return None

        # Collect header candidate lines: top 6 non-delimiter lines
        header_candidates = []
        for line in lines[:8]:
            t = line.text.strip()
            if ":" not in t and "tax invoice" not in t.lower() and "bill" not in t.lower():
                header_candidates.append(line)

        if not header_candidates:
            return None

        if fdef.position == "first_header":
            # First non-delimiter line is typically the brand/business name
            first = header_candidates[0]
            val = first.text.strip()
            return FieldValue(
                value=val,
                confidence=round(first.confidence * 0.95, 4),
                source=FieldSource.LAYOUT_KV,
                source_text=val,
                evidence=val,
                bbox=_bbox_to_list(first.bbox),
                page=1,
                method="position_header_first",
            )

        if fdef.position == "second_header":
            # Second line if it matches any configured tagline_markers
            if len(header_candidates) >= 2:
                sec = header_candidates[1]
                t = sec.text.strip()
                markers = fdef.tagline_markers
                if markers:
                    if any(m.upper() in t.upper() for m in markers):
                        return FieldValue(
                            value=t,
                            confidence=round(sec.confidence * 0.90, 4),
                            source=FieldSource.LAYOUT_KV,
                            source_text=t,
                            evidence=t,
                            bbox=_bbox_to_list(sec.bbox),
                            page=1,
                            method="position_header_second",
                        )
                else:
                    # No markers configured: return second line unconditionally
                    return FieldValue(
                        value=t,
                        confidence=round(sec.confidence * 0.85, 4),
                        source=FieldSource.LAYOUT_KV,
                        source_text=t,
                        evidence=t,
                        bbox=_bbox_to_list(sec.bbox),
                        page=1,
                        method="position_header_second",
                    )

        if fdef.position == "header_block":
            # Collect lines that match any configured address_keywords
            addr_keywords = fdef.address_keywords
            addr_parts = []
            bboxes = []
            for line in header_candidates[1:]:
                t = line.text.strip()
                if addr_keywords:
                    if any(kw.lower() in t.lower() for kw in addr_keywords):
                        addr_parts.append(t)
                        bboxes.append(line.bbox)
                else:
                    # No keywords configured: collect all header block lines
                    addr_parts.append(t)
                    bboxes.append(line.bbox)

            if addr_parts:
                combined_addr = ", ".join(addr_parts)
                b0_list = _bbox_to_list(bboxes[0]) or [0, 0, 0, 0]
                b_last_list = _bbox_to_list(bboxes[-1]) or [0, 0, 0, 0]
                comb_bbox = [
                    b0_list[0],
                    b0_list[1],
                    max((_bbox_to_list(b) or [0, 0, 0, 0])[2] for b in bboxes),
                    b_last_list[3],
                ]
                return FieldValue(
                    value=combined_addr,
                    confidence=0.90,
                    source=FieldSource.LAYOUT_KV,
                    source_text=combined_addr,
                    evidence=combined_addr,
                    bbox=comb_bbox,
                    page=1,
                    method="position_header_block",
                )

        return None

    def _extract_entity_match(self, lines: List[OCRLine], fdef: FieldDefinition) -> Optional[FieldValue]:
        """
        Match OCR lines against a catalog list.
        Catalog resolution order:
          1. fdef.catalog_ref  -> look up name in schema.entity_catalogs
          2. fdef.catalog_master -> inline list in field definition
        No hardcoded field-name checks.
        """
        catalog: List[str] = []

        # 1. Named catalog reference (from schema.entity_catalogs)
        if fdef.catalog_ref:
            catalog = self.schema.get_catalog(fdef.catalog_ref)

        # 2. Inline catalog list in field definition
        if not catalog and fdef.catalog_master:
            catalog = fdef.catalog_master

        if not catalog:
            return None

        best_entity = None
        best_score = 0.0
        best_line = None

        for line in lines:
            text = line.text.strip()
            if len(text) < 3:
                continue
            for entry in catalog:
                score = self.fuzzy_matcher.calculate_similarity(text, entry)
                if score > best_score and score >= 0.75:
                    best_score = score
                    best_entity = entry
                    best_line = line

        if best_entity and best_line:
            return FieldValue(
                value=best_entity,
                confidence=round(best_score * best_line.confidence, 4),
                source=FieldSource.FUZZY_MATCH,
                source_text=best_line.text,
                evidence=best_line.text,
                bbox=_bbox_to_list(best_line.bbox),
                page=1,
                method="entity_catalog_match",
            )

        return None

    def _normalize_type(self, val: Any, ftype: str) -> Tuple[Any, bool]:
        """Normalize extracted string value into expected schema data type."""
        if val is None:
            return None, False

        val_str = str(val).strip()
        if not val_str:
            return None, False

        if ftype == "currency" or ftype == "number":
            num = parse_numeric(val_str)
            if num is not None:
                return (int(num) if num % 1 == 0 and ftype == "number" else round(num, 2)), True
            return val_str, False

        if ftype == "phone":
            digits = re.sub(r"[^\d\+]", "", val_str)
            if len(digits) >= 8:
                return val_str, True
            return val_str, False

        if ftype == "date":
            dmatch = re.search(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b", val_str)
            if dmatch:
                return dmatch.group(1), True
            return val_str, True

        if ftype == "time":
            tmatch = re.search(r"\b(\d{1,2}:\d{2}(?:\s*[apAP][mM])?)\b", val_str)
            if tmatch:
                return tmatch.group(1), True
            return val_str, True

        # Default: string
        return val_str, True
