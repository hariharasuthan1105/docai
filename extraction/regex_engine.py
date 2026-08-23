"""
Regex Pattern Utilities for Schema-Driven and Baseline Extraction.

This module provides:
  - normalize_numeric_string(): shared numeric normalization
  - Individual pattern-extraction utility functions (tractor-compatible, used by baseline & tests)
  - RegexExtractionEngine: a GENERIC class that applies schema.field.regex_patterns
    to OCR lines, returning {field_name: FieldValue} where field names come from YAML

ARCHITECTURE NOTES:
  - RegexExtractionEngine does NOT return hardcoded field names. Names come from the schema.
  - Individual helper functions (extract_horse_power_from_line, etc.) are pure utilities
    with no hardcoded output dict. They are reused by the tractor baseline and tests.
  - The tractor-domain class TractorRegexBaseline (which returns a field-named dict) lives
    in evaluation/baselines/tractor_regex_baseline.py for ablation purposes only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult


# ---------------------------------------------------------------------------
# Re-usable Pattern Libraries (not domain assumptions — just regex primitives)
# Used by: tractor baseline, schema-defined field regexes, and unit tests
# ---------------------------------------------------------------------------

_HP_PATTERNS = [
    re.compile(
        r"(?:horse\s*power|engine\s*power|power|hp)\s*[:\-]\s*(\d{1,3}(?:\.\d+)?)\s*(?:h\.?p\.?|hp)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:horse\s*power|engine\s*power|power)\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:h\.?p\.?|hp)?",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:h\.?\s*p\.?|hp)\b", re.IGNORECASE),
    re.compile(r"\bhp\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)\b", re.IGNORECASE),
]

_PRIMARY_COST_PATTERNS = [
    re.compile(
        r"(?:asset\s*cost|total\s*cost|grand\s*total|invoice\s*total|net\s*total|invoice\s*value"
        r"|total\s*amount(?:\s*payable)?|amount\s*payable|total\s*(?:\([^)]*\))?)\s*[:\-]?\s*"
        r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/-)?",
        re.IGNORECASE,
    ),
]

_SECONDARY_COST_PATTERNS = [
    re.compile(
        r"(?:cost|amount|price|base\s*price|sub\s*total)\s*[:\-]?\s*"
        r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/-)?",
        re.IGNORECASE,
    ),
    re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/-)?", re.IGNORECASE),
]


def extract_horse_power_from_line(
    line_text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    page: int = 1,
) -> Optional[FieldValue]:
    """Extract horse power value from a single OCR line. Pure utility, no field dict."""
    for pattern in _HP_PATTERNS:
        match = pattern.search(line_text)
        if match:
            raw_val = match.group(1)
            num = normalize_numeric_string(raw_val)
            if num is not None and 5.0 <= num <= 250.0:
                return FieldValue(
                    value=num,
                    confidence=0.95,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method="regex_hp",
                )
    return None


def extract_asset_cost_from_line(
    line_text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    primary_only: bool = False,
    page: int = 1,
) -> Optional[FieldValue]:
    """Extract asset/invoice cost from a single OCR line. Pure utility, no field dict."""
    patterns = _PRIMARY_COST_PATTERNS if primary_only else (_PRIMARY_COST_PATTERNS + _SECONDARY_COST_PATTERNS)
    for pattern in patterns:
        match = pattern.search(line_text)
        if match:
            raw_val = match.group(1)
            num = normalize_numeric_string(raw_val)
            if num is not None and num >= 1000.0:
                return FieldValue(
                    value=num,
                    confidence=0.92,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method="regex_cost",
                )
    return None


def normalize_numeric_string(raw_str: str) -> Optional[float]:
    """Parse a numeric string supporting commas and Indian number formats."""
    if not raw_str:
        return None
    cleaned = raw_str.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_first_match(
    line_text: str,
    patterns: List[re.Pattern],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    page: int = 1,
    method: str = "regex",
    min_length: int = 1,
) -> Optional[FieldValue]:
    """
    Try each compiled pattern against line_text in order.
    Return a FieldValue for the first match, or None.

    This is a pure utility — caller supplies patterns and field semantics.
    """
    for pattern in patterns:
        match = pattern.search(line_text)
        if match:
            raw_val = match.group(1).strip() if match.groups() else match.group(0).strip()
            if len(raw_val) >= min_length:
                return FieldValue(
                    value=raw_val,
                    confidence=0.90,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method=method,
                )
    return None


class RegexExtractionEngine:
    """
    Generic schema-driven regex extractor.

    Applies schema-defined regex_patterns for each field in the schema,
    scanning each OCR line. Returns a dict[field_name -> FieldValue] where
    field names come from the schema, not from hardcoded strings in this class.

    Usage:
        engine = RegexExtractionEngine(schema)
        fields = engine.extract_from_ocr(ocr_result)
    """

    def __init__(self, schema=None):
        """
        schema: DocumentSchema instance. If None, returns empty dict (no patterns to apply).
        """
        self.schema = schema

    def extract_from_ocr(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """
        Apply schema field regex_patterns to OCR lines.
        Returns {field_name: FieldValue} where field_name is from schema YAML.
        """
        if not self.schema:
            return {}

        results: Dict[str, FieldValue] = {}

        for fname, fdef in self.schema.fields.items():
            compiled = fdef.get_compiled_regexes()
            if not compiled:
                continue

            for line in ocr_result.lines:
                text = line.text.strip()
                if not text:
                    continue
                page = getattr(line, "page", 1)
                bbox = tuple(line.bbox) if line.bbox else None

                match_result = extract_first_match(
                    text,
                    compiled,
                    bbox=bbox,
                    page=page,
                    method=f"regex_{fname}",
                )
                if match_result:
                    results[fname] = match_result
                    break  # First match per field wins

        return results

    # Keep backward-compat method name (used in some tests)
    def extract_from_ocr_result(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """Backward-compatible alias for extract_from_ocr()."""
        return self.extract_from_ocr(ocr_result)

    def extract(self, ocr_text: str) -> Dict[str, FieldValue]:
        """Extract from raw text string (no bbox info)."""
        lines = [
            OCRLine(text=line, bbox=(0.0, 0.0, 0.0, 0.0), confidence=1.0, page=1)
            for line in ocr_text.splitlines()
            if line.strip()
        ]
        return self.extract_from_ocr(OCRResult(lines=lines))


# Backward-compat alias
RegexEngine = RegexExtractionEngine
