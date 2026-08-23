"""
Tractor Invoice Regex Baseline Extractor.

IMPORTANT: This is a DOMAIN-SPECIFIC BASELINE for evaluation/ablation purposes only.
It is NOT used by the main generic pipeline.
The main pipeline uses SchemaExtractor with YAML-driven regex patterns instead.

This baseline extracts tractor invoice fields by name using hardcoded regex patterns.
It is preserved for:
  - Ablation studies comparing baseline vs. schema-driven extraction
  - Regression testing of tractor-specific OCR patterns
  - Reference implementation for building schema YAML regex entries

DO NOT import or call this from pipeline.py, schema_extractor.py, or any core module.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult

# --- Horse Power Patterns (tractor-specific) ----------------------------------
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

# --- Asset Cost Patterns (tractor invoice amounts) ----------------------------
_PRIMARY_COST_PATTERNS = [
    re.compile(
        r"(?:asset\s*cost|total\s*cost|grand\s*total|invoice\s*total|net\s*total|invoice\s*value|total\s*amount(?:\s*payable)?|amount\s*payable|total\s*(?:\([^)]*\))?)\s*[:\-]?\s*"
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

# --- Invoice & Date Patterns --------------------------------------------------
_INVOICE_NUM_PATTERNS = [
    re.compile(r"(?:invoice\s*(?:no\.?|num\.?|number|#)|bill\s*(?:no\.?|number)|inv\s*no\.?)\s*[:\-]?\s*([a-zA-Z0-9\-\/]+)", re.IGNORECASE),
    re.compile(r"\b(?:inv|bill)[/\-_](?:20\d{2}[/\-_])?[a-zA-Z0-9\-]+\b", re.IGNORECASE),
]
_DATE_PATTERNS = [
    re.compile(r"(?:date|dated|invoice\s*date|bill\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
]

# --- Entity Patterns ---------------------------------------------------------
_PHONE_PATTERNS = [
    re.compile(r"(?:phone|mobile|mob|contact|tel)\s*(?:no\.?|number)?\s*[:\-]?\s*(?:\+91[\-\s]?)?([6-9]\d{9})\b", re.IGNORECASE),
    re.compile(r"\b(?:\+91[\-\s]?)?([6-9]\d{4}[\-\s]?\d{5})\b"),
]
_DEALER_PATTERNS = [
    re.compile(r"(?:dealer\s*name|dealer|m/s\.?|authorized\s*dealer|authorised\s*dealer)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]
_MODEL_PATTERNS = [
    re.compile(r"(?:model\s*name|model|tractor\s*model)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]
_CUSTOMER_PATTERNS = [
    re.compile(r"(?:customer\s*name|buyer\s*name|bill\s*to|purchaser|sold\s*to)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]


def _normalize_numeric(raw: str) -> Optional[float]:
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


class TractorRegexBaseline:
    """
    Baseline tractor invoice extractor using hardcoded regex patterns.
    For evaluation/ablation ONLY. Do NOT use in the main pipeline.
    """

    def extract_from_ocr_result(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """
        Extract tractor invoice fields by name using hardcoded patterns.
        Returns a dict keyed by tractor-specific field names.
        """
        hp_field = FieldValue()
        cost_field = FieldValue()
        dealer_field = FieldValue()
        model_field = FieldValue()
        inv_no_field = FieldValue()
        date_field = FieldValue()
        cust_name_field = FieldValue()
        phone_field = FieldValue()

        lines = ocr_result.lines
        n_lines = len(lines)

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue
            page = getattr(line, "page", 1)
            bbox = list(line.bbox) if line.bbox else None

            if not hp_field.is_present():
                for pat in _HP_PATTERNS:
                    m = pat.search(text)
                    if m:
                        num = _normalize_numeric(m.group(1))
                        if num and 5.0 <= num <= 250.0:
                            hp_field = FieldValue(value=num, confidence=0.95, source=FieldSource.REGEX,
                                                  evidence=m.group(0), bbox=bbox, page=page, method="regex_hp")
                            break

            if not cost_field.is_present():
                for pat in _PRIMARY_COST_PATTERNS:
                    m = pat.search(text)
                    if m:
                        num = _normalize_numeric(m.group(1))
                        if num and num >= 1000.0:
                            cost_field = FieldValue(value=num, confidence=0.92, source=FieldSource.REGEX,
                                                    evidence=m.group(0), bbox=bbox, page=page, method="regex_cost")
                            break

            if not inv_no_field.is_present():
                for pat in _INVOICE_NUM_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = (m.group(1) if m.groups() else m.group(0)).strip()
                        if len(val) >= 3:
                            inv_no_field = FieldValue(value=val, confidence=0.90, source=FieldSource.REGEX,
                                                      evidence=m.group(0), bbox=bbox, page=page, method="regex_inv")
                            break

            if not date_field.is_present():
                for pat in _DATE_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1) if m.groups() else m.group(0)
                        date_field = FieldValue(value=val, confidence=0.91, source=FieldSource.REGEX,
                                                evidence=m.group(0), bbox=bbox, page=page, method="regex_date")
                        break

            if not phone_field.is_present():
                for pat in _PHONE_PATTERNS:
                    m = pat.search(text)
                    if m:
                        phone_field = FieldValue(value=m.group(1), confidence=0.92, source=FieldSource.REGEX,
                                                 evidence=m.group(0), bbox=bbox, page=page, method="regex_phone")
                        break

            if not dealer_field.is_present():
                for pat in _DEALER_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 3:
                            dealer_field = FieldValue(value=val, confidence=0.75, source=FieldSource.REGEX,
                                                      evidence=m.group(0), bbox=bbox, page=page, method="regex_dealer")
                            break

            if not model_field.is_present():
                for pat in _MODEL_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 2:
                            model_field = FieldValue(value=val, confidence=0.75, source=FieldSource.REGEX,
                                                     evidence=m.group(0), bbox=bbox, page=page, method="regex_model")
                            break

            if not cust_name_field.is_present():
                for pat in _CUSTOMER_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 3:
                            cust_name_field = FieldValue(value=val, confidence=0.75, source=FieldSource.REGEX,
                                                         evidence=m.group(0), bbox=bbox, page=page, method="regex_cust")
                            break

        return {
            "dealer_name": dealer_field,
            "model_name": model_field,
            "horse_power": hp_field,
            "asset_cost": cost_field,
            "invoice_number": inv_no_field,
            "invoice_date": date_field,
            "customer_name": cust_name_field,
            "phone_number": phone_field,
        }
