"""
Deterministic regex/rule-based extraction with OCR line bounding-box association.

Extracts Horse Power, Asset Cost, Invoice Number, Date, Customer Details,
Phone Number, Registration/Serial Number, Dealer, and Model candidates.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult

# --- Horse Power Patterns ----------------------------------------------------
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

# --- Asset Cost Patterns -----------------------------------------------------
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
    re.compile(
        r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/-)?",
        re.IGNORECASE,
    ),
]

# --- Invoice Number & Date Patterns ------------------------------------------
_INVOICE_NUM_PATTERNS = [
    re.compile(r"(?:invoice\s*(?:no\.?|num\.?|number|#)|bill\s*(?:no\.?|number)|inv\s*no\.?)\s*[:\-]?\s*([a-zA-Z0-9\-\/]+)", re.IGNORECASE),
    re.compile(r"\b(?:inv|bill)[/\-_](?:20\d{2}[/\-_])?[a-zA-Z0-9\-]+\b", re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r"(?:date|dated|invoice\s*date|bill\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", re.IGNORECASE),
    re.compile(r"(?:date|dated|invoice\s*date)\s*[:\-]?\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
]

# --- Phone & Registration / Serial Patterns ----------------------------------
_PHONE_PATTERNS = [
    re.compile(r"(?:phone|mobile|mob|contact|tel)\s*(?:no\.?|number)?\s*[:\-]?\s*(?:\+91[\-\s]?)?([6-9]\d{9})\b", re.IGNORECASE),
    re.compile(r"\b(?:\+91[\-\s]?)?([6-9]\d{4}[\-\s]?\d{5})\b"),
]

_REG_NO_PATTERNS = [
    re.compile(r"(?:reg(?:istration)?\s*no\.?|chassis\s*no\.?)\s*[:\-]?\s*([A-Z]{2}[0-9\s\-]{1,3}[A-Z]{0,3}[0-9]{3,4})", re.IGNORECASE),
    re.compile(r"\b([A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4})\b"),
]

_SERIAL_NO_PATTERNS = [
    re.compile(r"(?:serial\s*no\.?|sl\.?\s*no\.?|engine\s*no\.?|tractor\s*serial\s*no\.?)\s*[:\-]?\s*([a-zA-Z0-9\-\/]+)", re.IGNORECASE),
]

# --- Dealer & Model Line Patterns --------------------------------------------
_DEALER_PATTERNS = [
    re.compile(r"(?:dealer\s*name|dealer|m/s\.?|authorized\s*dealer|authorised\s*dealer)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]

_MODEL_PATTERNS = [
    re.compile(r"(?:model\s*name|model|item\s*name|item|tractor\s*model)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]

_CUSTOMER_PATTERNS = [
    re.compile(r"(?:customer\s*name|buyer\s*name|buyer|bill\s*to|purchaser|sold\s*to)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]

_ADDRESS_PATTERNS = [
    re.compile(r"(?:customer\s*address|buyer\s*address|address)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]


def normalize_numeric_string(raw_str: str) -> Optional[float]:
    """Parse a numeric string supporting commas and Indian number formats."""
    if not raw_str:
        return None
    cleaned = raw_str.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_horse_power_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
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


def extract_invoice_number_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
    for pat in _INVOICE_NUM_PATTERNS:
        m = pat.search(line_text)
        if m:
            val = m.group(1).strip() if m.groups() else m.group(0).strip()
            if len(val) >= 3:
                return FieldValue(
                    value=val,
                    confidence=0.90,
                    source=FieldSource.REGEX,
                    evidence=m.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method="regex_invoice_num",
                )
    return None


def extract_date_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
    for pat in _DATE_PATTERNS:
        m = pat.search(line_text)
        if m:
            val = m.group(1).strip()
            return FieldValue(
                value=val,
                confidence=0.91,
                source=FieldSource.REGEX,
                evidence=m.group(0).strip(),
                bbox=list(bbox) if bbox else None,
                page=page,
                method="regex_date",
            )
    return None


def extract_phone_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
    for pat in _PHONE_PATTERNS:
        m = pat.search(line_text)
        if m:
            val = re.sub(r"[^\d]", "", m.group(1))
            if len(val) == 10:
                return FieldValue(
                    value=val,
                    confidence=0.92,
                    source=FieldSource.REGEX,
                    evidence=m.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method="regex_phone",
                )
    return None


def extract_serial_no_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
    for pat in _SERIAL_NO_PATTERNS:
        m = pat.search(line_text)
        if m:
            val = m.group(1).strip()
            if len(val) >= 4:
                return FieldValue(
                    value=val,
                    confidence=0.88,
                    source=FieldSource.REGEX,
                    evidence=m.group(0).strip(),
                    bbox=list(bbox) if bbox else None,
                    page=page,
                    method="regex_serial_no",
                )
    return None


def extract_reg_no_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None, page: int = 1) -> Optional[FieldValue]:
    for pat in _REG_NO_PATTERNS:
        m = pat.search(line_text)
        if m:
            val = m.group(1).strip().replace(" ", "")
            return FieldValue(
                value=val,
                confidence=0.89,
                source=FieldSource.REGEX,
                evidence=m.group(0).strip(),
                bbox=list(bbox) if bbox else None,
                page=page,
                method="regex_reg_no",
            )
    return None


class RegexExtractionEngine:
    """Extracts all document fields and associates OCR line bounding boxes."""

    def extract_from_ocr_result(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        hp_field = FieldValue()
        cost_field = FieldValue()
        dealer_field = FieldValue()
        model_field = FieldValue()
        inv_no_field = FieldValue()
        date_field = FieldValue()
        cust_name_field = FieldValue()
        cust_addr_field = FieldValue()
        phone_field = FieldValue()
        reg_no_field = FieldValue()
        serial_no_field = FieldValue()

        lines = ocr_result.lines
        n_lines = len(lines)

        # Pass 1: Extract individual lines and 2-line adjacent windows
        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue

            page = getattr(line, "page", 1)

            if not hp_field.is_present():
                hp = extract_horse_power_from_line(text, line.bbox, page=page)
                if hp:
                    hp_field = hp

            if not cost_field.is_present():
                cost = extract_asset_cost_from_line(text, line.bbox, primary_only=True, page=page)
                if cost:
                    cost_field = cost

            if not inv_no_field.is_present():
                inv = extract_invoice_number_from_line(text, line.bbox, page=page)
                if inv:
                    inv_no_field = inv

            if not date_field.is_present():
                dt = extract_date_from_line(text, line.bbox, page=page)
                if dt:
                    date_field = dt

            if not phone_field.is_present():
                ph = extract_phone_from_line(text, line.bbox, page=page)
                if ph:
                    phone_field = ph

            if not reg_no_field.is_present():
                rg = extract_reg_no_from_line(text, line.bbox, page=page)
                if rg:
                    reg_no_field = rg

            if not serial_no_field.is_present():
                sn = extract_serial_no_from_line(text, line.bbox, page=page)
                if sn:
                    serial_no_field = sn

            # Dealer candidate
            if not dealer_field.is_present():
                for pat in _DEALER_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 3:
                            dealer_field = FieldValue(
                                value=val,
                                confidence=0.75,
                                source=FieldSource.REGEX,
                                evidence=m.group(0).strip(),
                                bbox=list(line.bbox) if line.bbox else None,
                                page=page,
                                method="regex_dealer_candidate",
                            )

            # Model candidate
            if not model_field.is_present():
                for pat in _MODEL_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 2:
                            model_field = FieldValue(
                                value=val,
                                confidence=0.75,
                                source=FieldSource.REGEX,
                                evidence=m.group(0).strip(),
                                bbox=list(line.bbox) if line.bbox else None,
                                page=page,
                                method="regex_model_candidate",
                            )

            # Customer candidate
            if not cust_name_field.is_present():
                for pat in _CUSTOMER_PATTERNS:
                    m = pat.search(text)
                    if m:
                        val = m.group(1).strip()
                        if len(val) >= 3:
                            cust_name_field = FieldValue(
                                value=val,
                                confidence=0.75,
                                source=FieldSource.REGEX,
                                evidence=m.group(0).strip(),
                                bbox=list(line.bbox) if line.bbox else None,
                                page=page,
                                method="regex_customer",
                            )

            # 2-line adjacent window for cost
            if not cost_field.is_present() and i + 1 < n_lines:
                next_line = lines[i + 1]
                combined_text = f"{text} {next_line.text.strip()}"
                combined_bbox = next_line.bbox or line.bbox
                extracted_cost = extract_asset_cost_from_line(combined_text, combined_bbox, primary_only=True, page=page)
                if extracted_cost:
                    cost_field = extracted_cost

        # Pass 2: Secondary cost patterns
        if not cost_field.is_present():
            for i, line in enumerate(lines):
                text = line.text.strip()
                page = getattr(line, "page", 1)
                extracted_cost = extract_asset_cost_from_line(text, line.bbox, primary_only=False, page=page)
                if extracted_cost:
                    cost_field = extracted_cost
                    break
                if i + 1 < n_lines:
                    next_line = lines[i + 1]
                    combined_text = f"{text} {next_line.text.strip()}"
                    combined_bbox = next_line.bbox or line.bbox
                    extracted_cost = extract_asset_cost_from_line(combined_text, combined_bbox, primary_only=False, page=page)
                    if extracted_cost:
                        cost_field = extracted_cost
                        break

        # Pass 3: Plausible numeric invoice total fallback
        if not cost_field.is_present():
            candidate_costs: List[Tuple[float, OCRLine]] = []
            for line in lines:
                text = line.text.strip()
                num = normalize_numeric_string(text)
                if num and 50_000.0 <= num <= 50_000_000.0:
                    candidate_costs.append((num, line))
            if candidate_costs:
                max_num, best_line = max(candidate_costs, key=lambda x: x[0])
                cost_field = FieldValue(
                    value=max_num,
                    confidence=0.85,
                    source=FieldSource.REGEX,
                    evidence=best_line.text.strip(),
                    bbox=list(best_line.bbox) if best_line.bbox else None,
                    page=getattr(best_line, "page", 1),
                    method="regex_numeric_fallback",
                )

        return {
            "dealer_name": dealer_field,
            "model_name": model_field,
            "horse_power": hp_field,
            "asset_cost": cost_field,
            "invoice_number": inv_no_field,
            "invoice_date": date_field,
            "customer_name": cust_name_field,
            "customer_address": cust_addr_field,
            "phone_number": phone_field,
            "registration_number": reg_no_field,
            "serial_number": serial_no_field,
        }

    def extract(self, ocr_text: str) -> Dict[str, FieldValue]:
        lines = [OCRLine(text=line, bbox=(0.0, 0.0, 0.0, 0.0), confidence=1.0, page=1) for line in ocr_text.splitlines() if line.strip()]
        return self.extract_from_ocr_result(OCRResult(lines=lines))


# Alias for backward compatibility
RegexEngine = RegexExtractionEngine
