"""
Deterministic regex/rule-based extraction with OCR line bounding-box association.

Extracts Horse Power, Asset Cost, Dealer Name, and Model Name candidates,
normalizing numbers and associating OCR bounding boxes with every extracted field.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult

# --- Horse Power Patterns ----------------------------------------------------
# Supports: "Horse Power: 50 HP", "Power: 50", "Engine Power: 50 H.P.",
# "50HP", "50 H.P", "50 HP", "HP - 50", "HP: 50", "Horse Power - 50"
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

# --- Dealer & Model Line Patterns --------------------------------------------
_DEALER_PATTERNS = [
    re.compile(r"(?:dealer\s*name|dealer|m/s\.?|authorized\s*dealer|authorised\s*dealer)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]

_MODEL_PATTERNS = [
    re.compile(r"(?:model\s*name|model|item\s*name|item|tractor\s*model)\s*[:\-]?\s*(.+)", re.IGNORECASE),
]


def normalize_numeric_string(raw_str: str) -> Optional[float]:
    """
    Parse a numeric string supporting commas and Indian number formats (e.g. 5,50,000.00).
    """
    if not raw_str:
        return None
    cleaned = raw_str.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_horse_power_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[FieldValue]:
    for pattern in _HP_PATTERNS:
        match = pattern.search(line_text)
        if match:
            raw_val = match.group(1)
            num = normalize_numeric_string(raw_val)
            if num is not None and 5.0 <= num <= 250.0:
                bbox_list = list(bbox) if bbox else None
                return FieldValue(
                    value=num,
                    confidence=0.95,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=bbox_list,
                )
    return None


def extract_asset_cost_from_line(
    line_text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    primary_only: bool = False,
) -> Optional[FieldValue]:
    patterns = _PRIMARY_COST_PATTERNS if primary_only else (_PRIMARY_COST_PATTERNS + _SECONDARY_COST_PATTERNS)
    for pattern in patterns:
        match = pattern.search(line_text)
        if match:
            raw_val = match.group(1)
            num = normalize_numeric_string(raw_val)
            if num is not None and num >= 1000.0:
                bbox_list = list(bbox) if bbox else None
                return FieldValue(
                    value=num,
                    confidence=0.92,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=bbox_list,
                )
    return None


def extract_dealer_name_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[FieldValue]:
    for pattern in _DEALER_PATTERNS:
        match = pattern.search(line_text)
        if match:
            val = match.group(1).strip()
            if len(val) >= 3:
                bbox_list = list(bbox) if bbox else None
                return FieldValue(
                    value=val,
                    confidence=0.75,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=bbox_list,
                )
    return None


def extract_model_name_from_line(line_text: str, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[FieldValue]:
    for pattern in _MODEL_PATTERNS:
        match = pattern.search(line_text)
        if match:
            val = match.group(1).strip()
            if len(val) >= 2:
                bbox_list = list(bbox) if bbox else None
                return FieldValue(
                    value=val,
                    confidence=0.75,
                    source=FieldSource.REGEX,
                    evidence=match.group(0).strip(),
                    bbox=bbox_list,
                )
    return None


class RegexExtractionEngine:
    """Extracts fields and associates OCR bounding boxes."""

    def extract(self, ocr_text: str) -> Dict[str, FieldValue]:
        """Extract from plain text without line bboxes."""
        lines = [OCRLine(text=line, bbox=(0.0, 0.0, 0.0, 0.0), confidence=1.0) for line in ocr_text.splitlines() if line.strip()]
        return self.extract_from_ocr_result(OCRResult(lines=lines))

    def extract_from_ocr_result(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """Extract fields matching line by line, capturing exact OCR bounding boxes."""
        hp_field = FieldValue()
        cost_field = FieldValue()
        dealer_field = FieldValue()
        model_field = FieldValue()

        lines = ocr_result.lines
        n_lines = len(lines)

        # Pass 1: Extract HP, Dealer candidate, Model candidate, and Primary Total Cost (1-line and 2-line windows)
        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue

            # Horse power
            if not hp_field.is_present():
                extracted_hp = extract_horse_power_from_line(text, line.bbox)
                if extracted_hp:
                    hp_field = extracted_hp

            # Primary Asset cost (explicit 'Asset Cost:', 'Total Cost:', 'Total (₹)' etc.)
            if not cost_field.is_present():
                extracted_cost = extract_asset_cost_from_line(text, line.bbox, primary_only=True)
                if extracted_cost:
                    cost_field = extracted_cost

            # Dealer candidate
            if not dealer_field.is_present():
                extracted_dealer = extract_dealer_name_from_line(text, line.bbox)
                if extracted_dealer:
                    dealer_field = extracted_dealer

            # Model candidate
            if not model_field.is_present():
                extracted_model = extract_model_name_from_line(text, line.bbox)
                if extracted_model:
                    model_field = extracted_model

            # 2-line adjacent window for primary cost (e.g. "Total (₹)" line followed by "732,780.00")
            if not cost_field.is_present() and i + 1 < n_lines:
                next_line = lines[i + 1]
                combined_text = f"{text} {next_line.text.strip()}"
                combined_bbox = next_line.bbox or line.bbox
                extracted_cost = extract_asset_cost_from_line(combined_text, combined_bbox, primary_only=True)
                if extracted_cost:
                    cost_field = extracted_cost

        # Pass 2: If primary cost was not found, check secondary cost patterns (single line & 2-line pairs)
        if not cost_field.is_present():
            for i, line in enumerate(lines):
                text = line.text.strip()
                if not text:
                    continue
                extracted_cost = extract_asset_cost_from_line(text, line.bbox, primary_only=False)
                if extracted_cost:
                    cost_field = extracted_cost
                    break
                if i + 1 < n_lines:
                    next_line = lines[i + 1]
                    combined_text = f"{text} {next_line.text.strip()}"
                    combined_bbox = next_line.bbox or line.bbox
                    extracted_cost = extract_asset_cost_from_line(combined_text, combined_bbox, primary_only=False)
                    if extracted_cost:
                        cost_field = extracted_cost
                        break

        # Pass 3: If still not found, check for numeric cost lines in reasonable tractor price range
        if not cost_field.is_present():
            candidate_costs: List[Tuple[float, OCRLine]] = []
            for line in lines:
                text = line.text.strip()
                num = normalize_numeric_string(text)
                if num and 50_000.0 <= num <= 50_000_000.0:
                    candidate_costs.append((num, line))
            if candidate_costs:
                # Pick max amount (typically grand invoice total)
                max_num, best_line = max(candidate_costs, key=lambda x: x[0])
                cost_field = FieldValue(
                    value=max_num,
                    confidence=0.85,
                    source=FieldSource.REGEX,
                    evidence=best_line.text.strip(),
                    bbox=list(best_line.bbox) if best_line.bbox else None,
                )

        return {
            "dealer_name": dealer_field,
            "model_name": model_field,
            "horse_power": hp_field,
            "asset_cost": cost_field,
        }
