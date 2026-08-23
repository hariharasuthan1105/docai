"""
Business rules layer.

Domain validation: plausible ranges, required-field checks, and data integrity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from docai.config import ASSET_COST_RANGE, HORSE_POWER_RANGE, REQUIRED_FIELDS
from docai.models.extraction_schema import FieldValue


def check_numeric_range(
    field_name: str, field_value: FieldValue, low: float, high: float
) -> List[str]:
    reasons = []
    if not field_value.is_present():
        return reasons
    try:
        numeric = float(field_value.value)
    except (TypeError, ValueError):
        reasons.append(f"{field_name}: value '{field_value.value}' is not numeric.")
        return reasons
    if not (low <= numeric <= high):
        reasons.append(
            f"{field_name}: value {numeric} is outside the plausible range "
            f"[{low}, {high}] -- please verify."
        )
    return reasons


def check_required_fields(fields: Dict[str, FieldValue]) -> List[str]:
    reasons = []
    for field_name in REQUIRED_FIELDS:
        field_value = fields.get(field_name, FieldValue())
        if not field_value.is_present():
            reasons.append(f"{field_name}: required field is missing.")
    return reasons


def apply_business_rules(fields: Dict[str, FieldValue]) -> Tuple[List[str], bool]:
    """
    Returns (reasons, blocking). `blocking` is True if any business rule was violated.
    """
    reasons: List[str] = []
    reasons += check_required_fields(fields)
    reasons += check_numeric_range(
        "horse_power", fields.get("horse_power", FieldValue()), *HORSE_POWER_RANGE
    )
    reasons += check_numeric_range(
        "asset_cost", fields.get("asset_cost", FieldValue()), *ASSET_COST_RANGE
    )

    blocking = len(reasons) > 0
    return reasons, blocking
