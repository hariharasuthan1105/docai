"""
Overall confidence scoring and review decisioning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from docai.config import AUTO_APPROVE_THRESHOLD, FIELD_WEIGHTS
from docai.models.extraction_schema import FieldValue


def compute_overall_confidence(fields: Dict[str, FieldValue]) -> float:
    """
    Computes weighted average confidence across all required fields.
    Missing fields contribute 0.0 confidence.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for field_name, weight in FIELD_WEIGHTS.items():
        field_value = fields.get(field_name, FieldValue())
        confidence = field_value.confidence if field_value.is_present() else 0.0
        weighted_sum += weight * confidence
        total_weight += weight
    return round(weighted_sum / total_weight, 4) if total_weight else 0.0


def needs_human_review(
    overall_confidence: float,
    rule_reasons: List[str],
) -> Tuple[bool, List[str]]:
    """
    Determines if human review is needed due to rule violations or low confidence.
    """
    reasons = list(rule_reasons)
    if overall_confidence < AUTO_APPROVE_THRESHOLD:
        reasons.append(
            f"Overall confidence {overall_confidence:.2f} is below auto-approve threshold {AUTO_APPROVE_THRESHOLD}."
        )
    return (len(reasons) > 0, reasons)
