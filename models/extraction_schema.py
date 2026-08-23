"""
Core data models for the deterministic Document AI pipeline.

Every extracted field carries value + confidence + source + evidence + bbox
for full banking / finance auditability.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class FieldSource(str, Enum):
    REGEX = "regex"
    FUZZY_MATCH = "fuzzy_match"
    EXACT_MATCH = "exact_match"
    RULE_DEFAULT = "rule_default"


class FieldValue(BaseModel):
    """A single extracted field with full provenance and OCR bounding box."""

    value: Optional[Any] = None
    confidence: float = 0.0
    source: Optional[FieldSource] = None
    evidence: Optional[str] = None
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]

    def is_present(self) -> bool:
        return self.value is not None and str(self.value).strip() != ""


class VisualMark(BaseModel):
    """Visual mark (signature / stamp) status."""

    status: str = "not_implemented"
    present: bool = False
    bbox: Optional[List[float]] = None
    confidence: float = 0.0


class FinalDocument(BaseModel):
    """The final structured JSON representation for one document."""

    dealer_name: FieldValue = Field(default_factory=FieldValue)
    model_name: FieldValue = Field(default_factory=FieldValue)
    horse_power: FieldValue = Field(default_factory=FieldValue)
    asset_cost: FieldValue = Field(default_factory=FieldValue)

    dealer_signature: VisualMark = Field(default_factory=VisualMark)
    dealer_stamp: VisualMark = Field(default_factory=VisualMark)

    overall_confidence: float = 0.0
    needs_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
