"""
Expanded and Calibrated Data Models for Document AI.

Every extracted field carries value + confidence + source + evidence + bbox + page index + method
for complete banking and audit compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FieldSource(str, Enum):
    LAYOUT_KV = "layout_kv"
    REGEX = "regex"
    FUZZY_MATCH = "fuzzy_match"
    EXACT_MATCH = "exact_match"
    RULE_DEFAULT = "rule_default"
    MANUAL = "manual"


class ReviewDecision(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW = "REVIEW"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class FieldValue(BaseModel):
    """A single extracted field with full provenance, spatial layout bbox, and audit trail."""

    value: Optional[Any] = None
    confidence: float = 0.0
    source: Optional[FieldSource] = None
    source_text: Optional[str] = None
    evidence: Optional[str] = None
    page: int = 1
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    method: str = "regex"
    validation_status: str = "valid"  # "valid", "warning", "invalid", "missing"
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)

    def is_present(self) -> bool:
        return self.value is not None and str(self.value).strip() != ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class VisualMark(BaseModel):
    """Visual mark (signature / stamp) status and bounding box."""

    mark_type: str = "visual_mark"  # "signature", "stamp"
    status: str = "not_implemented"  # "detected", "not_detected", "not_implemented"
    present: bool = False
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    confidence: float = 0.0
    page: int = 1
    details: Optional[Dict[str, Any]] = None


class FinalDocument(BaseModel):
    """
    The comprehensive structured output for an extracted document.
    """

    document_id: str = "doc_001"
    document: str = ""

    # Primary Equipment & Financial Fields
    dealer_name: FieldValue = Field(default_factory=FieldValue)
    model_name: FieldValue = Field(default_factory=FieldValue)
    horse_power: FieldValue = Field(default_factory=FieldValue)
    asset_cost: FieldValue = Field(default_factory=FieldValue)

    # Document Metadata & Customer Fields
    invoice_number: FieldValue = Field(default_factory=FieldValue)
    invoice_date: FieldValue = Field(default_factory=FieldValue)
    customer_name: FieldValue = Field(default_factory=FieldValue)
    customer_address: FieldValue = Field(default_factory=FieldValue)
    phone_number: FieldValue = Field(default_factory=FieldValue)
    registration_number: FieldValue = Field(default_factory=FieldValue)
    serial_number: FieldValue = Field(default_factory=FieldValue)

    # Visual Marks
    dealer_signature: VisualMark = Field(default_factory=VisualMark)
    dealer_stamp: VisualMark = Field(default_factory=VisualMark)

    # Confidence & Human-in-the-loop Decision
    overall_confidence: float = 0.0
    calibrated_confidence: Optional[float] = None
    decision: ReviewDecision = ReviewDecision.AUTO_APPROVE
    needs_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)

    # Telemetry
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_all_fields(self) -> Dict[str, FieldValue]:
        """Return dictionary mapping field name to FieldValue object."""
        return {
            "dealer_name": self.dealer_name,
            "model_name": self.model_name,
            "horse_power": self.horse_power,
            "asset_cost": self.asset_cost,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "customer_name": self.customer_name,
            "customer_address": self.customer_address,
            "phone_number": self.phone_number,
            "registration_number": self.registration_number,
            "serial_number": self.serial_number,
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Export clean legacy-compatible JSON structure."""
        fields_dict: Dict[str, Any] = {}
        for name, f_obj in self.get_all_fields().items():
            if f_obj.is_present():
                fields_dict[name] = {
                    "value": f_obj.value,
                    "confidence": round(f_obj.confidence, 4),
                    "bbox": f_obj.bbox,
                    "source": f_obj.source.value if f_obj.source else f_obj.method,
                    "page": f_obj.page,
                }
            else:
                fields_dict[name] = {
                    "value": None,
                    "confidence": 0.0,
                    "bbox": None,
                }

        fields_dict["dealer_signature"] = {
            "status": self.dealer_signature.status,
            "present": self.dealer_signature.present,
            "confidence": round(self.dealer_signature.confidence, 4),
            "bbox": self.dealer_signature.bbox,
        }
        fields_dict["dealer_stamp"] = {
            "status": self.dealer_stamp.status,
            "present": self.dealer_stamp.present,
            "confidence": round(self.dealer_stamp.confidence, 4),
            "bbox": self.dealer_stamp.bbox,
        }

        return {
            "document_id": self.document_id,
            "document": self.document,
            "fields": fields_dict,
            "validation": {
                "overall_confidence": round(self.overall_confidence, 4),
                "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
                "decision": self.decision.value,
                "needs_human_review": self.needs_human_review,
                "review_reasons": self.review_reasons,
            },
            "processing_time_ms": round(self.processing_time_ms, 2),
            "metadata": self.metadata,
        }
