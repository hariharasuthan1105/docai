"""
Expanded and Calibrated Data Models for Multi-Document AI.

Supports schema-driven generic document results (restaurant receipts, tractor invoices,
generic documents), multi-column tables, visual marks, and backward-compatible adapters.
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


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]


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


class TableRowResult(BaseModel):
    row_index: int
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    bbox: Optional[List[float]] = None


class TableResult(BaseModel):
    name: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0
    bbox: Optional[List[float]] = None


class ValidationSummary(BaseModel):
    is_valid: bool = True
    passed_rules: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    """
    Standardized, schema-driven multi-document result wrapper.
    """

    document_id: str = "doc_001"
    document: str = ""
    document_type: str = "generic"
    document_type_confidence: float = 1.0

    # Fields can be structured hierarchically by section or as a flat map
    sections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    fields: Dict[str, FieldValue] = Field(default_factory=dict)
    tables: List[TableResult] = Field(default_factory=list)

    # Visual Marks (if applicable)
    visual_marks: Dict[str, VisualMark] = Field(default_factory=dict)

    # Validation & Confidence
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    overall_confidence: float = 0.0
    calibrated_confidence: Optional[float] = None
    decision: ReviewDecision = ReviewDecision.AUTO_APPROVE
    review_required: bool = False
    review_reasons: List[str] = Field(default_factory=list)

    # Telemetry
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export clean standardized JSON structure."""
        out_fields: Dict[str, Any] = {}
        for fname, fval in self.fields.items():
            out_fields[fname] = {
                "value": fval.value,
                "confidence": round(fval.confidence, 4),
                "bbox": fval.bbox,
                "source": fval.source.value if fval.source else fval.method,
                "page": fval.page,
            }

        tables_data = []
        for t in self.tables:
            tables_data.append({
                "name": t.name,
                "rows": t.rows,
                "confidence": round(t.confidence, 4),
                "bbox": t.bbox,
            })

        marks_data = {}
        for mname, mark in self.visual_marks.items():
            marks_data[mname] = {
                "present": mark.present,
                "status": mark.status,
                "confidence": round(mark.confidence, 4),
                "bbox": mark.bbox,
            }

        return {
            "document_id": self.document_id,
            "document": self.document,
            "document_type": self.document_type,
            "document_type_confidence": round(self.document_type_confidence, 4),
            "sections": self.sections,
            "fields": out_fields,
            "tables": tables_data,
            "visual_marks": marks_data,
            "validation": {
                "is_valid": self.validation.is_valid,
                "passed_rules": self.validation.passed_rules,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "overall_confidence": round(self.overall_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4) if self.calibrated_confidence is not None else None,
            "decision": self.decision.value,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "metadata": self.metadata,
        }


class FinalDocument(BaseModel):
    """
    Backward-compatible structured output adapter for legacy tractor invoice pipelines.
    """

    document_id: str = "doc_001"
    document: str = ""
    document_type: str = "tractor_invoice"
    document_type_confidence: float = 1.0

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

    # Extended Tables and Generic Fields
    tables: List[TableResult] = Field(default_factory=list)
    generic_fields: Dict[str, FieldValue] = Field(default_factory=dict)
    sections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

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
        res = {
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
        res.update(self.generic_fields)
        return res

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

        tables_data = [{"name": t.name, "rows": t.rows} for t in self.tables]

        return {
            "document_id": self.document_id,
            "document": self.document,
            "document_type": self.document_type,
            "document_type_confidence": round(self.document_type_confidence, 4),
            "sections": self.sections,
            "fields": fields_dict,
            "tables": tables_data,
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
