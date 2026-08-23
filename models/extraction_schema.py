"""
Generic Data Models for Multi-Document AI.

DocumentResult is the single primary output model.
FinalDocument is an alias kept for import backward compatibility.

Domain-specific field classes (dealer_name, horse_power, etc.) have been
moved to docai/legacy/tractor_adapter.py.
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
    Generic schema-driven document extraction result.

    Works for any document type — restaurant receipts, tractor invoices,
    insurance claims, purchase orders, etc.  Domain fields are stored in
    the generic `fields` dict keyed by field name.  The schema YAML determines
    which fields exist; this class never enumerates them.
    """

    document_id: str = "doc_001"
    document: str = ""
    document_type: str = "generic"
    document_type_confidence: float = 1.0

    # All extracted fields as a generic dict: field_name -> FieldValue
    fields: Dict[str, FieldValue] = Field(default_factory=dict)

    # Detected tables
    tables: List[TableResult] = Field(default_factory=list)

    # Fields grouped by schema section for display
    sections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Visual marks detected by CV (e.g. signature, stamp) — keyed by mark name from schema
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

    def get_all_fields(self) -> Dict[str, FieldValue]:
        """Return all extracted fields dict (generic API)."""
        return dict(self.fields)

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

    # Backward-compat alias for code that calls .to_legacy_dict()
    def to_legacy_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    # Backward-compat: needs_human_review property
    @property
    def needs_human_review(self) -> bool:
        return self.review_required


# -----------------------------------------------------------------------
# FinalDocument is now an alias for DocumentResult.
# Kept for import backward compatibility only — no domain fields.
# -----------------------------------------------------------------------
FinalDocument = DocumentResult
