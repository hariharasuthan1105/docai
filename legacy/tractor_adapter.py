"""
Legacy Tractor Document Adapter.

Wraps a generic DocumentResult to provide the legacy FinalDocument API for
backward-compatible tooling (e.g. old batch scripts, older test fixtures).

The main production pipeline does NOT use this adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from docai.models.extraction_schema import (
    DocumentResult,
    FieldValue,
    ReviewDecision,
    VisualMark,
)


TRACTOR_LEGACY_FIELDS = [
    "dealer_name", "model_name", "horse_power", "asset_cost",
    "invoice_number", "invoice_date", "customer_name", "customer_address",
    "phone_number", "registration_number", "serial_number",
]


class LegacyTractorDocumentAdapter:
    """
    Thin compatibility shim around DocumentResult exposing the old
    FinalDocument attribute API for legacy tractor-invoice tooling.

    Usage:
        result: DocumentResult = pipeline.process(path)
        legacy = LegacyTractorDocumentAdapter(result)
        print(legacy.dealer_name.value)
        print(legacy.horse_power.value)
    """

    def __init__(self, doc: DocumentResult):
        self._doc = doc

    # --- Field attribute proxies ---
    def _fv(self, name: str) -> FieldValue:
        return self._doc.fields.get(name, FieldValue())

    @property
    def dealer_name(self) -> FieldValue:
        return self._fv("dealer_name")

    @property
    def model_name(self) -> FieldValue:
        return self._fv("model_name")

    @property
    def horse_power(self) -> FieldValue:
        return self._fv("horse_power")

    @property
    def asset_cost(self) -> FieldValue:
        return self._fv("asset_cost")

    @property
    def invoice_number(self) -> FieldValue:
        return self._fv("invoice_number")

    @property
    def invoice_date(self) -> FieldValue:
        return self._fv("invoice_date")

    @property
    def customer_name(self) -> FieldValue:
        return self._fv("customer_name")

    @property
    def customer_address(self) -> FieldValue:
        return self._fv("customer_address")

    @property
    def phone_number(self) -> FieldValue:
        return self._fv("phone_number")

    @property
    def registration_number(self) -> FieldValue:
        return self._fv("registration_number")

    @property
    def serial_number(self) -> FieldValue:
        return self._fv("serial_number")

    @property
    def dealer_signature(self) -> VisualMark:
        return self._doc.visual_marks.get("signature", VisualMark(mark_type="signature"))

    @property
    def dealer_stamp(self) -> VisualMark:
        return self._doc.visual_marks.get("stamp", VisualMark(mark_type="stamp"))

    # --- Passthrough attributes ---
    @property
    def document_type(self) -> str:
        return self._doc.document_type

    @property
    def document_type_confidence(self) -> float:
        return self._doc.document_type_confidence

    @property
    def overall_confidence(self) -> float:
        return self._doc.overall_confidence

    @property
    def calibrated_confidence(self) -> Optional[float]:
        return self._doc.calibrated_confidence

    @property
    def decision(self) -> ReviewDecision:
        return self._doc.decision

    @property
    def needs_human_review(self) -> bool:
        return self._doc.review_required

    @property
    def review_reasons(self) -> List[str]:
        return self._doc.review_reasons

    @property
    def tables(self):
        return self._doc.tables

    @property
    def sections(self):
        return self._doc.sections

    @property
    def processing_time_ms(self) -> float:
        return self._doc.processing_time_ms

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._doc.metadata

    def get_all_fields(self) -> Dict[str, FieldValue]:
        return dict(self._doc.fields)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Export legacy-compatible JSON matching old FinalDocument.to_legacy_dict() output."""
        return self._doc.to_dict()
