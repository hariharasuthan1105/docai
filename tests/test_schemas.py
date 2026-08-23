"""
Tests for Schema Loader, Schema Registry and Validation Engine.
"""

import pytest
from docai.layout.table_extractor import ExtractedTable, TableCell, TableRow
from docai.models.extraction_schema import FieldValue
from docai.schemas.schema_loader import get_schema_registry
from docai.validation.schema_validator import SchemaValidator


def test_schema_registry_loads_all():
    registry = get_schema_registry()
    doc_types = registry.list_document_types()

    assert "restaurant_receipt" in doc_types
    assert "tractor_invoice" in doc_types
    assert "generic" in doc_types

    r_schema = registry.get_schema("restaurant_receipt")
    assert r_schema is not None
    assert "merchant_name" in r_schema.fields
    assert "subtotal" in r_schema.fields
    assert "grand_total" in r_schema.fields
    assert "line_items" in r_schema.tables

    t_schema = registry.get_schema("tractor_invoice")
    assert t_schema is not None
    assert "horse_power" in t_schema.fields
    assert "asset_cost" in t_schema.fields


def test_schema_validator_restaurant_math():
    registry = get_schema_registry()
    schema = registry.get_schema("restaurant_receipt")
    validator = SchemaValidator(schema)

    # Valid totals
    fields = {
        "invoice_number": FieldValue(value="INV-001", confidence=0.95),
        "invoice_date": FieldValue(value="16/05/2025", confidence=0.95),
        "subtotal": FieldValue(value=1210.00, confidence=0.95),
        "discount_amount": FieldValue(value=121.00, confidence=0.95),
        "cgst_amount": FieldValue(value=27.23, confidence=0.95),
        "sgst_amount": FieldValue(value=27.23, confidence=0.95),
        "grand_total": FieldValue(value=1143.00, confidence=0.95),
    }

    report = validator.validate(fields)
    assert report.is_valid
    assert len(report.errors) == 0


def test_schema_validator_tractor_ranges():
    registry = get_schema_registry()
    schema = registry.get_schema("tractor_invoice")
    validator = SchemaValidator(schema)

    # Invalid HP
    fields = {
        "dealer_name": FieldValue(value="Mahindra", confidence=0.95),
        "model_name": FieldValue(value="Arjun 605", confidence=0.95),
        "horse_power": FieldValue(value=250.0, confidence=0.95),  # exceeds 150
        "asset_cost": FieldValue(value=600000.0, confidence=0.95),
    }

    report = validator.validate(fields)
    assert not report.is_valid
    assert any("horse_power" in err for err in report.errors)
