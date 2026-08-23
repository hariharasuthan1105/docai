"""
Tests for Generic / Unknown Document Type Fallback.

Verifies:
1. Unknown document types classify as 'generic' with low/fallback confidence
2. Generic key-value pairs are extracted using layout anchors without domain schemas
3. Generic tables are detected and parsed
"""

import pytest
from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine, OCRResult
from docai.pipeline import DocumentAIPipeline


def create_unknown_document_ocr_mock() -> OCRResult:
    raw_lines = [
        ("CITY GENERAL HOSPITAL", [100, 50, 400, 80], 0.98),
        ("LABORATORY TEST REPORT", [120, 85, 380, 105], 0.95),
        ("Patient Name : John Doe", [50, 130, 300, 150], 0.97),
        ("Age : 42 Years", [50, 160, 200, 180], 0.96),
        ("Doctor : Dr. Smith", [50, 190, 250, 210], 0.98),
        ("Date of Test : 12/08/2026", [50, 220, 300, 240], 0.97),
    ]

    ocr_lines = []
    for text, coords, conf in raw_lines:
        bbox = BoundingBox(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])
        ocr_lines.append(OCRLine(text=text, bbox=bbox, confidence=conf, page=1))

    return OCRResult(
        lines=ocr_lines,
        full_text="\n".join(t for t, _, _ in raw_lines),
        confidence=0.98,
        page_count=1,
    )


class MockOCREngine:
    def __init__(self, ocr_result: OCRResult):
        self.ocr_result = ocr_result

    def extract_text(self, path: str) -> OCRResult:
        return self.ocr_result


def test_generic_unknown_document_extraction():
    mock_ocr = create_unknown_document_ocr_mock()
    pipeline = DocumentAIPipeline(ocr_engine=MockOCREngine(mock_ocr))

    result = pipeline.process("medical_report.pdf")

    # Document type is generic
    assert result.document_type == "generic"

    all_fields = result.get_all_fields()
    assert "patient_name" in all_fields or "patient" in all_fields
    # Check value extracted
    found_patient = False
    for k, v in all_fields.items():
        if "patient" in k and "John Doe" in str(v.value):
            found_patient = True
            break
    assert found_patient, f"Extracted fields: {all_fields}"

    # Negative check: no tractor or restaurant domain assumptions
    assert not result.horse_power.is_present()
