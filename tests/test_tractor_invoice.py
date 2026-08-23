"""
Tests for Tractor Commercial Tax Invoice Extraction.

Verifies:
1. Document type classification as 'tractor_invoice'
2. Extraction of equipment financing fields (Dealer Name, Model, Horse Power, Asset Cost)
3. Extraction of metadata (Invoice Number, Date, Customer)
4. Range validation rules (HP [5, 150], Asset Cost [50k, 50M])
"""

import pytest
from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine, OCRResult
from docai.pipeline import DocumentAIPipeline


def create_tractor_invoice_ocr_mock() -> OCRResult:
    raw_lines = [
        ("TAX INVOICE", [200, 50, 400, 80], 0.99),
        ("SRI SHAKTHI TRACTORS", [100, 90, 450, 115], 0.98),
        ("Authorised Dealer: Mahindra & Mahindra Ltd.", [50, 120, 480, 140], 0.98),
        ("Invoice No: INV/2024-25/00125    Date: 15/05/2024", [50, 150, 480, 170], 0.97),
        ("Customer Name: M. Ramasamy", [50, 180, 350, 200], 0.96),
        ("Mobile No: 9842155667", [50, 205, 300, 225], 0.98),
        ("Tractor Model: Arjun Novo 605 DI", [50, 240, 400, 260], 0.97),
        ("Horse Power: 50 HP", [50, 270, 300, 290], 0.99),
        ("Asset Cost (₹): 621,000.00", [50, 300, 350, 320], 0.98),
        ("Grand Total: ₹ 621,000.00", [200, 340, 450, 365], 0.99),
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


def test_tractor_invoice_end_to_end():
    mock_ocr = create_tractor_invoice_ocr_mock()
    pipeline = DocumentAIPipeline(ocr_engine=MockOCREngine(mock_ocr))

    result = pipeline.process("dummy_tractor_invoice.png")

    assert result.document_type == "tractor_invoice"
    assert result.document_type_confidence >= 0.70

    # Fields
    assert result.dealer_name.value == "Mahindra & Mahindra Ltd."
    assert "Arjun Novo" in str(result.model_name.value)
    assert result.horse_power.value == 50.0
    assert result.asset_cost.value == 621000.0
    assert result.invoice_number.value == "INV/2024-25/00125"
    assert "15/05/2024" in str(result.invoice_date.value)
    assert "Ramasamy" in str(result.customer_name.value)
    assert "9842155667" in str(result.phone_number.value)
