"""
Tests for Table Boundary Detection and Multi-Column Table Extraction.
"""

import pytest
from docai.layout.table_detector import TableDetector
from docai.layout.table_extractor import TableExtractor
from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine, OCRResult


def test_table_detector_and_extractor():
    header_line = OCRLine(
        text="S.No. Item Name Qty Unit Price Amount (₹)",
        bbox=BoundingBox(x0=50, y0=100, x1=500, y1=120),
        confidence=0.98,
        page=1,
    )
    body_1 = OCRLine(
        text="1 Vegetable Spring Roll 1 120.00 120.00",
        bbox=BoundingBox(x0=50, y0=130, x1=500, y1=150),
        confidence=0.98,
        page=1,
    )
    body_2 = OCRLine(
        text="2 Tomato Soup 2 110.00 220.00",
        bbox=BoundingBox(x0=50, y0=160, x1=500, y1=180),
        confidence=0.98,
        page=1,
    )
    footer_line = OCRLine(
        text="Subtotal 340.00",
        bbox=BoundingBox(x0=200, y0=200, x1=500, y1=220),
        confidence=0.98,
        page=1,
    )

    ocr_res = OCRResult(
        lines=[header_line, body_1, body_2, footer_line],
        full_text="...",
        confidence=0.98,
        page_count=1,
    )

    detector = TableDetector()
    regions = detector.detect_tables(ocr_res)
    assert len(regions) == 1
    reg = regions[0]
    assert len(reg.body_lines) == 2

    # Check excluded indices
    table_indices = detector.get_table_line_indices(ocr_res, regions)
    assert 0 in table_indices  # header
    assert 1 in table_indices  # row 1
    assert 2 in table_indices  # row 2
    assert 3 not in table_indices  # footer is not in table

    # Table Extractor
    extractor = TableExtractor()
    extracted = extractor.extract_table(reg)
    assert len(extracted.rows) == 2
    assert extracted.rows[0].cells["item"].value == "Vegetable Spring Roll"
    assert extracted.rows[0].cells["qty"].value == 1
    assert extracted.rows[0].cells["amount"].value == 120.0
    assert extracted.rows[1].cells["item"].value == "Tomato Soup"
    assert extracted.rows[1].cells["qty"].value == 2
    assert extracted.rows[1].cells["amount"].value == 220.0
