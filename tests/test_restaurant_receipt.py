"""
Tests for Restaurant Receipt / Food Tax Invoice Extraction.

Verifies:
1. Document type classification as 'restaurant_receipt'
2. Merchant details (Name, Address, Phone, Tagline)
3. Metadata (Invoice No, Date, Time, Order No, Table No, Cashier)
4. Multi-column table line items (9 items)
5. Totals (Subtotal, Discount, CGST, SGST, Grand Total)
6. Payment Info (UPI, Ref No, Paid)
7. CRITICAL NEGATIVE TEST: No tractor-specific fields (model_name, horse_power, asset_cost)
"""

import os
import pytest
from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine, OCRResult
from docai.pipeline import DocumentAIPipeline
from docai.schemas.schema_loader import get_schema_registry


def create_restaurant_receipt_ocr_mock() -> OCRResult:
    """Mock OCR result matching Spice Garden Restaurant receipt."""
    raw_lines = [
        ("SPICE GARDEN RESTAURANT", [100, 50, 400, 80], 0.98),
        ("GOOD FOOD • GOOD MOOD", [120, 85, 380, 105], 0.95),
        ("123, Green Park Extension", [130, 110, 370, 130], 0.96),
        ("New Delhi - 110016", [160, 135, 340, 155], 0.96),
        ("Phone: 011-41234567", [150, 160, 350, 180], 0.97),
        ("TAX INVOICE (RECEIPT)", [140, 200, 360, 225], 0.99),
        ("Invoice No : SG/24-25/0789    Date : 16/05/2025", [50, 240, 450, 260], 0.98),
        ("Order No : 45    Time : 08:35 PM", [50, 265, 450, 285], 0.97),
        ("Table No : T-07    Cashier : Rahul", [50, 290, 450, 310], 0.97),
        ("S.No. Item Name Qty Unit Price Amount (₹)", [50, 330, 450, 350], 0.98),
        ("1 Vegetable Spring Roll 1 120.00 120.00", [50, 360, 450, 380], 0.98),
        ("2 Tomato Soup 1 110.00 110.00", [50, 385, 450, 405], 0.98),
        ("3 Paneer Butter Masala 1 240.00 240.00", [50, 410, 450, 430], 0.98),
        ("4 Dal Makhani 1 190.00 190.00", [50, 435, 450, 455], 0.98),
        ("5 Jeera Rice 1 140.00 140.00", [50, 460, 450, 480], 0.98),
        ("6 Garlic Naan 2 60.00 120.00", [50, 485, 450, 505], 0.98),
        ("7 Butter Roti 2 30.00 60.00", [50, 510, 450, 530], 0.98),
        ("8 Masala Cola 2 50.00 100.00", [50, 535, 450, 555], 0.98),
        ("9 Chocolate Brownie 1 130.00 130.00", [50, 560, 450, 580], 0.98),
        ("Subtotal 1,210.00", [200, 600, 450, 620], 0.98),
        ("Discount (10%) -121.00", [200, 625, 450, 645], 0.97),
        ("Taxable Amount 1,089.00", [200, 660, 450, 680], 0.98),
        ("CGST (2.5%) 27.23", [200, 685, 450, 705], 0.97),
        ("SGST (2.5%) 27.23", [200, 710, 450, 730], 0.97),
        ("Grand Total ₹ 1,143.00", [200, 745, 450, 770], 0.99),
        ("Amount in Words: One Thousand One Hundred Forty Three Only", [50, 785, 450, 805], 0.97),
        ("Payment Mode : UPI", [50, 825, 300, 845], 0.98),
        ("UPI Ref No : 512345678901", [50, 850, 350, 870], 0.98),
        ("Payment Status : Paid", [50, 875, 300, 895], 0.98),
        ("THANK YOU! VISIT AGAIN", [120, 920, 380, 945], 0.99),
    ]

    ocr_lines = []
    full_text_parts = []
    for text, coords, conf in raw_lines:
        bbox = BoundingBox(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])
        ocr_lines.append(OCRLine(text=text, bbox=bbox, confidence=conf, page=1))
        full_text_parts.append(text)

    return OCRResult(
        lines=ocr_lines,
        full_text="\n".join(full_text_parts),
        confidence=0.98,
        page_count=1,
    )


class MockOCREngine:
    def __init__(self, ocr_result: OCRResult):
        self.ocr_result = ocr_result

    def extract_text(self, path: str) -> OCRResult:
        return self.ocr_result


def test_restaurant_receipt_end_to_end():
    mock_ocr = create_restaurant_receipt_ocr_mock()
    pipeline = DocumentAIPipeline(ocr_engine=MockOCREngine(mock_ocr))

    result = pipeline.process("dummy_restaurant_invoice.png")

    # 1. Document Type Classification
    assert result.document_type == "restaurant_receipt"
    assert result.document_type_confidence >= 0.85

    all_fields = result.get_all_fields()

    # 2. Merchant Details
    assert all_fields["merchant_name"].value == "SPICE GARDEN RESTAURANT"
    assert "Green Park Extension" in str(all_fields["merchant_address"].value)
    assert "011-41234567" in str(all_fields["merchant_phone"].value)
    assert "GOOD FOOD" in str(all_fields["merchant_tagline"].value)

    # 3. Metadata & Multi-Field Line Splitting
    assert all_fields["invoice_number"].value == "SG/24-25/0789"
    assert "16/05/2025" in str(all_fields["invoice_date"].value)
    assert all_fields["order_number"].value == "45"
    assert "08:35" in str(all_fields["invoice_time"].value)
    assert all_fields["table_number"].value == "T-07"
    assert all_fields["cashier"].value == "Rahul"

    # 4. Multi-Column Table (9 Line Items)
    assert len(result.tables) >= 1
    line_items_table = next(t for t in result.tables if t.name == "line_items")
    assert len(line_items_table.rows) == 9

    row_1 = line_items_table.rows[0]
    assert row_1["item"] == "Vegetable Spring Roll"
    assert row_1["qty"] == 1
    assert row_1["unit_price"] == 120.00
    assert row_1["amount"] == 120.00

    row_6 = line_items_table.rows[5]
    assert row_6["item"] == "Garlic Naan"
    assert row_6["qty"] == 2
    assert row_6["unit_price"] == 60.00
    assert row_6["amount"] == 120.00

    # 5. Financial Totals & Taxes
    assert all_fields["subtotal"].value == 1210.00
    assert all_fields["discount_amount"].value == 121.00
    assert all_fields["taxable_amount"].value == 1089.00
    assert all_fields["cgst_amount"].value == 27.23
    assert all_fields["sgst_amount"].value == 27.23
    assert all_fields["grand_total"].value == 1143.00

    # 6. Payment Information
    assert all_fields["payment_mode"].value == "UPI"
    assert all_fields["upi_ref_no"].value == "512345678901"
    assert all_fields["payment_status"].value == "Paid"

    # 7. CRITICAL NEGATIVE TEST:
    # Ensure NO tractor-specific field values contaminated the restaurant receipt extraction.
    # In the generic DocumentResult, these fields simply won't be in the fields dict.
    assert "model_name" not in all_fields or not all_fields["model_name"].is_present(), \
        "model_name should not be present in restaurant receipt"
    assert "horse_power" not in all_fields or not all_fields["horse_power"].is_present(), \
        "horse_power should not be present in restaurant receipt"
    assert "asset_cost" not in all_fields or not all_fields["asset_cost"].is_present(), \
        "asset_cost should not be present in restaurant receipt"
