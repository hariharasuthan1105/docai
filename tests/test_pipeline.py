"""
Integration tests for the deterministic Document AI pipeline.
"""

from docai.models.extraction_schema import FieldSource
from docai.ocr.paddleocr_engine import OCRLine, OCRResult, StubOCREngine
from docai.pipeline import DocumentAIPipeline, run_pipeline


class TestDeterministicPipeline:
    def test_pipeline_with_complete_invoice(self):
        canned_lines = [
            OCRLine(text="TAX INVOICE", bbox=(10, 10, 200, 30), confidence=0.99),
            OCRLine(text="Dealer: Mahindra Tractors Ltd.", bbox=(10, 40, 320, 60), confidence=0.98),
            OCRLine(text="Model: Arjun Novo 605 DI", bbox=(10, 70, 280, 90), confidence=0.97),
            OCRLine(text="Horse Power: 50 HP", bbox=(10, 100, 180, 120), confidence=0.95),
            OCRLine(text="Asset Cost: Rs. 5,50,000.00", bbox=(10, 130, 260, 150), confidence=0.99),
        ]
        ocr = StubOCREngine(canned_lines=canned_lines)
        pipeline = DocumentAIPipeline(ocr_engine=ocr)

        doc = pipeline.process("dummy.png")

        # 1. Dealer
        assert doc.dealer_name.value == "Mahindra Tractors Ltd."
        assert doc.dealer_name.source == FieldSource.FUZZY_MATCH
        assert doc.dealer_name.bbox == [10.0, 40.0, 320.0, 60.0]

        # 2. Model
        assert doc.model_name.value == "Arjun Novo 605 DI"
        assert doc.model_name.source == FieldSource.EXACT_MATCH
        assert doc.model_name.bbox == [10.0, 70.0, 280.0, 90.0]

        # 3. Horse Power
        assert doc.horse_power.value == 50.0
        assert doc.horse_power.source == FieldSource.REGEX
        assert doc.horse_power.bbox == [10.0, 100.0, 180.0, 120.0]

        # 4. Asset Cost
        assert doc.asset_cost.value == 550000.0
        assert doc.asset_cost.source == FieldSource.REGEX
        assert doc.asset_cost.bbox == [10.0, 130.0, 260.0, 150.0]

        # 5. Visual marks (Stubs)
        assert doc.dealer_signature.status == "not_implemented"
        assert doc.dealer_signature.present is False
        assert doc.dealer_stamp.status == "not_implemented"
        assert doc.dealer_stamp.present is False

        # 6. Validation
        assert doc.overall_confidence >= 0.85
        assert doc.needs_human_review is False

    def test_pipeline_with_malformed_text(self):
        canned_lines = [
            OCRLine(text="### Random OCR Glitch ###", bbox=(0, 0, 0, 0), confidence=0.3),
            OCRLine(text="--- gibberish text ---", bbox=(0, 0, 0, 0), confidence=0.2),
        ]
        ocr = StubOCREngine(canned_lines=canned_lines)
        pipeline = DocumentAIPipeline(ocr_engine=ocr)

        doc = pipeline.process("dummy.png")

        assert doc.dealer_name.is_present() is False
        assert doc.model_name.is_present() is False
        assert doc.horse_power.is_present() is False
        assert doc.asset_cost.is_present() is False
        assert doc.needs_human_review is True
        assert doc.overall_confidence == 0.0

    def test_pipeline_with_missing_fields(self):
        canned_lines = [
            OCRLine(text="Dealer: Swaraj Tractors Ltd.", bbox=(10, 20, 200, 40), confidence=0.98),
            OCRLine(text="Power: 50 HP", bbox=(10, 50, 120, 70), confidence=0.95),
            # Missing model and asset cost
        ]
        ocr = StubOCREngine(canned_lines=canned_lines)
        pipeline = DocumentAIPipeline(ocr_engine=ocr)

        doc = pipeline.process("dummy.png")

        assert doc.dealer_name.value == "Swaraj Tractors Ltd."
        assert doc.horse_power.value == 50.0
        assert doc.model_name.is_present() is False
        assert doc.asset_cost.is_present() is False
        assert doc.needs_human_review is True
        assert any("model_name: required field is missing" in r for r in doc.review_reasons)
        assert any("asset_cost: required field is missing" in r for r in doc.review_reasons)
