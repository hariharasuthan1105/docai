"""
Unit tests for fuzzy dealer matching and exact model matching with OCR text normalization.
"""

from docai.extraction.fuzzy_matcher import EntityMatcher, normalize_text_for_matching
from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult


class TestEntityMatcher:
    def setup_method(self):
        self.dealer_catalog = [
            "Mahindra & Mahindra Ltd.",
            "Mahindra Tractors Ltd.",
            "Swaraj Tractors Ltd.",
            "John Deere India Pvt. Ltd.",
            "Tafe Tractors and Farm Equipment Ltd.",
        ]
        self.model_catalog = [
            "Arjun 605 DI",
            "Arjun Novo 605 DI",
            "Swaraj 744 FE",
            "Mahindra 575 DI",
            "John Deere 5050 D",
        ]
        self.matcher = EntityMatcher(
            dealer_master=self.dealer_catalog,
            model_master=self.model_catalog,
        )

    def test_normalize_text_for_matching(self):
        assert normalize_text_for_matching("  Arjun   NOVO  605 DI  ") == "arjun novo 605 di"
        assert normalize_text_for_matching("Mahindra & Mahindra Ltd.") == "mahindra & mahindra ltd."

    def test_fuzzy_dealer_exact_query(self):
        res = self.matcher.match_dealer_query("Mahindra Tractors Ltd.")
        assert res is not None
        matched_name, score = res
        assert matched_name == "Mahindra Tractors Ltd."
        assert score >= 99.0

    def test_fuzzy_dealer_noisy_ocr(self):
        # Slightly corrupted OCR string
        res = self.matcher.match_dealer_query("M/s Mahindra Tractor Ltd")
        assert res is not None
        matched_name, score = res
        assert matched_name == "Mahindra Tractors Ltd."
        assert score >= 80.0

    def test_fuzzy_dealer_unmatched(self):
        res = self.matcher.match_dealer_query("Completely Unrelated Auto Workshop")
        assert res is None

    def test_match_dealer_from_ocr_with_bbox(self):
        ocr = OCRResult(
            lines=[
                OCRLine(text="Invoice # 123", bbox=(10, 10, 100, 25), confidence=0.99),
                OCRLine(text="Authorized Dealer: Swaraj Tractors", bbox=(10, 30, 250, 45), confidence=0.98),
            ]
        )
        field = self.matcher.match_dealer_from_ocr(ocr)
        assert field is not None
        assert field.value == "Swaraj Tractors Ltd."
        assert field.source == FieldSource.FUZZY_MATCH
        assert field.bbox == [10.0, 30.0, 250.0, 45.0]

    def test_exact_model_matching(self):
        res = self.matcher.match_model_query("Arjun Novo 605 DI")
        assert res == "Arjun Novo 605 DI"

    def test_exact_model_matching_case_and_whitespace_insensitive(self):
        res = self.matcher.match_model_query("arjun   novo   605  di")
        assert res == "Arjun Novo 605 DI"

    def test_exact_model_matching_in_sentence(self):
        res = self.matcher.match_model_query("Supplied 1 Unit of Mahindra 575 DI with accessories")
        assert res == "Mahindra 575 DI"

    def test_match_model_from_ocr_with_bbox(self):
        ocr = OCRResult(
            lines=[
                OCRLine(text="Model: Arjun Novo 605 DI", bbox=(15, 60, 280, 80), confidence=0.97),
            ]
        )
        field = self.matcher.match_model_from_ocr(ocr)
        assert field is not None
        assert field.value == "Arjun Novo 605 DI"
        assert field.source == FieldSource.EXACT_MATCH
        assert field.bbox == [15.0, 60.0, 280.0, 80.0]
