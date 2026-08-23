"""
Unit tests for deterministic regex extraction and numeric normalization.
"""

import pytest
from docai.extraction.regex_engine import (
    extract_asset_cost_from_line,
    extract_horse_power_from_line,
    normalize_numeric_string,
)
from docai.models.extraction_schema import FieldSource
from docai.ocr.paddleocr_engine import OCRLine, OCRResult



class TestNumericNormalization:
    def test_normalize_plain_number(self):
        assert normalize_numeric_string("550000") == 550000.0
        assert normalize_numeric_string("50") == 50.0

    def test_normalize_commas_and_indian_format(self):
        assert normalize_numeric_string("5,50,000") == 550000.0
        assert normalize_numeric_string("55,00,000.50") == 5500000.50
        assert normalize_numeric_string("1,00,000.00") == 100000.0

    def test_normalize_invalid_string(self):
        assert normalize_numeric_string("abc") is None
        assert normalize_numeric_string("") is None


class TestHorsePowerRegex:
    @pytest.mark.parametrize(
        "text,expected_hp",
        [
            ("Horse Power: 50 HP", 50.0),
            ("50HP", 50.0),
            ("50 H.P.", 50.0),
            ("50 h.p.", 50.0),
            ("HP - 50", 50.0),
            ("HP: 55", 55.0),
            ("Power: 45", 45.0),
            ("Engine Power: 50.5 H.P.", 50.5),
            ("Horse Power - 60", 60.0),
            ("Total Power: 75 HP", 75.0),
        ],
    )
    def test_horse_power_variations(self, text, expected_hp):
        field = extract_horse_power_from_line(text, bbox=(10.0, 20.0, 100.0, 40.0))
        assert field is not None
        assert field.value == expected_hp
        assert field.confidence >= 0.90
        assert field.source == FieldSource.REGEX
        assert field.bbox == [10.0, 20.0, 100.0, 40.0]


class TestAssetCostRegex:
    @pytest.mark.parametrize(
        "text,expected_cost",
        [
            ("Asset Cost: Rs. 5,50,000.00", 550000.0),
            ("Total Cost: 550000", 550000.0),
            ("Cost: ₹5,50,000/-", 550000.0),
            ("Invoice Value: Rs 6,25,000/- only", 625000.0),
            ("Total Amount: INR 5,50,000", 550000.0),
            ("Amount Payable: 7,50,000.50", 750000.50),
            ("Price: Rs. 4,80,000", 480000.0),
        ],
    )
    def test_asset_cost_variations(self, text, expected_cost):
        field = extract_asset_cost_from_line(text, bbox=(10.0, 50.0, 200.0, 70.0))
        assert field is not None
        assert field.value == expected_cost
        assert field.confidence >= 0.90
        assert field.source == FieldSource.REGEX
        assert field.bbox == [10.0, 50.0, 200.0, 70.0]


class TestRegexExtractionEngine:
    def test_extract_from_ocr_result_with_bboxes(self):
        """
        Tests the TractorRegexBaseline (evaluation/baselines) which returns
        tractor-specific field names. The generic RegexExtractionEngine no longer
        returns hardcoded field names (those come from schema YAML).
        """
        from docai.evaluation.baselines.tractor_regex_baseline import TractorRegexBaseline

        lines = [
            OCRLine(text="Tax Invoice", bbox=(10, 10, 200, 30), confidence=0.99),
            OCRLine(text="Dealer Name: Mahindra Tractors Ltd.", bbox=(10, 40, 350, 60), confidence=0.98),
            OCRLine(text="Model Name: Arjun Novo 605 DI", bbox=(10, 70, 300, 90), confidence=0.98),
            OCRLine(text="Engine Power: 50 HP", bbox=(10, 100, 180, 120), confidence=0.97),
            OCRLine(text="Asset Cost: Rs. 5,50,000.00", bbox=(10, 130, 250, 150), confidence=0.99),
        ]
        baseline = TractorRegexBaseline()
        fields = baseline.extract_from_ocr_result(OCRResult(lines=lines))

        assert fields["horse_power"].value == 50.0
        assert fields["horse_power"].bbox == [10.0, 100.0, 180.0, 120.0]

        assert fields["asset_cost"].value == 550000.0
        assert fields["asset_cost"].bbox == [10.0, 130.0, 250.0, 150.0]

        assert fields["dealer_name"].value == "Mahindra Tractors Ltd."
        assert fields["dealer_name"].bbox == [10.0, 40.0, 350.0, 60.0]

        assert fields["model_name"].value == "Arjun Novo 605 DI"
        assert fields["model_name"].bbox == [10.0, 70.0, 300.0, 90.0]

