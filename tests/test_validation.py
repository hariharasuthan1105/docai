import pytest
from docai.models.extraction_schema import FieldSource, FieldValue
from docai.validation.business_rules import RuleEngine, RequiredFieldsRule, NumericRangeRule, DateFormatRule, PhoneNumberRule
from docai.validation.confidence import compute_overall_confidence, make_review_decision
from docai.models.extraction_schema import ReviewDecision

class TestBusinessRules:
    def test_valid_fields_pass(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=0.95),
            "model_name": FieldValue(value="Arjun Novo 605 DI", confidence=0.98),
            "horse_power": FieldValue(value=50.0, confidence=0.95),
            "asset_cost": FieldValue(value=550000.0, confidence=0.92),
        }
        engine = RuleEngine([
            RequiredFieldsRule(["dealer_name", "model_name", "horse_power", "asset_cost"]),
            NumericRangeRule("horse_power", 5.0, 150.0),
            NumericRangeRule("asset_cost", 50000.0, 50000000.0),
        ])
        reasons, blocking = engine.validate(fields)
        assert len(reasons) == 0
        assert blocking is False

    def test_missing_required_field_fails(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=0.95),
            "model_name": FieldValue(value="", confidence=0.0),
            "horse_power": FieldValue(value=50.0, confidence=0.95),
            "asset_cost": FieldValue(value=550000.0, confidence=0.92),
        }
        engine = RuleEngine([
            RequiredFieldsRule(["dealer_name", "model_name", "horse_power", "asset_cost"])
        ])
        reasons, blocking = engine.validate(fields)
        assert len(reasons) > 0
        assert blocking is True
        assert any("model_name: required field is missing" in r for r in reasons)

    def test_horse_power_out_of_range(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=0.95),
            "model_name": FieldValue(value="Arjun Novo 605 DI", confidence=0.98),
            "horse_power": FieldValue(value=600.0, confidence=0.95),  # Plausible max is 150
            "asset_cost": FieldValue(value=550000.0, confidence=0.92),
        }
        engine = RuleEngine([NumericRangeRule("horse_power", 5.0, 150.0)])
        reasons, blocking = engine.validate(fields)
        assert len(reasons) > 0
        assert blocking is True
        assert any("horse_power: value 600.0 is outside" in r for r in reasons)

    def test_asset_cost_out_of_range(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=0.95),
            "model_name": FieldValue(value="Arjun Novo 605 DI", confidence=0.98),
            "horse_power": FieldValue(value=50.0, confidence=0.95),
            "asset_cost": FieldValue(value=200.0, confidence=0.92),  # Plausible min is 50,000
        }
        engine = RuleEngine([NumericRangeRule("asset_cost", 50000.0, 50000000.0)])
        reasons, blocking = engine.validate(fields)
        assert len(reasons) > 0
        assert blocking is True
        assert any("asset_cost: value 200.0 is outside" in r for r in reasons)


class TestConfidenceScoring:
    def test_overall_confidence_calculation(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=1.0),
            "model_name": FieldValue(value="Arjun Novo 605 DI", confidence=1.0),
            "horse_power": FieldValue(value=50.0, confidence=0.90),
            "asset_cost": FieldValue(value=550000.0, confidence=0.90),
        }
        score = compute_overall_confidence(fields)
        assert score == 0.95

    def test_missing_field_drags_confidence_down(self):
        fields = {
            "dealer_name": FieldValue(value="Mahindra Tractors Ltd.", confidence=1.0),
            # Missing model_name
            "horse_power": FieldValue(value=50.0, confidence=0.90),
            "asset_cost": FieldValue(value=550000.0, confidence=0.90),
        }
        field_weights = {"dealer_name": 1.0, "model_name": 1.0, "horse_power": 1.0, "asset_cost": 1.0}
        score = compute_overall_confidence(fields, field_weights=field_weights)
        assert score < 0.75

    def test_needs_human_review_logic(self):
        # High confidence, no rule violations -> auto approved
        decision, review, reasons = make_review_decision(0.95, [])
        assert decision == ReviewDecision.AUTO_APPROVE
        assert review is False
        assert len(reasons) == 0

        # Low confidence -> needs review
        decision, review, reasons = make_review_decision(0.70, [])
        assert decision == ReviewDecision.REVIEW
        assert review is True
        assert any("below auto-approval threshold" in r for r in reasons)

