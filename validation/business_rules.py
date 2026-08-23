"""
Extensible Business Rules and Domain Validation Engine.

Provides pluggable validation rules for range checks, date integrity,
phone formats, and required field completeness.
"""

from __future__ import annotations

import datetime
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from docai.config import ASSET_COST_RANGE, HORSE_POWER_RANGE, REQUIRED_FIELDS
from docai.models.extraction_schema import FieldValue

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    field_name: str
    is_valid: bool
    severity: str  # "error", "warning", "info"
    message: str


class BaseValidationRule(ABC):
    """Abstract base class for extensible domain validation rules."""

    @abstractmethod
    def validate(self, fields: Dict[str, FieldValue]) -> List[ValidationResult]:
        raise NotImplementedError


class RequiredFieldsRule(BaseValidationRule):
    def __init__(self, required_fields: Optional[List[str]] = None):
        self.required_fields = required_fields or REQUIRED_FIELDS

    def validate(self, fields: Dict[str, FieldValue]) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for field_name in self.required_fields:
            field_val = fields.get(field_name, FieldValue())
            if not field_val.is_present():
                results.append(
                    ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        severity="error",
                        message=f"{field_name}: required field is missing",
                    )
                )
        return results


class NumericRangeRule(BaseValidationRule):
    def __init__(self, field_name: str, min_val: float, max_val: float, unit: str = ""):
        self.field_name = field_name
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit

    def validate(self, fields: Dict[str, FieldValue]) -> List[ValidationResult]:
        field_val = fields.get(self.field_name, FieldValue())
        if not field_val.is_present():
            return []

        try:
            val_num = float(field_val.value)
        except (ValueError, TypeError):
            return [
                ValidationResult(
                    field_name=self.field_name,
                    is_valid=False,
                    severity="error",
                    message=f"{self.field_name}: value '{field_val.value}' cannot be parsed as numeric",
                )
            ]

        if not (self.min_val <= val_num <= self.max_val):
            return [
                ValidationResult(
                    field_name=self.field_name,
                    is_valid=False,
                    severity="error",
                    message=(
                        f"{self.field_name}: value {val_num} is outside plausible range "
                        f"[{self.min_val}, {self.max_val}]"
                    ),
                )
            ]

        return []


class DateFormatRule(BaseValidationRule):
    def __init__(self, field_name: str = "invoice_date", min_year: int = 2000, max_year: int = 2035):
        self.field_name = field_name
        self.min_year = min_year
        self.max_year = max_year

    def validate(self, fields: Dict[str, FieldValue]) -> List[ValidationResult]:
        field_val = fields.get(self.field_name, FieldValue())
        if not field_val.is_present():
            return []

        date_str = str(field_val.value).strip()
        parsed_date = None

        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%d/%m/%y",
            "%d-%m-%y",
            "%d %b %Y",
            "%d %B %Y",
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue

        if parsed_date is None:
            return [
                ValidationResult(
                    field_name=self.field_name,
                    is_valid=False,
                    severity="warning",
                    message=f"Date '{date_str}' format unrecognized or unparseable.",
                )
            ]

        if not (self.min_year <= parsed_date.year <= self.max_year):
            return [
                ValidationResult(
                    field_name=self.field_name,
                    is_valid=False,
                    severity="error",
                    message=f"Date year {parsed_date.year} outside plausible range [{self.min_year}, {self.max_year}].",
                )
            ]

        return []


class PhoneNumberRule(BaseValidationRule):
    def __init__(self, field_name: str = "phone_number"):
        self.field_name = field_name

    def validate(self, fields: Dict[str, FieldValue]) -> List[ValidationResult]:
        field_val = fields.get(self.field_name, FieldValue())
        if not field_val.is_present():
            return []

        digits = re.sub(r"[^\d]", "", str(field_val.value))
        if len(digits) == 10 and digits[0] in "6789":
            return []
        elif len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
            return []

        return [
            ValidationResult(
                field_name=self.field_name,
                is_valid=False,
                severity="warning",
                message=f"Phone number '{field_val.value}' does not match standard 10-digit mobile pattern.",
            )
        ]


class RuleEngine:
    """
    Extensible validation engine executing registered domain rules.
    """

    def __init__(self, rules: Optional[List[BaseValidationRule]] = None):
        self.rules: List[BaseValidationRule] = rules if rules is not None else self._default_rules()

    @staticmethod
    def _default_rules() -> List[BaseValidationRule]:
        return [
            RequiredFieldsRule(REQUIRED_FIELDS),
            NumericRangeRule("horse_power", *HORSE_POWER_RANGE, unit="HP"),
            NumericRangeRule("asset_cost", *ASSET_COST_RANGE, unit="INR"),
            DateFormatRule("invoice_date"),
            PhoneNumberRule("phone_number"),
        ]

    def add_rule(self, rule: BaseValidationRule) -> None:
        self.rules.append(rule)

    def validate(self, fields: Dict[str, FieldValue]) -> Tuple[List[str], bool]:
        """
        Execute all rules. Returns (error_messages, blocking_flag).
        """
        messages: List[str] = []
        blocking = False

        for rule in self.rules:
            try:
                results = rule.validate(fields)
                for res in results:
                    if not res.is_valid:
                        messages.append(res.message)
                        if res.severity == "error":
                            blocking = True
                            # Update field validation status
                            if res.field_name in fields:
                                fields[res.field_name].validation_status = "invalid"
                        elif res.severity == "warning" and res.field_name in fields:
                            if fields[res.field_name].validation_status != "invalid":
                                fields[res.field_name].validation_status = "warning"
            except Exception as e:
                logger.error("Error executing rule %s: %s", type(rule).__name__, e)
                messages.append(f"Validation rule failure ({type(rule).__name__}): {e}")

        return messages, blocking


# Backward-compatible functional helpers
def check_numeric_range(
    field_name: str, field_value: FieldValue, low: float, high: float
) -> List[str]:
    rule = NumericRangeRule(field_name, low, high)
    results = rule.validate({field_name: field_value})
    return [r.message for r in results if not r.is_valid]


def check_required_fields(fields: Dict[str, FieldValue]) -> List[str]:
    rule = RequiredFieldsRule(REQUIRED_FIELDS)
    results = rule.validate(fields)
    return [r.message for r in results if not r.is_valid]


def apply_business_rules(fields: Dict[str, FieldValue]) -> Tuple[List[str], bool]:
    engine = RuleEngine()
    return engine.validate(fields)

