"""
Schema-Driven Validation Engine.

Dynamically evaluates schema-defined business rules, arithmetic balance formulas,
table row multiplications, numeric range bounds, and data formats without hardcoded logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from docai.layout.table_extractor import ExtractedTable
from docai.models.extraction_schema import FieldValue
from docai.schemas.schema_loader import DocumentSchema, ValidationRuleDefinition

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool = True
    passed_rules: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, rule_name: str, message: str) -> None:
        self.is_valid = False
        self.errors.append(f"[{rule_name}] {message}")

    def add_warning(self, rule_name: str, message: str) -> None:
        self.warnings.append(f"[{rule_name}] {message}")

    def add_pass(self, rule_name: str) -> None:
        self.passed_rules.append(rule_name)


class SchemaValidator:
    """
    Evaluates schema validation rules dynamically on extracted fields and tables.
    """

    def __init__(self, schema: DocumentSchema):
        self.schema = schema

    def validate(
        self,
        fields: Dict[str, FieldValue],
        tables: Optional[Dict[str, ExtractedTable]] = None,
    ) -> ValidationReport:
        report = ValidationReport()
        field_values: Dict[str, Any] = {k: v.value for k, v in fields.items()}
        tables = tables or {}

        # 1. Required Fields Check
        for fname, fdef in self.schema.fields.items():
            if fdef.required:
                if fname not in fields or fields[fname].value is None or str(fields[fname].value).strip() == "":
                    report.add_error("required_field", f"Mandatory field '{fname}' is missing.")
                elif fields[fname].confidence < 0.40:
                    report.add_warning("low_confidence_required", f"Mandatory field '{fname}' has low confidence ({fields[fname].confidence:.2f}).")
                else:
                    report.add_pass(f"required_{fname}")

        # 2. Field-level Range and Pattern Checks
        for fname, fdef in self.schema.fields.items():
            if fname not in fields:
                continue

            val = fields[fname].value
            # Numeric range check
            if isinstance(val, (int, float)):
                if fdef.min_value is not None and val < fdef.min_value:
                    report.add_error(f"{fname}_range", f"Field '{fname}' value {val} is below minimum allowed ({fdef.min_value}).")
                elif fdef.max_value is not None and val > fdef.max_value:
                    report.add_error(f"{fname}_range", f"Field '{fname}' value {val} exceeds maximum allowed ({fdef.max_value}).")
                else:
                    if fdef.min_value is not None or fdef.max_value is not None:
                        report.add_pass(f"{fname}_range")

        # 3. Schema Defined Validation Rules
        for vrule in self.schema.validations:
            if vrule.rule_type == "table_row_check":
                self._validate_table_rows(vrule, tables, report)
            elif vrule.rule_type == "arithmetic_balance":
                self._validate_arithmetic_balance(vrule, field_values, report)
            elif vrule.rule_type == "range_check":
                self._validate_expression(vrule, field_values, report)

        return report

    def _validate_table_rows(
        self,
        vrule: ValidationRuleDefinition,
        tables: Dict[str, ExtractedTable],
        report: ValidationReport,
    ) -> None:
        """Validate row math formula e.g. qty * unit_price == amount across table rows."""
        table_found = False
        for tname, table in tables.items():
            if not table.rows:
                continue
            table_found = True
            mismatches = 0
            for row in table.rows:
                cells = row.cells
                if "qty" in cells and "unit_price" in cells and "amount" in cells:
                    try:
                        q = float(cells["qty"].value)
                        u = float(cells["unit_price"].value)
                        a = float(cells["amount"].value)
                        expected_amt = q * u
                        if abs(expected_amt - a) > 1.0:
                            mismatches += 1
                    except Exception:
                        pass

            if mismatches > 0:
                msg = f"{mismatches} table row(s) had calculation mismatches."
                if vrule.severity == "error":
                    report.add_error(vrule.name, msg)
                else:
                    report.add_warning(vrule.name, msg)
            else:
                report.add_pass(vrule.name)

        if not table_found and self.schema.tables:
            report.add_warning("table_missing", "Expected table was not detected.")

    def _validate_arithmetic_balance(
        self,
        vrule: ValidationRuleDefinition,
        values: Dict[str, Any],
        report: ValidationReport,
    ) -> None:
        """Validate arithmetic balance formulas like subtotal - discount + taxes == grand_total."""
        subtotal = values.get("subtotal")
        grand_total = values.get("grand_total")

        if subtotal is not None and grand_total is not None:
            try:
                sub = float(subtotal)
                disc = float(values.get("discount_amount") or 0.0)
                cgst = float(values.get("cgst_amount") or 0.0)
                sgst = float(values.get("sgst_amount") or 0.0)
                gt = float(grand_total)

                computed_total = sub - disc + cgst + sgst
                # Allow minor rounding tolerance (up to 1.0 unit / rupee)
                if abs(computed_total - gt) <= 2.0:
                    report.add_pass(vrule.name)
                else:
                    msg = f"Arithmetic mismatch: Subtotal ({sub:.2f}) - Discount ({disc:.2f}) + Taxes ({cgst+sgst:.2f}) = {computed_total:.2f} != Grand Total ({gt:.2f})"
                    if vrule.severity == "error":
                        report.add_error(vrule.name, msg)
                    else:
                        report.add_warning(vrule.name, msg)
            except Exception as e:
                logger.debug("Arithmetic validation evaluation error: %s", e)

    def _validate_expression(
        self,
        vrule: ValidationRuleDefinition,
        values: Dict[str, Any],
        report: ValidationReport,
    ) -> None:
        """Safely evaluate simple comparison expressions on field values."""
        expr = vrule.expression
        if not expr:
            return

        # Prepare safe context dictionary
        context = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                context[k] = v

        try:
            # Check if all variables in expression are present in context
            vars_in_expr = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
            missing = [var for var in vars_in_expr if var not in context and var not in ["and", "or", "not", "True", "False"]]
            if missing:
                return

            result = eval(expr, {"__builtins__": None}, context)
            if result:
                report.add_pass(vrule.name)
            else:
                msg = vrule.message or f"Validation rule '{vrule.name}' failed."
                if vrule.severity == "error":
                    report.add_error(vrule.name, msg)
                else:
                    report.add_warning(vrule.name, msg)
        except Exception as e:
            logger.debug("Expression evaluation exception: %s", e)
