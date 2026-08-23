"""
Schema-Driven Validation Engine.

Dynamically evaluates schema-defined business rules, arithmetic balance formulas,
table row multiplications, numeric range bounds, and data formats.

ARCHITECTURE REQUIREMENT:
  This file must NOT contain any hardcoded field names (subtotal, grand_total,
  qty, unit_price, amount, cgst_amount, etc.).

  All domain-specific validation logic is declared in the schema YAML:
    - Field ranges:    schema.fields[x].min_value / max_value
    - Row formulas:    schema.tables[x].row_formula  (e.g. "qty * unit_price == amount")
    - Expressions:     schema.validations[x].expression
    - Required fields: schema.fields[x].required = true

  This validator interprets those declarations generically.

  Adding a new document type requires ONLY a new YAML schema, not changes here.
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


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_row_formula(formula: str) -> Optional[tuple]:
    """
    Parse a row formula of the form 'a * b == c' or 'a + b == c'.
    Returns (lhs_expression, rhs_var) or None if unparsable.

    Supported operators: *, +, -, /
    Example: 'qty * unit_price == amount' -> ('qty * unit_price', 'amount')
    """
    if not formula:
        return None
    m = re.match(r"^(.+?)\s*==\s*(\w+)\s*$", formula.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


class SchemaValidator:
    """
    Evaluates schema validation rules dynamically on extracted fields and tables.

    NO domain-specific field names appear in this class.
    All validation logic is driven by the schema YAML declarations.
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

        # 1. Required Fields Check — field names come from schema YAML
        for fname, fdef in self.schema.fields.items():
            if fdef.required:
                if fname not in fields or fields[fname].value is None or str(fields[fname].value).strip() == "":
                    report.add_error("required_field", f"Mandatory field '{fname}' is missing.")
                elif fields[fname].confidence < 0.40:
                    report.add_warning(
                        "low_confidence_required",
                        f"Mandatory field '{fname}' has low confidence ({fields[fname].confidence:.2f}).",
                    )
                else:
                    report.add_pass(f"required_{fname}")

        # 2. Field-level Range Checks — min_value / max_value from schema YAML
        for fname, fdef in self.schema.fields.items():
            if fname not in fields:
                continue
            val = fields[fname].value
            if isinstance(val, (int, float)):
                if fdef.min_value is not None and val < fdef.min_value:
                    report.add_error(
                        f"{fname}_range",
                        f"Field '{fname}' value {val} is below minimum allowed ({fdef.min_value}).",
                    )
                elif fdef.max_value is not None and val > fdef.max_value:
                    report.add_error(
                        f"{fname}_range",
                        f"Field '{fname}' value {val} exceeds maximum allowed ({fdef.max_value}).",
                    )
                else:
                    if fdef.min_value is not None or fdef.max_value is not None:
                        report.add_pass(f"{fname}_range")

        # 3. Schema-Declared Validation Rules
        for vrule in self.schema.validations:
            rule_type = vrule.rule_type

            if rule_type == "table_row_check":
                self._validate_table_rows_generic(vrule, tables, report)

            elif rule_type in ("arithmetic_balance", "range_check", "expression", "cross_field"):
                self._validate_expression(vrule, field_values, report)

            elif rule_type == "required":
                # Handled above via schema.fields; can also be declared as a rule for emphasis
                self._validate_required_rule(vrule, fields, report)

        return report

    def _validate_table_rows_generic(
        self,
        vrule: ValidationRuleDefinition,
        tables: Dict[str, ExtractedTable],
        report: ValidationReport,
    ) -> None:
        """
        Validate table row math using a formula from vrule.row_formula (from YAML).

        Formula syntax example:  "qty * unit_price == amount"
        The formula uses column key names as defined in the schema's table.columns.

        NO hardcoded column names appear here.
        """
        # Find row_formula from the schema table definition (preferred) or from vrule expression
        row_formula = vrule.expression  # YAML: expression field carries the formula

        if not row_formula:
            # Try to find it from the schema table definition
            for tdef in self.schema.tables.values():
                if tdef.row_formula:
                    row_formula = tdef.row_formula
                    break

        parsed = _parse_row_formula(row_formula) if row_formula else None

        table_found = False
        for tname, table in tables.items():
            if not table.rows:
                continue
            table_found = True
            mismatches = 0

            for row in table.rows:
                cells = row.cells

                if parsed:
                    lhs_expr, rhs_var = parsed
                    # Build a safe evaluation context from available cell values
                    context: Dict[str, float] = {}
                    skip = False
                    for var in re.findall(r"\b[a-zA-Z_]\w*\b", lhs_expr + " " + rhs_var):
                        if var in cells:
                            num = _safe_float(cells[var].value)
                            if num is not None:
                                context[var] = num
                            else:
                                skip = True
                                break
                        else:
                            skip = True
                            break

                    if skip:
                        continue

                    try:
                        lhs_val = eval(lhs_expr, {"__builtins__": None}, context)  # nosec: controlled context
                        rhs_val = context.get(rhs_var)
                        if rhs_val is not None and abs(lhs_val - rhs_val) > 1.0:
                            mismatches += 1
                    except Exception as e:
                        logger.debug("Row formula eval error: %s", e)

            if mismatches > 0:
                msg = f"{mismatches} table row(s) had calculation mismatches (formula: {row_formula})."
                if vrule.severity == "error":
                    report.add_error(vrule.name, msg)
                else:
                    report.add_warning(vrule.name, msg)
            else:
                report.add_pass(vrule.name)

        if not table_found and self.schema.tables:
            report.add_warning("table_missing", "Expected table was not detected.")

    def _validate_expression(
        self,
        vrule: ValidationRuleDefinition,
        values: Dict[str, Any],
        report: ValidationReport,
    ) -> None:
        """
        Safely evaluate any comparison expression on field values.

        The expression uses variable names that match the schema's field names.
        For example, from restaurant_receipt.yaml:
          expression: "subtotal - discount_amount + cgst_amount + sgst_amount == grand_total"

        NO field names are hardcoded here. The YAML provides all field names.
        """
        expr = vrule.expression
        if not expr:
            return

        # Build a safe numeric-only context from available field values
        context: Dict[str, float] = {}
        for k, v in values.items():
            num = _safe_float(v)
            if num is not None:
                context[k] = num

        try:
            # Find all variable names in the expression
            vars_in_expr = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
            skip_keywords = {"and", "or", "not", "True", "False", "abs", "round", "min", "max"}
            missing = [v for v in vars_in_expr if v not in context and v not in skip_keywords]
            if missing:
                logger.debug("Skipping rule '%s': missing fields %s", vrule.name, missing)
                return

            safe_builtins = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(expr, {"__builtins__": None, **safe_builtins}, context)  # nosec: controlled
            if result:
                report.add_pass(vrule.name)
            else:
                msg = vrule.message or f"Validation rule '{vrule.name}' failed. Expression: {expr}"
                if vrule.severity == "error":
                    report.add_error(vrule.name, msg)
                else:
                    report.add_warning(vrule.name, msg)
        except Exception as e:
            logger.debug("Expression '%s' evaluation exception: %s", expr, e)

    def _validate_required_rule(
        self,
        vrule: ValidationRuleDefinition,
        fields: Dict[str, FieldValue],
        report: ValidationReport,
    ) -> None:
        """Handle an explicit 'required' rule type in schema.validations."""
        expr = vrule.expression  # e.g. "invoice_number"
        if not expr:
            return
        field_name = expr.strip()
        if field_name not in fields or not fields[field_name].is_present():
            msg = vrule.message or f"Required field '{field_name}' is missing."
            if vrule.severity == "error":
                report.add_error(vrule.name, msg)
            else:
                report.add_warning(vrule.name, msg)
        else:
            report.add_pass(vrule.name)
