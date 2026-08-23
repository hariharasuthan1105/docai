"""
Dynamic Schema Loader and Models for Multi-Document AI.

Loads YAML schema definitions and compiles them into structured Pydantic models.
All domain-specific logic (catalogs, validation ranges, visual marks, address keywords)
is declared in YAML schemas, not in Python code.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Pattern, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FieldDefinition(BaseModel):
    name: str
    section: str = "metadata"
    aliases: List[str] = Field(default_factory=list)
    type: str = "string"  # "string", "number", "currency", "date", "time", "phone"
    required: bool = False
    extraction_strategies: List[str] = Field(default_factory=lambda: ["key_value", "regex"])
    regex_patterns: List[str] = Field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    format_pattern: Optional[str] = None
    confidence_weight: float = 1.0
    catalog_master: Optional[List[str]] = None
    catalog_ref: Optional[str] = None  # Named catalog reference from schema.entity_catalogs

    # Position-header extraction metadata (replaces hardcoded field-name checks)
    position: Optional[str] = None           # "first_header" | "second_header" | "header_block"
    address_keywords: List[str] = Field(default_factory=list)  # Keywords for address line detection
    tagline_markers: List[str] = Field(default_factory=list)   # Markers for tagline detection

    def get_compiled_regexes(self) -> List[Pattern]:
        compiled = []
        for pat in self.regex_patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE))
            except Exception as e:
                logger.error("Failed to compile regex '%s' for field '%s': %s", pat, self.name, e)
        return compiled

    def get_compiled_aliases(self) -> List[Pattern]:
        compiled = []
        for alias in self.aliases:
            escaped = re.escape(alias.strip())
            pat_str = rf"(?:^|[\b\s]){escaped}\s*[:\-]?"
            try:
                compiled.append(re.compile(pat_str, re.IGNORECASE))
            except Exception:
                pass
        return compiled


class TableColumnDefinition(BaseModel):
    key: str
    aliases: List[str] = Field(default_factory=list)
    type: str = "string"
    required: bool = False


class TableDefinition(BaseModel):
    name: str
    header_keywords: List[str] = Field(default_factory=list)
    columns: List[TableColumnDefinition] = Field(default_factory=list)
    row_formula: Optional[str] = None
    min_rows: int = 1


class ValidationRuleDefinition(BaseModel):
    name: str
    rule_type: str
    expression: Optional[str] = None
    severity: str = "error"
    message: str = ""


class VisualMarkDefinition(BaseModel):
    """Schema-configured visual mark (e.g. signature or stamp)."""
    name: str          # Unique key used in DocumentResult.visual_marks dict
    mark_type: str     # "signature" or "stamp"
    label: str = ""    # Human-readable label for CLI/demo display


class ClassificationHints(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    title_keywords: List[str] = Field(default_factory=list)
    required_anchors: List[str] = Field(default_factory=list)
    table_indicators: List[str] = Field(default_factory=list)
    min_score_threshold: float = 0.35


class DocumentSchema(BaseModel):
    document_type: str
    title: str
    description: str = ""
    version: str = "1.0"
    classification: ClassificationHints = Field(default_factory=ClassificationHints)
    sections: List[str] = Field(default_factory=lambda: ["metadata", "totals"])
    fields: Dict[str, FieldDefinition] = Field(default_factory=dict)
    tables: Dict[str, TableDefinition] = Field(default_factory=dict)
    validations: List[ValidationRuleDefinition] = Field(default_factory=list)
    auto_approve_threshold: float = 0.88
    review_threshold: float = 0.70

    # Schema-driven visual marks -- replaces hardcoded "dealer_signature"/"dealer_stamp"
    visual_marks: List[VisualMarkDefinition] = Field(default_factory=list)

    # Schema-driven table stop keywords -- replaces hardcoded list in table_detector.py
    stop_keywords: List[str] = Field(default_factory=list)

    # Named entity catalogs for fuzzy/exact matching (referenced by field.catalog_ref)
    entity_catalogs: Dict[str, List[str]] = Field(default_factory=dict)

    def get_field(self, field_name: str) -> Optional[FieldDefinition]:
        return self.fields.get(field_name)

    def get_fields_in_section(self, section: str) -> List[FieldDefinition]:
        return [f for f in self.fields.values() if f.section == section]

    def get_catalog(self, catalog_ref: str) -> List[str]:
        """Resolve a named catalog reference to its list."""
        return self.entity_catalogs.get(catalog_ref, [])


class SchemaRegistry:
    """Central repository of all registered document schemas."""

    def __init__(self, schemas_dir: Optional[str] = None):
        self.schemas_dir = schemas_dir or os.path.dirname(os.path.abspath(__file__))
        self.schemas: Dict[str, DocumentSchema] = {}
        self.registry_meta: Dict[str, Any] = {}
        self.load_all()

    def load_all(self) -> None:
        registry_path = os.path.join(self.schemas_dir, "registry.yaml")
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                self.registry_meta = yaml.safe_load(f) or {}

        doc_types = self.registry_meta.get("document_types", {})
        for dtype, meta in doc_types.items():
            schema_file = meta.get("schema")
            if schema_file:
                full_path = os.path.join(self.schemas_dir, schema_file)
                if os.path.exists(full_path):
                    schema = self.load_schema_file(full_path)
                    if schema:
                        self.schemas[schema.document_type] = schema

        for fname in os.listdir(self.schemas_dir):
            if fname.endswith(".yaml") and fname not in ["registry.yaml"]:
                full_path = os.path.join(self.schemas_dir, fname)
                schema = self.load_schema_file(full_path)
                if schema and schema.document_type not in self.schemas:
                    self.schemas[schema.document_type] = schema

    def load_schema_file(self, file_path: str) -> Optional[DocumentSchema]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                return None

            dtype = raw.get("document_type", os.path.splitext(os.path.basename(file_path))[0])
            title = raw.get("title", dtype.replace("_", " ").title())

            raw_fields = raw.get("fields", {})
            fields_dict = {}
            for fname, fmeta in raw_fields.items():
                if isinstance(fmeta, dict):
                    fmeta["name"] = fname
                    fields_dict[fname] = FieldDefinition(**fmeta)

            raw_tables = raw.get("tables", {})
            tables_dict = {}
            for tname, tmeta in raw_tables.items():
                if isinstance(tmeta, dict):
                    raw_cols = tmeta.get("columns", {})
                    cols = []
                    if isinstance(raw_cols, dict):
                        for ckey, cmeta in raw_cols.items():
                            if isinstance(cmeta, dict):
                                cmeta["key"] = ckey
                                cols.append(TableColumnDefinition(**cmeta))
                            elif isinstance(cmeta, list):
                                cols.append(TableColumnDefinition(key=ckey, aliases=cmeta))
                    elif isinstance(raw_cols, list):
                        for citem in raw_cols:
                            if isinstance(citem, dict):
                                cols.append(TableColumnDefinition(**citem))
                            elif isinstance(citem, str):
                                cols.append(TableColumnDefinition(key=citem, aliases=[citem]))
                    tmeta["columns"] = cols
                    tables_dict[tname] = TableDefinition(**tmeta)

            raw_val = raw.get("validations", [])
            validations_list = []
            if isinstance(raw_val, list):
                for v in raw_val:
                    if isinstance(v, dict):
                        validations_list.append(ValidationRuleDefinition(**v))

            raw_class = raw.get("classification", {})
            class_hints = ClassificationHints(**raw_class) if isinstance(raw_class, dict) else ClassificationHints()

            raw_vision = raw.get("vision", {})
            marks_list = []
            if isinstance(raw_vision, dict):
                for m in raw_vision.get("marks", []):
                    if isinstance(m, dict):
                        marks_list.append(VisualMarkDefinition(**m))

            stop_kws = raw.get("stop_keywords", [])
            if not isinstance(stop_kws, list):
                stop_kws = []

            raw_catalogs = raw.get("entity_catalogs", {})
            entity_catalogs = raw_catalogs if isinstance(raw_catalogs, dict) else {}

            return DocumentSchema(
                document_type=dtype,
                title=title,
                description=raw.get("description", ""),
                version=str(raw.get("version", "1.0")),
                classification=class_hints,
                sections=raw.get("sections", ["metadata", "totals"]),
                fields=fields_dict,
                tables=tables_dict,
                validations=validations_list,
                auto_approve_threshold=float(raw.get("auto_approve_threshold", 0.88)),
                review_threshold=float(raw.get("review_threshold", 0.70)),
                visual_marks=marks_list,
                stop_keywords=stop_kws,
                entity_catalogs=entity_catalogs,
            )
        except Exception as e:
            logger.error("Failed to parse schema file %s: %s", file_path, e)
            return None

    def get_schema(self, document_type: str) -> Optional[DocumentSchema]:
        return self.schemas.get(document_type)

    def list_document_types(self) -> List[str]:
        return list(self.schemas.keys())


_default_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SchemaRegistry()
    return _default_registry
