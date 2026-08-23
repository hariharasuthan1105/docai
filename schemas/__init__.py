"""
Schema Registry Package for Document AI.
"""

from docai.schemas.schema_loader import (
    ClassificationHints,
    DocumentSchema,
    FieldDefinition,
    SchemaRegistry,
    TableColumnDefinition,
    TableDefinition,
    ValidationRuleDefinition,
    get_schema_registry,
)

__all__ = [
    "ClassificationHints",
    "DocumentSchema",
    "FieldDefinition",
    "SchemaRegistry",
    "TableColumnDefinition",
    "TableDefinition",
    "ValidationRuleDefinition",
    "get_schema_registry",
]
