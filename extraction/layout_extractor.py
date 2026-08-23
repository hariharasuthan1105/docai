"""
Layered Multi-Strategy Field Extractor (Backward-Compatible Adapter).

Routes extraction through SchemaExtractor using a provided schema.
If no schema is provided, falls back to the generic_document schema.

NOTE: This module does NOT default to tractor_invoice.
A new document type requires only a new YAML schema, not changes here.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from docai.extraction.fuzzy_matcher import EntityMatcher, FuzzyMatcher
from docai.extraction.schema_extractor import SchemaExtractor
from docai.layout.kv_extractor import LayoutKVExtractor
from docai.models.extraction_schema import FieldValue
from docai.ocr.paddleocr_engine import OCRResult
from docai.schemas.schema_loader import DocumentSchema, get_schema_registry

logger = logging.getLogger(__name__)


class LayeredFieldExtractor:
    """
    Combines Spatial Layout Intelligence with Pattern/Regex and Entity Matching.
    Uses the supplied schema; defaults to generic_document if none provided.
    """

    def __init__(
        self,
        entity_matcher: Optional[EntityMatcher] = None,
        schema: Optional[DocumentSchema] = None,
    ):
        # Never assume tractor_invoice; fall back to generic_document
        if schema is None:
            schema = get_schema_registry().get_schema("generic_document") or get_schema_registry().get_schema("generic")
        self.schema = schema
        self.fuzzy_matcher = entity_matcher or FuzzyMatcher()
        self.schema_extractor = SchemaExtractor(schema=self.schema, fuzzy_matcher=self.fuzzy_matcher)

    def extract(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """
        Extract all target document fields using the schema-driven strategy.
        """
        return self.schema_extractor.extract(ocr_result)


# Alias for backward compatibility
ExtractionEngine = LayeredFieldExtractor
