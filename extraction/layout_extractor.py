"""
Layered Multi-Strategy Field Extractor (Backward-Compatible Adapter).

Routes extraction through SchemaExtractor using the tractor_invoice schema or provided schema.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from docai.extraction.fuzzy_matcher import EntityMatcher, FuzzyMatcher
from docai.extraction.regex_engine import RegexExtractionEngine
from docai.extraction.schema_extractor import SchemaExtractor
from docai.layout.kv_extractor import LayoutKVExtractor
from docai.models.extraction_schema import FieldValue
from docai.ocr.paddleocr_engine import OCRResult
from docai.schemas.schema_loader import DocumentSchema, get_schema_registry

logger = logging.getLogger(__name__)


class LayeredFieldExtractor:
    """
    Combines Spatial Layout Intelligence with Pattern/Regex and Entity Matching.
    """

    def __init__(
        self,
        entity_matcher: Optional[EntityMatcher] = None,
        regex_engine: Optional[RegexExtractionEngine] = None,
        layout_extractor: Optional[LayoutKVExtractor] = None,
        schema: Optional[DocumentSchema] = None,
    ):
        self.schema = schema or get_schema_registry().get_schema("tractor_invoice")
        self.fuzzy_matcher = entity_matcher or FuzzyMatcher()
        self.schema_extractor = SchemaExtractor(schema=self.schema, fuzzy_matcher=self.fuzzy_matcher)

    def extract(self, ocr_result: OCRResult) -> Dict[str, FieldValue]:
        """
        Extract all target document fields using the schema-driven strategy.
        """
        return self.schema_extractor.extract(ocr_result)


# Alias for backward compatibility
ExtractionEngine = LayeredFieldExtractor
