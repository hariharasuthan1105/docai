"""
Document Type Classification Engine.

Dynamically classifies documents into registered schema types (e.g. restaurant receipt,
tractor invoice, or generic/unknown) using semantic keyword density, title tokens,
and structural layout signals.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from docai.ocr.paddleocr_engine import OCRResult
from docai.schemas.schema_loader import DocumentSchema, SchemaRegistry, get_schema_registry

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    title: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    schema: Optional[DocumentSchema] = None


class DocumentClassifier:
    """
    Production document classifier matching OCR content against schema definitions.
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None):
        self.registry = registry or get_schema_registry()

    def classify(self, ocr_result: OCRResult) -> ClassificationResult:
        full_text = (ocr_result.full_text or "").lower()
        if not full_text.strip():
            return ClassificationResult(
                document_type="generic",
                confidence=0.10,
                title="Unknown Document",
                scores={"generic": 0.10},
                schema=self.registry.get_schema("generic"),
            )

        # Separate top header text (first 5 lines or lines in top 25% of page)
        header_lines = ocr_result.lines[:6]
        header_text = " ".join(line.text.lower() for line in header_lines)

        scores: Dict[str, float] = {}
        matched_kw_map: Dict[str, List[str]] = {}

        for doc_type, schema in self.registry.schemas.items():
            if doc_type == "generic":
                continue

            hints = schema.classification
            score = 0.0
            matched_kws = []

            # 1. Title Keywords (High Weight: 0.35)
            title_hits = 0
            for t_kw in hints.title_keywords:
                if t_kw.lower() in header_text:
                    title_hits += 1
                    matched_kws.append(f"title:{t_kw}")
                elif t_kw.lower() in full_text:
                    title_hits += 0.5
                    matched_kws.append(f"body_title:{t_kw}")

            if hints.title_keywords:
                score += min(0.35, (title_hits / max(1, len(hints.title_keywords))) * 0.70)

            # 2. General Body Keywords (Weight: 0.35)
            body_hits = 0
            for kw in hints.keywords:
                pattern = rf"\b{re.escape(kw.lower())}\b"
                if re.search(pattern, full_text):
                    body_hits += 1
                    matched_kws.append(kw)

            if hints.keywords:
                kw_ratio = min(1.0, body_hits / max(3, len(hints.keywords) * 0.40))
                score += kw_ratio * 0.35

            # 3. Required / Distinctive Anchors (Weight: 0.20)
            anchor_hits = 0
            for anchor in hints.required_anchors:
                if anchor.lower() in full_text:
                    anchor_hits += 1
                    matched_kws.append(f"anchor:{anchor}")

            if hints.required_anchors:
                anchor_ratio = anchor_hits / len(hints.required_anchors)
                score += anchor_ratio * 0.20

            # 4. Table Indicator Match (Weight: 0.10)
            table_hits = 0
            for tind in hints.table_indicators:
                if tind.lower() in full_text:
                    table_hits += 1
                    matched_kws.append(f"table:{tind}")

            if hints.table_indicators:
                table_ratio = min(1.0, table_hits / max(1, len(hints.table_indicators)))
                score += table_ratio * 0.10

            scores[doc_type] = round(score, 4)
            matched_kw_map[doc_type] = matched_kws

        if not scores:
            return ClassificationResult(
                document_type="generic",
                confidence=0.30,
                title="Generic Document",
                scores={"generic": 0.30},
                schema=self.registry.get_schema("generic"),
            )

        # Find best matching candidate
        best_doc_type, best_score = max(scores.items(), key=lambda item: item[1])
        best_schema = self.registry.get_schema(best_doc_type)

        threshold = best_schema.classification.min_score_threshold if best_schema else 0.30

        if best_score >= threshold:
            # Map score to calibrated confidence in [0.70, 0.98]
            conf = min(0.98, max(0.65, 0.60 + (best_score / 1.0) * 0.38))
            return ClassificationResult(
                document_type=best_doc_type,
                confidence=round(conf, 4),
                title=best_schema.title if best_schema else best_doc_type,
                matched_keywords=matched_kw_map.get(best_doc_type, []),
                scores=scores,
                schema=best_schema,
            )
        else:
            # Below confidence threshold: fallback cleanly to generic
            return ClassificationResult(
                document_type="generic",
                confidence=round(max(0.20, best_score), 4),
                title="Generic / Unknown Document",
                matched_keywords=[],
                scores=scores,
                schema=self.registry.get_schema("generic"),
            )
