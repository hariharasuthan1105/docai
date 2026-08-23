"""
Fuzzy matching for dealer names and exact matching for model names.

Pure deterministic entity resolution against master data catalogs with
bounding-box association.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rapidfuzz import fuzz, process

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult


_DEALER_PREFIXES = [
    re.compile(r"^(?:authorized\s*dealer(?:\s*for)?|dealer\s*name|dealer|m/s\.?)\s*[:\-]?\s*", re.IGNORECASE),
]


def clean_dealer_candidate_text(text: str) -> str:
    """Strip common dealer prefixes like 'Authorized Dealer:', 'Dealer Name:' etc."""
    cleaned = text.strip()
    for p in _DEALER_PREFIXES:
        cleaned = p.sub("", cleaned).strip()
    return cleaned


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by lowering case and collapsing whitespaces/punctuation."""
    if not text:
        return ""
    # Collapse non-alphanumeric to single spaces
    return re.sub(r"\s+", " ", text).strip().lower()


class EntityMatcher:
    """
    Performs fuzzy matching for dealer names and exact/normalized matching
    for model names against master lists.
    """

    def __init__(
        self,
        dealer_master: Optional[List[str]] = None,
        model_master: Optional[List[str]] = None,
        dealer_score_cutoff: float = 65.0,
    ):
        self.dealer_master = dealer_master or []
        self.model_master = model_master or []
        self.dealer_score_cutoff = dealer_score_cutoff

        # Precompute normalized models for fast exact matching
        self._normalized_models = {
            normalize_text_for_matching(m): m for m in self.model_master
        }

    # --- Dealer Fuzzy Matching ------------------------------------------------
    def match_dealer_query(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Fuzzy match a query string against dealer master list.
        Returns (matched_dealer_name, similarity_score_0_to_100).
        """
        if not query or not query.strip() or not self.dealer_master:
            return None

        clean_query = clean_dealer_candidate_text(query)
        if not clean_query:
            clean_query = query.strip()

        result = process.extractOne(
            clean_query,
            self.dealer_master,
            scorer=fuzz.WRatio,
            score_cutoff=self.dealer_score_cutoff,
        )
        if result:
            match_name, score, _ = result
            return match_name, score
        return None

    def match_dealer_from_ocr(
        self,
        ocr_result: OCRResult,
        candidate_dealer: Optional[FieldValue] = None,
    ) -> Optional[FieldValue]:
        """
        Match dealer name from candidate or by searching through OCR lines.
        """
        # 1. If we already have a regex candidate line, test that first
        if candidate_dealer and candidate_dealer.is_present():
            match = self.match_dealer_query(str(candidate_dealer.value))
            if match:
                matched_name, score = match
                conf = round(score / 100.0, 2)
                return FieldValue(
                    value=matched_name,
                    confidence=conf,
                    source=FieldSource.FUZZY_MATCH,
                    evidence=candidate_dealer.evidence or candidate_dealer.value,
                    bbox=candidate_dealer.bbox,
                )

        # 2. Scan each OCR line for dealer match
        best_match: Optional[Tuple[str, float, OCRLine]] = None
        for line in ocr_result.lines:
            text = line.text.strip()
            if len(text) < 4:
                continue

            match = self.match_dealer_query(text)
            if match:
                name, score = match
                if best_match is None or score > best_match[1]:
                    best_match = (name, score, line)

        if best_match:
            matched_name, score, line = best_match
            conf = round(score / 100.0, 2)
            return FieldValue(
                value=matched_name,
                confidence=conf,
                source=FieldSource.FUZZY_MATCH,
                evidence=line.text,
                bbox=list(line.bbox) if line.bbox else None,
            )

        return None

    # --- Model Exact Matching -------------------------------------------------
    def match_model_query(self, query: str) -> Optional[str]:
        """
        Exact match query against normalized model catalog.
        """
        if not query or not query.strip():
            return None

        norm_query = normalize_text_for_matching(query)

        # Direct normalized match
        if norm_query in self._normalized_models:
            return self._normalized_models[norm_query]

        # Check if any master model is contained exactly within the query
        # (sorted by length descending so longer specific models match first)
        sorted_models = sorted(self._normalized_models.keys(), key=len, reverse=True)
        for norm_model in sorted_models:
            # Word-boundary containment
            pattern = r"\b" + re.escape(norm_model) + r"\b"
            if re.search(pattern, norm_query):
                return self._normalized_models[norm_model]

        return None

    def match_model_from_ocr(
        self,
        ocr_result: OCRResult,
        candidate_model: Optional[FieldValue] = None,
    ) -> Optional[FieldValue]:
        """
        Find exact model match from candidate line or across OCR lines.
        """
        # 1. Test candidate line if present
        if candidate_model and candidate_model.is_present():
            matched_name = self.match_model_query(str(candidate_model.value))
            if matched_name:
                return FieldValue(
                    value=matched_name,
                    confidence=0.98,
                    source=FieldSource.EXACT_MATCH,
                    evidence=candidate_model.evidence or candidate_model.value,
                    bbox=candidate_model.bbox,
                )

        # 2. Scan OCR lines for exact model name
        for line in ocr_result.lines:
            text = line.text.strip()
            if not text:
                continue

            matched_name = self.match_model_query(text)
            if matched_name:
                return FieldValue(
                    value=matched_name,
                    confidence=0.97,
                    source=FieldSource.EXACT_MATCH,
                    evidence=line.text,
                    bbox=list(line.bbox) if line.bbox else None,
                )

        return None


# Backward-compatibility alias
FuzzyMatcher = EntityMatcher
