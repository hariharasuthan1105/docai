"""
Fuzzy and Exact Entity Matching Layer with Candidate Alternatives.

Matches extracted text against master dealer and equipment model catalogs
using RapidFuzz token sorting, partial ratios, and Levenshtein edit distance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, utils

from docai.models.extraction_schema import FieldSource, FieldValue
from docai.ocr.paddleocr_engine import OCRLine, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    name: str
    similarity_score: float
    confidence: float
    match_type: str
    metrics: Dict[str, float] = field(default_factory=dict)


def normalize_text_for_matching(raw_text: str) -> str:
    """Normalize text by trimming and collapsing multiple spaces."""
    if not raw_text:
        return ""
    text = raw_text.strip().lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EntityMatcher:
    """
    Production entity matcher for dealer and equipment model catalogs.
    """

    def __init__(
        self,
        dealer_master: List[str],
        model_master: List[str],
        dealer_threshold: float = 65.0,
        model_threshold: float = 75.0,
    ):
        self.dealer_master = dealer_master
        self.model_master = model_master
        self.dealer_threshold = dealer_threshold
        self.model_threshold = model_threshold

        self._norm_dealer_master = [normalize_text_for_matching(d) for d in dealer_master]
        self._norm_model_master = [normalize_text_for_matching(m) for m in model_master]

    def match_dealer_query(self, query: str) -> Optional[Tuple[str, float]]:
        """Match dealer name against catalog and return (name, similarity_score)."""
        cands = self.match_dealer_candidates(query, top_k=1)
        if cands:
            return cands[0].name, cands[0].similarity_score
        return None

    def match_dealer_candidates(self, query: str, top_k: int = 3) -> List[MatchCandidate]:
        """
        Rank dealer catalog items against query string.
        """
        if not query or not query.strip():
            return []

        norm_query = normalize_text_for_matching(query)
        candidates: List[MatchCandidate] = []

        for idx, (dealer, norm_dealer) in enumerate(zip(self.dealer_master, self._norm_dealer_master)):
            # Metric 1: Token Sort Ratio (word order invariant)
            ts_ratio = fuzz.token_sort_ratio(norm_query, norm_dealer)
            # Metric 2: Partial Ratio (substring matching)
            part_ratio = fuzz.partial_ratio(norm_query, norm_dealer)
            # Metric 3: Token Set Ratio (handles extra words like 'Authorized Dealer')
            tset_ratio = fuzz.token_set_ratio(norm_query, norm_dealer)

            composite = max(ts_ratio, (tset_ratio * 0.7 + part_ratio * 0.3))
            conf = min(1.0, max(0.0, composite / 100.0))

            if composite >= self.dealer_threshold:
                candidates.append(
                    MatchCandidate(
                        name=dealer,
                        similarity_score=round(composite, 2),
                        confidence=round(conf, 4),
                        match_type="fuzzy_token_sort",
                        metrics={
                            "token_sort_ratio": ts_ratio,
                            "partial_ratio": part_ratio,
                            "token_set_ratio": tset_ratio,
                        },
                    )
                )

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates[:top_k]

    def match_dealer_from_ocr(
        self,
        ocr_result: OCRResult,
        candidate_dealer: Optional[FieldValue] = None,
    ) -> Optional[FieldValue]:
        """
        Match dealer name from candidate line or scan OCR lines.
        """
        # 1. Candidate line if present
        if candidate_dealer and candidate_dealer.is_present():
            cands = self.match_dealer_candidates(str(candidate_dealer.value))
            if cands:
                top = cands[0]
                alt_list = [
                    {"name": c.name, "score": c.similarity_score, "confidence": c.confidence}
                    for c in cands[1:]
                ]
                return FieldValue(
                    value=top.name,
                    confidence=top.confidence,
                    source=FieldSource.FUZZY_MATCH,
                    source_text=str(candidate_dealer.value),
                    evidence=candidate_dealer.evidence or candidate_dealer.value,
                    bbox=candidate_dealer.bbox,
                    page=candidate_dealer.page,
                    method="fuzzy_candidate",
                    alternatives=alt_list,
                )

        # 2. Scan OCR lines for dealer mention
        all_cands: List[Tuple[MatchCandidate, OCRLine]] = []
        for line in ocr_result.lines:
            text = line.text.strip()
            if len(text) < 4:
                continue
            cands = self.match_dealer_candidates(text, top_k=1)
            if cands and cands[0].similarity_score >= 80.0:
                all_cands.append((cands[0], line))

        if all_cands:
            all_cands.sort(key=lambda x: x[0].similarity_score, reverse=True)
            best_cand, best_line = all_cands[0]
            return FieldValue(
                value=best_cand.name,
                confidence=best_cand.confidence,
                source=FieldSource.FUZZY_MATCH,
                source_text=best_line.text,
                evidence=best_line.text,
                bbox=list(best_line.bbox) if best_line.bbox else None,
                page=getattr(best_line, "page", 1),
                method="fuzzy_ocr_scan",
                alternatives=[],
            )

        return None

    def match_model_query(self, query: str) -> Optional[str]:
        """Exact substring / normalized match against model catalog."""
        if not query:
            return None
        norm_q = normalize_text_for_matching(query)
        for model, norm_model in zip(self.model_master, self._norm_model_master):
            if norm_model in norm_q or norm_q in norm_model:
                return model
        return None

    def match_model_candidates(self, query: str, top_k: int = 3) -> List[MatchCandidate]:
        """Rank model catalog items using token sort and partial ratios."""
        if not query:
            return []
        norm_q = normalize_text_for_matching(query)
        candidates: List[MatchCandidate] = []

        for model, norm_model in zip(self.model_master, self._norm_model_master):
            # Check exact substring first
            if norm_model == norm_q or norm_model in norm_q:
                candidates.append(
                    MatchCandidate(
                        name=model,
                        similarity_score=100.0,
                        confidence=0.98,
                        match_type="exact_normalized",
                        metrics={"exact_match": 1.0},
                    )
                )
                continue

            ts = fuzz.token_sort_ratio(norm_q, norm_model)
            pr = fuzz.partial_ratio(norm_q, norm_model)
            score = max(ts, pr)
            if score >= self.model_threshold:
                candidates.append(
                    MatchCandidate(
                        name=model,
                        similarity_score=round(score, 2),
                        confidence=round(score / 100.0 * 0.95, 4),
                        match_type="fuzzy_model",
                        metrics={"token_sort": ts, "partial_ratio": pr},
                    )
                )

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates[:top_k]

    def match_model_from_ocr(
        self,
        ocr_result: OCRResult,
        candidate_model: Optional[FieldValue] = None,
    ) -> Optional[FieldValue]:
        """Find model match from candidate line or across OCR lines."""
        # 1. Candidate line
        if candidate_model and candidate_model.is_present():
            cands = self.match_model_candidates(str(candidate_model.value))
            if cands:
                top = cands[0]
                alt_list = [
                    {"name": c.name, "score": c.similarity_score, "confidence": c.confidence}
                    for c in cands[1:]
                ]
                return FieldValue(
                    value=top.name,
                    confidence=top.confidence,
                    source=FieldSource.EXACT_MATCH if top.similarity_score >= 99.0 else FieldSource.FUZZY_MATCH,
                    source_text=str(candidate_model.value),
                    evidence=candidate_model.evidence or candidate_model.value,
                    bbox=candidate_model.bbox,
                    page=candidate_model.page,
                    method="model_candidate_match",
                    alternatives=alt_list,
                )

        # 2. Scan OCR lines
        all_cands: List[Tuple[MatchCandidate, OCRLine]] = []
        for line in ocr_result.lines:
            text = line.text.strip()
            if len(text) < 3:
                continue
            cands = self.match_model_candidates(text, top_k=1)
            if cands and cands[0].similarity_score >= 80.0:
                all_cands.append((cands[0], line))

        if all_cands:
            all_cands.sort(key=lambda x: x[0].similarity_score, reverse=True)
            best_cand, best_line = all_cands[0]
            return FieldValue(
                value=best_cand.name,
                confidence=best_cand.confidence,
                source=FieldSource.EXACT_MATCH if best_cand.similarity_score >= 99.0 else FieldSource.FUZZY_MATCH,
                source_text=best_line.text,
                evidence=best_line.text,
                bbox=list(best_line.bbox) if best_line.bbox else None,
                page=getattr(best_line, "page", 1),
                method="model_ocr_scan",
                alternatives=[],
            )

        return None


# Backward-compatibility alias
FuzzyMatcher = EntityMatcher
