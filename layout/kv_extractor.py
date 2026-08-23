"""
Geometric Key-Value Pair Extractor.

Extracts semantic Label -> Value relationships using 2D spatial reasoning:
- Horizontal right-neighbor association (e.g. `Horse Power: | 50 HP`)
- Vertical below-neighbor association (e.g. `Total (₹)` on line 1, `732,780.00` below)
- Table grid cell association
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

from docai.layout.spatial_index import BoundingBox, SpatialIndex
from docai.models.extraction_schema import FieldSource, FieldValue


@dataclass
class KeyValuePair:
    key_name: str
    label_text: str
    label_bbox: BoundingBox
    value_text: str
    value_bbox: BoundingBox
    alignment: str  # "right" or "below"
    layout_score: float
    page: int = 1


# Standard semantic anchor regexes for document labels
LABEL_ANCHORS: Dict[str, List[Pattern]] = {
    "horse_power": [
        re.compile(r"^(?:horse\s*power|engine\s*power|hp|power)\s*[:\-]?", re.IGNORECASE),
    ],
    "asset_cost": [
        re.compile(r"^(?:asset\s*cost|total\s*cost|grand\s*total|invoice\s*total|net\s*total|total\s*amount|total\s*(?:\([^)]*\))?|amount\s*payable)\s*[:\-]?", re.IGNORECASE),
        re.compile(r"^(?:sub\s*total|base\s*price)\s*[:\-]?", re.IGNORECASE),
    ],
    "dealer_name": [
        re.compile(r"^(?:authorised\s*dealer|authorized\s*dealer|dealer\s*name|dealer|m/s\.?)\s*[:\-]?", re.IGNORECASE),
    ],
    "model_name": [
        re.compile(r"^(?:model\s*name|tractor\s*model|model|item\s*name|description\s*of\s*goods)\s*[:\-]?", re.IGNORECASE),
    ],
    "invoice_number": [
        re.compile(r"^(?:invoice\s*(?:no\.?|num\.?|number|#)|bill\s*(?:no\.?|number)|inv\s*no\.?)\s*[:\-]?", re.IGNORECASE),
    ],
    "invoice_date": [
        re.compile(r"^(?:invoice\s*date|bill\s*date|dated|date)\s*[:\-]?", re.IGNORECASE),
    ],
    "customer_name": [
        re.compile(r"^(?:customer\s*name|buyer\s*name|buyer|bill\s*to|purchaser|sold\s*to)\s*[:\-]?", re.IGNORECASE),
    ],
    "customer_address": [
        re.compile(r"^(?:customer\s*address|buyer\s*address|address)\s*[:\-]?", re.IGNORECASE),
    ],
    "phone_number": [
        re.compile(r"^(?:phone\s*(?:no\.?|number)?|mobile\s*(?:no\.?|number)?|contact\s*no\.?|tel\.?)\s*[:\-]?", re.IGNORECASE),
    ],
    "registration_number": [
        re.compile(r"^(?:registration\s*no\.?|reg\s*no\.?|chassis\s*no\.?)\s*[:\-]?", re.IGNORECASE),
    ],
    "serial_number": [
        re.compile(r"^(?:serial\s*no\.?|sl\.?\s*no\.?|engine\s*no\.?|tractor\s*serial\s*no\.?)\s*[:\-]?", re.IGNORECASE),
    ],
}


class LayoutKVExtractor:
    """
    Spatial reasoning engine to extract key-value pairs from 2D OCR layout.
    """

    def __init__(self, anchors: Optional[Dict[str, List[Pattern]]] = None):
        self.anchors = anchors or LABEL_ANCHORS
        self.spatial_index = SpatialIndex()

    def extract_key_values(self, ocr_lines: List[Any], page: int = 1) -> List[KeyValuePair]:
        """
        Scan OCR lines for label anchors and match spatially adjacent values.
        """
        kv_pairs: List[KeyValuePair] = []
        if not ocr_lines:
            return kv_pairs

        self.spatial_index.set_items(ocr_lines)

        for item in ocr_lines:
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue

            item_box = self.spatial_index.get_bbox(item)

            for key_name, patterns in self.anchors.items():
                matched_label = False
                label_match_str = ""

                for pat in patterns:
                    m = pat.search(text)
                    if m:
                        matched_label = True
                        label_match_str = m.group(0).strip()
                        break

                if not matched_label:
                    continue

                # Check if the value is already present inline (e.g. "Horse Power: 50 HP")
                remaining_text = text[len(label_match_str) :].strip().lstrip(":- ").strip()
                if len(remaining_text) >= 1:
                    kv_pairs.append(
                        KeyValuePair(
                            key_name=key_name,
                            label_text=label_match_str,
                            label_bbox=item_box,
                            value_text=remaining_text,
                            value_bbox=item_box,
                            alignment="inline",
                            layout_score=0.95,
                            page=page,
                        )
                    )
                    continue

                # Check horizontal right neighbor
                right_cand = self.spatial_index.find_right_neighbor(item, ocr_lines, max_h_dist=500.0)
                if right_cand is not None:
                    cand_item, score = right_cand
                    cand_box = self.spatial_index.get_bbox(cand_item)
                    cand_text = getattr(cand_item, "text", "").strip()
                    if cand_text:
                        kv_pairs.append(
                            KeyValuePair(
                                key_name=key_name,
                                label_text=label_match_str or text,
                                label_bbox=item_box,
                                value_text=cand_text,
                                value_bbox=cand_box,
                                alignment="right",
                                layout_score=score,
                                page=page,
                            )
                        )
                        continue

                # Check vertical below neighbor
                below_cand = self.spatial_index.find_below_neighbor(item, ocr_lines, max_v_dist=120.0)
                if below_cand is not None:
                    cand_item, score = below_cand
                    cand_box = self.spatial_index.get_bbox(cand_item)
                    cand_text = getattr(cand_item, "text", "").strip()
                    if cand_text:
                        kv_pairs.append(
                            KeyValuePair(
                                key_name=key_name,
                                label_text=label_match_str or text,
                                label_bbox=item_box,
                                value_text=cand_text,
                                value_bbox=cand_box,
                                alignment="below",
                                layout_score=score,
                                page=page,
                            )
                        )

        return kv_pairs

    def extract_fields(self, ocr_lines: List[Any], page: int = 1) -> Dict[str, FieldValue]:
        """
        Convert extracted KeyValuePairs into structured FieldValue objects.
        """
        kv_pairs = self.extract_key_values(ocr_lines, page=page)
        fields: Dict[str, FieldValue] = {}

        for kv in kv_pairs:
            # If field already found with higher layout score, keep highest
            if kv.key_name in fields and fields[kv.key_name].confidence >= kv.layout_score:
                continue

            fields[kv.key_name] = FieldValue(
                value=kv.value_text,
                confidence=round(kv.layout_score, 4),
                source=FieldSource.LAYOUT_KV,
                source_text=f"{kv.label_text} -> {kv.value_text} ({kv.alignment})",
                evidence=kv.value_text,
                bbox=kv.value_bbox.to_list(),
                page=kv.page,
                method=f"layout_kv_{kv.alignment}",
            )

        return fields
