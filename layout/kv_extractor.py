"""
Geometric Key-Value Pair Extractor with Multi-Field Line Splitting.

Extracts semantic Label -> Value relationships using 2D spatial reasoning:
- Inline multi-field line splitting (e.g. `Invoice No : SG/24-25/0789   Date : 16/05/2025`)
- Horizontal right-neighbor association
- Vertical below-neighbor association
- Respects table exclusion boundaries to prevent table headers from becoming scalar fields
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from docai.layout.spatial_index import BoundingBox, SpatialIndex
from docai.models.extraction_schema import FieldSource, FieldValue
from docai.schemas.schema_loader import DocumentSchema

logger = logging.getLogger(__name__)


@dataclass
class KeyValuePair:
    key_name: str
    label_text: str
    label_bbox: BoundingBox
    value_text: str
    value_bbox: BoundingBox
    alignment: str  # "inline", "right", or "below"
    layout_score: float
    page: int = 1


def clean_extracted_value(text: str) -> str:
    """Clean colons, hyphens, and whitespace from extracted values."""
    cleaned = text.strip()
    # Strip leading punctuation commonly following a label
    cleaned = re.sub(r"^[:\-\|\s]+", "", cleaned)
    # Strip trailing punctuation
    cleaned = re.sub(r"[:\-\|\s]+$", "", cleaned)
    return cleaned.strip()


class LayoutKVExtractor:
    """
    Spatial reasoning engine to extract key-value pairs from 2D OCR layout.
    """

    def __init__(
        self,
        anchors: Optional[Dict[str, List[Pattern]]] = None,
        schema: Optional[DocumentSchema] = None,
    ):
        self.anchors: Dict[str, List[Pattern]] = {}
        if schema:
            self.load_schema_anchors(schema)
        elif anchors:
            self.anchors = anchors
        else:
            try:
                from docai.schemas.schema_loader import get_schema_registry
                default_s = get_schema_registry().get_schema("tractor_invoice")
                if default_s:
                    self.load_schema_anchors(default_s)
            except Exception:
                pass
        self.spatial_index = SpatialIndex()

    def load_schema_anchors(self, schema: DocumentSchema) -> None:
        """Compile regex anchors for all fields defined in the schema."""
        self.anchors = {}
        for fname, fdef in schema.fields.items():
            patterns = fdef.get_compiled_aliases()
            if patterns:
                self.anchors[fname] = patterns

    def extract_key_values(
        self,
        ocr_lines: List[Any],
        page: int = 1,
        excluded_line_indices: Optional[Set[int]] = None,
    ) -> List[KeyValuePair]:
        """
        Scan OCR lines for label anchors and match spatially adjacent values.
        Supports multi-field inline splitting and table line exclusions.
        """
        kv_pairs: List[KeyValuePair] = []
        if not ocr_lines:
            return kv_pairs

        excluded = excluded_line_indices or set()
        active_lines = [line for idx, line in enumerate(ocr_lines) if idx not in excluded]

        self.spatial_index.set_items(active_lines)

        for line_idx, item in enumerate(ocr_lines):
            if line_idx in excluded:
                continue

            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue

            item_box = self.spatial_index.get_bbox(item)

            # Step 1: Detect all label anchor matches and their span offsets on this line
            line_matches: List[Tuple[int, int, str, str]] = []  # (start_char, end_char, key_name, label_text)

            for key_name, patterns in self.anchors.items():
                for pat in patterns:
                    for m in pat.finditer(text):
                        line_matches.append((m.start(), m.end(), key_name, m.group(0)))

            if line_matches:
                # Sort matches by start position on the line
                line_matches.sort(key=lambda x: x[0])

                # De-duplicate overlapping matches (keep earliest/longest)
                unique_matches: List[Tuple[int, int, str, str]] = []
                for m in line_matches:
                    if not unique_matches:
                        unique_matches.append(m)
                    else:
                        last = unique_matches[-1]
                        if m[0] >= last[1]:  # No overlap
                            unique_matches.append(m)

                # Step 2: Multi-Field Line Splitting
                # Segment line between successive labels
                for idx, (m_start, m_end, key_name, label_text) in enumerate(unique_matches):
                    # Value starts at m_end and terminates before next label starts (or end of line)
                    if idx + 1 < len(unique_matches):
                        val_end = unique_matches[idx + 1][0]
                    else:
                        val_end = len(text)

                    val_raw = text[m_end:val_end]
                    val_cleaned = clean_extracted_value(val_raw)

                    # Compute proportional bounding box for the value
                    line_len = max(1, len(text))
                    w = item_box.x1 - item_box.x0
                    sub_x0 = item_box.x0 + (m_end / line_len) * w
                    sub_x1 = item_box.x0 + (val_end / line_len) * w
                    val_bbox = BoundingBox(x0=sub_x0, y0=item_box.y0, x1=sub_x1, y1=item_box.y1)

                    if val_cleaned:
                        kv_pairs.append(
                            KeyValuePair(
                                key_name=key_name,
                                label_text=label_text.strip(),
                                label_bbox=item_box,
                                value_text=val_cleaned,
                                value_bbox=val_bbox,
                                alignment="inline",
                                layout_score=0.96,
                                page=page,
                            )
                        )
                        continue

                    # If value was empty inline and this was the only label, check right & below neighbors
                    if len(unique_matches) == 1:
                        # Check horizontal right neighbor
                        right_cand = self.spatial_index.find_right_neighbor(item, active_lines, max_h_dist=500.0)
                        if right_cand is not None:
                            cand_item, score = right_cand
                            cand_box = self.spatial_index.get_bbox(cand_item)
                            cand_text = clean_extracted_value(getattr(cand_item, "text", ""))
                            if cand_text:
                                kv_pairs.append(
                                    KeyValuePair(
                                        key_name=key_name,
                                        label_text=label_text.strip(),
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
                        below_cand = self.spatial_index.find_below_neighbor(item, active_lines, max_v_dist=120.0)
                        if below_cand is not None:
                            cand_item, score = below_cand
                            cand_box = self.spatial_index.get_bbox(cand_item)
                            cand_text = clean_extracted_value(getattr(cand_item, "text", ""))
                            if cand_text:
                                kv_pairs.append(
                                    KeyValuePair(
                                        key_name=key_name,
                                        label_text=label_text.strip(),
                                        label_bbox=item_box,
                                        value_text=cand_text,
                                        value_bbox=cand_box,
                                        alignment="below",
                                        layout_score=score,
                                        page=page,
                                    )
                                )

        return kv_pairs

    def extract_fields(
        self,
        ocr_lines: List[Any],
        page: int = 1,
        excluded_line_indices: Optional[Set[int]] = None,
    ) -> Dict[str, FieldValue]:
        """
        Convert extracted KeyValuePairs into structured FieldValue objects.
        """
        kv_pairs = self.extract_key_values(ocr_lines, page=page, excluded_line_indices=excluded_line_indices)
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

    def extract_generic_key_values(
        self,
        ocr_lines: List[Any],
        page: int = 1,
        excluded_line_indices: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract arbitrary unknown label -> value pairs for generic/unclassified documents.
        Looks for standard ':' or '-' delimiters on lines not in tables.
        """
        generic_pairs = []
        excluded = excluded_line_indices or set()

        for idx, line in enumerate(ocr_lines):
            if idx in excluded:
                continue
            text = (getattr(line, "text", "") or "").strip()
            if ":" in text:
                parts = text.split(":", 1)
                label = parts[0].strip()
                val = clean_extracted_value(parts[1])
                if label and val:
                    b_list = [0.0, 0.0, 0.0, 0.0]
                    if hasattr(line, "bbox"):
                        b = line.bbox
                        if hasattr(b, "to_list"):
                            b_list = b.to_list()
                        elif isinstance(b, (list, tuple)):
                            b_list = [float(x) for x in b]
                    generic_pairs.append({
                        "label": label,
                        "value": val,
                        "confidence": getattr(line, "confidence", 0.90),
                        "bbox": b_list,
                        "page": page,
                    })
        return generic_pairs

