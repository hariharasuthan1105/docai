"""
Table Boundary and Header Detector for Document AI.

Identifies multi-column tabular regions, header lines, and column coordinate intervals.
Marks tabular lines so that scalar field extraction avoids false positives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine, OCRResult
from docai.schemas.schema_loader import TableDefinition

logger = logging.getLogger(__name__)

# Common table header keywords across invoice / receipt domains
STANDARD_HEADER_KEYWORDS = [
    "s.no",
    "sno",
    "sl no",
    "item",
    "item name",
    "description",
    "particulars",
    "qty",
    "quantity",
    "rate",
    "unit price",
    "price",
    "amount",
    "total",
    "hsn",
    "tax",
]


@dataclass
class ColumnBoundary:
    key: str
    name: str
    x_min: float
    x_max: float


@dataclass
class TableRegion:
    name: str
    header_line: OCRLine
    header_index: int
    y_top: float
    y_bottom: float
    columns: List[ColumnBoundary] = field(default_factory=list)
    body_lines: List[OCRLine] = field(default_factory=list)
    table_def: Optional[TableDefinition] = None


def get_bbox_coords(bbox: Any) -> Tuple[float, float, float, float]:
    if hasattr(bbox, "x0") and hasattr(bbox, "y0"):
        return float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return 0.0, 0.0, 0.0, 0.0


class TableDetector:
    """
    Identifies table header rows, column positions, and table bounding regions.
    """

    def __init__(self, table_defs: Optional[Dict[str, TableDefinition]] = None):
        self.table_defs = table_defs or {}

    def detect_tables(self, ocr_result: OCRResult) -> List[TableRegion]:
        lines = ocr_result.lines
        if not lines:
            return []

        detected_regions: List[TableRegion] = []

        for i, line in enumerate(lines):
            text_lower = line.text.lower().strip()

            # Check if line matches any schema-defined table header or standard table header
            matching_table_def = None
            matched_cols = 0

            # 1. Try schema table definitions
            for tname, tdef in self.table_defs.items():
                hits = sum(1 for kw in tdef.header_keywords if kw.lower() in text_lower)
                if hits >= 2:
                    matching_table_def = tdef
                    matched_cols = hits
                    break

            # 2. Try standard header keywords if no schema table matched
            if not matching_table_def:
                std_hits = sum(1 for kw in STANDARD_HEADER_KEYWORDS if kw.lower() in text_lower)
                if std_hits >= 3:
                    matched_cols = std_hits

            if matched_cols >= 2:
                # Found table header candidate
                x0, y0, x1, y1 = get_bbox_coords(line.bbox)
                y_top = y0
                header_line = line
                header_idx = i

                # Detect columns from header text/word positions
                columns = self._estimate_column_boundaries(header_line, matching_table_def)

                # Determine table body lines until totals or section footer
                body_lines = []
                y_bottom = y1

                for j in range(i + 1, len(lines)):
                    candidate = lines[j]
                    cand_text = candidate.text.lower().strip()

                    # Stop conditions: totals, footer, signature/stamp markers, blank
                    if re.search(r"^(?:subtotal|sub\s*total|grand\s*total|taxable|cgst|sgst|discount|total|thank\s*you)\b", cand_text):
                        break
                    if "authorized signatory" in cand_text or "for " in cand_text:
                        break

                    # If line starts with a number (S.No.) or matches column alignment
                    body_lines.append(candidate)
                    _, _, _, cand_y1 = get_bbox_coords(candidate.bbox)
                    y_bottom = max(y_bottom, cand_y1)

                region = TableRegion(
                    name=matching_table_def.name if matching_table_def else "generic_table",
                    header_line=header_line,
                    header_index=header_idx,
                    y_top=y_top,
                    y_bottom=y_bottom,
                    columns=columns,
                    body_lines=body_lines,
                    table_def=matching_table_def,
                )
                detected_regions.append(region)

        return detected_regions

    def _estimate_column_boundaries(
        self, header_line: OCRLine, table_def: Optional[TableDefinition]
    ) -> List[ColumnBoundary]:
        """Estimate horizontal column intervals across the header line."""
        text = header_line.text
        x0, y0, x1, y1 = get_bbox_coords(header_line.bbox)
        line_width = max(1.0, x1 - x0)

        # Split header line into tokens or keywords
        tokens = re.split(r"\s{2,}|\t|\|", text)
        if len(tokens) <= 1:
            tokens = [t for t in text.split(" ") if t.strip()]

        columns = []

        if table_def and table_def.columns:
            # Map schema columns across header width proportionally
            num_cols = len(table_def.columns)
            col_w = line_width / max(1, num_cols)
            for idx, col_def in enumerate(table_def.columns):
                c_x0 = x0 + idx * col_w
                c_x1 = c_x0 + col_w
                columns.append(ColumnBoundary(key=col_def.key, name=col_def.key, x_min=c_x0, x_max=c_x1))
        else:
            # Generic column division from tokens
            num_tokens = len(tokens)
            col_w = line_width / max(1, num_tokens)
            for idx, tok in enumerate(tokens):
                c_x0 = x0 + idx * col_w
                c_x1 = c_x0 + col_w
                columns.append(ColumnBoundary(key=f"col_{idx+1}", name=tok.strip(), x_min=c_x0, x_max=c_x1))

        return columns

    def get_table_line_indices(self, ocr_result: OCRResult, tables: List[TableRegion]) -> set[int]:
        """Return set of line indices that belong to detected table regions."""
        table_line_indices = set()
        for t in tables:
            table_line_indices.add(t.header_index)
            for bline in t.body_lines:
                for idx, oline in enumerate(ocr_result.lines):
                    if oline is bline or (oline.bbox == bline.bbox and oline.text == bline.text):
                        table_line_indices.add(idx)
        return table_line_indices
