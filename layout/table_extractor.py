"""
Generic Table Extractor for Multi-Document AI.

Extracts structured row/cell items from detected table regions with bounding boxes,
supporting schema-driven column keys or generic dynamic tabular parsing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from docai.layout.table_detector import TableRegion
from docai.models.extraction_schema import BoundingBox
from docai.ocr.paddleocr_engine import OCRLine
from docai.schemas.schema_loader import TableDefinition

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    key: str
    value: Any
    raw_text: str
    confidence: float
    bbox: Optional[BoundingBox] = None


@dataclass
class TableRow:
    row_index: int
    cells: Dict[str, TableCell] = field(default_factory=dict)
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: cell.value for k, cell in self.cells.items()}


@dataclass
class ExtractedTable:
    name: str
    rows: List[TableRow] = field(default_factory=list)
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.rows]


class TableExtractor:
    """
    Extracts structured rows and columns from detected TableRegions.
    """

    # Common multi-column invoice / receipt line item pattern:
    # Optional S.No (1-3 digits) + Item Description + Quantity + Unit Price + Total Amount
    LINE_ITEM_REGEX = re.compile(
        r"^(?:(?P<sno>\d{1,4})\s+)?(?P<item>[A-Za-z0-9\s\.\&\-\(\)\/]+?)\s+"
        r"(?P<qty>\d+(?:\.\d+)?)\s+"
        r"(?:(?:rs\.?|inr|₹)\s*)?(?P<unit_price>[\d,]+(?:\.\d{1,2})?)\s+"
        r"(?:(?:rs\.?|inr|₹)\s*)?(?P<amount>[\d,]+(?:\.\d{1,2})?)$",
        re.IGNORECASE,
    )

    def extract_table(self, region: TableRegion) -> ExtractedTable:
        rows: List[TableRow] = []
        conf_scores: List[float] = []

        table_def = region.table_def
        expected_cols = [c.key for c in table_def.columns] if table_def and table_def.columns else []

        for idx, line in enumerate(region.body_lines):
            text = line.text.strip()
            if not text:
                continue

            # Attempt regex extraction for standard itemized line
            match = self.LINE_ITEM_REGEX.match(text)
            if match:
                gd = match.groupdict()
                row_cells: Dict[str, TableCell] = {}

                # S.No
                sno_val = int(gd["sno"]) if gd.get("sno") else (idx + 1)
                row_cells["sno"] = TableCell(
                    key="sno",
                    value=sno_val,
                    raw_text=str(gd.get("sno") or sno_val),
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

                # Item Description
                item_name = gd["item"].strip()
                row_cells["item"] = TableCell(
                    key="item",
                    value=item_name,
                    raw_text=item_name,
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

                # Quantity
                qty_val = float(gd["qty"])
                row_cells["qty"] = TableCell(
                    key="qty",
                    value=qty_val if qty_val % 1 != 0 else int(qty_val),
                    raw_text=gd["qty"],
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

                # Unit Price
                u_str = gd["unit_price"].replace(",", "")
                unit_price_val = float(u_str)
                row_cells["unit_price"] = TableCell(
                    key="unit_price",
                    value=unit_price_val,
                    raw_text=gd["unit_price"],
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

                # Total Amount
                a_str = gd["amount"].replace(",", "")
                amount_val = float(a_str)
                row_cells["amount"] = TableCell(
                    key="amount",
                    value=amount_val,
                    raw_text=gd["amount"],
                    confidence=line.confidence,
                    bbox=line.bbox,
                )

                rows.append(TableRow(row_index=len(rows) + 1, cells=row_cells, confidence=line.confidence, bbox=line.bbox))
                conf_scores.append(line.confidence)
            else:
                # Fallback: token-based column splitting
                tokens = re.split(r"\s{2,}|\t|\|", text)
                if len(tokens) >= 3:
                    row_cells = {}
                    for c_idx, tok in enumerate(tokens):
                        col_key = expected_cols[c_idx] if c_idx < len(expected_cols) else f"col_{c_idx+1}"
                        tok_clean = tok.strip()
                        # Try parsing number
                        val: Any = tok_clean
                        clean_num = tok_clean.replace(",", "").replace("₹", "").replace("Rs.", "")
                        try:
                            val = float(clean_num)
                        except ValueError:
                            pass

                        row_cells[col_key] = TableCell(
                            key=col_key,
                            value=val,
                            raw_text=tok_clean,
                            confidence=line.confidence,
                            bbox=line.bbox,
                        )

                    rows.append(TableRow(row_index=len(rows) + 1, cells=row_cells, confidence=line.confidence, bbox=line.bbox))
                    conf_scores.append(line.confidence)

        avg_conf = sum(conf_scores) / max(1, len(conf_scores)) if conf_scores else 0.85
        hdr_bbox = region.header_line.bbox
        if hasattr(hdr_bbox, "x0"):
            hx0, hx1 = float(hdr_bbox.x0), float(hdr_bbox.x1)
        elif isinstance(hdr_bbox, (list, tuple)) and len(hdr_bbox) >= 4:
            hx0, hx1 = float(hdr_bbox[0]), float(hdr_bbox[2])
        else:
            hx0, hx1 = 0.0, 0.0
        t_bbox = BoundingBox(
            x0=hx0,
            y0=region.y_top,
            x1=hx1,
            y1=region.y_bottom,
        )
        return ExtractedTable(name=region.name, rows=rows, confidence=round(avg_conf, 4), bbox=t_bbox)
