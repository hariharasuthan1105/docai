"""
2D Spatial Index and Geometry Engine for Document OCR Layouts.

Enables reading order reconstruction, proximity queries, and neighbor searches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple_or_list(cls, coords: Sequence[float]) -> BoundingBox:
        if len(coords) < 4:
            return cls(0.0, 0.0, 0.0, 0.0)
        return cls(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection over Union (IoU)."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)

        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter_area = iw * ih

        union_area = (self.width * self.height) + (other.width * other.height) - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0

    def vertical_overlap(self, other: BoundingBox) -> float:
        """Compute vertical overlap fraction with another box."""
        iy0 = max(self.y0, other.y0)
        iy1 = min(self.y1, other.y1)
        overlap_h = max(0.0, iy1 - iy0)
        min_h = min(self.height, other.height)
        return overlap_h / min_h if min_h > 0 else 0.0

    def horizontal_overlap(self, other: BoundingBox) -> float:
        """Compute horizontal overlap fraction with another box."""
        ix0 = max(self.x0, other.x0)
        ix1 = min(self.x1, other.x1)
        overlap_w = max(0.0, ix1 - ix0)
        min_w = min(self.width, other.width)
        return overlap_w / min_w if min_w > 0 else 0.0


class SpatialIndex:
    """
    Spatial querying and reading order engine over OCR items.
    """

    def __init__(self, items: Optional[List[Any]] = None):
        self.items: List[Any] = items or []

    def set_items(self, items: List[Any]) -> None:
        self.items = items

    @staticmethod
    def get_bbox(item: Any) -> BoundingBox:
        if hasattr(item, "bbox"):
            bbox_val = item.bbox
            if isinstance(bbox_val, (tuple, list)):
                return BoundingBox.from_tuple_or_list(bbox_val)
            elif isinstance(bbox_val, BoundingBox):
                return bbox_val
        return BoundingBox(0.0, 0.0, 0.0, 0.0)

    def sort_reading_order(self, items: Optional[List[Any]] = None, y_tolerance_ratio: float = 0.5) -> List[Any]:
        """
        Sort OCR elements into natural top-to-bottom, left-to-right reading order
        by clustering items into horizontal text lines.
        """
        item_list = items if items is not None else self.items
        if not item_list:
            return []

        # Sort primarily by y0
        sorted_by_y = sorted(item_list, key=lambda it: self.get_bbox(it).y0)

        lines: List[List[Any]] = []
        for item in sorted_by_y:
            item_box = self.get_bbox(item)
            placed = False
            for line in lines:
                ref_box = self.get_bbox(line[0])
                avg_h = (ref_box.height + item_box.height) / 2.0
                y_tol = max(8.0, avg_h * y_tolerance_ratio)

                # Check if center_y is within tolerance or if vertical overlap is high
                if abs(item_box.center_y - ref_box.center_y) <= y_tol or item_box.vertical_overlap(ref_box) > 0.4:
                    line.append(item)
                    placed = True
                    break

            if not placed:
                lines.append([item])

        # Sort each horizontal line left to right, then flatten
        result: List[Any] = []
        for line in lines:
            sorted_line = sorted(line, key=lambda it: self.get_bbox(it).x0)
            result.extend(sorted_line)

        return result

    def find_right_neighbor(
        self,
        target_item: Any,
        candidates: Optional[List[Any]] = None,
        max_h_dist: float = 600.0,
        min_v_overlap: float = 0.3,
    ) -> Optional[Tuple[Any, float]]:
        """
        Find the nearest item to the right of `target_item` sharing horizontal baseline.
        Returns (item, distance_score).
        """
        target_box = self.get_bbox(target_item)
        cand_list = candidates if candidates is not None else self.items

        best_cand = None
        min_dist = float("inf")

        for cand in cand_list:
            if cand is target_item:
                continue
            cand_box = self.get_bbox(cand)

            # Must be strictly to the right
            h_dist = cand_box.x0 - target_box.x1
            if h_dist < -10.0 or h_dist > max_h_dist:
                continue

            # Must have vertical overlap or close vertical center
            v_overlap = target_box.vertical_overlap(cand_box)
            v_diff = abs(target_box.center_y - cand_box.center_y)
            avg_h = (target_box.height + cand_box.height) / 2.0

            if v_overlap >= min_v_overlap or v_diff <= max(10.0, avg_h * 0.6):
                # Euclidean distance between right-edge of target and left-edge of cand
                dist = math.hypot(max(0.0, h_dist), v_diff)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand

        if best_cand is not None:
            score = max(0.5, 1.0 - (min_dist / max_h_dist))
            return best_cand, score
        return None

    def find_below_neighbor(
        self,
        target_item: Any,
        candidates: Optional[List[Any]] = None,
        max_v_dist: float = 200.0,
        min_h_overlap: float = 0.2,
    ) -> Optional[Tuple[Any, float]]:
        """
        Find the nearest item directly below `target_item`.
        Returns (item, distance_score).
        """
        target_box = self.get_bbox(target_item)
        cand_list = candidates if candidates is not None else self.items

        best_cand = None
        min_dist = float("inf")

        for cand in cand_list:
            if cand is target_item:
                continue
            cand_box = self.get_bbox(cand)

            # Must be below
            v_dist = cand_box.y0 - target_box.y1
            if v_dist < -5.0 or v_dist > max_v_dist:
                continue

            # Must have horizontal overlap or close horizontal alignment
            h_overlap = target_box.horizontal_overlap(cand_box)
            h_diff = abs(target_box.x0 - cand_box.x0)

            if h_overlap >= min_h_overlap or h_diff <= 100.0:
                dist = math.hypot(h_diff * 0.5, max(0.0, v_dist))
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand

        if best_cand is not None:
            score = max(0.5, 1.0 - (min_dist / max_v_dist))
            return best_cand, score
        return None
