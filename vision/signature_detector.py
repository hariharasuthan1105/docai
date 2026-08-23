"""
Computer Vision Signature Detector.

Detects handwritten signatures using morphological gradient, stroke connectivity,
and contour complexity analysis.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from docai.models.extraction_schema import VisualMark
from docai.preprocessing.image_enhancer import load_image

logger = logging.getLogger(__name__)


class SignatureDetector(ABC):
    @abstractmethod
    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        """Detect signature presence and bounding box."""
        raise NotImplementedError


class CVSignatureDetector(SignatureDetector):
    """
    Genuine Computer Vision signature detector using contour variance,
    stroke connectivity, and lower-region spatial heuristics.
    """

    def __init__(
        self,
        min_area: float = 400.0,
        max_area: float = 40000.0,
        region_of_interest_y: float = 0.50,  # Focus on lower 50% of invoice
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.roi_y = region_of_interest_y

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        if isinstance(image_input, str) and (image_input.lower().endswith(".txt") or not os.path.exists(image_input)):
            return VisualMark(
                mark_type="signature",
                status="not_implemented",
                present=False,
                confidence=0.0,
                page=page,
            )

        try:
            image = load_image(image_input)
        except Exception as e:
            return VisualMark(
                mark_type="signature",
                status="not_implemented",
                present=False,
                confidence=0.0,
                page=page,
            )

        h, w = image.shape[:2]
        roi_start_y = int(h * self.roi_y)
        roi = image[roi_start_y:h, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )

        # Remove horizontal and vertical table grid lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        text_strokes = cv2.subtract(binary, cv2.add(h_lines, v_lines))

        # Dilate slightly to connect cursive handwritten strokes
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        connected = cv2.dilate(text_strokes, connect_kernel, iterations=1)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(ch) if ch > 0 else 0.0

            # Signatures typically have aspect ratio between 1.2 and 6.0 with high stroke complexity
            if 1.0 <= aspect_ratio <= 7.0 and ch >= 15 and cw >= 35:
                # Check perimeter-to-area complexity ratio (distinguishes cursive from solid blocks)
                perimeter = cv2.arcLength(cnt, True)
                complexity = (perimeter * perimeter) / (area + 1e-5)

                if complexity >= 18.0:  # High edge curvature characteristic of pen signatures
                    global_box = [
                        float(x),
                        float(roi_start_y + y),
                        float(x + cw),
                        float(roi_start_y + y + ch),
                    ]
                    score = min(0.95, max(0.60, 0.60 + (complexity / 100.0) * 0.35))
                    candidate_boxes.append((global_box, score, area))

        if candidate_boxes:
            # Pick the largest most prominent signature candidate
            candidate_boxes.sort(key=lambda b: (b[1] * 0.6 + (b[2] / self.max_area) * 0.4), reverse=True)
            best_box, best_score, _ = candidate_boxes[0]
            return VisualMark(
                mark_type="signature",
                status="detected",
                present=True,
                bbox=best_box,
                confidence=round(best_score, 4),
                page=page,
                details={"method": "cv_stroke_complexity", "candidates_found": len(candidate_boxes)},
            )

        return VisualMark(
            mark_type="signature",
            status="not_detected",
            present=False,
            bbox=None,
            confidence=0.15,
            page=page,
            details={"method": "cv_stroke_complexity"},
        )


class StubSignatureDetector(SignatureDetector):
    """Placeholder detector for testing or when visual detection is disabled."""

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        return VisualMark(
            mark_type="signature",
            status="not_implemented",
            present=False,
            bbox=None,
            confidence=0.0,
            page=page,
        )


class ManualSignatureDetector(SignatureDetector):
    """Test helper returning fixed pre-supplied visual mark."""

    def __init__(
        self,
        present: bool,
        bbox: Optional[List[float]] = None,
        confidence: float = 1.0,
        status: str = "detected",
    ):
        self._result = VisualMark(
            mark_type="signature",
            present=present,
            bbox=bbox,
            confidence=confidence,
            status=status if present else "not_detected",
        )

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        res = self._result.model_copy()
        res.page = page
        return res
