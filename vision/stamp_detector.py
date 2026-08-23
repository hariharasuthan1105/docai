"""
Computer Vision Rubber Stamp & Seal Detector.

Detects official dealer / bank stamps using HSV color segmentation
(blue, purple, and red ink stamps) and circular/elliptical contour geometry.
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


class StampDetector(ABC):
    @abstractmethod
    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        """Detect official stamp/seal presence and bounding box."""
        raise NotImplementedError


class CVStampDetector(StampDetector):
    """
    Genuine Computer Vision stamp detector using color segmentation
    and morphological contour analysis.
    """

    def __init__(
        self,
        min_stamp_area: float = 600.0,
        max_stamp_area: float = 120000.0,
    ):
        self.min_area = min_stamp_area
        self.max_area = max_stamp_area

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        if isinstance(image_input, str) and (image_input.lower().endswith(".txt") or not os.path.exists(image_input)):
            return VisualMark(
                mark_type="stamp",
                status="not_implemented",
                present=False,
                confidence=0.0,
                page=page,
            )

        try:
            image = load_image(image_input)
        except Exception as e:
            return VisualMark(
                mark_type="stamp",
                status="not_implemented",
                present=False,
                confidence=0.0,
                page=page,
            )

        if len(image.shape) != 3 or image.shape[2] != 3:
            # Grayscale images don't have color ink signals
            return VisualMark(
                mark_type="stamp",
                status="not_detected",
                present=False,
                confidence=0.10,
                page=page,
                details={"reason": "grayscale_image_no_color_ink"},
            )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask 1: Blue stamp ink (Hue ~95 to 135)
        lower_blue = np.array([95, 45, 45])
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Mask 2: Purple / Magenta stamp ink (Hue ~135 to 165)
        lower_purple = np.array([135, 45, 45])
        upper_purple = np.array([165, 255, 255])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        # Mask 3: Red stamp ink (Hue 0-10 and 165-180)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

        # Combine all stamp ink masks
        combined_ink_mask = cv2.add(cv2.add(mask_blue, mask_purple), mask_red)

        # Morphological closing to connect stamp borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined_ink_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0.0

            # Stamps are generally round, oval, or rectangular (aspect ratio 0.6 to 2.5)
            if 0.5 <= aspect_ratio <= 2.5 and w >= 25 and h >= 25:
                # Calculate ink fill density inside bounding box
                box_roi = combined_ink_mask[y : y + h, x : x + w]
                ink_pixel_count = cv2.countNonZero(box_roi)
                box_area = w * h
                density = ink_pixel_count / float(box_area) if box_area > 0 else 0.0

                if density >= 0.08:  # Minimum 8% color ink density
                    # Circularity metric
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
                    conf = min(0.96, max(0.65, 0.65 + density * 0.4 + (circularity if circularity <= 1.0 else 0.5) * 0.2))

                    candidates.append(([float(x), float(y), float(x + w), float(y + h)], conf, area))

        if candidates:
            candidates.sort(key=lambda c: (c[1] * 0.5 + (c[2] / self.max_area) * 0.5), reverse=True)
            best_box, best_conf, _ = candidates[0]
            return VisualMark(
                mark_type="stamp",
                status="detected",
                present=True,
                bbox=best_box,
                confidence=round(best_conf, 4),
                page=page,
                details={"method": "hsv_color_ink_contour", "stamps_found": len(candidates)},
            )

        return VisualMark(
            mark_type="stamp",
            status="not_detected",
            present=False,
            bbox=None,
            confidence=0.15,
            page=page,
            details={"method": "hsv_color_ink_contour"},
        )


class StubStampDetector(StampDetector):
    """Placeholder detector for testing or when visual detection is disabled."""

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        return VisualMark(
            mark_type="stamp",
            status="not_implemented",
            present=False,
            bbox=None,
            confidence=0.0,
            page=page,
        )


class ManualStampDetector(StampDetector):
    """Test helper returning fixed pre-supplied visual mark."""

    def __init__(
        self,
        present: bool,
        bbox: Optional[List[float]] = None,
        confidence: float = 1.0,
        status: str = "detected",
    ):
        self._result = VisualMark(
            mark_type="stamp",
            present=present,
            bbox=bbox,
            confidence=confidence,
            status=status if present else "not_detected",
        )

    def detect(self, image_input: Union[str, np.ndarray], page: int = 1) -> VisualMark:
        res = self._result.model_copy()
        res.page = page
        return res
