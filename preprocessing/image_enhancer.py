"""
Image Enhancer for Document AI.

Implements intelligent computer vision preprocessing:
1. Skew angle detection & correction (deskewing)
2. 90/180/270 degree orientation detection & correction
3. Adaptive contrast enhancement (CLAHE) for faint or low-contrast scans
4. Edge-preserving bilateral denoising for noisy/grainy captures
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """Load image from path or return existing array."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Unable to read image at path: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError(f"Expected file path str or np.ndarray, got {type(image_input)}")


def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect slight skew angle (-45 to +45 degrees) using minAreaRect on text contours.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Otsu thresholding with inversion
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect words into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of text lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        # OpenCV minAreaRect angle normalization
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    # Robust median skew
    median_angle = float(np.median(angles))
    return median_angle if abs(median_angle) >= 0.5 else 0.0


def correct_skew(image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Rotate image to correct detected skew angle.
    """
    if angle is None:
        angle = detect_skew_angle(image)

    if abs(angle) < 0.5:
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def detect_orientation_angle(image: np.ndarray) -> int:
    """
    Check if document is rotated by 90, 180, or 270 degrees using horizontal vs vertical projection variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Standard reading text has much higher variance in horizontal projection (lines) than vertical
    h_proj = np.sum(binary, axis=1)
    v_proj = np.sum(binary, axis=0)

    h_var = float(np.var(h_proj))
    v_var = float(np.var(v_proj))

    # If vertical variance is significantly higher than horizontal, the text lines are likely vertical (90 or 270 deg)
    if v_var > h_var * 1.6:
        return 90
    return 0


def enhance_contrast_adaptive(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for faint or shadowed scans.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        enhanced_lab = cv2.merge(lab_planes)
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def denoise_document(image: np.ndarray, diameter: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to remove scan noise while keeping text edges sharp.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def preprocess_document_image(
    image_input: Union[str, np.ndarray],
    enable_deskew: bool = True,
    enable_clahe: bool = True,
    enable_denoise: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Intelligent end-to-end preprocessing pipeline.
    Preserves original document structure while optimizing for OCR quality.
    """
    image = load_image(image_input)
    metadata: Dict[str, Any] = {
        "original_shape": list(image.shape),
        "skew_angle_corrected": 0.0,
        "orientation_angle_corrected": 0,
        "clahe_applied": False,
        "denoise_applied": False,
    }

    # 1. Orientation check
    orient_angle = detect_orientation_angle(image)
    if orient_angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        metadata["orientation_angle_corrected"] = 90
    elif orient_angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        metadata["orientation_angle_corrected"] = 270

    # 2. Deskew check
    if enable_deskew:
        skew_angle = detect_skew_angle(image)
        if abs(skew_angle) >= 0.5:
            image, corrected_angle = correct_skew(image, skew_angle)
            metadata["skew_angle_corrected"] = round(corrected_angle, 2)

    # 3. Check contrast level
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    std_dev = float(np.std(gray))
    if enable_clahe and std_dev < 60.0:  # low contrast image
        image = enhance_contrast_adaptive(image)
        metadata["clahe_applied"] = True

    # 4. Check noise level (Laplacian variance of blur)
    if enable_denoise:
        image = denoise_document(image)
        metadata["denoise_applied"] = True

    metadata["final_shape"] = list(image.shape)
    return image, metadata
