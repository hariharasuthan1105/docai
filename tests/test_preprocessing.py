"""
Unit tests for Document Preprocessing & Image Enhancements.
"""

import numpy as np
import pytest

from docai.preprocessing.image_enhancer import (
    correct_skew,
    denoise_document,
    detect_orientation_angle,
    detect_skew_angle,
    enhance_contrast_adaptive,
    load_image,
    preprocess_document_image,
)


def test_load_image_array():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    loaded = load_image(img)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)


def test_load_image_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file.png")


def test_clahe_contrast_enhancement():
    # Low contrast gray image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    enhanced = enhance_contrast_adaptive(img)
    assert enhanced.shape == (200, 200, 3)


def test_denoise_document():
    img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    denoised = denoise_document(img)
    assert denoised.shape == (150, 150, 3)


def test_detect_skew_on_blank():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    angle = detect_skew_angle(img)
    assert isinstance(angle, float)
    assert abs(angle) < 1.0


def test_correct_skew():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    corrected, ang = correct_skew(img, 0.0)
    assert corrected.shape == (200, 200, 3)
    assert ang == 0.0


def test_detect_orientation():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    orient = detect_orientation_angle(img)
    assert orient in [0, 90, 180, 270]


def test_preprocess_document_image():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 200
    res_img, meta = preprocess_document_image(img)
    assert isinstance(res_img, np.ndarray)
    assert "original_shape" in meta
    assert "final_shape" in meta
