"""
Unit tests for Computer Vision Signature and Stamp Detectors.
"""

import cv2
import numpy as np
import pytest

from docai.vision.signature_detector import CVSignatureDetector, ManualSignatureDetector, StubSignatureDetector
from docai.vision.stamp_detector import CVStampDetector, ManualStampDetector, StubStampDetector


def test_stub_detectors():
    sig_stub = StubSignatureDetector()
    stamp_stub = StubStampDetector()

    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res_sig = sig_stub.detect(img)
    res_stamp = stamp_stub.detect(img)

    assert res_sig.status == "not_implemented"
    assert res_stamp.status == "not_implemented"
    assert not res_sig.present
    assert not res_stamp.present


def test_manual_detectors():
    sig_manual = ManualSignatureDetector(present=True, bbox=[10, 20, 50, 40], confidence=0.95)
    stamp_manual = ManualStampDetector(present=False, confidence=0.1)

    img = np.ones((100, 100, 3), dtype=np.uint8)
    res_sig = sig_manual.detect(img)
    res_stamp = stamp_manual.detect(img)

    assert res_sig.present
    assert res_sig.confidence == 0.95
    assert not res_stamp.present


def test_cv_stamp_detector_on_blank():
    detector = CVStampDetector()
    blank_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    res = detector.detect(blank_img)

    assert not res.present
    assert res.mark_type == "stamp"


def test_cv_stamp_detector_on_simulated_stamp():
    detector = CVStampDetector(min_stamp_area=100.0)
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    # Draw blue circular stamp
    cv2.circle(img, (300, 300), 50, (200, 50, 40), thickness=4)
    cv2.putText(img, "VERIFIED", (270, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 50, 40), 1)

    res = detector.detect(img)
    assert res.present
    assert res.confidence > 0.60
    assert res.bbox is not None


def test_cv_signature_detector_on_blank():
    detector = CVSignatureDetector()
    blank_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    res = detector.detect(blank_img)

    assert not res.present
    assert res.mark_type == "signature"


def test_cv_signature_detector_on_simulated_strokes():
    detector = CVSignatureDetector(min_area=100.0, region_of_interest_y=0.4)
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255
    # Draw cursive curved polyline in bottom region
    pts = np.array([[200, 600], [230, 570], [260, 620], [290, 580], [330, 610]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 180), thickness=2)

    res = detector.detect(img)
    assert res.present
    assert res.confidence > 0.50
    assert res.bbox is not None
