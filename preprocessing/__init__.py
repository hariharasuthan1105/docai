"""
Document Preprocessing Module.

Provides intelligent image enhancements:
- Skew detection and rotation correction
- Adaptive contrast enhancement (CLAHE)
- Bilateral edge-preserving denoising
- Multi-page PDF rendering and coordinate scaling
"""

from docai.preprocessing.image_enhancer import (
    correct_skew,
    detect_orientation_angle,
    enhance_contrast_adaptive,
    denoise_document,
    preprocess_document_image,
)
from docai.preprocessing.pdf_handler import render_pdf_to_images, PDFPageImage

__all__ = [
    "correct_skew",
    "detect_orientation_angle",
    "enhance_contrast_adaptive",
    "denoise_document",
    "preprocess_document_image",
    "render_pdf_to_images",
    "PDFPageImage",
]
