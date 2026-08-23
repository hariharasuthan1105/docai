"""
PDF Document Handler.

Renders multi-page PDFs to high-resolution RGB/BGR images for OCR and computer vision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pypdfium2 as pdfium

    _PDFIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFIUM_AVAILABLE = False


@dataclass
class PDFPageImage:
    page_number: int  # 1-indexed
    image: np.ndarray  # BGR numpy array
    width: int
    height: int
    scale: float


def render_pdf_to_images(
    pdf_path: str,
    scale: float = 2.0,  # 2.0 corresponds to ~144-200 DPI
    max_pages: Optional[int] = None,
) -> List[PDFPageImage]:
    """
    Render all pages of a PDF file into a list of NumPy BGR images.
    """
    if not _PDFIUM_AVAILABLE:
        raise ImportError(
            "pypdfium2 is required for PDF rendering. "
            "Install with `pip install pypdfium2`."
        )

    pages: List[PDFPageImage] = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        logger.error("Failed to open PDF document at '%s': %s", pdf_path, e)
        raise ValueError(f"Corrupt or invalid PDF file: {pdf_path}") from e

    num_pages = len(pdf)
    pages_to_process = min(num_pages, max_pages) if max_pages else num_pages

    for page_idx in range(pages_to_process):
        page = pdf[page_idx]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        rgb_np = np.array(pil_image)
        # Convert RGB to BGR for OpenCV consistency
        bgr_np = rgb_np[:, :, ::-1].copy() if rgb_np.ndim == 3 else rgb_np

        h, w = bgr_np.shape[:2]
        pages.append(
            PDFPageImage(
                page_number=page_idx + 1,
                image=bgr_np,
                width=w,
                height=h,
                scale=scale,
            )
        )

    return pages
