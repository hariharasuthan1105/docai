"""
Enhanced PaddleOCR Engine with Preprocessing & Multi-page Geometry Tracking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure stable execution on Windows CPU by avoiding oneDNN PIR instruction bug
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from docai.preprocessing.image_enhancer import load_image, preprocess_document_image
from docai.preprocessing.pdf_handler import render_pdf_to_images

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PADDLE_AVAILABLE = False


@dataclass
class OCRLine:
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float
    page: int = 1


@dataclass
class OCRResult:
    lines: List[OCRLine] = field(default_factory=list)
    page_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    full_text_override: Optional[str] = None
    confidence: float = 1.0

    @property
    def full_text(self) -> str:
        if self.full_text_override is not None:
            return self.full_text_override
        return "\n".join(line.text for line in self.lines)

    def __init__(
        self,
        lines: Optional[List[OCRLine]] = None,
        page_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        full_text: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.lines = lines if lines is not None else []
        self.page_count = page_count
        self.metadata = metadata if metadata is not None else {}
        self.full_text_override = full_text
        self.confidence = confidence

    def get_page_lines(self, page_num: int) -> List[OCRLine]:
        return [line for line in self.lines if line.page == page_num]


class OCREngine:
    """
    Production-ready OCR engine with intelligent preprocessing and multi-page support.
    """

    def __init__(
        self,
        lang: str = "en",
        use_textline_orientation: bool = True,
        enable_preprocessing: bool = True,
        use_angle_cls: Optional[bool] = None,
    ):
        if not _PADDLE_AVAILABLE:
            raise ImportError(
                "paddleocr is required for OCREngine. "
                "Install with `pip install paddleocr paddlepaddle`."
            )
        self.lang = lang
        self.enable_preprocessing = enable_preprocessing
        orientation_flag = use_textline_orientation if use_angle_cls is None else use_angle_cls

        try:
            self._ocr = PaddleOCR(
                use_textline_orientation=orientation_flag,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=lang,
            )
        except (TypeError, ValueError):
            try:
                self._ocr = PaddleOCR(use_textline_orientation=orientation_flag, lang=lang)
            except (TypeError, ValueError):
                try:
                    self._ocr = PaddleOCR(use_angle_cls=orientation_flag, lang=lang)
                except (TypeError, ValueError):
                    self._ocr = PaddleOCR(lang=lang)

    def _call_ocr(self, img_or_path: Any) -> Any:
        """Robustly call PaddleOCR across different library versions."""
        try:
            return self._ocr.ocr(img_or_path)
        except (TypeError, Exception):
            try:
                return self._ocr.predict(img_or_path)
            except Exception:
                return self._ocr.ocr(img_or_path, cls=True)

    def _parse_raw_ocr(self, raw: Any, page: int = 1) -> List[OCRLine]:
        lines: List[OCRLine] = []
        if not raw:
            return lines

        def parse_dict_like(page_dict: Any) -> List[OCRLine]:
            parsed: List[OCRLine] = []
            texts = (
                page_dict.get("rec_texts")
                or page_dict.get("rec_text")
                or page_dict.get("texts")
                or []
            )
            scores = (
                page_dict.get("rec_scores")
                or page_dict.get("rec_score")
                or page_dict.get("scores")
                or []
            )
            boxes = (
                page_dict.get("rec_polys")
                or page_dict.get("rec_boxes")
                or page_dict.get("dt_polys")
                or page_dict.get("dt_boxes")
                or page_dict.get("boxes")
                or []
            )
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 1.0
                box = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                x_coords = [p[0] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                y_coords = [p[1] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                if not x_coords or not y_coords:
                    bbox = (0.0, 0.0, 0.0, 0.0)
                else:
                    bbox = (
                        float(min(x_coords)),
                        float(min(y_coords)),
                        float(max(x_coords)),
                        float(max(y_coords)),
                    )
                parsed.append(OCRLine(text=str(text), bbox=bbox, confidence=score, page=page))
            return parsed

        if isinstance(raw, (list, tuple)):
            for p_elem in raw:
                if not p_elem:
                    continue
                if isinstance(p_elem, dict) or (hasattr(p_elem, "get") and callable(getattr(p_elem, "get"))):
                    lines.extend(parse_dict_like(p_elem))
                elif isinstance(p_elem, list):
                    for detection in p_elem:
                        if isinstance(detection, (list, tuple)) and len(detection) == 2:
                            box, text_info = detection
                            if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                                text, conf = text_info
                            else:
                                text, conf = str(text_info), 1.0
                            if hasattr(box, "tolist"):
                                box = box.tolist()
                            x_coords = [p[0] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            y_coords = [p[1] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                            if not x_coords or not y_coords:
                                bbox = (0.0, 0.0, 0.0, 0.0)
                            else:
                                bbox = (
                                    float(min(x_coords)),
                                    float(min(y_coords)),
                                    float(max(x_coords)),
                                    float(max(y_coords)),
                                )
                            lines.append(OCRLine(text=str(text), bbox=bbox, confidence=float(conf), page=page))
        elif isinstance(raw, dict) or (hasattr(raw, "get") and callable(getattr(raw, "get"))):
            lines.extend(parse_dict_like(raw))

        return lines

    def extract_text(self, input_path: str) -> OCRResult:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Text file
        if input_path.lower().endswith(".txt"):
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = [
                OCRLine(text=line, bbox=(0.0, 0.0, 0.0, 0.0), confidence=1.0, page=1)
                for line in content.splitlines()
                if line.strip()
            ]
            return OCRResult(lines=lines, page_count=1, metadata={"source_type": "text"})

        # Multi-page PDF
        if input_path.lower().endswith(".pdf"):
            try:
                pages = render_pdf_to_images(input_path)
                all_lines: List[OCRLine] = []
                preprocess_logs = []

                for p_img in pages:
                    img_to_ocr = p_img.image
                    if self.enable_preprocessing:
                        img_to_ocr, meta = preprocess_document_image(img_to_ocr)
                        preprocess_logs.append(meta)

                    raw = self._call_ocr(img_to_ocr)
                    all_lines.extend(self._parse_raw_ocr(raw, page=p_img.page_number))

                return OCRResult(
                    lines=all_lines,
                    page_count=len(pages),
                    metadata={"source_type": "pdf", "pages": len(pages), "preprocessing": preprocess_logs},
                )
            except Exception as e:
                logger.warning("PDF rendering fallback due to: %s", e)

        # Image file (PNG / JPG / TIFF)
        img_np = load_image(input_path)
        meta = {}
        if self.enable_preprocessing:
            img_np, meta = preprocess_document_image(img_np)

        raw = self._call_ocr(img_np)
        lines = self._parse_raw_ocr(raw, page=1)
        return OCRResult(
            lines=lines,
            page_count=1,
            metadata={"source_type": "image", "preprocessing": meta},
        )

    def run(self, input_path: str) -> OCRResult:
        return self.extract_text(input_path)


class StubOCREngine:
    """Drop-in mock OCR engine for fast testing without PaddleOCR models."""

    def __init__(self, canned_lines: Optional[List[OCRLine]] = None, canned_text: Optional[str] = None):
        self._canned_lines = canned_lines
        self._canned_text = canned_text

    def extract_text(self, input_path: str) -> OCRResult:
        if self._canned_lines is not None:
            return OCRResult(lines=self._canned_lines, page_count=1)

        text = self._canned_text
        if text is None and os.path.exists(input_path):
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = ""

        lines = [
            OCRLine(text=line, bbox=(0.0, 0.0, 0.0, 0.0), confidence=1.0, page=1)
            for line in (text or "").splitlines()
            if line.strip()
        ]
        return OCRResult(lines=lines, page_count=1)

    def run(self, input_path: str) -> OCRResult:
        return self.extract_text(input_path)


# Alias
PaddleOCREngine = OCREngine
