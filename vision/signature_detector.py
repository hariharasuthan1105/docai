"""
Signature detection interface and stub implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from docai.models.extraction_schema import VisualMark


class SignatureDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> VisualMark:
        """Return whether a dealer signature is present and where."""
        raise NotImplementedError


class StubSignatureDetector(SignatureDetector):
    """
    Placeholder that clearly reports status="not_implemented".
    Does not falsely claim an actual detector ran.
    """

    def detect(self, image_path: str) -> VisualMark:  # noqa: ARG002
        return VisualMark(
            status="not_implemented",
            present=False,
            bbox=None,
            confidence=0.0,
        )


class ManualSignatureDetector(SignatureDetector):
    """Useful for tests: returns a fixed, pre-supplied result."""

    def __init__(
        self,
        present: bool,
        bbox: Optional[List[float]] = None,
        confidence: float = 1.0,
        status: str = "detected",
    ):
        self._result = VisualMark(
            present=present,
            bbox=bbox,
            confidence=confidence,
            status=status if present else "not_detected",
        )

    def detect(self, image_path: str) -> VisualMark:  # noqa: ARG002
        return self._result
