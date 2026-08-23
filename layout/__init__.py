"""
Document Layout Intelligence Package.

Provides 2D spatial indexing, reading order reconstruction,
and geometric Label -> Value association.
"""

from docai.layout.spatial_index import SpatialIndex, BoundingBox
from docai.layout.kv_extractor import LayoutKVExtractor, KeyValuePair

__all__ = [
    "SpatialIndex",
    "BoundingBox",
    "LayoutKVExtractor",
    "KeyValuePair",
]
