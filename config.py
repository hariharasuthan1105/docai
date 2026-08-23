"""
Generic pipeline configuration for Document AI.

All domain-specific catalogs (dealer_master, model_master) and
validation ranges (HORSE_POWER_RANGE, ASSET_COST_RANGE) have been
moved into the YAML schema files (schemas/tractor_invoice.yaml, etc.).

This module now contains only domain-agnostic pipeline settings.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

# Global auto-approval confidence threshold (can be overridden per-schema)
AUTO_APPROVE_THRESHOLD: float = 0.85


def load_custom_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load optional pipeline-level configuration from a JSON file.

    Returns a dict with any overrides the caller cares about.
    Supported keys: "auto_approve_threshold", "ocr_lang", etc.
    """
    if not config_path or not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
