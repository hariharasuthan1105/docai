"""
Master data catalogs and business configuration for Document AI.

Pure deterministic configuration without any external API or LLM dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

# Default master dealer catalog for fuzzy matching
DEFAULT_DEALER_MASTER: List[str] = [
    "Mahindra & Mahindra Ltd.",
    "Mahindra Tractors Ltd.",
    "Mahindra Farm Equipment",
    "Swaraj Tractors Ltd.",
    "Sonalika Tractors Ltd.",
    "Escorts Kubota Ltd.",
    "John Deere India Pvt. Ltd.",
    "Tafe Tractors and Farm Equipment Ltd.",
    "New Holland Agriculture",
    "Eicher Tractors",
    "Force Motors Ltd.",
    "VST Tillers Tractors Ltd.",
    "Preet Tractors Pvt. Ltd.",
    "Indo Farm Equipment Ltd.",
    "Captain Tractors Pvt. Ltd.",
    "Gujarat Tractors Ltd.",
    "Sri Shakthi Tractors",
    "Sri Shakthi Tractors Pvt. Ltd.",
    "ABC Tractors Pvt. Ltd.",
    "ABC Tractors",
]

# Default master model catalog for exact matching
DEFAULT_MODEL_MASTER: List[str] = [
    "Arjun 605",
    "Arjun 605 DI",
    "Arjun Novo 605 DI",
    "Arjun Novo 605 DI-i",
    "Arjun Novo 605 DI-PS",
    "Yuvraj 215",
    "Yuvraj 215 NXT",
    "Mahindra Yuvo Tech+",
    "Mahindra Yuvo Tech",
    "Mahindra Yuvo 575 DI",
    "Mahindra Yuvo 415 DI",
    "Mahindra Yuvo 475 DI",
    "Mahindra Yuvo 585 DI",
    "Yuvo Tech+",
    "Model XYZ",
    "Swaraj 744",
    "Swaraj 744 FE",
    "Swaraj 744 XT",
    "Swaraj 855",
    "Swaraj 855 FE",
    "Swaraj 735 FE",
    "Mahindra 575 DI",
    "Mahindra 575 DI XP Plus",
    "Mahindra 275 DI",
    "Mahindra 275 DI TU",
    "Mahindra 475 DI",
    "Mahindra Jivo 245 DI",
    "Mahindra OJA 3140",
    "John Deere 5050 D",
    "John Deere 5310",
    "Sonalika DI 745 III",
    "Sonalika Sikander RX 50",
    "Eicher 380",
    "Eicher 485",
    "Eicher 557",
    "Farmtrac 60",
    "Powertrac Euro 50",
    "Kubota MU4501",
]

# Business rules validation ranges
HORSE_POWER_RANGE: Tuple[float, float] = (10.0, 150.0)
ASSET_COST_RANGE: Tuple[float, float] = (50_000.0, 50_000_000.0)  # 50k to 5 Crore

REQUIRED_FIELDS: List[str] = ["dealer_name", "model_name", "horse_power", "asset_cost"]

FIELD_WEIGHTS: Dict[str, float] = {
    "dealer_name": 0.25,
    "model_name": 0.25,
    "horse_power": 0.25,
    "asset_cost": 0.25,
}

AUTO_APPROVE_THRESHOLD: float = 0.85


def load_custom_config(config_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Load dealer and model master lists from an optional JSON config file.
    """
    if not config_path or not os.path.exists(config_path):
        return DEFAULT_DEALER_MASTER, DEFAULT_MODEL_MASTER

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            dealers = data.get("dealers", DEFAULT_DEALER_MASTER)
            models = data.get("models", DEFAULT_MODEL_MASTER)
            return dealers, models
    except Exception:
        return DEFAULT_DEALER_MASTER, DEFAULT_MODEL_MASTER
