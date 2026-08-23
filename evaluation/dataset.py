"""
Evaluation Dataset Generator and Loader.

Creates standardized sample documents (PDF/images/text) with exact
ground-truth annotations for reproducibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class GroundTruthDocument:
    doc_id: str
    file_path: str
    ground_truth: Dict[str, Any]


SAMPLE_TEMPLATES = [
    {
        "doc_id": "sample_01",
        "dealer_name": "Mahindra & Mahindra Ltd.",
        "model_name": "Arjun Novo 605 DI",
        "horse_power": 50.0,
        "asset_cost": 550000.0,
        "invoice_number": "INV-2026-001",
        "invoice_date": "15/02/2026",
        "customer_name": "Ramesh Kumar",
        "phone_number": "9876543210",
        "has_signature": True,
        "has_stamp": True,
    },
    {
        "doc_id": "sample_02",
        "dealer_name": "Swaraj Tractors Ltd.",
        "model_name": "Swaraj 744 FE",
        "horse_power": 48.0,
        "asset_cost": 680000.0,
        "invoice_number": "SW-78901",
        "invoice_date": "20/01/2026",
        "customer_name": "Suresh Patel",
        "phone_number": "9123456780",
        "has_signature": True,
        "has_stamp": False,
    },
    {
        "doc_id": "sample_03",
        "dealer_name": "John Deere India Pvt. Ltd.",
        "model_name": "John Deere 5050 D",
        "horse_power": 50.0,
        "asset_cost": 790000.0,
        "invoice_number": "JD/2026/88",
        "invoice_date": "05/03/2026",
        "customer_name": "Balwant Singh",
        "phone_number": "9845012345",
        "has_signature": False,
        "has_stamp": True,
    },
    {
        "doc_id": "sample_04",
        "dealer_name": "Escorts Kubota Ltd.",
        "model_name": "Farmtrac 60",
        "horse_power": 55.0,
        "asset_cost": 720000.0,
        "invoice_number": "EK-5541",
        "invoice_date": "12/02/2026",
        "customer_name": "Manpreet Kaur",
        "phone_number": "9765432109",
        "has_signature": True,
        "has_stamp": True,
    },
    {
        "doc_id": "sample_05",
        "dealer_name": "Sonalika Tractors Ltd.",
        "model_name": "Sonalika Sikander RX 50",
        "horse_power": 50.0,
        "asset_cost": 640000.0,
        "invoice_number": "SON-9921",
        "invoice_date": "28/02/2026",
        "customer_name": "Vikram Rathore",
        "phone_number": "9811223344",
        "has_signature": True,
        "has_stamp": False,
    },
]


def render_synthetic_invoice_image(meta: Dict[str, Any], output_path: str) -> None:
    """Render a realistic invoice PNG with simulated text, stamps, and signatures."""
    w, h = 1000, 1400
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header Box
    draw.rectangle([40, 40, w - 40, 160], outline=(0, 0, 0), width=2)
    draw.text((60, 60), f"TAX INVOICE / QUOTATION", fill=(0, 0, 0))
    draw.text((60, 95), f"Authorised Dealer: {meta['dealer_name']}", fill=(0, 0, 0))
    draw.text((60, 125), f"Phone: {meta['phone_number']}", fill=(0, 0, 0))

    # Invoice Details Grid
    draw.rectangle([40, 180, w - 40, 290], outline=(0, 0, 0), width=1)
    draw.text((60, 200), f"Invoice No: {meta['invoice_number']}", fill=(0, 0, 0))
    draw.text((550, 200), f"Invoice Date: {meta['invoice_date']}", fill=(0, 0, 0))
    draw.text((60, 240), f"Customer Name: {meta['customer_name']}", fill=(0, 0, 0))

    # Equipment Table Header
    draw.rectangle([40, 310, w - 40, 360], fill=(240, 240, 240), outline=(0, 0, 0), width=1)
    draw.text((60, 325), "Sl No.", fill=(0, 0, 0))
    draw.text((160, 325), "Description of Equipment / Model", fill=(0, 0, 0))
    draw.text((600, 325), "Horse Power", fill=(0, 0, 0))
    draw.text((780, 325), "Total Amount (Rs.)", fill=(0, 0, 0))

    # Equipment Table Row
    draw.rectangle([40, 360, w - 40, 600], outline=(0, 0, 0), width=1)
    draw.text((60, 400), "01", fill=(0, 0, 0))
    draw.text((160, 400), f"{meta['model_name']} Tractor", fill=(0, 0, 0))
    draw.text((600, 400), f"{meta['horse_power']} HP", fill=(0, 0, 0))
    draw.text((780, 400), f"{meta['asset_cost']:,.2f}", fill=(0, 0, 0))

    # Total Box
    draw.rectangle([40, 600, w - 40, 660], fill=(245, 245, 245), outline=(0, 0, 0), width=1)
    draw.text((550, 620), "Grand Total (Rs.):", fill=(0, 0, 0))
    draw.text((780, 620), f"{meta['asset_cost']:,.2f}", fill=(0, 0, 0))

    # Signatures & Stamps in Footer
    draw.text((100, 1150), "Customer Signature", fill=(100, 100, 100))
    draw.text((650, 1150), "Authorised Dealer Sign & Seal", fill=(100, 100, 100))

    # Add simulated stamp if requested
    img_np = np.array(img)
    if meta.get("has_stamp"):
        # Draw blue circular stamp
        cv2.circle(img_np, (750, 1260), 65, (210, 45, 40), thickness=3)
        cv2.circle(img_np, (750, 1260), 55, (210, 45, 40), thickness=1)
        cv2.putText(
            img_np, "OFFICIAL SEAL", (705, 1265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 45, 40), 1
        )

    # Add simulated signature if requested
    if meta.get("has_signature"):
        # Draw cursive pen stroke curves
        pts = np.array(
            [[670, 1220], [700, 1200], [730, 1240], [760, 1210], [800, 1230], [840, 1215]],
            np.int32,
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_np, [pts], isClosed=False, color=(20, 20, 180), thickness=2)

    cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


def create_sample_evaluation_dataset(base_dir: str = "data") -> List[GroundTruthDocument]:
    """
    Generate synthetic sample invoice files and corresponding ground-truth JSON files.
    """
    samples_dir = os.path.join(base_dir, "samples")
    gt_dir = os.path.join(base_dir, "ground_truth")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    documents: List[GroundTruthDocument] = []

    for tpl in SAMPLE_TEMPLATES:
        doc_id = tpl["doc_id"]
        img_path = os.path.join(samples_dir, f"{doc_id}.png")
        gt_path = os.path.join(gt_dir, f"{doc_id}.json")

        render_synthetic_invoice_image(tpl, img_path)

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(tpl, f, indent=2)

        documents.append(
            GroundTruthDocument(
                doc_id=doc_id,
                file_path=img_path,
                ground_truth=tpl,
            )
        )

    return documents


def load_evaluation_dataset(base_dir: str = "data") -> List[GroundTruthDocument]:
    """
    Load ground-truth documents from directory.
    If directory is missing or empty, creates the dataset automatically.
    """
    samples_dir = os.path.join(base_dir, "samples")
    gt_dir = os.path.join(base_dir, "ground_truth")

    if not os.path.exists(samples_dir) or not os.listdir(samples_dir):
        return create_sample_evaluation_dataset(base_dir)

    docs: List[GroundTruthDocument] = []
    for f in os.listdir(gt_dir):
        if f.endswith(".json"):
            doc_id = os.path.splitext(f)[0]
            gt_path = os.path.join(gt_dir, f)
            with open(gt_path, "r", encoding="utf-8") as fp:
                gt_data = json.load(fp)

            # Look for corresponding sample image or pdf
            sample_path = os.path.join(samples_dir, f"{doc_id}.png")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(samples_dir, f"{doc_id}.pdf")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(samples_dir, f"{doc_id}.txt")

            if os.path.exists(sample_path):
                docs.append(
                    GroundTruthDocument(
                        doc_id=doc_id,
                        file_path=sample_path,
                        ground_truth=gt_data,
                    )
                )

    return docs if docs else create_sample_evaluation_dataset(base_dir)
