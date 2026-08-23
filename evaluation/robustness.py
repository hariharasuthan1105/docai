"""
Robustness Benchmark for Document AI.

Evaluates system resilience under realistic document degradations:
- Gaussian Blur
- Additive Gaussian Noise
- Rotations (90, 180, 270 degrees)
- Low Contrast / Shadowing
- Downscaling / Low Resolution
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from docai.evaluation.dataset import GroundTruthDocument, load_evaluation_dataset
from docai.evaluation.evaluator import DocumentAIEvaluator
from docai.pipeline import DocumentAIPipeline

logger = logging.getLogger(__name__)


@dataclass
class RobustnessMetric:
    condition_name: str
    description: str
    doc_accuracy: float
    avg_f1: float
    accuracy_drop: float


def apply_degradation(image: np.ndarray, condition: str) -> np.ndarray:
    """Apply synthetic degradation to an image array."""
    h, w = image.shape[:2]

    if condition == "clean":
        return image.copy()
    elif condition == "blur":
        return cv2.GaussianBlur(image, (9, 9), sigmaX=3.0)
    elif condition == "noise":
        noise = np.random.normal(0, 20, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    elif condition == "rotation_90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif condition == "rotation_180":
        return cv2.rotate(image, cv2.ROTATE_180)
    elif condition == "low_contrast":
        # Darken / reduce dynamic range
        return np.clip(image.astype(np.float32) * 0.45 + 50, 0, 255).astype(np.uint8)
    elif condition == "downscale":
        down = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        return image.copy()


def run_robustness_benchmark(dataset: Optional[List[GroundTruthDocument]] = None) -> List[RobustnessMetric]:
    """
    Execute robustness benchmark across synthetic image degradations.
    """
    docs = dataset if dataset is not None else load_evaluation_dataset()
    pipeline = DocumentAIPipeline(enable_preprocessing=True)
    evaluator = DocumentAIEvaluator(pipeline=pipeline)

    conditions = [
        ("clean", "Baseline (Clean scans)"),
        ("blur", "Gaussian Blur (Defocus / Camera blur)"),
        ("noise", "Additive Noise (Sensor grain / Low light)"),
        ("rotation_90", "90-degree Rotation"),
        ("low_contrast", "Low Contrast / Underexposure"),
        ("downscale", "50% Downscaling (Low-res camera photo)"),
    ]

    results: List[RobustnessMetric] = []
    baseline_acc = 0.0

    for idx, (cond_key, cond_desc) in enumerate(conditions):
        degraded_docs: List[GroundTruthDocument] = []
        temp_files: List[str] = []

        try:
            for d in docs:
                img = cv2.imread(d.file_path)
                if img is None:
                    continue
                deg_img = apply_degradation(img, cond_key)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    cv2.imwrite(tmp.name, deg_img)
                    temp_files.append(tmp.name)
                    degraded_docs.append(
                        GroundTruthDocument(
                            doc_id=f"{d.doc_id}_{cond_key}",
                            file_path=tmp.name,
                            ground_truth=d.ground_truth,
                        )
                    )

            if degraded_docs:
                metrics = evaluator.evaluate_dataset(degraded_docs)
                avg_f1 = float(np.mean([fm.f1_score for fm in metrics.field_metrics.values()]))

                if cond_key == "clean":
                    baseline_acc = metrics.document_level_accuracy
                    drop = 0.0
                else:
                    drop = max(0.0, baseline_acc - metrics.document_level_accuracy)

                results.append(
                    RobustnessMetric(
                        condition_name=cond_key,
                        description=cond_desc,
                        doc_accuracy=metrics.document_level_accuracy,
                        avg_f1=round(avg_f1, 4),
                        accuracy_drop=round(drop, 4),
                    )
                )
        finally:
            for tf in temp_files:
                if os.path.exists(tf):
                    try:
                        os.remove(tf)
                    except Exception:
                        pass

    return results


def format_robustness_markdown(results: List[RobustnessMetric]) -> str:
    """Format robustness results into markdown report."""
    lines = [
        "# Document AI Degradation & Robustness Benchmark",
        "",
        "| Degradation Condition | Description | Doc Accuracy | Macro F1 | Performance Drop |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        drop_str = f"-{r.accuracy_drop * 100:.1f}%" if r.accuracy_drop > 0 else "Baseline"
        lines.append(
            f"| **{r.condition_name}** | {r.description} | **{r.doc_accuracy * 100:.1f}%** | {r.avg_f1 * 100:.1f}% | {drop_str} |"
        )

    return "\n".join(lines)
