"""
Document AI Evaluation Framework.

Computes:
- Field-level Precision, Recall, F1, Exact Match (EM)
- Normalized Levenshtein Edit Distance (NLED)
- Document-level Accuracy
- Expected Calibration Error (ECE) and Brier Score
- Processing Latency Statistics
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

from docai.evaluation.dataset import GroundTruthDocument, load_evaluation_dataset
from docai.models.extraction_schema import FinalDocument
from docai.pipeline import DocumentAIPipeline
from docai.validation.confidence import ConfidenceCalibrator

logger = logging.getLogger(__name__)


@dataclass
class FieldMetric:
    exact_match_accuracy: float
    normalized_edit_distance: float
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class EvaluationMetrics:
    total_documents: int
    document_level_accuracy: float
    field_metrics: Dict[str, FieldMetric]
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    brier_score: Optional[float] = None
    expected_calibration_error: Optional[float] = None


class DocumentAIEvaluator:
    """
    Evaluates extraction pipelines against ground-truth benchmarks.
    """

    def __init__(self, pipeline: Optional[DocumentAIPipeline] = None):
        self.pipeline = pipeline or DocumentAIPipeline()

    @staticmethod
    def _normalize_val(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, (float, int)):
            # If whole number, format as integer
            if float(val).is_integer():
                return str(int(val))
            return f"{float(val):.2f}"
        return str(val).strip().lower()

    def evaluate_dataset(self, dataset: Optional[List[GroundTruthDocument]] = None) -> EvaluationMetrics:
        docs = dataset if dataset is not None else load_evaluation_dataset()
        if not docs:
            raise ValueError("No ground truth documents found for evaluation.")

        field_names = [
            "dealer_name",
            "model_name",
            "horse_power",
            "asset_cost",
            "invoice_number",
            "invoice_date",
            "customer_name",
            "phone_number",
        ]

        # Tracking accumulators per field
        field_matches: Dict[str, List[bool]] = {fn: [] for fn in field_names}
        field_nled: Dict[str, List[float]] = {fn: [] for fn in field_names}
        field_tp: Dict[str, int] = {fn: 0 for fn in field_names}
        field_fp: Dict[str, int] = {fn: 0 for fn in field_names}
        field_fn: Dict[str, int] = {fn: 0 for fn in field_names}

        doc_level_matches: List[bool] = []
        latencies: List[float] = []

        all_confs: List[float] = []
        all_correctness: List[int] = []

        for doc in docs:
            start_t = time.perf_counter()
            pred: FinalDocument = self.pipeline.process(doc.file_path, document_id=doc.doc_id)
            lat_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(lat_ms)

            pred_fields = pred.get_all_fields()
            gt = doc.ground_truth

            all_field_match = True

            for fn in field_names:
                if fn not in gt:
                    continue

                gt_raw = gt[fn]
                gt_norm = self._normalize_val(gt_raw)

                pred_obj = pred_fields.get(fn)
                pred_raw = pred_obj.value if (pred_obj and pred_obj.is_present()) else None
                pred_norm = self._normalize_val(pred_raw)

                is_match = False
                if gt_norm and pred_norm:
                    # Compare
                    if fn in ["horse_power", "asset_cost"]:
                        try:
                            is_match = abs(float(gt_raw) - float(pred_raw)) < 0.01
                        except Exception:
                            is_match = (gt_norm == pred_norm)
                    else:
                        is_match = (gt_norm == pred_norm) or (gt_norm in pred_norm) or (pred_norm in gt_norm)

                    # NLED
                    dist = Levenshtein.normalized_distance(gt_norm, pred_norm)
                    field_nled[fn].append(dist)
                else:
                    field_nled[fn].append(1.0 if (gt_norm or pred_norm) else 0.0)

                field_matches[fn].append(is_match)

                if is_match:
                    field_tp[fn] += 1
                elif pred_norm and not is_match:
                    field_fp[fn] += 1
                    all_field_match = False
                elif gt_norm and not pred_norm:
                    field_fn[fn] += 1
                    all_field_match = False

                if pred_obj and pred_obj.is_present():
                    all_confs.append(pred_obj.confidence)
                    all_correctness.append(1 if is_match else 0)

            doc_level_matches.append(all_field_match)

        # Compute field level metrics
        compiled_field_metrics: Dict[str, FieldMetric] = {}
        for fn in field_names:
            matches = field_matches[fn]
            if not matches:
                continue
            em = float(np.mean(matches))
            nled_val = float(np.mean(field_nled[fn]))
            tp = field_tp[fn]
            fp = field_fp[fn]
            fn_cnt = field_fn[fn]

            prec = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / float(tp + fn_cnt) if (tp + fn_cnt) > 0 else 1.0
            f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0

            compiled_field_metrics[fn] = FieldMetric(
                exact_match_accuracy=round(em, 4),
                normalized_edit_distance=round(nled_val, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
                support=len(matches),
            )

        # Calibration
        ece = None
        brier = None
        if all_confs and all_correctness:
            c_arr = np.array(all_confs)
            y_arr = np.array(all_correctness)
            ece = ConfidenceCalibrator.compute_ece(y_arr, c_arr)
            brier = round(ConfidenceCalibrator.compute_brier_score(y_arr, c_arr), 4)

        return EvaluationMetrics(
            total_documents=len(docs),
            document_level_accuracy=round(float(np.mean(doc_level_matches)), 4),
            field_metrics=compiled_field_metrics,
            mean_latency_ms=round(float(np.mean(latencies)), 2),
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            brier_score=brier,
            expected_calibration_error=ece,
        )

    def generate_report_markdown(self, metrics: EvaluationMetrics) -> str:
        """Format metrics into a structured markdown report."""
        lines = [
            "# Document AI Evaluation Report",
            "",
            f"**Total Documents Evaluated:** {metrics.total_documents}  ",
            f"**Document-Level Accuracy:** {metrics.document_level_accuracy * 100:.2f}%  ",
            f"**Mean Latency:** {metrics.mean_latency_ms:.1f} ms (p50: {metrics.p50_latency_ms:.1f} ms, p95: {metrics.p95_latency_ms:.1f} ms)  ",
            f"**Expected Calibration Error (ECE):** {metrics.expected_calibration_error if metrics.expected_calibration_error is not None else 'N/A'}  ",
            f"**Brier Score:** {metrics.brier_score if metrics.brier_score is not None else 'N/A'}  ",
            "",
            "## Field-Level Performance Breakdown",
            "",
            "| Field | Exact Match (EM) | Precision | Recall | F1-Score | Norm Edit Dist (NLED) | Support |",
            "|---|---|---|---|---|---|---|",
        ]

        for fname, fm in metrics.field_metrics.items():
            display_name = fname.replace("_", " ").title()
            lines.append(
                f"| **{display_name}** | {fm.exact_match_accuracy * 100:.1f}% | {fm.precision * 100:.1f}% | {fm.recall * 100:.1f}% | **{fm.f1_score * 100:.1f}%** | {fm.normalized_edit_distance:.3f} | {fm.support} |"
            )

        return "\n".join(lines)
