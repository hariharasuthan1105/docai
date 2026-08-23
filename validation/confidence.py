"""
Confidence Calibration, Multi-Signal Scoring & Review Decision Engine.

Implements:
- Multi-signal confidence calculation (OCR score + method weight + spatial layout + validation)
- Probability calibration (Platt scaling & Isotonic regression when ground truth is provided)
- Expected Calibration Error (ECE) and Brier Score metrics
- Human-in-the-loop review decisioning with transparent audit reasons
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docai.config import AUTO_APPROVE_THRESHOLD, FIELD_WEIGHTS
from docai.models.extraction_schema import FieldValue, ReviewDecision

logger = logging.getLogger(__name__)

# Method reliability weights
METHOD_WEIGHTS: Dict[str, float] = {
    "layout_kv": 1.00,
    "layout_kv_inline": 1.00,
    "layout_kv_right": 0.96,
    "layout_kv_below": 0.92,
    "exact_match": 0.98,
    "exact_normalized": 0.98,
    "regex": 0.92,
    "regex_hp": 0.95,
    "regex_cost": 0.92,
    "regex_invoice_num": 0.90,
    "regex_date": 0.91,
    "regex_phone": 0.92,
    "regex_reg_no": 0.89,
    "regex_serial_no": 0.88,
    "fuzzy_candidate": 0.88,
    "fuzzy_ocr_scan": 0.82,
    "model_candidate_match": 0.90,
    "model_ocr_scan": 0.85,
    "regex_numeric_fallback": 0.80,
    "rule_default": 0.50,
}


def compute_field_confidence(field_value: FieldValue) -> float:
    """
    Compute multi-signal calibrated confidence for a single field.
    Integrates raw confidence, method reliability, and validation status.
    """
    if not field_value.is_present():
        return 0.0

    raw_conf = field_value.confidence
    method = field_value.method or "regex"
    method_factor = METHOD_WEIGHTS.get(method, 0.85)

    # Base score
    score = raw_conf * 0.7 + method_factor * 0.3

    # Validation penalty/bonus
    if field_value.validation_status == "invalid":
        score *= 0.40  # Severe penalty for violating domain rules
    elif field_value.validation_status == "warning":
        score *= 0.80

    return min(1.0, max(0.0, round(score, 4)))


def compute_overall_confidence(fields: Dict[str, FieldValue]) -> float:
    """
    Compute weighted average confidence across all required fields.
    Missing fields contribute 0.0 confidence.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for field_name, weight in FIELD_WEIGHTS.items():
        field_value = fields.get(field_name, FieldValue())
        conf = field_value.confidence if field_value.is_present() else 0.0
        weighted_sum += weight * conf
        total_weight += weight

    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


def make_review_decision(
    overall_confidence: float,
    validation_reasons: List[str],
    fields: Optional[Dict[str, FieldValue]] = None,
    auto_approve_threshold: float = 0.88,
    review_threshold: float = 0.70,
) -> Tuple[ReviewDecision, bool, List[str]]:
    """
    Determine the human-in-the-loop decision with complete explanatory reasoning.
    """
    reasons = list(validation_reasons)

    # Check individual low-confidence fields
    if fields:
        for fname, fval in fields.items():
            if fval.is_present():
                f_conf = compute_field_confidence(fval)
                if f_conf < 0.65:
                    reasons.append(
                        f"Field '{fname}' confidence ({f_conf:.2f}) is critically low (extracted via '{fval.method}')."
                    )

    if overall_confidence >= auto_approve_threshold and not validation_reasons:
        return (ReviewDecision.AUTO_APPROVE, False, reasons)
    elif overall_confidence >= review_threshold:
        if overall_confidence < auto_approve_threshold:
            reasons.append(
                f"Overall confidence ({overall_confidence:.2f}) is below auto-approve threshold ({auto_approve_threshold:.2f})."
            )
        return (ReviewDecision.REVIEW, True, reasons)
    else:
        reasons.append(
            f"Overall confidence ({overall_confidence:.2f}) is below review threshold ({review_threshold:.2f}) - manual inspection required."
        )
        return (ReviewDecision.MANUAL_REVIEW, True, reasons)


def needs_human_review(overall_confidence: float, rule_reasons: List[str]) -> Tuple[bool, List[str]]:
    """Backward-compatible helper."""
    _, required, reasons = make_review_decision(
        overall_confidence,
        rule_reasons,
        auto_approve_threshold=AUTO_APPROVE_THRESHOLD,
    )
    return required, reasons


class ConfidenceCalibrator:
    """
    Probability calibration engine implementing Platt Scaling,
    Isotonic Regression, and calibration error evaluation (ECE, Brier Score).
    """

    def __init__(self, method: str = "platt"):
        self.method = method
        self._is_fitted = False
        self._model: Any = None

    def fit(self, heuristic_scores: np.ndarray, ground_truth_labels: np.ndarray) -> None:
        """
        Fit calibration model on validation set.
        heuristic_scores: 1D array of uncalibrated confidence scores in [0, 1].
        ground_truth_labels: 1D binary array (1 = correct, 0 = incorrect).
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.isotonic import IsotonicRegression

            if self.method == "isotonic":
                self._model = IsotonicRegression(out_of_bounds="clip")
                self._model.fit(heuristic_scores, ground_truth_labels)
            else:
                self._model = LogisticRegression(C=1.0, max_iter=200)
                self._model.fit(heuristic_scores.reshape(-1, 1), ground_truth_labels)

            self._is_fitted = True
        except Exception as e:
            logger.error("Failed to fit confidence calibrator: %s", e)

    def calibrate(self, score: float) -> float:
        """Calibrate a single confidence score to an empirical probability."""
        if not self._is_fitted or self._model is None:
            return score  # Return heuristic when uncalibrated

        if self.method == "isotonic":
            cal = float(self._model.predict(np.array([score]))[0])
        else:
            cal = float(self._model.predict_proba(np.array([[score]]))[0, 1])

        return min(1.0, max(0.0, round(cal, 4)))

    @staticmethod
    def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Brier score (mean squared error of probability predictions)."""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        """
        bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            bin_min, bin_max = bin_limits[i], bin_limits[i + 1]
            mask = (y_prob >= bin_min) & (y_prob < bin_max if i < n_bins - 1 else y_prob <= bin_max)
            bin_size = int(np.sum(mask))

            if bin_size > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

        return round(float(ece), 4)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return ConfidenceCalibrator.compute_brier_score(y_true, y_prob)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    return ConfidenceCalibrator.compute_ece(y_true, y_prob, n_bins=n_bins)

