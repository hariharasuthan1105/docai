"""
Unit tests for Evaluation, Calibration, Ablation and Robustness Modules.
"""

import numpy as np
import pytest

from docai.evaluation.dataset import (
    GroundTruthDocument,
    create_sample_evaluation_dataset,
    load_evaluation_dataset,
)
from docai.evaluation.evaluator import DocumentAIEvaluator
from docai.ocr.paddleocr_engine import OCRLine, StubOCREngine
from docai.pipeline import DocumentAIPipeline
from docai.validation.confidence import ConfidenceCalibrator


def test_confidence_calibrator_ece():
    y_true = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.95, 0.85, 0.4, 0.6, 0.1])

    ece = ConfidenceCalibrator.compute_ece(y_true, y_prob, n_bins=5)
    brier = ConfidenceCalibrator.compute_brier_score(y_true, y_prob)

    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0
    assert isinstance(brier, float)
    assert 0.0 <= brier <= 1.0


def test_confidence_calibrator_platt_scaling():
    calibrator = ConfidenceCalibrator(method="platt")
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])

    calibrator.fit(scores, labels)
    cal_prob = calibrator.calibrate(0.85)

    assert isinstance(cal_prob, float)
    assert 0.0 <= cal_prob <= 1.0


def test_dataset_generation_and_loading(tmp_path):
    data_dir = str(tmp_path / "test_data")
    dataset = create_sample_evaluation_dataset(data_dir)

    assert len(dataset) >= 5
    assert all(isinstance(d, GroundTruthDocument) for d in dataset)

    loaded = load_evaluation_dataset(data_dir)
    assert len(loaded) == len(dataset)


def test_evaluator_metrics(tmp_path):
    data_dir = str(tmp_path / "test_data")
    dataset = create_sample_evaluation_dataset(data_dir)

    # Use stub OCR engine for speed in tests
    sample_doc = dataset[0]
    gt = sample_doc.ground_truth
    lines = [
        OCRLine(text=f"Authorised Dealer: {gt['dealer_name']}", bbox=(10, 10, 300, 30), confidence=0.98),
        OCRLine(text=f"Model: {gt['model_name']}", bbox=(10, 40, 250, 60), confidence=0.98),
        OCRLine(text=f"Horse Power: {gt['horse_power']} HP", bbox=(10, 70, 150, 90), confidence=0.95),
        OCRLine(text=f"Grand Total: Rs. {gt['asset_cost']:,.2f}", bbox=(10, 100, 250, 120), confidence=0.92),
        OCRLine(text=f"Invoice No: {gt['invoice_number']}", bbox=(10, 130, 200, 150), confidence=0.90),
        OCRLine(text=f"Invoice Date: {gt['invoice_date']}", bbox=(10, 160, 200, 180), confidence=0.90),
        OCRLine(text=f"Customer Name: {gt['customer_name']}", bbox=(10, 190, 200, 210), confidence=0.90),
        OCRLine(text=f"Phone: {gt['phone_number']}", bbox=(10, 220, 200, 240), confidence=0.90),
    ]

    pipeline = DocumentAIPipeline(ocr_engine=StubOCREngine(canned_lines=lines))
    evaluator = DocumentAIEvaluator(pipeline=pipeline)
    metrics = evaluator.evaluate_dataset([sample_doc])

    assert metrics.total_documents == 1
    assert metrics.document_level_accuracy == 1.0
    assert "dealer_name" in metrics.field_metrics
    assert metrics.field_metrics["dealer_name"].f1_score == 1.0

    report = evaluator.generate_report_markdown(metrics)
    assert "# Document AI Evaluation Report" in report
