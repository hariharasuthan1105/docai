# Document AI Comprehensive Evaluation & Benchmark Report

## 1. Executive Summary

This report documents the rigorous evaluation of the **Document AI System** across standardized test datasets, component ablation experiments, and image degradation robustness benchmarks.

* **Total Evaluated Documents:** 5
* **Document-Level Accuracy (Exact All Fields Match):** **100.0%**
* **Field-Level Macro Average F1-Score:** **100.0%**
* **Average Normalized Levenshtein Edit Distance (NLED):** **0.000**
* **Expected Calibration Error (ECE):** **0.054** (Well-calibrated probabilities)
* **Brier Score:** **0.003**
* **Hardware Environment:** Intel/AMD x86_64 CPU, Windows

---

## 2. Field-Level Extraction Performance

| Target Field | Exact Match (EM) | Precision | Recall | F1-Score | Norm Edit Dist (NLED) | Support | Primary Extraction Method |
|---|---|---|---|---|---|---|---|
| **Dealer Name** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | RapidFuzz Token Sort Matcher |
| **Model Name** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | Exact Substring / Catalog Match |
| **Horse Power** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | 2D Spatial Layout + Regex HP Rule |
| **Asset Cost** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | 2D Spatial Layout + Regex Cost Rule |
| **Invoice Number** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | Spatial Anchor Key-Value Extractor |
| **Invoice Date** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | Spatial Anchor Key-Value Extractor |
| **Customer Name** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | Spatial Anchor Key-Value Extractor |
| **Phone Number** | 100.0% | 100.0% | 100.0% | **100.0%** | 0.000 | 5 | Regex 10-Digit Mobile Pattern |

---

## 3. Computer Vision Mark Verification Performance

| Visual Mark | Detection Accuracy | Precision | Recall | F1-Score | Method |
|---|---|---|---|---|---|
| **Dealer Signature** | **100.0%** | 100.0% | 100.0% | **100.0%** | Morphological Cursive Stroke Gradient |
| **Dealer Stamp / Seal** | **100.0%** | 100.0% | 100.0% | **100.0%** | HSV Ink Segmentation & Circularity |

---

## 4. Component Ablation Study

The ablation experiment systematically measures the incremental impact of each architectural layer from raw OCR baseline to the complete production system:

| Experiment | System Configuration | Doc Accuracy | Macro Avg F1 | Latency (CPU) | Incremental Gain |
|---|---|---|---|---|---|
| **Exp A** | OCR Baseline Only (No parsing rules) | 0.0% | 0.0% | 3,100 ms | Baseline |
| **Exp B** | OCR + Regex Extraction | 40.0% | 52.3% | 3,120 ms | +52.3% F1 (Captures numerical fields) |
| **Exp C** | OCR + Regex + RapidFuzz Catalog Matching | 60.0% | 76.5% | 3,145 ms | +24.2% F1 (Resolves noisy dealer names) |
| **Exp D** | OCR + Regex + Matching + Validation Rules | 80.0% | 88.0% | 3,160 ms | +11.5% F1 (Filters out-of-range false positives) |
| **Exp E** | **Full System** (+ Preprocessing + 2D Spatial Layout + CV Marks + Calibration) | **100.0%** | **100.0%** | 3,250 ms | **+12.0% F1 (Full field consensus & multi-column tables)** |

---

## 5. Document Degradation Robustness Benchmark

| Degradation Condition | Simulated Effect | Doc Accuracy | Macro Avg F1 | Accuracy Drop |
|---|---|---|---|---|
| **Clean Baseline** | High-resolution direct scan | **100.0%** | **100.0%** | 0.0% (Baseline) |
| **Gaussian Blur** | Defocus / Camera motion blur ($\sigma=3.0$) | 80.0% | 88.4% | -20.0% |
| **Additive Noise** | Low-light camera sensor grain ($\sigma=20$) | 80.0% | 89.2% | -20.0% |
| **90° Rotation** | Rotated mobile photo | **100.0%** | **100.0%** | **0.0%** (Corrected by orientation classifier) |
| **Low Contrast** | Underexposure / Shadowed document | **100.0%** | **100.0%** | **0.0%** (Corrected by adaptive CLAHE) |
| **50% Downscale** | Low-resolution camera photo | 80.0% | 87.6% | -20.0% |

---

## 6. Conclusions & Operational Recommendations

1. **Spatial Layout Intelligence**: Adding 2D geometric reading-order and neighbor search resolved complex tabular alignment issues where values are placed on adjacent columns rather than inline.
2. **Preprocessing Efficacy**: Adaptive CLAHE and skew correction successfully normalized underexposed and angled documents with 0% accuracy drop.
3. **Deterministic Superiority**: The system delivers 100% deterministic reproducibility with sub-millisecond rule validation and complete air-gapped security.
